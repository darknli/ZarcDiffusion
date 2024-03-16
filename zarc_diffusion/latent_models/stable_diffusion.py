import warnings
import os
import torch
from torch.nn import functional as F
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTextModel
from zarc_diffusion.utils.utils_model import to_model_device, cast_training_params
from .base import BaseModel, DiffusionTrainer
from zarc_diffusion.utils.calculatron import compute_snr


class StableDiffision(BaseModel):
    """
    封装stable diffusion model的类

    Parameters
    ----------
    config_diffusion : dict. diffusion model的配置
        * pretrained_model_name_or_path: str. 与diffusers一致
        * train_unet: bool, optional. 是否训练unet
        * unet_dtype: torch.dtype, optional. {'fp16', 'bf16', 'fp32'}, 默认fp16
        * train_text_encoder: bool, optional. 是否训练text_encoder
        * text_encoder_dtype: torch.dtype, optional. {'fp16', 'bf16', 'fp32'}, 默认fp16
    config_vae : dict. vae设置
        * pretrained_vae_name_or_path: str, optional. 不做特别设置则保持和`config_diffusion`一致
        * vae_dtype: torch.dtype, optional. {'fp16', 'bf16', 'fp32'}, 默认fp16
    config_scheduler : dict. noise_scheduler设置
        * pretrained_model_name_or_path: str, optional. 不做特别设置则保持和`config_diffusion`一致
    config_lora : dict. lora设置
        * rank: int. lora的rank
    config_adapters : dict. 待实现
    prediction_type : str, default None. 'epsilon' or 'v_prediction' or leave `None`
    snr_gamma: float, default None. 用于加速收敛, https://arxiv.org/abs/2303.09556
    noise_offset: float, default None. 参考https://www.crosslabs.org//blog/diffusion-with-offset-noise
    """
    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict = None,
                 config_scheduler: dict = None,
                 config_lora: dict = None,
                 config_adapters: dict = None,
                 prediction_type: str = None,
                 snr_gamma: float = None,
                 noise_offset: float = None,
                 ):
        self.unet = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        self.lora = None
        super().__init__(config_diffusion, config_vae, config_scheduler, config_lora, config_adapters, prediction_type,
                         snr_gamma, noise_offset)

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        if "unet_dtype" in config:
            if config["unet_dtype"] == "fp16":
                self.unet.to(self.device, torch.float16)
            elif config["unet_dtype"] == "bf16":
                self.unet.to(self.device, torch.bfloat16)
            elif config["unet_dtype"] == "fp32":
                self.unet.to(self.device, torch.float32)
            else:
                warnings.warn("unet_dtype config not in (`fp16`, `bf16`), set to fp16")
                self.unet.to(self.device, torch.float16)
        else:
            self.unet.to(self.device, torch.float16)

        if "text_encoder_dtype" in config:
            if config["text_encoder_dtype"] == "fp16":
                self.text_encoder.to(self.device, torch.float16)
            elif config["text_encoder_dtype"] == "bf16":
                self.text_encoder.to(self.device, torch.bfloat16)
            elif config["text_encoder_dtype"] == "fp32":
                self.text_encoder.to(self.device, torch.float32)
            else:
                warnings.warn("text_encoder_dtype config not in (`fp16`, `bf16`), set to fp16")
                self.text_encoder.to(self.device, torch.float16)
        else:
            self.text_encoder.to(self.device, self.unet.dtype)

        # freeze unet
        if "train_unet" not in config or not config["train_unet"]:
            print("freeze unet")
            self.unet.requires_grad_(False)
        else:
            self.trainable_params = cast_training_params(self.unet)

        if "train_text_encoder" not in config or not config["train_text_encoder"]:
            print("freeze text_encoder")
            self.text_encoder.requires_grad_(False)
        else:
            self.trainable_params = cast_training_params(self.text_encoder)

    def init_vae(self, config):
        if config is None:
            config = {}
        if "pretrained_vae_name_or_path" in config:
            # 单独指定vae预训练模型
            pretrained_vae_name_or_path = config["pretrained_vae_name_or_path"]
        else:
            pretrained_vae_name_or_path = self.config_diffusion["pretrained_model_name_or_path"]
            config["pretrained_vae_name_or_path"] = pretrained_vae_name_or_path
            self.config_vae = config
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path, subfolder="vae")
        if "vae_dtype" in config:
            if config["vae_dtype"] == "fp16":
                self.vae.to(self.device, torch.float16)
            elif config["vae_dtype"] == "bf16":
                self.vae.to(self.device, torch.bfloat16)
            elif config["vae_dtype"] == "fp32":
                self.vae.to(self.device, torch.float32)
            else:
                warnings.warn("vae_dtype config not in (`fp16`, `bf16`), set to fp16")
                self.vae.to(self.device, torch.float16)
        else:
            self.vae.to(self.device, torch.float16)
        self.vae.requires_grad_(False)

    def init_scheduler(self, config):
        if config is None:
            config = {}
        if "pretrained_model_name_or_path" not in config:
            pretrained_model_name_or_path = self.config_diffusion["pretrained_model_name_or_path"]
            config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
            self.config_scheduler = config
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    def init_lora(self, config):
        if "train_unet" in self.config_diffusion and self.config_diffusion["train_unet"]:
            raise ValueError("不要既训练unet又训练lora!")
        rank = config["rank"]
        unet_lora_parameters = []
        for attn_processor_name, attn_processor in self.unet.attn_processors.items():
            # Parse the attention module.
            attn_module = self.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Set the `lora_layer` attribute of the attention-related matrices.
            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_q.in_features, out_features=attn_module.to_q.out_features, rank=rank
                ).to(self.unet.device)
            )
            attn_module.to_k.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_k.in_features, out_features=attn_module.to_k.out_features, rank=rank
                ).to(self.unet.device)
            )

            attn_module.to_v.set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_v.in_features, out_features=attn_module.to_v.out_features, rank=rank
                ).to(self.unet.device)
            )
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=rank,
                ).to(self.unet.device)
            )

            # Accumulate the LoRA params to optimize.
            unet_lora_parameters.extend(attn_module.to_q.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_k.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_v.lora_layer.parameters())
            unet_lora_parameters.extend(attn_module.to_out[0].lora_layer.parameters())
        self.lora = unet_lora_parameters
        self.trainable_params.extend(unet_lora_parameters)

    def forward(self, batch):
        latents = self.vae.encode(batch["image_origin"].to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        latents = latents.to(dtype=self.unet.dtype)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]

        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.noise_scheduler.register_to_config(prediction_type=self.prediction_type)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.unet.dtype)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler.alphas_cumprod, timesteps)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                    torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss


class SDTrainer(DiffusionTrainer):
    def save_checkpoint(self, file_name: str, save_single_model: bool = True,
                        print_info: bool = False) -> None:
        """注意，file_name是目录不再是文件"""
        save_path = os.path.join(self.ckpt_dir, os.path.splitext(file_name)[0])
        os.makedirs(save_path, exist_ok=True)
        if self.model.lora is not None:
            self.model.unet.save_attn_procs(save_path,
                                            weight_name="lora.safetensors")
        else:
            if self.model.text_encoder.__class__.__name__ == "DistributedDataParallel":
                text_encoder = self.accelerator.unwrap_model(self.model.text_encoder)
            else:
                text_encoder = self.model.text_encoder

            if self.model.unet.__class__.__name__ == "DistributedDataParallel":
                unet = self.accelerator.unwrap_model(self.model.unet)
            else:
                unet = self.model.unet

            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                text_encoder=text_encoder,
                vae=self.model.vae,
                unet=unet,
            )
            pipeline.save_pretrained(os.path.join(save_path, "stable_diffusion"))
