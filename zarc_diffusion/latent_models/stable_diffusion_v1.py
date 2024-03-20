import os
import copy
import torch
from torch.nn import functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,
                       StableDiffusionControlNetPipeline)
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTextModel
from zarc_diffusion.utils.utils_model import str2torch_dtype, cast_training_params
from .base import BaseModel, DiffusionTrainer
from zarc_diffusion.utils.calculatron import compute_snr


class StableDiffision(BaseModel):
    """
    封装stable diffusion model的类

    Parameters
    ----------
    config_diffusion : dict. diffusion model的配置
        * pretrained_model_name_or_path: str. 与diffusers一致
        * train_unet: Optional[bool], default False. 是否训练unet
        * unet_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
        * train_text_encoder: Optional[bool], default False. 是否训练text_encoder
        * text_encoder_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
    config_vae : dict, default None. vae设置, 不做特别设置则保持和`config_diffusion`一致
        * pretrained_vae_name_or_path: Optional[str]. 不做特别设置则保持和`config_diffusion`一致
        * vae_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
    config_scheduler : dict, default None. noise_scheduler设置, 不做特别设置则保持和`config_diffusion`一致
        * pretrained_model_name_or_path: Optional[str]. 不做特别设置则保持和`config_diffusion`一致
    config_lora : dict, default None. lora设置
        * rank: int. lora的rank
    config_adapters : List[dict], default None. 支持多个adapters!!!，每个dict类型的adapter配置如下：
        * train_adapter: Optional[bool], default False.
        * adapter_method: str. 目前只支持control, 未来也会支持t2i-adapter和ip-adapter
        * pretrained_adapter_name_or_path: Optional[str]. 同diffusers一致，如果不做特别设置则不使用任何预训练模型
        * adapter_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
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
        self.adapters = None
        super().__init__(config_diffusion, config_vae, config_scheduler, config_lora, config_adapters, prediction_type,
                         snr_gamma, noise_offset)

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.unet.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))

        # freeze unet
        if "train_unet" not in config:
            config["train_unet"] = False
        if not config["train_unet"]:
            print("freeze unet")
            self.unet.requires_grad_(False)
        else:
            self.trainable_params = cast_training_params(self.unet)

        if "train_text_encoder" not in config:
            config["train_text_encoder"] = False
        if not config["train_text_encoder"]:
            print("freeze text_encoder")
            self.text_encoder.requires_grad_(False)
        else:
            self.trainable_params.extend(cast_training_params(self.text_encoder))

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
        if "vae_dtype" not in config:
            config["vae_dtype"] = None
        self.vae.to(self.device, str2torch_dtype(config["vae_dtype"], default=self.weight_dtype))
        self.vae.requires_grad_(False)

    def init_scheduler(self, config):
        if config is None:
            config = {}
        if "pretrained_model_name_or_path" not in config:
            pretrained_model_name_or_path = self.config_diffusion["pretrained_model_name_or_path"]
            config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
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

    def init_adapter(self, config):
        if config is None:
            config = []
        adapters = []
        for cfg_adapter in config:
            train_adapter = cfg_adapter.get("train_adapter", False)
            pretrained_adapter_name_or_path = cfg_adapter.get("pretrained_adapter_name_or_path", None)
            if not (train_adapter or pretrained_adapter_name_or_path):
                raise ValueError("没有使用预训练参数的adapter需要训练!")
            if "train_unet" in self.config_diffusion and self.config_diffusion["train_unet"] and train_adapter:
                raise ValueError("通常不会既训练unet又训练adapter!")
            if cfg_adapter["adapter_method"] == "control":
                from diffusers import ControlNetModel
                if pretrained_adapter_name_or_path:
                    adapter = ControlNetModel.from_pretrained(pretrained_adapter_name_or_path)
                    print(f"controlnet加载{pretrained_adapter_name_or_path}权重")
                else:
                    adapter = ControlNetModel.from_unet(self.unet)
                    print(f"从unet中初始化controlnet")
            else:
                raise NotImplementedError
            if "adapter_dtype" not in cfg_adapter:
                cfg_adapter["adapter_dtype"] = None
            adapter.to(self.device, str2torch_dtype(cfg_adapter["adapter_dtype"], default=self.weight_dtype))

            if train_adapter:
                adapter.train()
                self.trainable_params.extend(cast_training_params(adapter))
            else:
                adapter.requires_grad_(False)
            adapters.append(adapter)
        self.adapters = adapters

    def forward(self, batch):
        latents = self.run_vae(batch["image_origin"])
        timesteps = self.run_timesteps(latents.shape[0])
        noise, noisy_latents = self.sample_noise(latents, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]

        down_block_res_samples, mid_block_res_sample = self.run_adapter(batch, noisy_latents, timesteps,
                                                                        encoder_hidden_states)

        model_pred = self.run_unet(noisy_latents, timesteps, encoder_hidden_states, down_block_res_samples,
                                   mid_block_res_sample)
        loss = self.run_loss(model_pred, noise, latents, timesteps)
        return loss

    def run_vae(self, pixel_values):
        latents = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.unet.dtype)
        return latents

    def run_timesteps(self, bsz):
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()
        return timesteps

    def sample_noise(self, latents, timesteps):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        return noise, noisy_latents

    def run_adapter(self, batch, noisy_latents, timesteps, encoder_hidden_states, **kwargs):
        if not self.adapters:
            return None, None
        # 防止乱序
        controlnet_cond = sorted(batch.items(), key=lambda x: x[0])
        controlnet_cond = [v for k, v in controlnet_cond if "image" in k and k != "image_origin"]
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.adapters[0].dtype)
        noisy_latents = noisy_latents.to(dtype=self.adapters[0].dtype)

        down_block_res_samples, mid_block_res_sample = None, None
        for image, adapter in zip(controlnet_cond, self.adapters):
            image = image.to(dtype=adapter.dtype)
            down_samples, mid_sample = adapter(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=image,
                return_dict=False,
            )

            # merge samples
            if down_block_res_samples is None and mid_block_res_sample is None:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample

    def run_unet(self, noisy_latents, timesteps, encoder_hidden_states, down_block_res_samples, mid_block_res_sample):
        # Predict the noise residual and compute loss
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.unet.dtype)
        if down_block_res_samples:
            down_block_res_samples = [
                sample.to(dtype=self.unet.dtype) for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample.to(dtype=self.unet.dtype)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        return model_pred

    def run_loss(self, model_pred, noise, latents, timesteps):
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
    def save_checkpoint(self, dirname: str, save_single_model: bool = True,
                        print_info: bool = False) -> None:
        """注意，file_name是目录不再是文件"""
        save_path = os.path.join(self.ckpt_dir, dirname)
        os.makedirs(save_path, exist_ok=True)
        if self.model.lora is not None:
            self.model.unet.save_attn_procs(save_path,
                                            weight_name="lora.safetensors")
        if self.model.adapters is not None:
            for i, (adapter, cfg_adapter) in enumerate(zip(self.model.adapters, self.model.config_adapters)):
                if cfg_adapter.get("train_adapter", False):
                    save_adapter_path = os.path.join(save_path, f"adapter_{i}")
                    adapter.save_pretrained(save_adapter_path)
        if self.model.config_diffusion["train_unet"]:
            self.model.unet.save_pretrained(os.path.join(save_path, "unet"))
        if self.model.config_diffusion["train_text_encoder"]:
            self.model.text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))

    def get_same_dtype_model(self, model, dtype: torch.dtype):
        """给定model，返回dtype类型的model（需要是trainer内部的model）"""
        model = self.accelerator.unwrap_model(model)
        if model.dtype != dtype or (isinstance(model, UNet2DConditionModel) and self.model.config_lora):
            model = copy.deepcopy(model).to(dtype)
        return model

    def get_pipeline(self):
        """给出当前模型组合而成的pipline"""
        # create pipeline
        unet = self.get_same_dtype_model(self.model.unet, dtype=torch.float16)
        text_encoder = self.get_same_dtype_model(self.model.text_encoder, dtype=torch.float16)
        if self.model.adapters:
            controlnets = []
            for adapter in self.model.adapters:
                controlnets.append(self.get_same_dtype_model(adapter, dtype=torch.float16))
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=unet.to(torch.float16),
                text_encoder=text_encoder.to(torch.float16),
                controlnet=controlnets,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=unet.to(torch.float16),
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
        return pipeline
