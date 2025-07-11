import os
import copy
import torch
from torch.nn import functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel,
                       StableDiffusionControlNetPipeline, StableDiffusionInpaintPipeline)
from transformers import CLIPTextModel
from zarc_diffusion.utils.utils_model import str2torch_dtype, cast_training_params
from zarc_diffusion.models.ip_adapter import IPAdaperEncoder, ValidIPAdapter
from .base import BaseModel, DiffusionTrainer
from zarc_diffusion.utils.calculatron import compute_snr
from peft import LoraConfig


class StableDiffision(BaseModel):
    """
    封装stable diffusion model的类，实现文生图模型&训练loss模块代码

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
    config_ip_adapter : dict, default None. ip-adapter设置
        * image_encoder_path: str. image_encoder预训练模型路径
        * train_ip_adapter: Optional[bool], default False. 是否训练ip_adapter
        * pretrain_model: Optional[str]. ip-adapter预训练模型
        * ip_adapter_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
    config_controls : List[dict], default None. 支持多个controls!!!，每个dict类型的control配置如下：
        * image_key: str. control_image在数据对应的key, 需要有指定
        * train_control: Optional[bool], default False.
        * pretrained_control_name_or_path: Optional[str]. 同diffusers一致，如果不做特别设置则不使用任何预训练模型
        * control_dtype: Optional[str], default fp16. {'fp16', 'bf16', 'fp32'}
    prediction_type : str, default None. 'epsilon' or 'v_prediction' or leave `None`
    snr_gamma: float, default None. 用于加速收敛, https://arxiv.org/abs/2303.09556
    noise_offset: float, default None. 参考https://www.crosslabs.org//blog/diffusion-with-offset-noise
    """

    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict = None,
                 config_scheduler: dict = None,
                 config_lora: dict = None,
                 config_ip_adapter: dict = None,
                 config_controls: dict = None,
                 prediction_type: str = None,
                 snr_gamma: float = None,
                 noise_offset: float = None,
                 ):
        self.latent_diffusion_model = None
        self.text_encoder = None
        self.vae = None
        self.noise_scheduler = None
        self.lora = None
        self.ip_encoder = None
        self.adapter_modules = None
        self.controls = None
        super().__init__(config_diffusion, config_vae, config_scheduler, config_lora, config_ip_adapter,
                         config_controls, prediction_type, snr_gamma, noise_offset)

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.latent_diffusion_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.latent_diffusion_model.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))

        # freeze unet
        if "train_unet" not in config:
            config["train_unet"] = False
        if not config["train_unet"]:
            print("freeze unet")
            self.latent_diffusion_model.requires_grad_(False)
        else:
            if config.get("enable_gradient_checkpointing", False):
                self.latent_diffusion_model.enable_gradient_checkpointing()
            self.trainable_params = cast_training_params(self.latent_diffusion_model)

        if "train_text_encoder" not in config:
            config["train_text_encoder"] = False
        if not config["train_text_encoder"]:
            print("freeze text_encoder")
            self.text_encoder.requires_grad_(False)
        else:
            if config.get("enable_gradient_checkpointing", False):
                if hasattr(self.text_encoder, 'enable_gradient_checkpointing'):
                    self.text_encoder.enable_gradient_checkpointing()
                if hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                    self.text_encoder.gradient_checkpointing_enable()
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
        if config.get("enable_gradient_checkpointing", False):
            self.latent_diffusion_model.enable_gradient_checkpointing()
        rank = config["rank"]
        if "target_modules" in config:
            target_modules = config["target_modules"]
        else:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
            print(f"lora配置里没有发现`target_modules`设置，采用默认配置:{target_modules}")
        unet_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.latent_diffusion_model.add_adapter(unet_lora_config)
        unet_lora_parameters = cast_training_params(self.latent_diffusion_model)
        # self.latent_diffusion_model.enable_gradient_checkpointing()
        self.lora = unet_lora_parameters
        self.trainable_params.extend(unet_lora_parameters)

    def init_ip_adapter(self, config):
        try:
            from zarc_diffusion.models.ip_adapter.utils import is_torch2_available
        except ImportError:
            raise ImportError("需要安装ip_adapter，pip install git+https://github.com/tencent-ailab/IP-Adapter.git")
        if is_torch2_available():
            from zarc_diffusion.models.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, \
                AttnProcessor2_0 as AttnProcessor
        else:
            from zarc_diffusion.models.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
        assert not self.config_lora, "暂不支持使用ip-adapter时训练lora"
        image_encoder_path = config["image_encoder_path"]
        self.ip_encoder = IPAdaperEncoder(image_encoder_path=image_encoder_path,
                                          cross_attention_dim=self.latent_diffusion_model.config.cross_attention_dim)
        if "ip_adapter_dtype" not in config:
            config["ip_adapter_dtype"] = None
        self.ip_encoder.to(self.device, str2torch_dtype(config["ip_adapter_dtype"], default=self.weight_dtype))

        # init adapter modules
        attn_procs = {}
        unet_sd = self.latent_diffusion_model.state_dict()
        for name in self.latent_diffusion_model.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.latent_diffusion_model.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.latent_diffusion_model.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.latent_diffusion_model.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.latent_diffusion_model.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        self.latent_diffusion_model.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(self.latent_diffusion_model.attn_processors.values())
        adapter_modules.to(self.device, str2torch_dtype(config["ip_adapter_dtype"], default=self.weight_dtype))

        if "pretrain_model" in config and config["pretrain_model"]:
            pretrain_model = config["pretrain_model"]
            state_dict = torch.load(pretrain_model, map_location=self.device)
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        if "train_ip_adapter" not in config:
            config["train_ip_adapter"] = False
        if not config["train_ip_adapter"]:
            print("freeze train_ip_adapter")
            self.ip_encoder.requires_grad_(False)
            adapter_modules.requires_grad_(False)
        else:
            if config.get("enable_gradient_checkpointing", False):
                self.ip_encoder.enable_gradient_checkpointing()
                adapter_modules.enable_gradient_checkpointing()
            self.trainable_params.extend(cast_training_params([self.ip_encoder.image_proj_model, adapter_modules]))
        self.adapter_modules = adapter_modules

    def init_controlnet(self, config):
        if config is None:
            config = []
        controlnets = []
        from diffusers import ControlNetModel
        for cfg_control in config:
            train_control = cfg_control.get("train_control", False)
            pretrained_control_name_or_path = cfg_control.get("pretrained_control_name_or_path", None)
            if not (train_control or pretrained_control_name_or_path):
                raise ValueError("没有使用预训练参数的control需要训练!")
            if "train_unet" in self.config_diffusion and self.config_diffusion["train_unet"] and train_control:
                raise ValueError("通常不会既训练unet又训练control!")
            if pretrained_control_name_or_path:
                control = ControlNetModel.from_pretrained(pretrained_control_name_or_path)
                print(f"controlnet加载{pretrained_control_name_or_path}权重")
            else:
                control = ControlNetModel.from_unet(self.latent_diffusion_model)
                print(f"从unet中初始化controlnet")
            if "control_dtype" not in cfg_control:
                cfg_control["control_dtype"] = None
            control.to(self.device, str2torch_dtype(cfg_control["control_dtype"], default=self.weight_dtype))

            if train_control:
                control.train()
                if config.get("enable_gradient_checkpointing", False):
                    control.enable_gradient_checkpointing()
                self.trainable_params.extend(cast_training_params(control))
            else:
                control.requires_grad_(False)
            controlnets.append(control)
        self.controls = controlnets

    def forward(self, batch):
        latents = self.run_vae(batch["image_origin"])
        timesteps = self.run_timesteps(latents.shape[0])
        noise, noisy_latents = self.sample_noise(latents, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]

        down_block_res_samples, mid_block_res_sample = self.run_control(batch, noisy_latents, timesteps,
                                                                        encoder_hidden_states)

        if self.ip_encoder:
            encoder_hidden_states = self.ip_encoder(
                batch["ip_adapter_image"], encoder_hidden_states, no_drop_arr=batch["ip_adapter_no_drop"])

        model_pred = self.run_diffusion_model(noisy_latents, timesteps, encoder_hidden_states, down_block_res_samples,
                                   mid_block_res_sample)
        loss = self.run_loss(model_pred, noise, latents, timesteps)
        return loss

    def run_vae(self, pixel_values):
        latents = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.latent_diffusion_model.dtype)
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

    def run_control(self, batch, noisy_latents, timesteps, encoder_hidden_states, **kwargs):
        if not self.controls:
            return None, None
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.controls[0].dtype)
        noisy_latents = noisy_latents.to(dtype=self.controls[0].dtype)

        down_block_res_samples, mid_block_res_sample = None, None
        for cfg, control in zip(self.config_controls, self.controls):
            image = batch[cfg["image_key"]]
            image = image.to(dtype=control.dtype)
            down_samples, mid_sample = control(
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

    def run_diffusion_model(
            self, noisy_latents, timesteps, encoder_hidden_states, down_block_res_samples, mid_block_res_sample
    ):
        # Predict the noise residual and compute loss
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)
        if down_block_res_samples:
            down_block_res_samples = [
                sample.to(dtype=self.latent_diffusion_model.dtype) for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample.to(dtype=self.latent_diffusion_model.dtype)

        model_pred = self.latent_diffusion_model(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        return model_pred

    def run_loss(self, model_pred, noise, latents, timesteps, weights=None):
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
            if weights is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                weights = F.interpolate(weights, size=model_pred.shape[2:])
                loss = (weights * F.mse_loss(model_pred.float(), target.float(), reduction="none")).mean()
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
            if weights is not None:
                loss = loss * weights
            loss = loss.mean()
        return loss


class StableDiffusionInpainting(StableDiffision):
    def forward(self, batch):
        latents = self.run_vae(batch["image_origin"])
        timesteps = self.run_timesteps(latents.shape[0])
        noise, noisy_latents = self.sample_noise(latents, timesteps)

        # inpainting的unet in_channel是8通道
        mask = batch["mask"].to(dtype=latents.dtype)
        latents_cond = self.run_vae(batch["image_cond"])
        if len(mask.shape) == 3:
            mask = mask[:, None]
        mask = F.interpolate(mask, size=latents_cond.shape[2:])
        noisy_with_cond = torch.cat([noisy_latents, mask, latents_cond], dim=1)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]

        down_block_res_samples, mid_block_res_sample = self.run_control(batch, noisy_latents, timesteps,
                                                                        encoder_hidden_states)

        if self.ip_encoder:
            encoder_hidden_states = self.ip_encoder(
                batch["ip_adapter_image"], encoder_hidden_states, no_drop_arr=batch["ip_adapter_no_drop"])

        model_pred = self.run_diffusion_model(
            noisy_with_cond, timesteps, encoder_hidden_states, down_block_res_samples, mid_block_res_sample
        )
        loss = self.run_loss(model_pred, noise, latents, timesteps, weights=batch.get("weights", None))
        return loss


class SDTrainer(DiffusionTrainer):
    """封装stable diffusion model的类，实现inpainting/outpainting训练&loss定义模块"""
    def save_checkpoint(self, dirname: str, save_single_model: bool = True,
                        print_info: bool = False) -> None:
        """注意，file_name是目录不再是文件"""
        save_path = os.path.join(self.ckpt_dir, dirname)
        os.makedirs(save_path, exist_ok=True)
        if self.model.lora is not None:
            self.model.latent_diffusion_model.save_attn_procs(save_path,
                                            weight_name="lora.safetensors")
        if self.model.ip_encoder is not None:
            image_proj_model = self.model.ip_encoder.image_proj_model
            torch.save(
                {
                    "image_proj": image_proj_model.state_dict(),
                    "ip_adapter": self.model.adapter_modules.state_dict()
                },
                os.path.join(save_path, "ip-adapter.pth")
            )
        if self.model.controls is not None:
            for i, (control, cfg_control) in enumerate(zip(self.model.controls, self.model.config_controls)):
                if cfg_control.get("train_control", False):
                    save_control_path = os.path.join(save_path, f"control_{i}")
                    control.save_pretrained(save_control_path)
        if self.model.config_diffusion["train_unet"]:
            self.model.latent_diffusion_model.save_pretrained(os.path.join(save_path, "unet"))
        if self.model.config_diffusion["train_text_encoder"]:
            self.model.text_encoder.save_pretrained(os.path.join(save_path, "text_encoder"))

    def get_same_dtype_model(self, model, dtype: torch.dtype):
        """给定model，返回dtype类型的model（需要是trainer内部的model）"""
        model = self.accelerator.unwrap_model(model)
        if model.dtype != dtype or (isinstance(model, UNet2DConditionModel) and
                                    (self.model.config_lora or self.model.config_ip_adapter)):
            model = copy.deepcopy(model).to(dtype)
        return model

    def get_pipeline(self):
        """给出当前模型组合而成的pipline"""
        # create pipeline
        latent_diffusion_model = self.get_same_dtype_model(self.model.latent_diffusion_model, dtype=torch.float16)
        text_encoder = self.get_same_dtype_model(self.model.text_encoder, dtype=torch.float16)
        if isinstance(self.model, StableDiffusionInpainting):
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=latent_diffusion_model.to(torch.float16),
                torch_dtype=torch.float16,
                revision=None
            )
        elif self.model.ip_encoder:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=latent_diffusion_model.to(torch.float16),
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            proj_model = copy.deepcopy(self.model.ip_encoder.image_proj_model)
            for p in proj_model.parameters():
                p.data = p.to(torch.float16)
            pipeline = ValidIPAdapter(pipeline,
                                      self.get_same_dtype_model(self.model.ip_encoder.image_encoder, torch.float16),
                                      proj_model,
                                      self.accelerator.device
                                      )
        elif self.model.controls:
            controlnets = []
            for control in self.model.controls:
                controlnets.append(self.get_same_dtype_model(control, dtype=torch.float16))
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=latent_diffusion_model.to(torch.float16),
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
                unet=latent_diffusion_model.to(torch.float16),
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
        return pipeline
