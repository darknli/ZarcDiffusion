import os
import copy
import torch
from typing import Union
from torch.nn import functional as F
from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline, FluxTransformer2DModel,
                       StableDiffusionControlNetPipeline, StableDiffusionInpaintPipeline)
from diffusers.models.lora import LoRALinearLayer
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel
from zarc_diffusion.utils.utils_model import str2torch_dtype, cast_training_params, flush_vram
from zarc_diffusion.models.ip_adapter import IPAdaperEncoder, ValidIPAdapter
from .stable_diffusion_v1 import StableDiffision, DiffusionTrainer
from zarc_diffusion.utils.calculatron import compute_snr
from peft import LoraConfig
from einops import rearrange, repeat


class CustomFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_noise_sigma = 1.0

        with torch.no_grad():
            # create weights for timesteps
            num_timesteps = 1000
            # Bell-Shaped Mean-Normalized Timestep Weighting
            # bsmntw? need a better name

            x = torch.arange(num_timesteps, dtype=torch.float32)
            y = torch.exp(-2 * ((x - num_timesteps / 2) / num_timesteps) ** 2)

            # Shift minimum to 0
            y_shifted = y - y.min()

            # Scale to make mean 1
            bsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # only do half bell
            hbsmntw_weighing = y_shifted * (num_timesteps / y_shifted.sum())

            # flatten second half to max
            hbsmntw_weighing[num_timesteps // 2:] = hbsmntw_weighing[num_timesteps // 2:].max()

            # Create linear timesteps from 1000 to 0
            timesteps = torch.linspace(1000, 0, num_timesteps, device='cpu')

            self.linear_timesteps = timesteps
            self.linear_timesteps_weights = bsmntw_weighing
            self.linear_timesteps_weights2 = hbsmntw_weighing
            pass

    def get_weights_for_timesteps(self, timesteps: torch.Tensor, v2=False) -> torch.Tensor:
        # Get the indices of the timesteps
        step_indices = [(self.timesteps == t).nonzero().item() for t in timesteps]

        # Get the weights for the timesteps
        if v2:
            weights = self.linear_timesteps_weights2[step_indices].flatten()
        else:
            weights = self.linear_timesteps_weights[step_indices].flatten()

        return weights

    def get_sigmas(self, timesteps: torch.Tensor, n_dim, dtype, device) -> torch.Tensor:
        sigmas = self.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)

        return sigma

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        ## ref https://github.com/huggingface/diffusers/blob/fbe29c62984c33c6cf9cf7ad120a992fe6d20854/examples/dreambooth/train_dreambooth_sd3.py#L1578
        ## Add noise according to flow matching.
        ## zt = (1 - texp) * x + texp * z1

        # sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        # noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # timestep needs to be in [0, 1], we store them in [0, 1000]
        # noisy_sample = (1 - timestep) * latent + timestep * noise
        t_01 = (timesteps / 1000).to(original_samples.device)
        noisy_model_input = (1 - t_01) * original_samples + t_01 * noise

        # n_dim = original_samples.ndim
        # sigmas = self.get_sigmas(timesteps, n_dim, original_samples.dtype, original_samples.device)
        # noisy_model_input = (1.0 - sigmas) * original_samples + sigmas * noise
        return noisy_model_input

    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        return sample

    def set_train_timesteps(self, num_timesteps, device, linear=False):
        if linear:
            timesteps = torch.linspace(1000, 0, num_timesteps, device=device)
            self.timesteps = timesteps
            return timesteps
        else:
            # distribute them closer to center. Inference distributes them as a bias toward first
            # Generate values from 0 to 1
            t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

            # Scale and reverse the values to go from 1000 to 0
            timesteps = ((1 - t) * 1000)

            # Sort the timesteps in descending order
            timesteps, _ = torch.sort(timesteps, descending=True)

            self.timesteps = timesteps.to(device=device)

            return timesteps


class Flux(StableDiffision):
    """
    封装Flux model的类，实现文生图模型&训练loss模块代码

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
        self.min_denoising_steps = 0
        self.max_denoising_steps = 1000
        self.standardize_noise = False  # 是否标准化noise
        super().__init__(config_diffusion, config_vae, config_scheduler, config_lora, config_ip_adapter,
                         config_controls, prediction_type, snr_gamma, noise_offset)

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2")
        self.latent_diffusion_model = FluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
            )

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.latent_diffusion_model.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder_2.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))
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
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder_2.eval()
        else:
            for text_encoder in [self.text_encoder, self.text_encoder_2]:
                if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                    text_encoder.enable_gradient_checkpointing()
                if hasattr(text_encoder, "gradient_checkpointing_enable"):
                    text_encoder.gradient_checkpointing_enable()
            self.trainable_params.extend(cast_training_params(self.text_encoder))
            self.trainable_params.extend(cast_training_params(self.text_encoder_2))
        flush_vram()

    def init_scheduler(self, config):
        # 根据http://github.com/ostris/ai-toolkit.git，以下面方式调用scheduler
        config = {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "_diffusers_version": "0.29.0.dev0",
            "num_train_timesteps": 1000,
            "shift": 3.0,
            'prediction_type': 'epsilon'
        }
        self.noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_config(config)

    def run_timesteps(self, bsz):
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        if self.min_denoising_steps == self.max_denoising_steps:
            timestep_indices = torch.ones((batch_size,), device=self.device) * self.min_noise_steps
        else:
            # todo, some schedulers use indices, otheres use timesteps. Not sure what to do here
            timestep_indices = torch.randint(
                self.min_denoising_steps + 1,
                self.max_denoising_steps - 1,
                (bsz,),
                device=self.device
            )
        timestep_indices = timestep_indices.long()
        timesteps = [self.noise_scheduler.timesteps[x.item()] for x in timestep_indices]
        timesteps = torch.stack(timesteps, dim=0)
        return timesteps

    def sample_noise(self, latents, timesteps):
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )

        if self.standardize_noise:
            std = noise.std(dim=(2, 3), keepdim=True)
            normalizer = 1 / (std + 1e-6)
            noise = noise * normalizer

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if len(timesteps.shape) == 1:
            timesteps = timesteps.reshape(-1, 1, 1, 1)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        return noise, noisy_latents

    def run_diffusion_model(self, noisy_latents, timesteps, encoder_hidden_states, pooled_encoder_hidden_states):
        # Predict the noise residual and compute loss
        noisy_latents = noisy_latents.to(self.device, dtype=self.latent_diffusion_model.dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)
        guidance = torch.tensor([1.]).to(device=self.device, dtype=self.latent_diffusion_model.dtype)
        timesteps = timesteps.to(self.device)

        bs, c, h, w = noisy_latents.shape
        latent_model_input_packed = rearrange(
            noisy_latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=2,
            pw=2
        )
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs).to(
            self.device, dtype=self.latent_diffusion_model.dtype)
        txt_ids = torch.zeros(bs, encoder_hidden_states.shape[1], 3).to(
            self.device, dtype=self.latent_diffusion_model.dtype)

        model_pred = self.latent_diffusion_model(
            hidden_states=latent_model_input_packed,  # [1, 4096, 64]
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            # todo make sure this doesnt change
            timestep=timesteps / 1000,  # timestep is 1000 scale
            encoder_hidden_states=encoder_hidden_states,
            # [1, 512, 4096]
            pooled_projections=pooled_encoder_hidden_states,  # [1, 768]
            txt_ids=txt_ids,  # [1, 512, 3]
            img_ids=img_ids,  # [1, 4096, 3]
            guidance=guidance,
            return_dict=False,
        )[0]
        noise_pred = rearrange(
            model_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=noisy_latents.shape[2] // 2,
            w=noisy_latents.shape[3] // 2,
            ph=2,
            pw=2,
            c=noisy_latents.shape[1],
        )
        return noise_pred

    def run_loss(self, model_pred, noise, latents, timesteps, weights=None):
        target = noise - latents
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean()
        return loss

    def forward(self, batch):
        latents = self.run_vae(batch["image_origin"])
        timesteps = self.run_timesteps(latents.shape[0])
        noise, noisy_latents = self.sample_noise(latents, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device), output_hidden_states=False)
        pooled_prompt_embeds = encoder_hidden_states.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device)
        encoder_hidden_states = self.text_encoder_2(batch["input_ids2"].to(self.device), output_hidden_states=False)[0]

        down_block_res_samples, mid_block_res_sample = self.run_control(batch, noisy_latents, timesteps,
                                                                        encoder_hidden_states)

        if self.ip_encoder:
            encoder_hidden_states = self.ip_encoder(
                batch["ip_adapter_image"], encoder_hidden_states, no_drop_arr=batch["ip_adapter_no_drop"])

        model_pred = self.run_diffusion_model(noisy_latents, timesteps, encoder_hidden_states, pooled_prompt_embeds)
        loss = self.run_loss(model_pred, noise, latents, timesteps)
        return loss
