import torch
import os
from transformers import CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline
from .stable_diffusion_v1 import StableDiffision, SDTrainer
from zarc_diffusion.utils.utils_model import cast_training_params, str2torch_dtype


class StableDiffusionXl(StableDiffision):
    """
    和stableDiffusion在代码区别就两点：
    1. 将图片原尺寸和crop的左上角点编码到模型里了
    2. text encoder从1个 -> 2个
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

        self.text_encoder1 = None
        self.text_encoder2 = None
        super().__init__(config_diffusion, config_vae, config_scheduler, config_lora, config_adapters, prediction_type,
                         snr_gamma, noise_offset)
        # sd_v1的text_encoder被sdxl的text_encoder1和text_encoder2代替
        del self.text_encoder

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.text_encoder1 = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.unet.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder1.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))
        self.text_encoder2.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))

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
            self.text_encoder1.requires_grad_(False)
            self.text_encoder2.requires_grad_(False)
        else:
            self.trainable_params.extend(cast_training_params(self.text_encoder1))
            self.trainable_params.extend(cast_training_params(self.text_encoder2))

    def forward(self, batch):
        latents = self.run_vae(batch["image_origin"])
        timesteps = self.run_timesteps(latents.shape[0])
        noise, noisy_latents = self.sample_noise(latents, timesteps)

        # Get the text embedding for conditioning
        prompt_embeds, unet_added_conditions = self.run_text_encoder(batch["input_ids1"], batch["input_ids2"],
                                                                     batch["original_sizes"], batch["crop_top_lefts"])

        down_block_res_samples, mid_block_res_sample = self.run_adapter(batch, noisy_latents, timesteps,
                                                                        prompt_embeds)

        model_pred = self.run_unet(noisy_latents, timesteps, prompt_embeds, down_block_res_samples,
                                   mid_block_res_sample, unet_added_conditions=unet_added_conditions)
        loss = self.run_loss(model_pred, noise, latents, timesteps)
        return loss

    def run_text_encoder(self, input_ids1, input_ids2, original_sizes, crop_top_lefts, resolution=1024):
        original_sizes = torch.stack(original_sizes, 1)
        crop_top_lefts = torch.stack(crop_top_lefts, 1)
        resolution_ids = torch.full((original_sizes.shape[0], 2), resolution,
                                    device=self.device, dtype=original_sizes.dtype)

        add_time_ids = torch.concat([
            original_sizes,
            crop_top_lefts,
            resolution_ids,
        ], -1)
        prompt_embeds_list = []

        text_input_ids_list = [input_ids1, input_ids2]
        encoder_list = [self.text_encoder1, self.text_encoder2]
        for text_input_ids, text_encoder in zip(text_input_ids_list, encoder_list):
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_prompt_embeds
        }
        return prompt_embeds, unet_added_conditions

    def run_unet(self, noisy_latents, timesteps, prompt_embeds,
                 down_block_res_samples, mid_block_res_sample, **kwargs):
        # Predict the noise residual and compute loss
        unet_added_conditions = kwargs.get("unet_added_conditions")
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype)
        if down_block_res_samples:
            down_block_res_samples = [
                sample.to(dtype=self.unet.dtype) for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample.to(dtype=self.unet.dtype)

        model_pred = self.unet(
            noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        return model_pred


class SDXLTrainer(SDTrainer):
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
            self.model.text_encoder1.save_pretrained(os.path.join(save_path, "text_encoder"))
            self.model.text_encoder2.save_pretrained(os.path.join(save_path, "text_encoder_2"))

    def get_pipeline(self):
        """给出当前模型组合而成的pipline"""
        # create pipeline
        unet = self.get_same_dtype_model(self.model.unet, dtype=torch.float16)
        text_encoder1 = self.get_same_dtype_model(self.model.text_encoder1, dtype=torch.float16)
        text_encoder2 = self.get_same_dtype_model(self.model.text_encoder2, dtype=torch.float16)
        if self.model.adapters:
            controlnets = []
            for adapter in self.model.adapters:
                controlnets.append(self.get_same_dtype_model(adapter, dtype=torch.float16))
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=unet.to(torch.float16),
                text_encoder=text_encoder1.to(torch.float16),
                text_encoder_2=text_encoder2.to(torch.float16),
                controlnet=controlnets,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
        else:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                self.model.config_diffusion["pretrained_model_name_or_path"],
                unet=unet.to(torch.float16),
                text_encoder=text_encoder1.to(torch.float16),
                text_encoder_2=text_encoder2.to(torch.float16),
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline = pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
        return pipeline
