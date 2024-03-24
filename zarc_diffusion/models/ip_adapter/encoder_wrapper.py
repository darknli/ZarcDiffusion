from torch import nn
from typing import List, Union
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from .ip_adapter import ImageProjModel, IPAttnProcessor
import torch
from PIL import Image


class IPAdaperEncoder(nn.Module):
    def __init__(self, image_encoder_path, cross_attention_dim):
        super().__init__()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        self.image_encoder.requires_grad_(False)
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )

    def forward(self, clip_image, encoder_hidden_states, no_drop_arr=None):
        with torch.no_grad():
            image_embeds = self.image_encoder(
                clip_image.to(self.image_encoder.device, dtype=self.image_encoder.dtype)).image_embeds
        if no_drop_arr is not None:
            image_embeds = image_embeds * no_drop_arr[:, None]
        ip_tokens = self.image_proj_model(image_embeds.to(dtype=self.image_proj_model.norm.weight.dtype))
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return encoder_hidden_states


class ValidIPAdapter:
    def __init__(self, sd_pipe, image_encoder, image_proj_model, device, num_tokens=4):
        self.device = device
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)

        # load image encoder
        self.image_encoder = image_encoder
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        pil_image=None,
        clip_image_embeds=None,
        scale=1.0,
        num_images_per_prompt=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=None,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
