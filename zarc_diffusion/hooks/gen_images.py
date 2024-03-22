from PIL import Image
import numpy as np
import os
from torch_frame.hooks import HookBase
from torch_frame import logger
import torch
from torchvision.utils import make_grid
from zarc_diffusion.utils.utils_model import get_gpu_free_memory


class GenHook(HookBase):
    def __init__(self, save_dir, validation_prompt, validation_images=None,
                 num_validation_images=1, seed=0, period=1, merge_result=True):
        self.save_dir = save_dir
        self.validation_prompt = validation_prompt
        self.validation_images = validation_images
        self.num_validation_images = num_validation_images
        self.seed = seed
        self.period = period
        self.merge_result = merge_result
        self.out_type = "pt" if merge_result else "pil"

    def after_epoch(self, *args, **kwargs):
        if self.trainer.accelerator.is_main_process and (self.every_n_epochs(self.period) or self.is_last_epoch()):
            logger.info(
                f"Running validation... \n Generating {self.num_validation_images} images with prompt:"
                f" {self.validation_prompt}."
            )

            generator = torch.Generator(device=self.trainer.accelerator.device)
            if self.seed is not None:
                generator = generator.manual_seed(self.seed)
            memory = get_gpu_free_memory()
            # 不足2G会清空缓存
            if memory < 2:
                torch.cuda.empty_cache()
            # create pipeline
            pipeline = self.trainer.get_pipeline()
            if self.trainer.model.controls:
                assert self.validation_images, "control必须要有image"
                validation_images = [Image.open(image).convert("RGB") for image in self.validation_images]
                images = pipeline(self.validation_prompt, validation_images, num_inference_steps=20,
                                  generator=generator, num_images_per_prompt=self.num_validation_images,
                                  output_type=self.out_type).images
            else:
                images = pipeline(self.validation_prompt, num_inference_steps=30, generator=generator,
                                  num_images_per_prompt=self.num_validation_images,
                                  output_type=self.out_type).images
            image_dir = os.path.join(self.trainer.work_dir, self.save_dir)
            os.makedirs(image_dir, exist_ok=True)
            if self.merge_result:
                nrow = max(int(len(images)**0.5 + 0.5), 1)
                result = make_grid(images, nrow)
                result = result.cpu().numpy().transpose((1, 2, 0))
                result = (result * 255).astype(np.uint8)
                filename = os.path.join(image_dir, "valid_{:04d}.jpg".format(self.trainer.epoch))
                Image.fromarray(result).save(filename)
            else:
                for i, img in enumerate(images):
                    filename = os.path.join(image_dir, "valid_{:04d}_{}.jpg".format(self.trainer.epoch, i))
                    img.save(filename)
            del pipeline
            torch.cuda.empty_cache()
