import os.path
import cv2
from PIL import Image
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torch_frame.hooks import CheckpointerHook
import torch
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from zarc_diffusion.utils.utils_model import get_gpu_free_memory
import glob


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return image


class FIDHook(CheckpointerHook):
    def __init__(self,
                 prompts: list,
                 real_image_dir: str,
                 period: int = 1,
                 max_to_keep: Optional[int] = None,
                 save_last: bool = True,
                 prefix: str = "eval",
                 seed: int = 0):
        self.prefix = prefix+"_"
        super(FIDHook, self).__init__(period, max_to_keep, self.prefix + "fid", False, save_last)
        real_images = []
        for path in glob.glob(os.path.join(real_image_dir, "*.[jp][pn]*g")):
            image = np.array(Image.open(path))
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image).unsqueeze(0)
            image = image.permute(0, 3, 1, 2) / 255.0
            real_images.append(image)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.real_images = torch.cat(real_images).to(self.device)
        self.prompts = prompts
        self.fid = FrechetInceptionDistance(normalize=True)
        self.fid.inception.to(self.device)
        self.seed = seed

    def fid_update(self, imgs, real):
        imgs = (imgs * 255).byte() if self.fid.normalize else imgs
        features = self.fid.inception(imgs).cpu()
        self.fid.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.fid.real_features_sum += features.sum(dim=0)
            self.fid.real_features_cov_sum += features.t().mm(features)
            self.fid.real_features_num_samples += imgs.shape[0]
        else:
            self.fid.fake_features_sum += features.sum(dim=0)
            self.fid.fake_features_cov_sum += features.t().mm(features)
            self.fid.fake_features_num_samples += imgs.shape[0]

    @torch.no_grad()
    def _do_eval(self):
        generator = torch.Generator(device=self.trainer.accelerator.device)
        if self.seed is not None:
            generator = generator.manual_seed(self.seed)
        memory = get_gpu_free_memory()
        # 不足2G会清空缓存
        if memory < 2:
            torch.cuda.empty_cache()
        # create pipeline
        pipeline = self.trainer.get_pipeline()
        fake_images = []
        for prompt in self.prompts:
            images = pipeline(prompt, num_inference_steps=30, generator=generator,
                              num_images_per_prompt=4,
                              output_type="np").images
            fake_images.append(images)
        fake_images = np.concatenate(fake_images, 0)
        fake_images = torch.from_numpy(fake_images).permute(0, 3, 1, 2).to(self.device)
        self.fid_update(self.real_images, real=True)
        self.fid_update(fake_images, real=False)
        kwargs = {self.save_metric: self.fid.compute()}
        self.log(self.trainer.epoch, **kwargs, smooth=False, window_size=1)
        self.fid.reset()

    def after_epoch(self) -> None:
        if self.every_n_epochs(self._period) or self.is_last_epoch():
            self._do_eval()
            self.save_model()