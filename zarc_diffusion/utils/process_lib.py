"""放一些数据处理"""
import random
import torch
from torchvision.transforms import v2
from transformers import CLIPTokenizer, T5Tokenizer, AutoTokenizer, CLIPImageProcessor
from PIL import Image
from typing import Tuple
import numpy as np


class NormalImageOperator:
    def __init__(self, size, enable_flip=True):
        self.size = size
        self.transform_resize = v2.Compose(
            [
                v2.Resize(self.size),
                v2.CenterCrop(self.size),
            ]
        )
        if enable_flip:
            self.transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(),
                    v2.ToTensor(),
                ]
            )
        else:
            self.transform = v2.ToTensor()

    def __call__(self, data, target_size=None):
        data_new = {}
        key_arr, value_arr = list(zip(*data.items()))
        if target_size is None:
            value_arr = self.transform_resize(value_arr)
        else:
            w, h = value_arr[0].size
            tw, th = target_size
            ratio = max(tw / w, th / h)
            nw, nh = int(w * ratio + 0.5), int(h * ratio + 0.5)
            value_arr = v2.Resize((nh, nw))(value_arr)
            value_arr = v2.CenterCrop((th, tw))(value_arr)
        value_arr = self.transform(value_arr)
        for key, value in zip(key_arr, value_arr):
            data_new[key] = value
        return data_new


class IPOprator:
    def __init__(self, drop_rate=0.1):
        self.drop_rate = drop_rate
        self.clip_image_processor = CLIPImageProcessor()

    def __call__(self, image):
        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        item = {
            "ip_adapter_image": clip_image[0],
            "ip_adapter_no_drop": int(random.random() > self.drop_rate)
        }
        return item


class SDOperator:
    def __init__(
            self,
            tokenizer: CLIPTokenizer = None,
            tokenizer_name_or_path: str = None,
            size: int = 512,
            key_caption: str = "caption",
            enable_flip: bool = True,
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name_or_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer"
            )
        else:
            raise ValueError
        self.image_opt = NormalImageOperator(size=size, enable_flip=enable_flip)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption
        self.ip_opt = IPOprator()

    def __call__(self, data, target_size=None):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k and "ip_adapter" not in k}
        ip_adapter_dict = self.ip_opt(data_images["image_origin"])
        data_images = self.image_opt(data_images, target_size)
        data_images["image_origin"] = self.normalize(data_images["image_origin"])

        input_ids = self.tokenizer(
            data[self.key_caption], max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        output = {"input_ids": input_ids}
        output.update(data_images)
        output.update(ip_adapter_dict)
        return output


class SDInpaintingOperator:
    def __init__(
            self,
            tokenizer: CLIPTokenizer = None,
            tokenizer_name_or_path: str = None,
            size: int = 512,
            key_caption: str = "caption",
            enable_flip: bool = True,
         ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name_or_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer"
            )
        else:
            raise ValueError
        self.image_opt = NormalImageOperator(size=size, enable_flip=enable_flip)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption
        self.ip_opt = IPOprator()

    def __call__(self, data, target_size=None):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k and "ip_adapter" not in k}
        ip_adapter_dict = self.ip_opt(data_images["image_origin"])
        data_images = self.image_opt(data_images, target_size)
        data_images["image_origin"] = self.normalize(data_images["image_origin"])

        # mask中间那块区域置1
        h, w = data_images["image_origin"].shape[1:3]
        mask = torch.zeros((h, w), dtype=torch.float32)
        bh, eh = int(h * 0.35), int(h * 0.65)
        bw, ew = int(w * 0.35), int(w * 0.65)
        mask[bh: eh, bw: ew] = 1

        # mask=1的区域都mask掉
        image_cond = data_images["image_origin"] * (mask[None] < 0.5)

        input_ids = self.tokenizer(
            data[self.key_caption], max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        output = {
            "input_ids": input_ids, "mask": mask, "image_cond": image_cond, 
        }
        output.update(data_images)
        output.update(ip_adapter_dict)
        return output


class SDXLOperator:
    def __init__(
            self,
            tokenizer: Tuple[CLIPTokenizer] = None,
            tokenizer_name_or_path: str = None,
            size: int = 1024,
            key_caption: str = "caption",
            enable_flip: bool = True,
    ):
        if tokenizer is not None:
            self.tokenizer1, self.tokenizer2 = tokenizer
        elif tokenizer_name_or_path:
            self.tokenizer1 = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer"
            )
            self.tokenizer2 = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer_2"
            )
        else:
            raise ValueError
        self.image_opt = NormalImageOperator(size=size, enable_flip=enable_flip)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption

    def __call__(self, data, target_size=None):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k and "ip_adapter" not in k}
        origin_size = np.array(Image.open(data["image_origin"])).shape[:2]
        data_images = self.image_opt(data_images, target_size)
        data_images["image_origin"] = self.normalize(data_images["image_origin"])

        input_ids1 = self.tokenizer1(
            data[self.key_caption], max_length=self.tokenizer1.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        input_ids2 = self.tokenizer2(
            data[self.key_caption], max_length=self.tokenizer2.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        output = {"input_ids1": input_ids1, "input_ids2": input_ids2}
        output.update(data_images)
        output["original_sizes"] = origin_size
        output["crop_top_lefts"] = [0, 0]
        return output


class FluxOperator:
    def __init__(
            self,
            tokenizer: CLIPTokenizer = None,
            tokenizer2: T5Tokenizer = None,
            tokenizer_name_or_path: str = None,
            size: int = 512,
            key_caption: str = "caption",
            enable_flip: bool = True,
    ):
        if tokenizer is not None and tokenizer2 is not None:
            self.tokenizer = tokenizer
            self.tokenizer2 = tokenizer2
        elif tokenizer_name_or_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer"
            )
            self.tokenizer2 = T5Tokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer_2"
            )
        else:
            raise ValueError
        self.image_opt = NormalImageOperator(size=size, enable_flip=enable_flip)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption
        self.ip_opt = IPOprator()

    def __call__(self, data, target_size=None):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k and "ip_adapter" not in k}
        ip_adapter_dict = self.ip_opt(data_images["image_origin"])
        data_images = self.image_opt(data_images, target_size)
        data_images["image_origin"] = self.normalize(data_images["image_origin"])

        input_ids = self.tokenizer(
            data[self.key_caption],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids

        input_ids2 = self.tokenizer2(
            data[self.key_caption],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids

        output = {
            "input_ids": input_ids,
            "input_ids2": input_ids2,
        }
        output.update(data_images)
        output.update(ip_adapter_dict)
        return output
