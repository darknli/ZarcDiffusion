"""放一些数据处理"""
from torchvision.transforms import v2
from transformers import CLIPTokenizer, AutoTokenizer
from PIL import Image
from typing import Tuple
import numpy as np


class NormalImageOperator:
    def __init__(self, size):
        self.transform = v2.Compose(
            [
                v2.Resize(size),
                v2.CenterCrop(size),
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
            ]
        )

    def __call__(self, data):
        data_new = {}
        key_arr, value_arr = list(zip(*data.items()))
        value_arr = self.transform(value_arr)
        for key, value in zip(key_arr, value_arr):
            data_new[key] = value
        return data_new


class SDOperator:
    def __init__(self,
                 tokenizer: CLIPTokenizer = None,
                 tokenizer_name_or_path: str = None,
                 size: int = 512,
                 key_caption: str = "caption"):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name_or_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_name_or_path, subfolder="tokenizer"
            )
        else:
            raise ValueError
        self.image_opt = NormalImageOperator(size=size)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption

    def __call__(self, data):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k}
        data_images = self.image_opt(data_images)
        data_images["image_origin"] = self.normalize(data_images["image_origin"])

        input_ids = self.tokenizer(
            data[self.key_caption], max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        output = {"input_ids": input_ids}
        output.update(data_images)
        return output


class SDXLOperator:
    def __init__(self, tokenizer: Tuple[CLIPTokenizer] = None,
                 tokenizer_name_or_path: str = None,
                 size: int = 1024,
                 key_caption: str = "caption"):
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
        self.image_opt = NormalImageOperator(size=size)
        self.normalize = v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.key_caption = key_caption

    def __call__(self, data):
        data_images = {k: Image.open(v) for k, v in data.items() if "image" in k}
        origin_size = np.array(Image.open(data["image_origin"])).shape[:2]
        data_images = self.image_opt(data_images)
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

