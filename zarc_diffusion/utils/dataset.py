"""用于数据打包，由于生成式训练常常条件非常多，打包数据能更好结构化"""
import cv2
import numpy as np
from torchvision.transforms import v2
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
from glob import glob
import pickle
from tqdm import tqdm


def get_data(filename):
    if "image" in filename:
        data = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    elif "txt" in filename:
        with open(filename) as f:
            data = f.read()
    else:
        data = None
    return data


def packing_dataset(root_src, root_dst):
    """
    将零碎文件打包成高度整合的文件，注意：这种方式会占用大量硬盘容量
    1. 以文件名来决定数据类型
    root_src的目录下需要是这样的形式：
    -- name1
    ---- image_origin.png
    ---- image_cond1.png
    ---- image_cond2.png
    ---- caption1.txt
    -- name2
    ---- image_origin.png
    ---- image_cond1.png
    ---- image_cond2.png
    ---- caption1.txt
    ...
    其中image_origin是必要的，其他都是选填的
    注意图片是opencv可正常读取的格式即可，文本需要txt

    2. 保存格式
    函数会把一套图象文本等数据打包成一条数据（根据name1、name2等）
    """
    os.makedirs(root_dst, exist_ok=True)
    for name in os.listdir(root_src):
        item = {}
        key_name = os.path.basename(name).rsplit(".", 1)[0]
        for filename in glob(os.path.join(root_src, name)):
            key_type = os.path.basename(filename).rsplit(".", 1)[0]
            data = get_data(filename)
            if data is None:
                continue
            item[key_type] = data

        with open(os.path.join(root_dst, key_name + ".pkl"), "wb") as f:
            pickle.dump(item, f)


def packing_addition_data(root_add, root_origin,
                          trict_match: bool = True, overwrite: bool = False):
    """
    将新的meta数据打包到已有的数据集中，相当于是把root_origin文件夹下每个文件都添加一个新的condition

    Parameters
    ----------
    root_add : str. 添加的condition目录，目录名称保持和希望添加的condition名称相同，如'image_cond12'
    root_origin : str. 原本的数据集，需要已经包含了全部样本
    trict_match : bool, default True. 如果是True，原数据集中存在匹配不到新condition数据的样本时会抛出异常
    overwrite: bool, default False. 如果新添加的condition名称已经在原数据集中存在了，是否允许覆盖
    """
    key_type = os.path.basename(root_add)
    for filename_origin in glob(os.path.join(root_origin, "*")):
        basename = os.path.basename(filename_origin).rsplit(".", 1)[0]
        files = glob(os.path.join(root_add, basename+".*"))
        assert len(files) > 1, f"出现多个{basename}对应的condition文件"
        if len(files) == 0:
            assert not trict_match, f"没有{basename}对应的condition文件"
            continue
        with open(filename_origin, "rb") as f:
            data_origin = pickle.load(f)
        if key_type in data_origin and not overwrite:
            raise ValueError(f"{key_type}condition已经在原数据集中存在，如覆盖，需要overwrite=True")

        filename_add = files[0]
        data_add = get_data(filename_add)
        data_origin[key_type] = data_add

        with open(filename_origin, "rb") as f:
            pickle.dump(data_origin, f)


def unpacking_dataset(root_src, root_dst, image_ext=".jpg"):
    """
    packing_dataset函数的逆输出版本
    """
    for filename in glob(os.path.join(root_src, "*.pkl")):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        for key, value in data.items():
            if "image" in key:
                cv2.imwrite(os.path.join(root_dst, f"{key}{image_ext}"), value)
            else:
                with open(os.path.join(root_dst, f"{key}.txt"), "w") as f:
                    f.write(value)


def gen_metalist(root: str, meta_path: str):
    """给出图片&caption文本目录，生成metalist.csv"""
    data = []
    match_format = [".png", ".jpg", ".jpeg", ".webp"]
    caption_list = glob(os.path.join(root, "*.txt"))
    with tqdm(caption_list) as pbar:
        for caption_path in pbar:
            with open(caption_path) as f:
                caption = f.read()
            for ext in match_format:
                image_path = caption_path.replace(".txt", ext)
                if os.path.exists(image_path):
                    data.append([caption, image_path])
                    break
            else:
                raise ValueError(f"caption {caption}没有与之匹配的图片")
    df = pd.DataFrame(data, columns=["caption", "image_origin"])
    df.to_csv(meta_path, index=False)


def vertical2horizontal(root_src, root_dst):
    """
    把数据从图片+metalist.csv的形式转换成层级式形式
    """
    raise NotImplementedError


class PackageDataset(Dataset):
    """
    数据集应该是packing_dataset函数处理过的格式
    """
    def __init__(self, root, lambda_func):
        self.data_list = []
        for filename in glob(os.path.join(root, "*.pkl")):
            with open(filename, "rb") as f:
                item = pickle.load(f)
            self.data_list.append(item)
        self.lambda_func = lambda_func

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        item = self.lambda_func(data)
        return item


class MetaListDataset(Dataset):
    """
    数据集应该是图片+metalist.csv的格式
    """
    def __init__(self, metalist_path: str, tokenizer, size):
        self.df = pd.read_csv(metalist_path)
        self.tokenizer = tokenizer
        self.transform = v2.Compose(
            [
                v2.Resize(size),
                v2.CenterCrop(size),
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx].to_dict()
        image_keys = []
        image_values = []
        captions = []
        for k, v in data.items():
            if "image" in k:
                image_keys.append(k)
                image_values.append(Image.open(v))
            elif "caption" in k:
                captions.append(v)
        captions_str = ",".join(captions)

        input_ids = self.tokenizer(
            captions_str, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        item = dict(zip(image_keys, self.transform(image_values)))
        # 需要再把origin image缩放到-1~1之间
        item["image_origin"] = (item["image_origin"] * 2) - 1
        item["input_ids"] = input_ids
        return item
