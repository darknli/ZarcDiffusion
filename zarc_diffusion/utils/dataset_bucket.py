"""提供分辨率分桶装进batch的方法"""
import random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class AspectRatioBucketDataset(Dataset, ABC):
    """
    根据数据集中的图片分辨率做bucket
    * 这个Dataset类的dataloader需要batch_size=None。
    * 继承了这个类的派生类不要重写__len__, 但要重写process（实现原先__getitem__的内容）
    * 把self.build_batch_indices方法放到hook的before_epoch中

    Parameters
    -------------------------------
    buckets : List[Tuple[int, int]]. 每个Tuple都包含width和height。后续如果走默认None的话会自动聚类，生成一个bucket
    max_batchsize : int, default 1. 分桶之后不能保证batchsize数量稳定，因此只能设一个最大的batch size数量
    shuffle: bool, default False. 是否打乱顺序
    """
    def __init__(self, buckets: List[Tuple[int, int]] = None, max_batchsize: int = 1, shuffle: bool = False):
        assert buckets is not None
        self.buckets = buckets
        self.asp_buckets = np.array([w / h for w, h in buckets])
        self.bid2row = {}
        # 每行数据对应的bucket id
        self.item2bid = []
        self.max_batchsize = max_batchsize
        self.shuffle = shuffle
        self.batch_indices = []
        self.set_bucket()
        self.build_batch_indices()

    def set_bucket(self):
        self.item2bid = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            if "width" in row and "height" in row:
                width = row["width"]
                height = row["height"]
            else:
                image = Image.open(row["image_origin"])
                width, height = image.size
            ratio = width / height
            # 找到最合适的那个bucket
            bid = np.argmin(np.abs(self.asp_buckets - ratio))
            self.item2bid.append(bid)
            if bid not in self.bid2row:
                self.bid2row[bid] = []
            self.bid2row[bid].append(idx)

    def build_batch_indices(self):
        print("build_batch...")
        if len(self.batch_indices) > 0 and not self.shuffle:
            return
        self.batch_indices = []
        for bid, bucket in self.bid2row.items():
            if self.shuffle:
                random.shuffle(bucket)
            for start_idx in range(0, len(bucket), self.max_batchsize):
                end_idx = min(start_idx + self.max_batchsize, len(bucket))
                batch = bucket[start_idx:end_idx]
                self.batch_indices.append(batch)
        if self.shuffle:
            random.shuffle(self.batch_indices)

    @abstractmethod
    def process(self, idx):
        pass

    def __getitem__(self, bid):
        assert self.batch_indices
        data = [self.process(idx) for idx in self.batch_indices[bid]]
        return data

    def __len__(self):
        return len(self.batch_indices)