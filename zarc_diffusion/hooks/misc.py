from torch_frame.hooks import HookBase

class ShuffleBucketHook(HookBase):
    """Bucket dataset专用hook"""
    def __init__(self, dataset):
        self.dataset = dataset

    def before_epoch(self):
        # 打乱顺序
        self.dataset.build_batch_indices()
