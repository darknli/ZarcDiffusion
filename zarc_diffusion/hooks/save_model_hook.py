from torch_frame.hooks import CheckpointerHook
from torch_frame.trainer import logger
import os
import shutil


class DiffusersCheckpointerHook(CheckpointerHook):
    """和`CheckpointerHook`的区别在于这个类传给trainer的是目录而不是文件，对标diffusers的api"""
    def save_model(self):
        if self.save_last:
            self.trainer.save_checkpoint("latest", False)

        # 如果当前epoch指标没有更好, 则不保存模型
        if self.save_metric is not None:
            if not self.is_better(self.trainer.metric_storage[self.save_metric]):
                return
            self.cur_best = self.trainer.metric_storage[self.save_metric].avg
            logger.info(f"{self.save_metric} update to {round(self.cur_best, 4)}")
            self.trainer.save_checkpoint("best.pth")

        if self._max_to_keep is not None and self._max_to_keep >= 1:
            epoch = self.trainer.epoch  # ranged in [0, max_epochs - 1]
            checkpoint_name = f"epoch_{epoch}"
            self.trainer.save_checkpoint(checkpoint_name)
            self._recent_checkpoints.append(checkpoint_name)
            if len(self._recent_checkpoints) > self._max_to_keep:
                # delete the oldest checkpoint
                dirname = self._recent_checkpoints.pop(0)
                dirname = os.path.join(self.trainer.ckpt_dir, dirname)
                if os.path.exists(dirname):
                    shutil.rmtree(dirname)
