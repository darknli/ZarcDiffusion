from typing import Union, List
import gc
import torch
from torch_frame.utils.dist_utils import get_rank


def str2torch_dtype(dtype: str, default=torch.float32) -> torch.dtype:
    """字符串形式转成torch.dtype形式"""
    if dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp32":
        return torch.float32
    return default


def to_model_device(dtype: torch.dtype, *args) -> list:
    """将数据复制到和model一样的device上"""
    return [v.to(dtype=dtype) for v in args]


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]],
                         dtype=torch.float32) -> list:
    """获取model中所有可训练参数"""
    if not isinstance(model, list):
        model = [model]
    training_params = []
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)
                training_params.append(param)
    return training_params


def get_gpu_free_memory():
    """查看gpu剩余显存"""
    current_device = torch.device(get_rank())
    memory_total = torch.cuda.get_device_properties(current_device).total_memory
    memory_uesd = torch.cuda.memory_cached(current_device)
    memory_free = memory_total - memory_uesd
    memory_free_g = (memory_free / 1024 / 1024 / 1024)
    return memory_free_g


def flush_vram():
    """清除显存缓存"""
    torch.cuda.empty_cache()
    gc.collect()
