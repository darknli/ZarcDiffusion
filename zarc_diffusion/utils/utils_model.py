from typing import Union, List
import torch


def to_model_device(dtype, *args):
    """将数据复制到和model一样的device上"""
    return [v.to(dtype=dtype) for v in args]


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
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
