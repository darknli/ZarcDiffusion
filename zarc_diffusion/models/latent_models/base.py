from abc import abstractmethod
import torch
import torch_frame
import warnings
import copy
from accelerate import Accelerator
from zarc_diffusion.utils.utils_model import str2torch_dtype


class BaseModel(torch.nn.Module):
    """模型基类"""
    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict,
                 config_scheduler: dict,
                 config_lora: dict = None,
                 config_ip_adapter: dict = None,
                 config_controls: dict = None,
                 prediction_type: str = None,
                 snr_gamma: float = None,
                 noise_offset: float = None
                 ):
        super().__init__()
        if torch_frame.utils.dist_utils.get_world_size() == 1:
            self.device = "cuda"
            self.weight_dtype = torch.float16
        else:
            accelerator = Accelerator()
            self.device = copy.deepcopy(accelerator.device)
            self.weight_dtype = str2torch_dtype(accelerator.mixed_precision, torch.float16)
            del accelerator
        self.cache = {}
        self.trainable_params = []

        # 保存配置
        self.config_diffusion = copy.deepcopy(config_diffusion)
        self.config_vae = copy.deepcopy(config_vae)
        self.config_scheduler = copy.deepcopy(config_scheduler)
        self.config_lora = copy.deepcopy(config_lora)
        self.config_ip_adapter = copy.deepcopy(config_ip_adapter)
        self.config_controls = copy.deepcopy(config_controls)
        self.snr_gamma = snr_gamma
        self.prediction_type = prediction_type
        self.noise_offset = noise_offset

        self.init_diffusion(self.config_diffusion)
        self.init_vae(self.config_vae)
        self.init_scheduler(self.config_scheduler)
        if self.config_controls:
            self.init_controlnet(self.config_controls)
        if self.config_lora:
            self.init_lora(self.config_lora)
        if self.config_ip_adapter:
            self.init_ip_adapter(self.config_ip_adapter)
        assert len(self.trainable_params) > 0, "No trainable parameters"

    @abstractmethod
    def init_diffusion(self, config):
        ...

    @abstractmethod
    def init_vae(self, config):
        ...

    @abstractmethod
    def init_scheduler(self, config):
        ...

    def init_lora(self, config):
        ...

    def init_ip_adapter(self, config):
        ...

    def init_controlnet(self, config):
        ...

    def save(self, *args):
        warnings.warn("Function not implemented")

    def train(self, training=True):
        ...


class DiffusionTrainer(torch_frame.AccelerateTrainer):
    """diffusion trainer的基类"""
    def prepare_model(self):
        self.optimizer, self.data_loader, self.lr_scheduler, self.model.trainable_params = self.accelerator.prepare(
            self.optimizer, self.data_loader, self.lr_scheduler, self.model.trainable_params
        )
