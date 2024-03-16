from abc import abstractmethod
import torch
import torch_frame
import warnings
import copy


class BaseModel(torch.nn.Module):
    """模型基类"""
    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict,
                 config_scheduler: dict,
                 config_lora: dict = None,
                 config_adapters: dict = None,
                 prediction_type: str = None,
                 snr_gamma: float = None,
                 noise_offset: float = None
                 ):
        super().__init__()
        self.device = torch.device("cuda", torch_frame.utils.dist_utils.get_rank())
        self.cache = {}
        self.trainable_params = []

        # 保存配置
        self.config_diffusion = copy.deepcopy(config_diffusion)
        self.config_vae = copy.deepcopy(config_diffusion)
        self.config_scheduler = copy.deepcopy(config_scheduler)
        self.config_lora = copy.deepcopy(config_lora)
        self.config_adapters = copy.deepcopy(config_adapters)
        self.snr_gamma = snr_gamma
        self.prediction_type = prediction_type
        self.noise_offset = noise_offset

        self.init_diffusion(config_diffusion)
        self.init_vae(config_vae)
        self.init_scheduler(config_scheduler)
        if config_adapters:
            self.init_adapter(config_adapters)
        if config_lora:
            self.init_lora(config_lora)
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

    def init_adapter(self, config):
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
