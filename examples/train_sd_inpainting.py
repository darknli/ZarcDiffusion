from zarc_diffusion.models.latent_models import StableDiffusionInpainting, SDTrainer
from zarc_diffusion.utils import SDInpaintingOperator
from zarc_diffusion.hooks import ShuffleBucketHook
from utils.dataset_bucket import AspectRatioBucketDataset
from torch.utils.data import DataLoader
from torch_frame import LoggerHook
from zarc_diffusion.hooks import DiffusersCheckpointerHook, FIDHook
import yaml
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/sd_lora.yaml",
        required=False,
        help="Path to config",
    )
    args = parser.parse_args()
    return args


class BucketDataset(AspectRatioBucketDataset):
    """
    数据集应该是packing_dataset函数处理过的格式
    """
    def __init__(self, metalist_path: str, lambda_func, max_batchsize=2):
        bucket = [(512, 512)]
        w = [(64 * r, 512) for r in range(8, 15)]
        h = [(512, 64 * r) for r in range(8, 15)]
        bucket.extend(w)
        bucket.extend(h)
        self.df = pd.read_csv(metalist_path)
        self.lambda_func = lambda_func
        super().__init__(bucket, max_batchsize=max_batchsize)

    def process(self, idx):
        data = self.df.iloc[idx].to_dict()
        bid = self.item2bid[idx]
        w, h = self.buckets[bid]
        item = self.lambda_func(data, (w, h))
        return item


def collection(data):
    key_list = list(data[0].keys())
    data_batch = {}
    for k in key_list:
        vs = [d[k] for d in data]
        data_batch[k] = torch.from_numpy(np.stack(vs, 0))
    return data_batch


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config_model = config["model"]
    config_diffusion = config_model["diffusion"]
    config_lora = config_model.get("lora", None)
    config_ip_adapter = config_model.get("ip_adapter", None)
    config_controls = config_model.get("controls", None)
    config_train = config["train"]
    model = StableDiffusionInpainting(config_diffusion,
                                      config_lora=config_lora,
                                      config_ip_adapter=config_ip_adapter,
                                      config_controls=config_controls,
                                      snr_gamma=config_model.get("snr_gamma", None),
                                      noise_offset=config_model.get("noise_offset", None))
    sd_opt = SDInpaintingOperator(tokenizer_name_or_path=config_diffusion["pretrained_model_name_or_path"],
                        size=config_train["size"])
    train_dataset = BucketDataset(config_train["dataset_path"], sd_opt)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              persistent_workers=True,
                              batch_size=config_train["train_batch_size"],
                              num_workers=config_train["num_workers"])

    if config_train["use_8bit_adam"]:
        print("使用8bit")
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "需要安装bitsandbytes; windows系统可以执行包名是bitsandbytes-windows"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.trainable_params,
        lr=1e-04,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-08,
    )

    hooks = [
        ShuffleBucketHook(train_dataset),
        # GenHook("valid_images",
        #         config_train["validation_prompt"],
        #         config_train.get("validation_images", None),
        #         period=config_train["period"]),
        DiffusersCheckpointerHook(
            period=config_train["period"],
            max_to_keep=config_train["max_to_keep"]),
        # FIDHook(
        #     [
        #         "prompt1",
        #         "prompt2",
        #      ],
        #     "directory",
        #     period=config_train["period"]
        # ),
        LoggerHook(),
    ]
    scheduler = config_train["lr_scheduler"]

    trainer = SDTrainer(model, optimizer, scheduler, train_loader, config_train["max_epoch"],
                config_train["workspace"], config_train["max_grad_norm"],
                mixed_precision=config_train["mixed_precision"], hooks=hooks,
                gradient_accumulation_steps=config_train["gradient_accumulation_steps"],
                )
    trainer.log_param(**config)
    trainer.train()


if __name__ == '__main__':
    main()
