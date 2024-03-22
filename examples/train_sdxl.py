from zarc_diffusion.latent_models.stable_diffusion_xl import StableDiffusionXl, SDXLTrainer
from zarc_diffusion.utils.dataset import MetaListXLDataset
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from torch_frame import LoggerHook
from zarc_diffusion.hooks import GenHook, DiffusersCheckpointerHook
import yaml
import torch


def main():
    with open("examples/configs/sdxl_lora.yaml") as f:
        config = yaml.safe_load(f)
    config_model = config["model"]
    config_diffusion = config_model["diffusion"]
    config_vae = config_model["vae"]
    config_lora = config_model.get("lora", None)
    config_controls = config_model.get("controls", None)
    config_train = config["train"]
    model = StableDiffusionXl(config_diffusion,
                              config_vae=config_vae,
                              config_lora=config_lora,
                              config_controls=config_controls,
                              snr_gamma=config_model.get("snr_gamma", None),
                              noise_offset=config_model.get("noise_offset", None))
    tokenizer1 = CLIPTokenizer.from_pretrained(
        config_diffusion["pretrained_model_name_or_path"], subfolder="tokenizer"
    )
    tokenizer2 = CLIPTokenizer.from_pretrained(
        config_diffusion["pretrained_model_name_or_path"], subfolder="tokenizer_2"
    )

    train_dataset = MetaListXLDataset(config_train["dataset_path"], tokenizer1, tokenizer2, config_train["size"])
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
        GenHook("valid_images",
                config_train["validation_prompt"],
                config_train.get("validation_images", None),
                num_validation_images=config_train.get("num_validation_images", 1),
                period=config_train["period"]),
        DiffusersCheckpointerHook(
            period=config_train["period"],
            max_to_keep=config_train["max_to_keep"]),
        LoggerHook(),
    ]
    scheduler = config_train["lr_scheduler"]

    trainer = SDXLTrainer(model, optimizer, scheduler, train_loader, config_train["max_epoch"],
                          config_train["workspace"], config_train["max_grad_norm"],
                          mixed_precision=config_train["mixed_precision"], hooks=hooks,
                          gradient_accumulation_steps=config_train["gradient_accumulation_steps"],
                          )
    trainer.log_param(**config)
    trainer.train()


if __name__ == '__main__':
    main()
