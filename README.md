# ZarcDiffusion
利用diffusers等库快速完成训练
# 准备工作
安装环境
```bash
pip install git+https://github.com/darknli/ZarcDiffusion.git
```
# 训练模型示例
训练sdv1.5的各种任务可以通过不同的配置来实现，如果要训练sdxl，把脚本改成examples/train_sdxl.py即可
```bash
accelerate launch --mixed_precision="fp16" examples/train_sd.py --config examples/configs/sd_unet.yaml
```
训练sdv1.5 lora
```bash
accelerate launch --mixed_precision="fp16" examples/train_sd.py --config examples/configs/sd_lora.yaml
```
训练sdv1.5 contolnet，这个controlnet可以训练多个，示例里默认是1个
```bash
accelerate launch --mixed_precision="fp16" examples/train_sd.py --config examples/configs/sd_controlnet.yaml
```
训练sdv1.5 ip-adapter
```bash
accelerate launch --mixed_precision="fp16" examples/train_sd.py --config examples/configs/sd_ip_adapter.yaml
```
# 支持模型
- stable diffusion 1
- stable diffusion xl
- flux