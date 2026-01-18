# ImageNet JEPA/IWM Pretraining

This repository contains a clean, ImageNet-only pretraining pipeline based on JEPA with IWM conditioning and dual-view masked prediction.

## What is included

- JEPA-style target encoder + predictor
- IWM conditioning on parameterized augmentations
- Dual-view masked prediction strategy
- ViT backbones (`vit_tiny`/`vit_small`/`vit_base`/`vit_large`/`vit_huge`)

```bash


torchrun --nproc_per_node=4 train_jepa.py \
    --exp_name iwm_dual_vitb \
    --ssl_type iwm_dual_easy \
    --model vit_base \
    --batch_size 64 \
    --epochs 100 \
    --warmup_epochs 40 \
    --blr 1e-3 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --ema 0.996 \
    --ema_end 1.0 \
    --amp \
```


