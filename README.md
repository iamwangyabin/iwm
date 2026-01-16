# ImageNet JEPA/IWM Pretraining

This repository contains a clean, ImageNet-only pretraining pipeline based on JEPA with IWM conditioning and dual-view masked prediction.

## What is included

- JEPA-style target encoder + predictor
- IWM conditioning on parameterized augmentations
- Dual-view masked prediction strategy
- ViT backbones (`vit_tiny`/`vit_small`/`vit_base`/`vit_large`/`vit_huge`)

## Data layout

ImageNet should follow the standard `ImageFolder` layout:

```
/path/to/imagenet/
  train/
    n01440764/...
    n01443537/...
  val/            # optional, not used for pretraining
```

## Quick start

See `PRETRAIN.md` for a full command example.

## Smoke test

Run a short forward-pass sanity check (CPU by default):

```bash
python scripts/smoke_pretrain.py --data_path <PATH_TO_IMAGENET> --ssl_type iwm_dual_easy --steps 1
```
