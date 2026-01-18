import os
import random
import warnings
from torchvision import datasets
from torch.utils.data import Dataset


class ImageNetHFPretrain(Dataset):
    """ImageNet 预训练数据集封装，基于 Hugging Face datasets。"""
    def __init__(
        self,
        dataset_name="ILSVRC/imagenet-1k",
        split="train",
        transform=None,
        data_pct=1.0,
        dataset_seed=42,
        cache_dir=None,
    ):
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        if data_pct < 1.0:
            keep = max(1, int(len(ds) * data_pct))
            ds = ds.shuffle(seed=dataset_seed).select(range(keep))
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        sample = item["image"]
        # Some HF ImageNet shards store grayscale images; force RGB for normalization.
        # Also silence noisy EXIF warnings from PIL on a few corrupt files.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning)
            if hasattr(sample, "mode") and sample.mode != "RGB":
                sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
