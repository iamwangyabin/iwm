import os
import random
from torchvision import datasets
from torch.utils.data import Dataset


class ImageNetPretrain(datasets.ImageFolder):
    """ImageNet 预训练数据集封装，支持子集抽样。

    使用位置: `data_utils/__init__.py:build_pretrain_dataset` 构造训练集。
    """
    def __init__(self, root, transform=None, split="train", data_pct=1.0, dataset_seed=42):
        super().__init__(os.path.join(root, split), transform=transform)
        if data_pct < 1.0:
            rng = random.Random(dataset_seed)
            keep = max(1, int(len(self.samples) * data_pct))
            indices = list(range(len(self.samples)))
            rng.shuffle(indices)
            keep_set = set(indices[:keep])
            self.samples = [s for i, s in enumerate(self.samples) if i in keep_set]
            self.targets = [t for i, t in enumerate(self.targets) if i in keep_set]
            self.imgs = self.samples

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


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
        if hasattr(sample, "mode") and sample.mode != "RGB":
            sample = sample.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
