import os
import random
from torchvision import datasets


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
