from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch
from timm.data import create_transform
import random
from .randaugment import RandomAugment


class GaussianBlur(object):
    """按概率应用高斯模糊的变换。

    使用位置: `build_aug_trans` 中按配置加入增强队列。
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))
        )


class Solarization(object):
    """按概率应用 Solarize 的变换。

    使用位置: `build_aug_trans` 中按配置加入增强队列。
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img


class RandomGaussianNoise:
    """添加随机高斯噪声的变换。

    使用位置: `build_aug_trans` 中按配置加入增强队列。
    """
    def __init__(self, noise_range=(0.05, 0.2)):
        self.noise_range = noise_range

    def __call__(self, img):
        sig = float(torch.empty(1).uniform_(self.noise_range[0], self.noise_range[1]))
        return add_gaussian_noise_pil(img, sig)


def add_gaussian_noise_pil(img, sig):
    arr = np.asarray(img)
    arr1 = arr + np.random.randn(*arr.shape) * sig * 255
    arr1 = np.clip(arr1, 0, 255)
    arr1 = arr1.astype(arr.dtype)
    return Image.fromarray(arr1)


def get_mean_std(args):
    if args.norm_type == 'default':
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif args.norm_type == 'inception':
        mean = IMAGENET_INCEPTION_MEAN
        std = IMAGENET_INCEPTION_STD
    else:
        raise NotImplementedError
    return mean, std


class Identity:
    """恒等变换，占位使用。

    使用位置: `build_world_model_transform` 在 `aug_type=none` 时使用。
    """
    def __call__(self, x):
        return x


def build_aug_trans(args):
    aug_trans = []
    if args.rot > 0:
        aug_trans.append(transforms.RandomRotation(degrees=args.rot, interpolation=3))
    has_blur = 'blur' in args.aug_type
    has_sol = 'sol' in args.aug_type
    if has_blur or has_sol:
        if has_blur and has_sol:
            t = transforms.RandomChoice([Solarization(p=1.0), GaussianBlur(p=1.0)])
        elif has_blur:
            t = GaussianBlur(p=1.0)
        else:
            t = Solarization(p=1.0)
        aug_trans.append(transforms.RandomApply([t], p=args.iwm_blur_prob))
    if 'jit' in args.aug_type:
        aug_trans.append(
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter)], p=1.0
            )
        )
    if 'noise' in args.aug_type:
        aug_trans.append(transforms.RandomApply([RandomGaussianNoise(noise_range=args.iwm_noise_range)], p=args.iwm_noise_prob))
    if 'ra' in args.aug_type:
        trans = RandomAugment(
            2,
            7,
            isPIL=True,
            augs=[
                'Identity','AutoContrast','Equalize','Brightness','Sharpness',
                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
            ]
        )
        aug_trans.append(trans)
    return transforms.Compose(aug_trans)


def build_transform_crop(args, is_train=True):
    img_size = args.input_size
    resize_size = args.input_size * 512 // 448 if args.resize_size is None else args.resize_size
    if is_train:
        if args.crop_type == 'rc':
            crop_trans = transforms.Compose([
                transforms.Resize(resize_size, interpolation=Image.BICUBIC),
                transforms.RandomCrop((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
            ])
        elif args.crop_type == 'rrc':
            crop_trans = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(args.scale_min, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            raise NotImplementedError
    else:
        crop_trans = transforms.Compose([
            transforms.Resize(resize_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop((img_size, img_size)),
        ])
    return crop_trans


def build_transform_rgb(args, is_train, ten_crop=False):
    mean, std = get_mean_std(args)
    post_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if args.aug_type == 'timm' and is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=(args.scale_min, 1.0),
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=0,
            mean=mean,
            std=std,
        )
        return transform

    crop_trans = build_transform_crop(args, is_train)
    if is_train:
        aug_trans = build_aug_trans(args)
    else:
        aug_trans = Identity()

    if ten_crop:
        transform = transforms.Compose([
            transforms.Resize(args.resize_size, interpolation=Image.BICUBIC),
            transforms.TenCrop(args.input_size),
            transforms.Lambda(lambda crops: torch.stack([post_trans(crop) for crop in crops]))
        ])
    else:
        transform = transforms.Compose([crop_trans, aug_trans, post_trans])
    return transform
