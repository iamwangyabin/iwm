from .imagenet import ImageNetPretrain, ImageNetHFPretrain
from .transforms import build_transform_rgb
from .world_model_transform import build_world_model_transform
from .dual_transform import build_world_model_dual_transform


def build_pretrain_dataset(args, ssl_type):
    # Choose transform based on SSL variant; IWM needs parameterized augmentation.
    if ssl_type == 'iwm':
        transform = build_world_model_transform(args)
    elif 'iwm_dual' in ssl_type:
        transform = build_world_model_dual_transform(args)
    else:
        transform = build_transform_rgb(args, is_train=True)

    if getattr(args, "imagenet_hf", False):
        cache_dir = args.hf_cache_dir or None
        return ImageNetHFPretrain(
            dataset_name=args.hf_dataset,
            transform=transform,
            split='train',
            data_pct=args.data_pct,
            dataset_seed=args.dataset_seed,
            cache_dir=cache_dir
        )

    return ImageNetPretrain(
        root=args.data_path,
        transform=transform,
        split='train',
        data_pct=args.data_pct,
        dataset_seed=args.dataset_seed
    )
