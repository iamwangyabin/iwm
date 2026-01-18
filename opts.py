import argparse

parser = argparse.ArgumentParser('JEPA/IWM pre-training', add_help=False)

# dataset
parser.add_argument('--data_pct', type=float, default=1.0)
parser.add_argument('--dataset_seed', type=int, default=42)
parser.add_argument('--hf_dataset', default='ILSVRC/imagenet-1k', type=str, help='HF dataset name')
parser.add_argument('--hf_cache_dir', default='', type=str, help='HF datasets cache dir')

# run config
parser.add_argument('--output_dir', default='./output', help='path where to save, empty for no saving')
parser.add_argument('--exp_name', default='', type=str)
parser.add_argument('--print_freq', default=20, type=int)
parser.add_argument('--eval_freq', default=1, type=int)
parser.add_argument('--eval_list', type=int, nargs='*', default=[])
parser.add_argument('--seed', type=int, default=2048)
parser.add_argument('--swanlab_project', type=str, default='IWM')
parser.add_argument('--swanlab_name', type=str, default='')

# model
parser.add_argument('--model', default='vit_base', type=str, metavar='MODEL')
parser.add_argument('--ssl_type', default='jepa', type=str)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--input_size', default=224, type=int)
parser.add_argument('--resize_size', default=None, type=int)
parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT')
parser.add_argument('--stop_grad_conv1', action='store_true', default=False)
parser.add_argument('--stop_grad_norm1', action='store_true', default=False)
parser.add_argument('--pred_depth', type=int, default=6)
parser.add_argument('--pred_emb_dim', type=int, default=384)
parser.add_argument('--mae_init_weights', action='store_true', default=False)
parser.add_argument('--unify_embed', action='store_true', default=False)
parser.add_argument('--cond_type', type=str, default='concat')
parser.add_argument('--pretrained', type=str, default='')

# optimizer
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999))
parser.add_argument('--weight_decay', type=float, default=0.04)
parser.add_argument('--weight_decay_end', type=float, default=0.4)
parser.add_argument('--lr', type=float, default=None, metavar='LR')
parser.add_argument('--start_lr', type=float, default=0.0)
parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR')
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--accum_iter', default=1, type=int)
parser.add_argument('--clip_grad', type=float, default=None)
parser.add_argument('--amp', dest='amp', action='store_true', default=True)
parser.add_argument('--no_amp', dest='amp', action='store_false')
parser.add_argument('--dist_backend', type=str, default='nccl')

# IO and resume
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--new_start', action='store_true', default=False)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--stop_epoch', default=None, type=int)
parser.add_argument('--num_workers', default=10, type=int)

# augment
parser.add_argument('--norm_type', type=str, default='default')
parser.add_argument('--crop_type', type=str, default='rrc', choices=['rc', 'rrc'])
parser.add_argument('--aug_type', type=str, default='jit')
parser.add_argument('--scale_min', type=float, default=0.08)
parser.add_argument('--color_jitter', type=float, default=0.2)
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME')
parser.add_argument('--rot', type=float, default=10)

# IWM augment params
parser.add_argument('--iwm_jitter_prob', type=float, default=0.8)
parser.add_argument('--iwm_blur_prob', type=float, default=0.2)
parser.add_argument('--iwm_noise_prob', type=float, default=0.0)
parser.add_argument('--iwm_noise_range', type=float, nargs=2, default=(0.05, 0.2))
parser.add_argument('--iwm_version', type=str, default='v1')
parser.add_argument('--iwm_aug_norm', action='store_true', default=False)
parser.add_argument('--iwm_disable', action='store_true', default=False)

# dual-view params
parser.add_argument('--extra_global_scale', type=float, nargs=2, default=(0.3, 1.0))
parser.add_argument('--min_overlap', type=float, default=-1)
parser.add_argument('--flip_prob', type=float, default=0.5)
parser.add_argument('--rel_pos_disable', action='store_true', default=False)
parser.add_argument('--reverse_pred', action='store_true', default=False)

# mask params
parser.add_argument('--mask_type', type=str, default='multi_multiblock')
parser.add_argument('--mask_nenc', type=int, default=1)
parser.add_argument('--mask_npred', type=int, default=4)
parser.add_argument('--mask_min_keep', type=int, default=10)
parser.add_argument('--mask_max_keep', type=int, default=None)
parser.add_argument('--mask_rand_keep', action='store_true', default=False)
parser.add_argument('--mask_merge', action='store_true', default=False)
parser.add_argument('--enc_mask_scale', type=float, nargs=2, default=(0.85, 1.0))
parser.add_argument('--pred_mask_scale', type=float, nargs=2, default=(0.15, 0.2))

# loss + target
parser.add_argument('--loss_type', type=str, default='l2')
parser.add_argument('--target_last_k', type=int, default=1)
parser.add_argument('--target_norm_type', type=str, default='avg_ln')
parser.add_argument('--extra_loss_weight', type=float, default=1.0)
parser.add_argument('--extra_mean', action='store_true', default=False)
parser.add_argument('--reg_weight', type=float, default=0.0)

# schedules
parser.add_argument('--ema', type=float, default=0.996)
parser.add_argument('--ema_end', type=float, default=1.0)
parser.add_argument('--ipe_scale', type=float, default=1.0)
