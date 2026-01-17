import argparse
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.utils.data
import torch.optim
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import util.misc as misc
from util.logger import AverageMeter
from util.misc import NativeScalerWithGradNormCount as NativeScaler, get_grad_norm_
from util import add_weight_decay, cosine_scheduler, nested_to_gpu
from data_utils import build_pretrain_dataset
from models import init_jepa_model
from masks import build_mask_collator

from opts import parser


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    return float(value)

def _maybe_init_swanlab(args):
    if misc.get_rank() != 0:
        return None
    import swanlab
    import inspect

    run_name = args.swanlab_name or args.exp_name
    init_kwargs = {
        "project": args.swanlab_project,
        "config": vars(args),
        "name": run_name,
        "experiment_name": run_name,
        "run_name": run_name,
        "dir": args.output_dir,
        "logdir": args.output_dir,
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v}
    try:
        sig = inspect.signature(swanlab.init)
    except (TypeError, ValueError):
        sig = None
    if sig:
        allows_kwargs = any(
            param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
        )
        if not allows_kwargs:
            allowed = set(sig.parameters.keys())
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in allowed}
    swanlab.init(**init_kwargs)
    return swanlab


class FP32Scaler:
    """Simple grad scaler for non-AMP training to match AMP call sites."""
    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        return None


def main(args):
    misc.init_dist_pytorch(args)
    device = torch.device(args.gpu)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    swanlab = _maybe_init_swanlab(args)

    # Build ImageNet pretrain dataset with the selected SSL variant (jepa/iwm/iwm_dual).
    dataset_train = build_pretrain_dataset(args, args.ssl_type)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset_train,
        num_replicas=args.world_size,
        rank=args.rank)



    # Collator generates encoder/predictor masks per batch.
    mask_collator = build_mask_collator(args)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn=mask_collator,
        sampler=train_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True)
    
    # Initialize JEPA/IWM model (encoder + predictor + target encoder).
    model = init_jepa_model(
        device=device,
        args=args,
        ssl_type=args.ssl_type
    )
    
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    model = DistributedDataParallel(model, static_graph=True)
    model_without_ddp = model.module

    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=args.betas)
    
    # Scale iterations per epoch to control total steps without changing epochs.
    iters_per_epoch = int(args.ipe_scale * len(data_loader_train))
    lr_schedule = cosine_scheduler(
        args.lr,  # linear scaling rule
        args.min_lr,
        args.epochs, iters_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.start_lr
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, iters_per_epoch,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    # EMA schedule for target encoder update.
    momentum_schedule = cosine_scheduler(args.ema, args.ema_end, args.epochs, iters_per_epoch)
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = FP32Scaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, new_start=args.new_start)
    
    
    start_time = time.time()
    stop_epoch = args.stop_epoch if args.stop_epoch else args.epochs
    last_step = None
    for epoch in range(args.start_epoch, stop_epoch):
        epoch_start = time.time()
        data_loader_train.sampler.set_epoch(epoch)
        # data_time = AverageMeter('Data Time', ":6.3f")
        batch_time = AverageMeter('Time', ':6.3f')
        loss_meter = AverageMeter(f'Loss', ':.4f')
        loss_intra_meter = AverageMeter(f'Loss_Intra', ':.4f')
        loss_extra_meter = AverageMeter(f'Loss_Extra', ':.4f')
        maskA_meter = AverageMeter(f'Mask_Enc', ':.1f')
        maskB_meter = AverageMeter(f'Mask_Pred', ':.1f')
        norm_meter = AverageMeter(f'Grad_Norm', ':.2f')
        var_meter = AverageMeter('Pred_Var', ':.2f')
        end = time.time()
        # for data_iter_step, (imgs, masks_enc, masks_pred) in enumerate(data_loader_train):
        for data_iter_step, data in enumerate(data_loader_train):
            it = len(data_loader_train) * epoch + data_iter_step
            last_step = it
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if param_group['weight_decay'] > 0:
                    param_group["weight_decay"] = wd_schedule[it]
            data = nested_to_gpu(data, device)
            # imgs = imgs.to(device, non_blocking=True)
            # masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            # masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]
            masks_enc, masks_pred = data[-2], data[-1]

            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=torch.bfloat16):
                # loss = model(imgs, masks_enc, masks_pred, args.loss_type, args.target_last_k, args.target_norm_type, args.target_type)
                outputs = model(data, args)
                loss = outputs['loss']
                pred_var = outputs['pred_var']
            loss_value = loss.item()
            loss = loss / args.accum_iter
            update_grad = ((it+1) % args.accum_iter == 0)
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=update_grad)
            if update_grad:
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)
                if args.pretrained == '':
                    model.module.update_target_encoder(momentum_schedule[it])
                norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            loss_meter.update(loss_value)
            if 'loss_intra' in outputs:
                loss_intra_meter.update(outputs['loss_intra'])
                loss_extra_meter.update(outputs['loss_extra'])
            var_meter.update(pred_var)
            torch.cuda.synchronize()
            if swanlab is not None and (
                (data_iter_step + 1) % args.print_freq == 0
                or data_iter_step == len(data_loader_train) - 1
            ):
                payload = {
                    "train/loss": _to_float(loss_value),
                    "train/loss_avg": _to_float(loss_meter.avg),
                    "train/pred_var": _to_float(pred_var),
                    "train/pred_var_avg": _to_float(var_meter.avg),
                    "train/mask_enc": _to_float(maskA_meter.val),
                    "train/mask_enc_avg": _to_float(maskA_meter.avg),
                    "train/mask_pred": _to_float(maskB_meter.val),
                    "train/mask_pred_avg": _to_float(maskB_meter.avg),
                    "train/lr": _to_float(lr_schedule[it]),
                    "train/weight_decay": _to_float(wd_schedule[it]),
                    "train/ema": _to_float(momentum_schedule[it]),
                }
                if grad_norm is not None:
                    payload["train/grad_norm"] = _to_float(grad_norm)
                    payload["train/grad_norm_avg"] = _to_float(norm_meter.avg)
                if 'loss_intra' in outputs:
                    payload["train/loss_intra"] = _to_float(outputs['loss_intra'])
                    payload["train/loss_extra"] = _to_float(outputs['loss_extra'])
                    payload["train/loss_intra_avg"] = _to_float(loss_intra_meter.avg)
                    payload["train/loss_extra_avg"] = _to_float(loss_extra_meter.avg)
                swanlab.log(payload, step=it)
            end = time.time()
        
        if args.rank == 0:
            if (epoch+1) % args.eval_freq == 0 or epoch == args.epochs - 1 or epoch in args.eval_list:
                to_save = {
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': args
                            }
                torch.save(to_save, os.path.join(args.output_dir, f'epoch_{epoch}.pth.tar'))
            misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, is_best=False)
        epoch_time = time.time() - epoch_start
        if swanlab is not None:
            epoch_payload = {
                "time/epoch_sec": _to_float(epoch_time),
                "train/epoch": epoch,
                "train/epoch_loss_avg": _to_float(loss_meter.avg),
                "train/epoch_pred_var_avg": _to_float(var_meter.avg),
                "train/epoch_mask_enc_avg": _to_float(maskA_meter.avg),
                "train/epoch_mask_pred_avg": _to_float(maskB_meter.avg),
            }
            if norm_meter.count > 0:
                epoch_payload["train/epoch_grad_norm_avg"] = _to_float(norm_meter.avg)
            if loss_intra_meter.count > 0:
                epoch_payload["train/epoch_loss_intra_avg"] = _to_float(loss_intra_meter.avg)
                epoch_payload["train/epoch_loss_extra_avg"] = _to_float(loss_extra_meter.avg)
            if last_step is not None:
                epoch_payload["train/epoch_lr"] = _to_float(lr_schedule[last_step])
                epoch_payload["train/epoch_weight_decay"] = _to_float(wd_schedule[last_step])
                epoch_payload["train/epoch_ema"] = _to_float(momentum_schedule[last_step])
            swanlab.log(epoch_payload, step=last_step)
    total_time = time.time() - start_time
    if swanlab is not None:
        swanlab.log({"time/total_sec": _to_float(total_time)}, step=last_step)
        if hasattr(swanlab, "finish"):
            swanlab.finish()
    if args.rank == 0 and stop_epoch == args.epochs:
        os.remove(os.path.join(args.output_dir, 'checkpoint.pth'))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.exp_name:
        args.output_dir = os.path.join(args.output_dir, args.exp_name, f'seed{args.seed}')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
