# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import shutil

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def init_dist_pytorch(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    # args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.rank), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        device_id=args.gpu,
    )
    torch.distributed.barrier()

class NativeScalerWithGradNormCount:
    """AMP 梯度缩放器封装，支持可选梯度裁剪。

    使用位置: `train_jepa.py` 训练循环中用于混合精度与梯度裁剪。
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, is_best=False, with_epoch=False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    # if loss_scaler is not None:
    if with_epoch:
        checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)
    else:
        checkpoint_path = output_dir / ('checkpoint.pth')
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }
    torch.save(to_save, checkpoint_path)
    if is_best:
        best_path = output_dir / 'checkpoint-best.pth'  
        shutil.copy(checkpoint_path, best_path) 


def load_model(args, model_without_ddp, optimizer, loss_scaler, new_start=False):
    if args.resume == '' and os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        print('Auto Resume Activated!')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['state_dict'])
        else:
            raise Exception('No State Dict!')
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")
        if new_start:
            print('New Start from pretrained.')
            args.start_epoch = 0
