import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from opts import parser as base_parser
from models import vit_backbone
from data_utils.imagenet import ImageNetHFPretrain
from data_utils.transforms import build_transform_rgb
from util import cosine_scheduler
from util.logger import AverageMeter


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
        torch.cuda.set_device(args.gpu)
        args.dist_url = "env://"
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            device_id=args.gpu,
        )
        dist.barrier()
    else:
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False


def is_main_process(args):
    return args.rank == 0


def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def build_dataset(args, split, is_train, data_pct=1.0):
    transform = build_transform_rgb(args, is_train=is_train)
    cache_dir = args.hf_cache_dir or None
    return ImageNetHFPretrain(
        dataset_name=args.hf_dataset,
        split=split,
        transform=transform,
        data_pct=data_pct,
        dataset_seed=args.dataset_seed,
        cache_dir=cache_dir,
        return_label=True,
    )


def load_encoder_weights(args, encoder):
    if not args.pretrained:
        raise ValueError("--pretrained 不能为空，需要提供 JEPA 训练得到的 checkpoint 路径。")
    try:
        checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    prefix = f"{args.encoder_source}."
    enc_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
    if not enc_state:
        raise ValueError(
            f"未在 checkpoint 中找到以 '{prefix}' 开头的权重，请检查 --encoder_source 是否正确。"
        )
    load_result = encoder.load_state_dict(enc_state, strict=False)
    missing = getattr(load_result, "missing_keys", [])
    unexpected = getattr(load_result, "unexpected_keys", [])
    if is_main_process(args):
        if missing:
            print("Missing keys when loading encoder:", missing)
        if unexpected:
            print("Unexpected keys when loading encoder:", unexpected)


def main(args):
    init_distributed_mode(args)
    device = torch.device("cuda", args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    cudnn.benchmark = True

    if args.exp_name:
        args.output_dir = os.path.join(args.output_dir, args.exp_name, f"seed{args.seed}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed + args.rank)

    dataset_train = build_dataset(args, args.train_split, is_train=True, data_pct=args.data_pct)
    dataset_val = build_dataset(args, args.val_split, is_train=False, data_pct=args.val_pct)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, num_replicas=args.world_size, rank=args.rank, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    encoder = vit_backbone.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
    ).to(device)
    load_encoder_weights(args, encoder)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    probe = torch.nn.Sequential(
        torch.nn.LayerNorm(encoder.embed_dim),
        torch.nn.Linear(encoder.embed_dim, args.probe_num_classes),
    ).to(device)

    if args.distributed:
        probe = DistributedDataParallel(probe, device_ids=[args.gpu], static_graph=True)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_wd,
        betas=args.betas,
    )

    iters_per_epoch = len(data_loader_train)
    lr_schedule = cosine_scheduler(
        args.probe_lr,
        args.min_lr,
        args.epochs,
        iters_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.start_lr,
    )

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    best_acc1 = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        probe.train()
        loss_meter = AverageMeter("Loss", ":.4f")
        acc1_meter = AverageMeter("Acc@1", ":.2f")
        acc5_meter = AverageMeter("Acc@5", ":.2f")
        for step, (images, labels) in enumerate(data_loader_train):
            it = epoch * iters_per_epoch + step
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with torch.no_grad():
                feats = encoder(images).mean(dim=1)
            if scaler is None:
                logits = probe(feats)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    logits = probe(feats)
                    loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

        if args.distributed:
            loss_meter.synchronize_between_processes()
            acc1_meter.synchronize_between_processes()
            acc5_meter.synchronize_between_processes()

        if is_main_process(args):
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Train Loss {loss_meter.avg:.4f} "
                f"Acc@1 {acc1_meter.avg:.2f} "
                f"Acc@5 {acc5_meter.avg:.2f}"
            )

        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
            acc1 = validate(args, encoder, probe, data_loader_val, device)
            if is_main_process(args):
                is_best = acc1 > best_acc1
                best_acc1 = max(best_acc1, acc1)
                ckpt = {
                    "epoch": epoch,
                    "probe": probe.module.state_dict() if isinstance(probe, DistributedDataParallel) else probe.state_dict(),
                    "best_acc1": best_acc1,
                    "args": args,
                }
                torch.save(ckpt, os.path.join(args.output_dir, "checkpoint.pth"))
                if is_best:
                    torch.save(ckpt, os.path.join(args.output_dir, "checkpoint-best.pth"))


def validate(args, encoder, probe, data_loader, device):
    encoder.eval()
    probe.eval()
    loss_meter = AverageMeter("Loss", ":.4f")
    acc1_meter = AverageMeter("Acc@1", ":.2f")
    acc5_meter = AverageMeter("Acc@5", ":.2f")

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            feats = encoder(images).mean(dim=1)
            logits = probe(feats)
            loss = F.cross_entropy(logits, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))

    if args.distributed:
        loss_meter.synchronize_between_processes()
        acc1_meter.synchronize_between_processes()
        acc5_meter.synchronize_between_processes()

    if is_main_process(args):
        print(
            f"Val Loss {loss_meter.avg:.4f} "
            f"Acc@1 {acc1_meter.avg:.2f} "
            f"Acc@5 {acc5_meter.avg:.2f}"
        )
    return acc1_meter.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Linear Probe on ImageNet", parents=[base_parser])
    parser.add_argument("--encoder_source", type=str, default="encoder", choices=["encoder", "target_encoder"])
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--val_pct", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
  # torchrun --nproc_per_node=8 linear_probe.py \
  #   --pretrained /path/to/checkpoint.pth \
  #   --model vit_base \
  #   --batch_size 256 \
  #   --epochs 100 \
  #   --probe_lr 1e-3 \
  #   --hf_dataset ILSVRC/imagenet-1k