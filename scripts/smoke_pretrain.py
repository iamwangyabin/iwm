import argparse
import torch
from torch.utils.data import DataLoader

from opts import parser as base_parser
from data_utils import build_pretrain_dataset
from masks import build_mask_collator
from models import init_jepa_model
from util import nested_to_gpu


def parse_args():
    parser = argparse.ArgumentParser('Smoke test for JEPA/IWM pretraining', parents=[base_parser])
    parser.add_argument('--device', default='cpu', type=str, help='cpu or cuda')
    parser.add_argument('--steps', default=1, type=int, help='number of iterations')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.imagenet_hf and not args.data_path:
        raise SystemExit('Please set --data_path to ImageNet root (with train/) or use --imagenet_hf.')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but not available.')

    dataset = build_pretrain_dataset(args, args.ssl_type)
    collator = build_mask_collator(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        drop_last=True
    )

    model = init_jepa_model(args=args, device=device, ssl_type=args.ssl_type)
    model.eval()

    it = 0
    with torch.no_grad():
        for data in loader:
            data = nested_to_gpu(data, device)
            outputs = model(data, args)
            if isinstance(outputs, dict):
                loss = outputs['loss']
                extra = f" loss_intra={outputs.get('loss_intra', 0):.4f} loss_extra={outputs.get('loss_extra', 0):.4f}"
            else:
                loss = outputs
                extra = ''
            print(f"step={it} loss={loss.item():.6f}{extra}")
            it += 1
            if it >= args.steps:
                break


if __name__ == '__main__':
    main()
