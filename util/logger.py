import torch
import torch.distributed as dist
from collections import deque

class AverageMeter(object):
    """计算并保存当前值与平均值的计量器。

    使用位置: `train_jepa.py` 训练循环中的指标统计。
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sync=False):
        self.val = val
        self.sum += val * n
        self.count += n
        if sync:
            self.synchronize_between_processes()
        self.avg = self.sum / self.count


    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
