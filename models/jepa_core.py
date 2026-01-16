from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
import math
from functools import partial
from einops import rearrange
from .utils import apply_masks, repeat_interleave_batch, AllReduce
from .unigrad import compute_unigrad_loss



class JEPA(nn.Module):
    """JEPA 主模型，包含上下文编码器、目标编码器与预测器。

    用途: 计算目标分支特征并预测被 mask 的 token，返回自监督损失。
    使用位置: `models/__init__.py:init_jepa_model` 根据 `ssl_type` 选择该类，
    由 `train_jepa.py` 的训练循环调用。
    """
    def __init__(self, encoder, predictor, is_mmb=True, list_mask=False):
        super().__init__()
        self.encoder = encoder
        self.target_encoder = deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = predictor
        self.is_mmb = is_mmb
        self.list_mask = list_mask
    
    def forward_target(self, imgs, masks_enc, masks_pred, target_last_k, target_norm_type, return_raw=False):
        with torch.no_grad():

            h_list = self.target_encoder(imgs, layer_results=True)
            h_raw = h_list[-1]
            h_list = h_list[-target_last_k:]
            if target_norm_type == 'avg_ln':
                h = sum(h_list) / len(h_list) # from data2vec
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            elif target_norm_type == 'in_avg_ln':
                h_list = [F.instance_norm(h)  for h in h_list]  # normalize over feature-dim
                h = sum(h_list) / len(h_list) # from data2vec
                h = F.layer_norm(h, (h.size(-1),))
            else:
                raise NotImplementedError
            B = len(h)
            
            if self.list_mask:
                h_list = [apply_masks(h, masks_pred_sub) for masks_pred_sub in masks_pred]
                return h_list
            else:
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                if not self.is_mmb:
                    h = repeat_interleave_batch(h, B, repeat=len(masks_enc)) #[npred*nenc*B]
                else:
                    pass #[nenc*npred*B]
                if return_raw:
                    return h, h_raw
                else:
                    return h
    
    def forward_context(self, imgs, masks_enc, masks_pred):
        if self.list_mask:
            z_list = []
            for masks_enc_sub, masks_pred_sub in zip(masks_enc, masks_pred):
                z = self.encoder(imgs, masks_enc_sub)
                z = self.predictor(z, masks_enc_sub, masks_pred_sub, is_mmb=True)
                z_list.append(z)
            return z_list
        else:
            z = self.encoder(imgs, masks_enc)
            z = self.predictor(z, masks_enc, masks_pred, is_mmb=self.is_mmb)
            return z
    
    def loss_fn(self, z, h, loss_type):
        def _loss(zz, hh):
            if loss_type == 'l1':
                loss = F.smooth_l1_loss(zz, hh)
            elif loss_type == 'l2':
                loss = F.mse_loss(zz, hh)
            elif loss_type == 'unigrad':
                with torch.cuda.amp.autocast(enabled=False):
                    loss = compute_unigrad_loss(zz.float(), hh.float())
            else:
                raise NotImplementedError
            return loss
        if self.list_mask:
            losses = [_loss(z1, h1) for z1, h1 in zip(z, h)]
            loss = sum(losses) / len(losses)
        else:
            loss = _loss(z, h)
        loss = AllReduce.apply(loss)
        return loss

    
    def forward(self, data, args):
        imgs, masks_enc, masks_pred = data
        h = self.forward_target(imgs, masks_enc, masks_pred, args.target_last_k, args.target_norm_type)
        z = self.forward_context(imgs, masks_enc, masks_pred)
        loss = self.loss_fn(z, h, args.loss_type)
        return loss

    def update_target_encoder(self, m):
        # with torch.no_grad():
        #     for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
        #         param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
        with torch.no_grad():
            torch._foreach_mul_(list(self.target_encoder.parameters()), m)
            torch._foreach_add_(list(self.target_encoder.parameters()), list(self.encoder.parameters()), alpha=1 - m)
