import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import *
from .utils import apply_masks, repeat_interleave_batch, trunc_normal_
from .vit_backbone import VisionTransformerPredictor

class VisionTransformerPredictorConditioned(VisionTransformerPredictor):
    """带增强参数条件的预测器，支持多种条件融合方式。

    使用位置: `models/__init__.py:init_jepa_model` 在 IWM 模式下通过
    `vit_predictor_conditioned` 构造。
    """
    def __init__(self, policy_dim=4, policy_net_layers=3, unify_embed=False, cond_type='feat', **kwargs):
        super().__init__(**kwargs)
        if cond_type == 'concat':
            cond_type = 'feat'
        self.policy_dim = policy_dim
        self.cond_type = cond_type
        if policy_dim > 0:
            if cond_type == 'feat' or cond_type == 'feat_res':
                layers = [nn.Linear(self.predictor_embed_dim+self.policy_dim, self.predictor_embed_dim)]
                for _ in range(policy_net_layers-1):
                    layers.extend([nn.ReLU(), nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim)])
                self.policy_net = nn.Sequential(*layers)
            elif cond_type == 'feat_silu':
                layers = [nn.Linear(self.predictor_embed_dim+self.policy_dim, self.predictor_embed_dim)]
                for _ in range(policy_net_layers-1):
                    layers.extend([nn.SiLU(), nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim)])
                self.policy_net = nn.Sequential(*layers)
            elif cond_type == 'token':
                dummy_aug_param = torch.rand(1, policy_dim)
                encoding = fourier_encode(dummy_aug_param).view(-1)
                self.policy_net = nn.Sequential(
                    nn.Linear(encoding.shape[-1], 4 * self.predictor_embed_dim),
                    nn.SiLU(),
                    nn.Linear(4 * self.predictor_embed_dim, self.predictor_embed_dim),
                )
            elif cond_type == 'token_bare':
                self.policy_net = nn.Sequential(
                    nn.Linear(policy_dim, self.predictor_embed_dim),
                    nn.SiLU(),
                    nn.Linear(self.predictor_embed_dim, self.predictor_embed_dim),
                )
            else:
                raise NotImplementedError
        else:
            self.policy_net = None
        if unify_embed:
            rel_pos_21 = torch.zeros((1, 4), dtype=torch.float)
            pos_embs = get_2d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
            self.predictor_pos_embed.data.copy_(pos_embs)
    
    def forward(self, x, aug_params, masks_x, masks, is_mmb=False,rel_pos_21=None):
        assert (masks is not None), 'Cannot run predictor without mask indices'
        # assert aug_params.shape[-1] == self.policy_dim, f'Policy dim mismatch! {aug_params.shape[-1]} vs {self.policy_dim}'
        no_input_mask = (masks_x is None)
        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens, original ones
        if no_input_mask:
            x += self.interpolate_pos_encoding(x, self.predictor_pos_embed)
        else:
            x_pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
            x += apply_masks(x_pos_embs, masks_x) #[nenc*B]

        _, N_ctxt, D = x.shape

        if self.extra:
            # use unified embedding for mask token
            if self.is_3d:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 3), dtype=torch.float, device=x.device)
                pos_embs = get_3d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1], round(self.num_patches**(1/3)))
            elif 'easy' in self.ssl_type:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 4), dtype=torch.float, device=x.device)
                pos_embs = get_2d_sincos_pos_embed_relative_easy(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
            else:
                if rel_pos_21 is None:
                    rel_pos_21 = torch.zeros((B, 6), dtype=torch.float, device=x.device)
                    rel_pos_21[:, 2:4] = 1 # delta h,w
                pos_embs = get_2d_sincos_pos_embed_relative(rel_pos_21, self.predictor_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5)) #[B, L, H]
                pos_embs = self.predictor_pos_mlp(pos_embs.float())
        else:
            raise NotImplementedError
            # pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        # -- concat mask tokens to x
        pos_embs = apply_masks(pos_embs, masks) #[npred*B]
        if not is_mmb:
            pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x)) #[npred*nenc*B]
        else:
            pass #[nenc*npred*B]
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        
        # concat aug params
        if self.policy_net is not None:
            if 'feat' in self.cond_type:
                aug_params = aug_params.unsqueeze(1).repeat(pred_tokens.shape[0]//B, pred_tokens.shape[1], 1)
                if self.cond_type == 'feat_res':
                    pred_tokens = pred_tokens + self.policy_net(torch.cat([pred_tokens, aug_params], dim=-1))
                else:
                    pred_tokens = torch.cat([pred_tokens, aug_params], dim=-1)
                    pred_tokens = self.policy_net(pred_tokens)
            elif self.cond_type == 'token':
                aug_embedding = self.policy_net(fourier_encode(aug_params).view(B, -1)).unsqueeze(1)
                pred_tokens = torch.cat([pred_tokens, aug_embedding], dim=1)
            elif self.cond_type == 'token_bare':
                aug_embedding = self.policy_net(aug_params).unsqueeze(1)
                pred_tokens = torch.cat([pred_tokens, aug_embedding], dim=1)
        
        if not is_mmb:
            x = x.repeat(len(masks), 1, 1) #[npred*nenc*B]
        else:
            x = repeat_interleave_batch(x, B, repeat=len(masks) // len(masks_x)) #[nenc*npred*B]
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        if 'token' in self.cond_type:
            x = x[:, :-1]
        x = self.predictor_proj(x)

        return x



def vit_predictor_conditioned(**kwargs):
    model = VisionTransformerPredictorConditioned(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
