import torch
from torch import nn
from timm.layers import trunc_normal_

from . import vit_backbone
from .vit_backbone import vit_predictor
from .vit_predictor_conditioned import vit_predictor_conditioned
from .jepa_core import JEPA
from .jepa_iwm import IWM, IWM_Dual


def mae_init_weights(model):
    if hasattr(model, 'patch_embed'):
        w = model.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    if hasattr(model, 'cls_token'):
        torch.nn.init.normal_(model.cls_token, std=.02)
    if hasattr(model, 'mask_token'):
        torch.nn.init.normal_(model.mask_token, std=.02)

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(_init_weights)


def init_jepa_model(args, device, ssl_type):
    encoder = vit_backbone.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path,
    )

    if args.stop_grad_conv1:
        encoder.patch_embed.proj.weight.requires_grad = False
        encoder.patch_embed.proj.bias.requires_grad = False
    if args.stop_grad_norm1:
        encoder.blocks[0].norm1.weight.requires_grad = False
        encoder.blocks[0].norm1.bias.requires_grad = False

    args.extra = ('extra' in ssl_type) or ('dual' in ssl_type)

    common_args = dict(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=args.pred_emb_dim,
        depth=args.pred_depth,
        num_heads=encoder.num_heads,
    )

    # IWM uses a conditioned predictor that takes augmentation parameters.
    if 'iwm' in ssl_type:
        predictor = vit_predictor_conditioned(
            policy_dim=args.policy_dim if not args.iwm_disable else 0,
            policy_net_layers=3,
            **common_args,
            extra=args.extra,
            ssl_type=ssl_type,
            unify_embed=args.unify_embed,
            cond_type=args.cond_type
        )
    else:
        predictor = vit_predictor(extra=args.extra, ssl_type=ssl_type, **common_args)



    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    if args.mae_init_weights:
        mae_init_weights(encoder)
        mae_init_weights(predictor)
    else:
        for m in encoder.modules():
            init_weights(m)
        for m in predictor.modules():
            init_weights(m)
    list_mask = ('list' in args.mask_type)

    # Map SSL type to the corresponding forward logic.
    model_cls = {
        'iwm': IWM,
        'jepa': JEPA,
        'iwm_dual': IWM_Dual,
        'iwm_dual_easy': IWM_Dual,
    }[ssl_type]
    model = model_cls(
        encoder,
        predictor,
        list_mask=list_mask
    )


    if args.pretrained:
        sd = torch.load(args.pretrained, map_location='cpu')['model']
        encoder_sd = {k.replace('target_encoder.', ''): v for k, v in sd.items() if k.startswith('target_encoder.')}
        model.target_encoder.load_state_dict(encoder_sd)

    model.to(device)
    return model
