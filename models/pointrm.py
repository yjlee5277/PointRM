import math
from functools import partial

import torch
import torch.nn as nn

from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .block import Block
from .build import MODELS
from knn_cuda import KNN
import torch.nn.functional as F


# (B=32, N=1024, 3) ---> (B=32, S=64, K=32, 3+3)  (B=32, S=64, 3)
class LocalGrouper(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

        self.alpha = nn.Parameter(torch.ones([1, 1, 1, 3]))
        self.beta = nn.Parameter(torch.zeros([1, 1, 1, 3]))

    def forward(self, xyz):  # (B=32, N=1024, 3)
        batch_size, num_points, _ = xyz.shape

        # FPS: sample center points
        center = misc.fps(xyz, self.num_group)  # (B=32, S=64, 3)

        # KNN: select neighbor points
        _, idx = self.knn(xyz, center)  # (B=32, S=64, K=32)

        assert idx.size(1) == self.num_group  # S=64
        assert idx.size(2) == self.group_size  # K=32
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points  # (B=32, 1, 1)

        idx = idx + idx_base  # (B=32, S=64, K=32)
        idx = idx.view(-1)  # (B*S*K=65536)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, -1).contiguous()  # (B=32, S=64, K=32, 3)

        mean = center.unsqueeze(2)  # (B=32, S=64, 1, 3)
        std = torch.std((neighborhood - mean).reshape(batch_size, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        neighborhood = (neighborhood - mean) / (std + 1e-5)  # (B=32, S=64, K=32, 3)
        neighborhood = self.alpha * neighborhood + self.beta  # (B=32, S=64, K=32, 3)

        neighborhood = torch.cat([neighborhood, center.view(batch_size, self.num_group, 1, -1).repeat(1, 1, self.group_size, 1)], dim=-1)
        return neighborhood, center  # (B=32, S=64, K=32, 3+3)  (B=32, S=64, 3)


# x(B, S, K=12, 3+3) ---> x(B, D, S)
class Extraction(nn.Module):
    def __init__(self, encoder_channels):
        super(Extraction, self).__init__()
        self.transfer = nn.Sequential(
            nn.Conv1d(3 + 3, encoder_channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(encoder_channels),
            nn.ReLU(inplace=True),
        )

        self.pre_operation = ConvBNReLURes1D(encoder_channels)
        self.pos_operation = ConvBNReLURes1D(encoder_channels)

    def forward(self, x):  # (B, S, K, 3+3)
        b, s, k, d = x.size()
        x = x.permute(0, 1, 3, 2)  # (B, S, 3+3, K)
        x = x.reshape(-1, d, k)  # (B*S, 3+3, K)
        x = self.transfer(x)  # (B*S, C, K)

        bs, _, _ = x.size()
        x = self.pre_operation(x)  # (B*S, C, K)
        x = F.adaptive_max_pool1d(x, 1).view(bs, -1)  # (B*S, C)
        x = x.reshape(b, s, -1)  # (B, S, C)
        x = x.permute(0, 2, 1)  # (B, C, S)

        x = self.pos_operation(x)  # (B, C, S)
        return x.permute(0, 2, 1)  # (B, S, C)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, res_expansion=1.0):
        super(ConvBNReLURes1D, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, int(channel * res_expansion), kernel_size=1, bias=True),
            nn.BatchNorm1d(int(channel * res_expansion)),
            nn.ReLU(inplace=True)
        )

        self.net2 = nn.Sequential(
            nn.Conv1d(int(channel * res_expansion), channel, kernel_size=1, bias=True),
            nn.BatchNorm1d(channel)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # (B, C, N)
        return self.act(self.net2(self.net1(x)) + x)  # (B, C, N)


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(self, d_model: int, n_layer: int, ssm_cfg=None, norm_epsilon: float = 1e-5, rms_norm: bool = False,
                 initializer_cfg=None, fused_add_norm=False, residual_in_fp32=False, drop_out_in_block: int = 0.,
                 drop_path: int = 0.1, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(partial(_init_weights, n_layer=n_layer, **(initializer_cfg if initializer_cfg is not None else {})))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids
        residual = None
        hidden_states = hidden_states + pos
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual, inference_params=inference_params)
            hidden_states = self.drop_out_in_block(hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm


class CalImportance(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.cal_imp_score = nn.Sequential(
            nn.Conv1d(encoder_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
        )

    def forward(self, patch):  # (B, D, S)
        patch = patch.transpose(2, 1)  # (B, S, D)

        map_patch_feature = self.projection(patch)  # (B, 256, S)
        pred_importance = self.cal_imp_score(patch)  # (B, 1, S)

        return map_patch_feature.transpose(2, 1), pred_importance.transpose(2, 1)  # (B, S, 256)  (B, S, 1)


class GLR_loss(nn.Module):
    def __init__(self):
        super(GLR_loss, self).__init__()
        self.cossim = nn.CosineSimilarity(dim=-1)

    def forward(self, global_feature, local_feature, mask=None):
        local_feature = local_feature.transpose(2, 1)
        B, D, N = local_feature.shape
        device = global_feature.device
        local_feature = local_feature.transpose(1, 0).reshape(D, -1)
        score = torch.matmul(global_feature, local_feature) * 64.
        score = score.view(B, -1).transpose(1, 0)
        label = torch.arange(B).unsqueeze(1).expand(B, N).reshape(-1).to(device)

        if mask is None:
            CELoss = nn.CrossEntropyLoss()
            loss = CELoss(score, label)
        else:
            CELoss = nn.CrossEntropyLoss(reduction='none')
            loss = (CELoss(score, label) * mask.reshape(-1)).sum() / mask.sum()
        return loss


@MODELS.register_module()
class PointRM(nn.Module):
    def __init__(self, config, **kwargs):
        super(PointRM, self).__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.cls_dim = config.cls_dim

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = LocalGrouper(num_group=self.num_group, group_size=self.group_size)  # lee
        self.encoder = Extraction(encoder_channels=self.encoder_dims)  # lee

        self.importance_cal_block = CalImportance(self.encoder_dims)  # lee

        self.use_cls_token = False if not hasattr(self.config, "use_cls_token") else self.config.use_cls_token
        self.drop_path = 0. if not hasattr(self.config, "drop_path") else self.config.drop_path
        self.rms_norm = False if not hasattr(self.config, "rms_norm") else self.config.rms_norm
        self.drop_out_in_block = 0. if not hasattr(self.config, "drop_out_in_block") else self.config.drop_out_in_block

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.drop_out = nn.Dropout(config.drop_out) if "drop_out" in config else nn.Dropout(0)

        self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)

        self.norm = nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1

        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim * self.HEAD_CHANEL, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        # lee
        self.global_projection = nn.Sequential(
            nn.Conv1d(self.trans_dim * self.HEAD_CHANEL, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            Normalize(dim=1)
        )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_glr = GLR_loss()
        self.loss_importance = nn.SmoothL1Loss(reduction='none')

    def get_loss_acc(self, ret, gt, patch_f, global_f, pred_score, cos_sim):
        loss_CE = self.loss_ce(ret, gt.long())  # CE loss
        loss_GLR = self.loss_glr(global_f, patch_f)  # global-to-local align loss

        # importance regression loss
        loss_SCORE = self.loss_importance(pred_score.squeeze(-1), cos_sim).sum(dim=-1).mean()

        # predict
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss_CE, loss_GLR, loss_SCORE, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Mamba')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Mamba'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Mamba')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Mamba'
                )

            print_log(f'[Mamba] Successful Loading the ckpt from {bert_ckpt_path}', logger='Mamba')
        else:
            print_log('Training from scratch!!!', logger='Mamba')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):  # (B, N=1024, 3)
        neighborhood, center = self.group_divider(pts)  # (B, S, K, 3+3) (B, S, 3)
        group_input_tokens = self.encoder(neighborhood)  # (B, S, C)
        pos = self.pos_embed(center)  # (B, S, C)

        # 2.Bi-reordering Strategy   Lee
        map_patch_feature, cal_importance = self.importance_cal_block(group_input_tokens)  # (B, S, 256) (B, S, 1)

        importance_order1 = cal_importance.argsort(dim=1, descending=True)  # (B, S, 1)
        sort_cal_importance1 = cal_importance.gather(dim=1, index=torch.tile(importance_order1, (
            1, 1, cal_importance.shape[-1])))  # (B, S, 1)
        group_input_tokens1 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order1, (
            1, 1, group_input_tokens.shape[-1])))  # (B, S, C)
        pos1 = pos.gather(dim=1, index=torch.tile(importance_order1, (1, 1, pos.shape[-1])))  # (B, S, C)

        importance_order2 = cal_importance.argsort(dim=1, descending=False)  # (B, S, 1)
        sort_cal_importance2 = cal_importance.gather(dim=1, index=torch.tile(importance_order2, (
            1, 1, cal_importance.shape[-1])))  # (B, S, 1)
        group_input_tokens2 = group_input_tokens.gather(dim=1, index=torch.tile(importance_order2, (
            1, 1, group_input_tokens.shape[-1])))  # (B, S, C)
        pos2 = pos.gather(dim=1, index=torch.tile(importance_order2, (1, 1, pos.shape[-1])))  # (B, S, C)

        sort_cal_importance = torch.cat([sort_cal_importance1, sort_cal_importance2], dim=1)
        group_input_tokens = torch.cat([group_input_tokens1, group_input_tokens2], dim=1)  # (B, 2*S, C)
        pos = torch.cat([pos1, pos2], dim=1)  # (B, 2*S, C)

        # 3.Mamba Blocks
        x = group_input_tokens  # (B, 2*S, C)
        x = self.drop_out(x)  # (B, 2*S, C)
        x = self.blocks(x, pos)  # (B, 2*S, C)
        x = self.norm(x)  # (B, 2*S, C)

        # lee
        weight = sort_cal_importance.clamp(0, 1).repeat(1, 1, x.shape[-1])  # (B, 2*S, C)
        global_feature = (x * weight).sum(dim=1)  # (B, C)
        map_global_feature = self.global_projection(global_feature.unsqueeze(-1)).squeeze(-1)  # (B, 256)

        # Classification
        ret = self.cls_head(global_feature)  # (B, 40)

        return ret, map_patch_feature, map_global_feature, cal_importance
        # (B, 40)  (B, S, 256)  (B, 256)  (B, S, 1)


