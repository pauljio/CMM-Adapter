# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Callable, Dict, List, Optional, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.init import normal_

from .adapter import Expert, TopKRouter
from .ops.modules import MSDeformAttn
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn


# MSDeformAttn Transformer encoder in deformable detr
# class MSDeformAttnTransformerEncoderOnly(nn.Module):
#     def __init__(self, d_model=256, nhead=8,
#                  num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
#                  activation="relu",
#                  num_feature_levels=4, enc_n_points=4,
#         ):
#         super().__init__()

#         self.d_model = d_model
#         self.nhead = nhead

#         encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
#                                                             dropout, activation,
#                                                             num_feature_levels, nhead, enc_n_points)
#         self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

#         self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         normal_(self.level_embed)

#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio

#     def forward(self, srcs, pos_embeds):
#         masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
#         # prepare input for encoder
#         src_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
#             bs, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             src = src.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             src_flatten.append(src)
#             mask_flatten.append(mask)
#         src_flatten = torch.cat(src_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

#         # encoder
#         memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

#         return memory, spatial_shapes, level_start_index


# class MSDeformAttnTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4):
#         super().__init__()

#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
#         # self attention
#         src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         return src


# class MSDeformAttnTransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers

#     @staticmethod
#     def get_reference_points(spatial_shapes, valid_ratios, device):
#         reference_points_list = []
#         for lvl, (H_, W_) in enumerate(spatial_shapes):

#             ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#                                           torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points

#     def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
#         output = src
#         reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
#         for _, layer in enumerate(self.layers):
#             output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

#         return output


# @SEM_SEG_HEADS_REGISTRY.register()
# class MSDeformAttnPixelDecoder(nn.Module):
#     @configurable
#     def __init__(
#         self,
#         input_shape: Dict[str, ShapeSpec],
#         *,
#         transformer_dropout: float,
#         transformer_nheads: int,
#         transformer_dim_feedforward: int,
#         transformer_enc_layers: int,
#         conv_dim: int,
#         mask_dim: int,
#         norm: Optional[Union[str, Callable]] = None,
#         # deformable transformer encoder args
#         transformer_in_features: List[str],
#         common_stride: int,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             input_shape: shapes (channels and stride) of the input features
#             transformer_dropout: dropout probability in transformer
#             transformer_nheads: number of heads in transformer
#             transformer_dim_feedforward: dimension of feedforward network
#             transformer_enc_layers: number of transformer encoder layers
#             conv_dims: number of output channels for the intermediate conv layers.
#             mask_dim: number of output channels for the final conv layer.
#             norm (str or callable): normalization for all conv layers
#         """
#         super().__init__()
#         transformer_input_shape = {
#             k: v for k, v in input_shape.items() if k in transformer_in_features
#         }

#         # this is the input shape of pixel decoder
#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
#         self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
#         self.feature_strides = [v.stride for k, v in input_shape]
#         self.feature_channels = [v.channels for k, v in input_shape]
        
#         # this is the input shape of transformer encoder (could use less features than pixel decoder
#         transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
#         self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
#         transformer_in_channels = [v.channels for k, v in transformer_input_shape]
#         self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers

#         self.transformer_num_feature_levels = len(self.transformer_in_features)
#         if self.transformer_num_feature_levels > 1:
#             input_proj_list = []
#             # from low resolution to high resolution (res5 -> res2)
#             for in_channels in transformer_in_channels[::-1]:
#                 input_proj_list.append(nn.Sequential(
#                     nn.Conv2d(in_channels, conv_dim, kernel_size=1),
#                     nn.GroupNorm(32, conv_dim),
#                 ))
#             self.input_proj = nn.ModuleList(input_proj_list)
#         else:
#             self.input_proj = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
#                     nn.GroupNorm(32, conv_dim),
#                 )])

#         for proj in self.input_proj:
#             nn.init.xavier_uniform_(proj[0].weight, gain=1)
#             nn.init.constant_(proj[0].bias, 0)

#         self.transformer = MSDeformAttnTransformerEncoderOnly(
#             d_model=conv_dim,
#             dropout=transformer_dropout,
#             nhead=transformer_nheads,
#             dim_feedforward=transformer_dim_feedforward,
#             num_encoder_layers=transformer_enc_layers,
#             num_feature_levels=self.transformer_num_feature_levels,
#         )
#         N_steps = conv_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

#         self.mask_dim = mask_dim
#         # use 1x1 conv instead
#         self.mask_features = Conv2d(
#             conv_dim,
#             mask_dim,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#         )
#         weight_init.c2_xavier_fill(self.mask_features)
        
#         self.maskformer_num_feature_levels = 3  # always use 3 scales
#         self.common_stride = common_stride

#         # extra fpn levels
#         stride = min(self.transformer_feature_strides)
#         # stride = max(self.transformer_feature_strides)
#         self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

#         lateral_convs = []
#         output_convs = []

#         use_bias = norm == ""
#         for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
#             lateral_norm = get_norm(norm, conv_dim)
#             output_norm = get_norm(norm, conv_dim)

#             lateral_conv = Conv2d(
#                 in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
#             )
#             output_conv = Conv2d(
#                 conv_dim,
#                 conv_dim,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=use_bias,
#                 norm=output_norm,
#                 activation=F.relu,
#             )
#             weight_init.c2_xavier_fill(lateral_conv)
#             weight_init.c2_xavier_fill(output_conv)
#             self.add_module("adapter_{}".format(idx + 1), lateral_conv)
#             self.add_module("layer_{}".format(idx + 1), output_conv)

#             lateral_convs.append(lateral_conv)
#             output_convs.append(output_conv)
#         # Place convs into top-down order (from low to high resolution)
#         # to make the top-down computation in forward clearer.
#         self.lateral_convs = lateral_convs[::-1]
#         self.output_convs = output_convs[::-1]

#     @classmethod
#     def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
#         ret = {}
#         ret["input_shape"] = {
#             k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
#         }
#         ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
#         ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
#         ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
#         ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
#         ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
#         # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
#         ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
#         ret[
#             "transformer_enc_layers"
#         ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
#         ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
#         ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
#         return ret

#     @autocast(enabled=False)
#     def forward_features(self, features):
#         srcs = []
#         pos = []
#         # Reverse feature maps into top-down order (from low to high resolution)
#         for idx, f in enumerate(self.transformer_in_features[::-1]):
#             x = features[f].float()  # deformable detr does not support half precision
#             srcs.append(self.input_proj[idx](x))
#             pos.append(self.pe_layer(x))

#         y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
#         bs = y.shape[0]

#         split_size_or_sections = [None] * self.transformer_num_feature_levels
#         for i in range(self.transformer_num_feature_levels):
#             if i < self.transformer_num_feature_levels - 1:
#                 split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
#             else:
#                 split_size_or_sections[i] = y.shape[1] - level_start_index[i]
#         y = torch.split(y, split_size_or_sections, dim=1)

#         out = []
#         multi_scale_features = []
#         num_cur_levels = 0
#         for i, z in enumerate(y):
#             out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

#         # append `out` with extra FPN levels
#         # Reverse feature maps into top-down order (from low to high resolution)
#         for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
#             x = features[f].float()
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]
#             cur_fpn = lateral_conv(x)
#             # Following FPN implementation, we use nearest upsampling here
#             y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
#             y = output_conv(y)
#             out.append(y)

#         for o in out:
#             if num_cur_levels < self.maskformer_num_feature_levels:
#                 multi_scale_features.append(o)
#                 num_cur_levels += 1

#         return self.mask_features(out[-1]), out[0], multi_scale_features


class MSDeformAttnTransformerAdapterEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 adapter_enabled=True,
                 dim_adapter_mid=64,  # cross modal adapter mid dim
                 adapter_count=-1,  # number of shared adapters across encoder layers
                 num_experts=4, fixed_experts=2, top_k=2, patches=4, # MoE args
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        # transformer encoder with adapter
        encoder_layer = MSDeformAttnTransformerAdapterEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points,
            adapter_enabled, dim_adapter_mid, num_experts, fixed_experts, top_k, patches,
        )
        self.encoder = MSDeformAttnTransformerAdapterEncoder(
            encoder_layer,
            num_encoder_layers,
            adapter_enabled=adapter_enabled,
            adapter_count=adapter_count,
            d_model=d_model,
            d_mid=dim_adapter_mid,
            n_experts=num_experts,
            fixed_experts=fixed_experts,
            top_k=top_k,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs_r, srcs_x, pos_embeds_r, pos_embeds_x):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs_r]
        # prepare input for encoder
        src_flatten_r = []
        src_flatten_x = []
        mask_flatten = []
        lvl_pos_embed_flatten_r = []
        lvl_pos_embed_flatten_x = []
        spatial_shapes = []
        for lvl, (src_r, src_x, mask, pos_embed_r, pos_embed_x) in enumerate(
                zip(srcs_r, srcs_x, masks, pos_embeds_r, pos_embeds_x)):
            bs, c, h, w = src_r.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src_r = src_r.flatten(2).transpose(1, 2)
            src_x = src_x.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed_r = pos_embed_r.flatten(2).transpose(1, 2)
            pos_embed_x = pos_embed_x.flatten(2).transpose(1, 2)
            lvl_pos_embed_r = pos_embed_r + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_x = pos_embed_x + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten_r.append(lvl_pos_embed_r)
            lvl_pos_embed_flatten_x.append(lvl_pos_embed_x)
            src_flatten_r.append(src_r)
            src_flatten_x.append(src_x)
            mask_flatten.append(mask)
        src_flatten_r = torch.cat(src_flatten_r, 1)
        src_flatten_x = torch.cat(src_flatten_x, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten_r = torch.cat(lvl_pos_embed_flatten_r, 1)
        lvl_pos_embed_flatten_x = torch.cat(lvl_pos_embed_flatten_x, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten_r.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory_r, memory_x = self.encoder(
            src_flatten_r, src_flatten_x, spatial_shapes, level_start_index,
            valid_ratios, lvl_pos_embed_flatten_r, lvl_pos_embed_flatten_x, mask_flatten
        )

        return memory_r, memory_x, spatial_shapes, level_start_index


class MSDeformAttnTransformerAdapterEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 adapter_enabled=True,
                 d_mid=64,  # cross modal adapter mid dim
                 n_experts=4, fixed_experts=2, top_k=2, patches=4, # MoE args
                 ):
        super().__init__()
        self.adapter_enabled = adapter_enabled

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross modal adapter -- no-sharing
        # self.adapter_r2x = CrossModalAdapter(d_model, d_mid, dropout=dropout)  # RGB -> X
        # self.adapter_x2r = CrossModalAdapter(d_model, d_mid, dropout=dropout)  # X -> RGB

        # cross modal MoE adapter
        self.top_k = top_k
        self.patches = patches
        if self.adapter_enabled:
            # routers are always per-layer (never shared)
            self.router_r = TopKRouter(d_model, n_experts, top_k, fixed_experts=fixed_experts)  # router for RGB
            self.router_x = TopKRouter(d_model, n_experts, top_k, fixed_experts=fixed_experts)  # router for X
        else:
            self.router_r = None
            self.router_x = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_moe(self, src, router, moe_adapter, spatial_shapes, level_start_index):
        if not self.adapter_enabled or router is None or moe_adapter is None:
            return torch.zeros_like(src)
        probs, top_k_indices = router(src, spatial_shapes, level_start_index, self.patches)
        # flatten to concat every batch
        src_flatten = src.view(-1, src.shape[-1])
        probs_flatten = probs.view(-1, probs.shape[-1])

        final_output = torch.zeros_like(src)
        for i, expert in enumerate(moe_adapter.experts):
            expert_mask = (top_k_indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_output = expert(src_flatten[flat_mask])
                gating_scores = probs_flatten[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output

    def forward(
            self, src_r, src_x, pos_r, pos_x, reference_points,
            spatial_shapes, level_start_index,
            padding_mask=None,
            moe_adapter=None,
    ):
        # attention
        src_r_attn = self.self_attn(
            self.with_pos_embed(src_r, pos_r), reference_points, src_r, spatial_shapes, level_start_index, padding_mask
        )
        src_r_attn = src_r + self.dropout1(src_r_attn)
        src_r_attn = self.norm1(src_r_attn)

        src_x_attn = self.self_attn(
            self.with_pos_embed(src_x, pos_x), reference_points, src_x, spatial_shapes, level_start_index, padding_mask
        )
        src_x_attn = src_x + self.dropout1(src_x_attn)
        src_x_attn = self.norm1(src_x_attn)

        # # cross modal adapter1
        # src_r_attn = src_r_attn + self.adapter_x2r(src_x)  # X -> RGB
        # src_x_attn = src_x_attn + self.adapter_r2x(src_r)  # RGB -> X

        # FFN
        src_r_ffn = self.forward_ffn(src_r_attn)
        src_x_ffn = self.forward_ffn(src_x_attn)

        # # cross modal adapter2
        # src_r_out = src_r_ffn + self.adapter_x2r(src_x_attn)  # X -> RGB
        # src_x_out = src_x_ffn + self.adapter_r2x(src_r_attn)  # RGB -> X

        # cross modal MoE adapter
        if self.adapter_enabled:
            src_r_out = src_r_ffn + self.forward_moe(
                src_x_attn, self.router_x, moe_adapter, spatial_shapes, level_start_index
            )  # X -> RGB
            src_x_out = src_x_ffn + self.forward_moe(
                src_r_attn, self.router_r, moe_adapter, spatial_shapes, level_start_index
            )  # RGB -> X
        else:
            src_r_out = src_r_ffn
            src_x_out = src_x_ffn

        # only one cross modal adapter
        # src_r_out = src_r_ffn + self.adapter_x2r(src_x)  # X -> RGB
        # src_x_out = src_x_ffn + self.adapter_r2x(src_r)  # RGB -> X

        # # only one cross modal MoE adapter
        # src_r_out = src_r_ffn + self.forward_moe(src_x, self.router_x, spatial_shapes, level_start_index)
        # src_x_out = src_x_ffn + self.forward_moe(src_r, self.router_r, spatial_shapes, level_start_index)

        return src_r_out, src_x_out


class MSDeformAttnTransformerAdapterEncoder(nn.Module):
    class _MoEAdapter(nn.Module):
        def __init__(self, d_model, d_mid, n_experts):
            super().__init__()
            self.experts = nn.ModuleList([Expert(d_model, d_mid, dropout=0.1) for _ in range(n_experts)])

    def __init__(
            self,
            encoder_layer,
            num_layers,
            adapter_enabled=True,
            adapter_count=-1,
            d_model=256,
            d_mid=64,
            n_experts=4,
            fixed_experts=2,
            top_k=2,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.adapter_enabled = adapter_enabled

        if self.adapter_enabled:
            if adapter_count <= 0:
                adapter_count = self.num_layers
            if adapter_count < 1 or adapter_count > self.num_layers:
                raise ValueError(
                    f"adapter_count must be in [1, {self.num_layers}] for encoder with {self.num_layers} layers, got {adapter_count}."
                )
            self.adapter_count = adapter_count
        else:
            self.adapter_count = 0

        # Shared MoE adapters controlled by COUNT; routers remain inside each layer.
        self.moe_adapters = nn.ModuleList()
        self.layer_to_adapter_idx = []
        if self.adapter_enabled and self.adapter_count > 0:
            for _ in range(self.adapter_count):
                self.moe_adapters.append(
                    self._MoEAdapter(d_model, d_mid, n_experts)
                )
            self.layer_to_adapter_idx = [
                min(i * self.adapter_count // self.num_layers, self.adapter_count - 1)
                for i in range(self.num_layers)
            ]

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
            self, src_r, src_x, spatial_shapes, level_start_index,
            valid_ratios, pos_r=None, pos_x=None, padding_mask=None
    ):
        output_r = src_r
        output_x = src_x
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_r.device)
        for i, layer in enumerate(self.layers):
            moe_adapter = None
            if self.adapter_enabled and self.adapter_count > 0:
                moe_adapter = self.moe_adapters[self.layer_to_adapter_idx[i]]
            output_r, output_x = layer(
                output_r, output_x, pos_r, pos_x,
                reference_points, spatial_shapes, level_start_index,
                padding_mask,
                moe_adapter,
            )

        return output_r, output_x


@SEM_SEG_HEADS_REGISTRY.register()
class MSDeformAttnAdapterPixelDecoder(nn.Module):
    @configurable
    def __init__(
            self,
            input_shape: Dict[str, ShapeSpec],
            *,
            transformer_dropout: float,
            transformer_nheads: int,
            transformer_dim_feedforward: int,
            transformer_enc_layers: int,
            conv_dim: int,
            mask_dim: int,
            norm: Optional[Union[str, Callable]] = None,
            # deformable transformer encoder args
            transformer_in_features: List[str],
            common_stride: int,
            # cross modal MoE adapter args
            adapter_enabled: bool,
            adapter_mid_dim: int,
            adapter_count: int,
            num_experts: int,
            fixed_experts: int,  # number of fixed experts
            top_k: int,
            patches: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
            transformer_in_features: input features for transformer encoder
            common_stride: common stride for extra FPN layers
            adapter_mid_dim: cross modal adapter mid dimension
            adapter_count: number of adapters shared across encoder layers.
                If <= 0, use one adapter per encoder layer.
            num_experts: number of experts in MoE
            fixed_experts: number of fixed experts in MoE
            top_k: top k experts in MoE
            patches: number of patches for MoE routing
        """
        super().__init__()
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # ["res2", "res3", "res4", "res5"]
        self.feature_strides = [v.stride for k, v in input_shape]  # [4, 8, 16, 32]
        self.feature_channels = [v.channels for k, v in input_shape]  # [192, 384, 768, 1536]

        # this is the input shape of transformer encoder (could use less features than pixel decoder)
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # ["res3", "res4", "res5"]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]  # [384, 768, 1536]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # [8, 16, 32]

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerAdapterEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            adapter_enabled=adapter_enabled,
            dim_adapter_mid=adapter_mid_dim,
            adapter_count=adapter_count,
            num_experts=num_experts,
            fixed_experts=fixed_experts,
            top_k=top_k,
            patches=patches,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))  # log2(8) - log2(4) = 1

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret["transformer_dim_feedforward"] = 1024  # use 1024 for deformable transformer encoder
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_in_features"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        # moe adapter config
        adapter_cfg = cfg.MODEL.SEM_SEG_HEAD.ADAPTER
        ret["adapter_enabled"] = adapter_cfg.ENABLED
        ret["adapter_mid_dim"] = adapter_cfg.DIM
        ret["adapter_count"] = adapter_cfg.COUNT
        ret["num_experts"] = adapter_cfg.NUM_EXPERTS
        ret["fixed_experts"] = adapter_cfg.FIXED_EXPERTS
        ret["top_k"] = adapter_cfg.TOP_K
        ret["patches"] = adapter_cfg.PATCHES

        return ret

    @autocast(enabled=False)
    def forward_features(self, features_r, features_x):
        srcs_r = []
        srcs_x = []
        pos_r = []
        pos_x = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x_r = features_r[f].float()  # deformable detr does not support half precision
            x_x = features_x[f].float()
            srcs_r.append(self.input_proj[idx](x_r))
            srcs_x.append(self.input_proj[idx](x_x))
            pos_r.append(self.pe_layer(x_r))
            pos_x.append(self.pe_layer(x_x))

        # modify `transformer` to accept two sets of features
        y_r, y_x, spatial_shapes, level_start_index = self.transformer(srcs_r, srcs_x, pos_r, pos_x)

        bs = y_r.shape[0]

        # split y_r and y_x into multiscale features
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y_r.shape[1] - level_start_index[i]
        y_r = torch.split(y_r, split_size_or_sections, dim=1)
        y_x = torch.split(y_x, split_size_or_sections, dim=1)

        # convert feature tensors into feature maps
        out_r = []
        out_x = []
        multi_scale_features_r = []
        multi_scale_features_x = []
        for i, (z_r, z_x) in enumerate(zip(y_r, y_x)):
            # B, N, C -> B, C, H, W
            out_r.append(z_r.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
            out_x.append(z_x.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x_r = features_r[f].float()
            x_x = features_x[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn_r = lateral_conv(x_r)
            cur_fpn_x = lateral_conv(x_x)
            # Following FPN implementation, we use nearest upsampling here
            y_r = cur_fpn_r + F.interpolate(out_r[-1], size=cur_fpn_r.shape[-2:], mode="bilinear", align_corners=False)
            y_x = cur_fpn_x + F.interpolate(out_x[-1], size=cur_fpn_x.shape[-2:], mode="bilinear", align_corners=False)
            y_r = output_conv(y_r)
            y_x = output_conv(y_x)
            out_r.append(y_r)
            out_x.append(y_x)
            
        # only keep the first `maskformer_num_feature_levels`(3) scales
        num_cur_levels = 0
        for o_r, o_x in zip(out_r, out_x):
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features_r.append(o_r)
                multi_scale_features_x.append(o_x)
                num_cur_levels += 1

        return (
            self.mask_features(out_r[-1]), out_r[0], multi_scale_features_r,
            self.mask_features(out_x[-1]), out_x[0], multi_scale_features_x
        )