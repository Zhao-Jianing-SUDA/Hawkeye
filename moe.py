from abc import ABC, abstractmethod

import torch
import torch.nn as nn
# import torch_geometric.nn as gnn
# from torch_geometric.nn import MessagePassing
# from torch_geometric.data import Data
# from torch_geometric.utils import add_self_loops, degree
#
# from .multimodal_encoder.builder import build_image_tower, build_video_tower
# from .multimodal_projector.builder import build_vision_projector
from dataclasses import dataclass
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from typing import Optional, Tuple
from flash_attn import flash_attn_func
import torch.nn.functional as F
import copy


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


default_linear_init = nn.init.xavier_uniform_


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                             [: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim -
                  1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
        )

        self.flash = True
        self.k_cache, self.v_cache = None, None

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.k_cache is None or self.v_cache is None:
            keys, values = xk, xv
        else:
            self.k_cache = self.k_cache.to(xk)
            self.v_cache = self.v_cache.to(xv)
            self.k_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xk
            self.v_cache[:bsz, start_pos: start_pos + seqlen, :, :] = xv
            keys = self.k_cache[:bsz, :start_pos + seqlen]
            values = self.v_cache[:bsz, :start_pos + seqlen]

        output = flash_attn_func(
            xq, keys, values, dropout_p=0.0, causal=mask is not None)
        output = output.contiguous().view(bsz, seqlen, -1)

        return self.wo(output)

    def allocate_kv_cache(self, max_batch_size: int, max_seq_len: int) -> None:
        kv_cache_shape = (max_batch_size, max_seq_len,
                          self.n_local_heads, self.head_dim)
        if self.k_cache is None or self.k_cache.size() != kv_cache_shape:
            self.k_cache = torch.empty(kv_cache_shape)
        if self.v_cache is None or self.v_cache.size() != kv_cache_shape:
            self.v_cache = torch.empty(kv_cache_shape)

    def destroy_kv_cache(self) -> None:
        self.k_cache, self.v_cache = None, None


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * \
                     ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def _silu_gating(self, x, y):
        return F.silu(x) * y

    def forward(self, x):
        return self.w2(self._silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def _forward_ffn(self, h):
        return h + self.feed_forward(self.ffn_norm(h))

    def _forward_attention(self, x, start_pos, freqs_cis, mask, prompt):
        return x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, prompt)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                prompt=None):
        h = self._forward_attention(x, start_pos, freqs_cis, mask, prompt)
        out = self._forward_ffn(h)
        return out


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MOE(nn.Module):
    def __init__(self, params):
        super(MOE, self).__init__()
        self.resample_layers = nn.ModuleDict()
        self.num_experts = 2
        self.num_resample_layers = 2
        for expert in range(self.num_experts):
            expert = str(expert)
            self.resample_layers[expert] = nn.ModuleList()
            resampler_params = copy.deepcopy(params)
            resampler_params.n_heads = 16
            for layer_id in range(self.num_resample_layers):
                self.resample_layers[expert].append(
                    TransformerBlock(layer_id, resampler_params))

        self.resample_tokens = nn.ParameterDict()
        self.routers = nn.ModuleDict()
        self.clip_proj1 = nn.ModuleDict()
        self.clip_proj2 = nn.ModuleDict()
        self.routers = nn.ModuleDict()
        self.start_tag = nn.ParameterDict()
        self.end_tag = nn.ParameterDict()

        for modal in ['pose', 'scene']:
            self.routers[modal] = Mlp(
                4096, 4096 * 4, self.num_experts)

            self.resample_tokens[modal] = nn.Parameter(
                torch.empty([1, 30, resampler_params.dim]))
            nn.init.normal_(self.resample_tokens[modal], std=0.02)

            self.clip_proj1[modal] = nn.Sequential(
                nn.Linear(4096, resampler_params.dim),
                nn.LayerNorm(resampler_params.dim))

            self.clip_proj2[modal] = nn.Sequential(
                nn.Linear(resampler_params.dim, params.dim),
                nn.LayerNorm(params.dim))

            self.start_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))
            self.end_tag[modal] = nn.Parameter(torch.rand(1, 1, params.dim))

    def forward(self, pose_feat, scene_feat):
        image_feats = torch.cat((pose_feat, scene_feat), dim=1)
        routing_weights = self.routers['pose'](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        image_feats_experts = []

        for expert_id in range(self.num_experts):
            image_feats_expert = image_feats
            for layer in self.resample_layers[str(expert_id)]:
                image_feats_expert = layer(image_feats_expert, 0, None, None)

            image_feats_expert = image_feats_expert[:, :self.resample_tokens['pose'].size(1)]
            routing_weight = routing_weights[:, :self.resample_tokens['pose'].size(
                1), expert_id]
            # [B, L, D] * [B, L, 1]
            image_feats_expert = image_feats_expert * routing_weight[:, :, None]

            image_feats_experts.append(image_feats_expert)

        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2['pose'](image_feats)

        return image_feats.reshape(-1, 4096)


if __name__ == "__main__":
    model = MOE(ModelArgs()).cuda()
    model = model.to(torch.bfloat16)
    pose_feat = torch.rand(5, 4096).unsqueeze(0).cuda()
    scene_feat = torch.rand(3, 4096).unsqueeze(0).cuda()
    pose_feat = pose_feat.to(torch.bfloat16)
    scene_feat = scene_feat.to(torch.bfloat16)
    image_feats = model(pose_feat, scene_feat)
    print(image_feats.shape)