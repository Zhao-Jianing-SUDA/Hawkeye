from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

from .multimodal_encoder.builder import build_image_tower, build_video_tower
from .multimodal_projector.builder import build_vision_projector
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

from llava.constants import IGNORE_INDEX, X_TOKEN_INDEX, DEFAULT_X_PATCH_TOKEN, DEFAULT_X_START_TOKEN, \
    DEFAULT_X_END_TOKEN

CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']


class pose_feat(nn.Module):
    def __init__(self):
        super(pose_feat, self).__init__()
        self.pose_projector = nn.Linear(85, 4096)
        self.pose_projector.requires_grad_(True)

    def forward(self, pose_feat):
        pose_feat = pose_feat.view(pose_feat.size(0), -1)
        pose_feat = self.pose_projector(pose_feat)
        return pose_feat


def build_pose_tower():
    return pose_feat()


def build_pose_projector():
    return nn.Linear(4096, 4096)


class GTNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_attr_dim):
        super(GTNLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)
        self.edge_attr_linear = nn.Linear(edge_attr_dim, in_channels)
        self.edge_attr_dim = edge_attr_dim

    def forward(self, x, edge_index, edge_attr=None):
        # 添加自环并更新边属性
        edge_index, edge_attr = self.add_self_loops_with_edge_attr(edge_index, edge_attr, x.size(0), self.edge_attr_dim)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.linear(x)
        return x

    def message(self, x_j, edge_index, edge_attr):
        if edge_attr is not None:
            edge_attr_transformed = self.edge_attr_linear(edge_attr.to(dtype=x_j.dtype))
            return x_j + edge_attr_transformed
        else:
            return x_j

    @staticmethod
    def add_self_loops_with_edge_attr(edge_index, edge_attr, num_nodes, edge_attr_dim):
        # 添加自环
        self_loops = torch.eye(num_nodes, dtype=torch.long)
        self_loops = self_loops.nonzero(as_tuple=False).t().contiguous().cuda()

        # 为自环创建边属性（例如，可以使用全零向量）
        self_loop_attr = torch.zeros((num_nodes, edge_attr_dim))

        # 合并原始边和自环
        edge_index = torch.cat([edge_index.cuda(), self_loops.cuda()], dim=1)
        edge_attr = torch.cat([edge_attr.cuda(), self_loop_attr.cuda()], dim=0) if edge_attr is not None else None

        return edge_index, edge_attr


class GTN(nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, edge_attr_dim):
        super(GTN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.num_layers = num_layers

        # 创建多层GTNLayer
        channels = in_channels
        for _ in range(num_layers):
            self.conv_layers.append(GTNLayer(channels, hidden_channels, edge_attr_dim))
            channels = hidden_channels

        self.linear = nn.Linear(hidden_channels, out_channels)
        # self.linear = nn.Linear(353, 4096)
        self.linear.requires_grad_(True)

        for p in self.conv_layers.parameters():
            p.requires_grad = True
        self.linear.requires_grad_(True)

    def forward(self, scene_feat):
        nodes = []
        edges = []
        probas, probas_sub, probas_obj = scene_feat[:, :51], scene_feat[:, 51:202], scene_feat[:, 202:]
        node_features = []
        edge_features = probas

        for i in range(probas.shape[0]):
            sub = CLASSES[probas_sub[i].argmax()]
            obj = CLASSES[probas_obj[i].argmax()]
            if sub not in nodes:
                nodes.append(sub)
                node_features.append(probas_sub[i])
                # node_features.append(video_feat[0])
            if obj not in nodes:
                nodes.append(obj)
                node_features.append(probas_obj[i])
                # node_features.append(video_feat[0])
            edges.append((sub, obj))
        edge_index = torch.tensor([[nodes.index(src), nodes.index(dst)] for src, dst in edges],
                                  dtype=torch.long).t().contiguous()
        node_features = torch.stack(node_features, dim=0)
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_attr)

        x = gnn.global_mean_pool(x, torch.arange(0, x.size(0), dtype=torch.long, device=x.device))

        x = self.linear(x)
        return x


def build_scene_tower():
    num_layers = 2
    in_channels = 151
    hidden_channels = 4096
    out_channels = 4096
    edge_attr_dim = 51
    return GTN(num_layers, in_channels, hidden_channels, out_channels, edge_attr_dim)


def build_scene_projector():
    return nn.Linear(4096, 4096)


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

    max_batch_size: int = 8
    max_seq_len: int = 256


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

        self.fc1 = nn.Linear(in_features, out_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(out_features, out_features)

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
        self.num_resample_layers = 1
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

        for modal in ['pose']:
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

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, pose_feat, scene_feat):
        # print(pose_feat.shape,scene_feat.shape)#torch.Size([1, 5, 4096]) torch.Size([1, 3, 4096])
        image_feats = torch.cat((pose_feat, scene_feat), dim=1)
        # print('image_feats:')
        # print(image_feats)
        # print(image_feats.shape)
        routing_weights = self.routers['pose'](image_feats).sigmoid()
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # print('routing_weights:')
        # print(routing_weights)
        # print(routing_weights.shape)
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
            # print(expert_id)
            # print(routing_weight)
            image_feats_experts.append(image_feats_expert)
        image_feats = sum(image_feats_experts)
        image_feats = self.clip_proj2['pose'](image_feats)

        return image_feats.reshape(-1, 4096)


def build_moe():
    moe = MOE(ModelArgs())
    for param in moe.parameters():
        param.requires_grad = True
    return moe


def build_moe_projector():
    return nn.Linear(4096, 4096)


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_image_tower"):
            self.image_tower = build_image_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_video_tower"):
            self.video_tower = build_video_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        if hasattr(config, "mm_pose_tower"):
            self.pose_tower = build_pose_tower()
            self.pose_projector = build_pose_projector()
        if hasattr(config, "mm_scene_tower"):
            self.scene_tower = build_scene_tower()
            self.scene_projector = build_scene_projector()
        if hasattr(config, "mm_moe"):
            self.moe = build_moe()
            self.moe_projector = build_moe_projector()

    def get_moe(self):
        moe = getattr(self, 'moe', None)
        if type(moe) is list:
            moe = moe[0]
        return moe

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def get_pose_tower(self):
        pose_tower = getattr(self, 'pose_tower', None)
        if type(pose_tower) is list:
            pose_tower = pose_tower[0]
        return pose_tower

    def get_scene_tower(self):
        scene_tower = getattr(self, 'scene_tower', None)
        if type(scene_tower) is list:
            scene_tower = scene_tower[0]
        return scene_tower

    def initialize_image_modules(self, model_args, fsdp=None):
        image_tower = model_args.image_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_image_tower = image_tower

        image_tower = build_image_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.image_tower = [image_tower]
        else:
            self.image_tower = image_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = image_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_video_modules(self, model_args, fsdp=None):
        video_tower = model_args.video_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_video_tower = video_tower

        video_tower = build_video_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.video_tower = [video_tower]
        else:
            self.video_tower = video_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = video_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_pose_modules(self, model_args, fsdp=None):
        pose_tower = model_args.pose_tower
        self.config.mm_pose_tower = pose_tower

        pose_tower = build_pose_tower()
        if fsdp is not None and len(fsdp) > 0:
            self.pose_tower = [pose_tower]
        else:
            self.pose_tower = pose_tower

        self.pose_projector = build_pose_projector()

    def initialize_scene_modules(self, model_args, fsdp=None):
        scene_tower = model_args.scene_tower
        self.config.mm_scene_tower = scene_tower

        scene_tower = build_scene_tower()
        if fsdp is not None and len(fsdp) > 0:
            self.scene_tower = [scene_tower]
        else:
            self.scene_tower = scene_tower

        self.scene_projector = build_scene_projector()

    def initialize_moe_modules(self, model_args, fsdp=None):
        moe = model_args.moe
        self.config.mm_moe = moe
        moe = build_moe()
        if fsdp is not None and len(fsdp) > 0:
            self.moe = [moe]
        else:
            self.moe = moe

        self.moe_projector = build_moe_projector()


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def get_pose_tower(self):
        return self.get_model().get_pose_tower()

    def get_scene_tower(self):
        return self.get_model().get_scene_tower()

    def get_moe(self):
        return self.get_model().get_moe()

    def get_all_tower(self, keys):
        tower = {key: getattr(self, f'get_{key}_tower') for key in keys}
        return tower

    def encode_images(self, images):
        print(self.get_model().get_image_tower(),'777777777777777')
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):
        # print(videos.shape)
        video_features = self.get_model().get_video_tower()(videos)
        # print(video_features.shape)
        video_features = self.get_model().mm_projector(video_features)
        return video_features

    def encode_poses(self, poses):
        pose_features = self.get_model().get_pose_tower()(poses)
        pose_features = self.get_model().pose_projector(pose_features)
        return pose_features

    def encode_scenes(self, scenes):
        scene_features = self.get_model().get_scene_tower()(scenes)
        scene_features = self.get_model().scene_projector(scene_features)
        return scene_features

    def moe_route(self, pose_feat, scene_feat):
        moe_featers = self.get_model().get_moe()(pose_feat.unsqueeze(0), scene_feat.unsqueeze(0))
        moe_featers = self.get_model().moe_projector(moe_featers)
        return moe_featers

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, X_modalities
    ):
        '''
        X_modalities [
        [img_feature, img_feature, video_feature, audio_feature],
        ['image', 'image', 'video', 'audio']
        ]
        '''
        # Xs, keys = X_modalities
        # Xs, poses, keys = X_modalities
        # print('prepare_inputs_labels_for_multimoda')
        Xs, poses, scenes, keys = X_modalities
        print(Xs[0].shape)


        all_tower = self.get_all_tower(set(keys)) if len(keys) > 0 else None
        # print(2.5)
        if all_tower is None or X_modalities[0][0] is None or input_ids.shape[1] == 1:
            if past_key_values is not None and all_tower is not None and Xs is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                            dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
        try:
            X_features_video = [getattr(self, 'encode_videos')(X.unsqueeze(0)).flatten(0, 1) for X in
                                Xs]  # expand to get batchsize
                
        except Exception as e:
            X_features_video = [getattr(self, 'encode_images')(X.unsqueeze(0)).flatten(0, 1) for X in Xs]

        X_features = []

        #print(X_features_video[0].shape)
        for i in range(len(X_features_video)):

            if poses[i] != None:
                X_features_pose = getattr(self, 'encode_poses')(poses[i])
                X_features_scene = getattr(self, 'encode_scenes')(scenes[i])
                # print(X_features_pose.shape,X_features_scene.shape)
                X_moe_feat = getattr(self, 'moe_route')(X_features_pose, X_features_scene)

                X_features.append(torch.cat((X_features_video[i], X_moe_feat), dim=0))
                # X_features.append(torch.cat((X_features_video[i], X_features_pose), dim=0))
                # X_features.append(X_features_video[i])
            else:
                # X_moe_feat = torch.zeros(10, 4096).cuda()
                X_features.append(X_features_video[i])
        # X_features = [torch.cat((X_features_video[i], X_features_scene[i]), dim=0) for i in range(len(X_features_video))]
        # X_features = [getattr(self, f'encode_{key}s')(X.unsqueeze(0)) for X, key in zip(Xs, keys)]
        # X_features = [x.flatten(0, 1) for x in X_features]
        # for i in X_features:
        #     print(i.shape)
        # print(torch.any(torch.stack([input_ids[0] == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_X_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print(333333)
            if (
            torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0)).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_X_features = X_features[cur_X_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_X_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_X_idx += 1
                continue
            X_token_indices = \
            torch.where(torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]
            cur_new_input_embeds = []
            if labels is not None:
                # print(labels, batch_idx)
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # print(4444444444)
            # print(cur_labels)
            # print(cur_input_ids)
            # print(X_token_indices)
            while X_token_indices.numel() > 0:
                cur_X_features = X_features[cur_X_idx]
                X_token_start = X_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end',
                                                                                  False):  # 不走这
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start - 1]).detach())
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start - 1:X_token_start]))
                    cur_new_input_embeds.append(cur_X_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[X_token_start + 1:X_token_start + 2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[X_token_start:X_token_start + 1])
                        cur_labels = cur_labels[X_token_start + 2:]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:X_token_start]))  # imgtoken之前的text拿出来，好像都是模板套话
                    cur_new_input_embeds.append(cur_X_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:X_token_start])
                        # print(cur_new_labels)
                        # print(cur_X_features.shape[0])
                        # cur_new_labels.append(torch.full((cur_X_features.shape[0],), vid_label[batch_idx], device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(torch.full((cur_X_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                                         dtype=labels.dtype))
                        # print(cur_new_labels)
                        cur_labels = cur_labels[X_token_start + 1:]
                cur_X_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end',
                                                                                  False):
                    cur_input_ids = cur_input_ids[X_token_start + 2:]
                else:
                    cur_input_ids = cur_input_ids[X_token_start + 1:]  # imgtoken之后的text拿出来，是真的question
                    # print(cur_input_ids)
                X_token_indices = torch.where(
                    torch.any(torch.stack([cur_input_ids == X_TOKEN_INDEX[key.upper()] for key in keys]), dim=0))[0]

            # print(55555555555555555)
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_x_start_end',
                                                                                  False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]  # 前面text+图片+后面question
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed,
                                           torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label,
                                               torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX,
                                                          dtype=cur_new_label.dtype, device=cur_new_label.device)),
                                              dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels,
                                                                                    new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True,
                                                        dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                                                         False, dtype=attention_mask.dtype,
                                                         device=attention_mask.device)
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]
        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_X_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_x_patch_token:
            for x in model_args.X:
                tokenizer.add_tokens([DEFAULT_X_PATCH_TOKEN[x.upper()]], special_tokens=True)
            # tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_x_start_end:
            num_new_tokens = 0
            for x in model_args.X:
                num_new_tokens += tokenizer.add_tokens(
                    [DEFAULT_X_START_TOKEN[x.upper()], DEFAULT_X_END_TOKEN[x.upper()]], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_x_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False