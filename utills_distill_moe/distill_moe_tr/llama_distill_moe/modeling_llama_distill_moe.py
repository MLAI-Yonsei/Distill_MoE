# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
from dataclasses import dataclass

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama_distill_moe import DistillLlamaConfig

import torch.nn.functional as F
logger = logging.get_logger(__name__)
def simple_nan_check(t, name):
    if t is None:
        print(f"[DEBUG] {name}: None")
        return
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(
            f"[DEBUG][NaN] {name}: "
            f"mean={t.mean().item():.4e}, "
            f"std={t.std().item():.4e}, "
            f"max={t.abs().max().item():.4e}"
        )
    else:
        print(f"[DEBUG] {name}: OK")

@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: DistillLlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# class StudentMoELayer(nn.Module):
#     def __init__(self, config: DistillLlamaConfig):
#         super().__init__()
#         self.num_experts = config.num_experts
#         self.top_k = config.num_selects
#         self.hidden_size = config.hidden_size

#         # Output rescaling (전통적인 MoE에서 자주 쓰는 스케일링)
#         self.scaling_factor = self.num_experts / self.top_k

#         # 1) Student Experts
#         expert_config = DistillLlamaConfig(**config.to_dict())
#         expert_config.intermediate_size = config.intermediate_size // config.num_experts
#         self.experts = nn.ModuleList(
#             [LlamaMLP(expert_config) for _ in range(self.num_experts)]
#         )

#         # 2) Router
#         self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
#         self.router_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,          # [B, S, H]
#         teacher_output: Optional[torch.Tensor] = None,  # [B, S, H]
#     ):
#         batch_size, seq_len, hidden_dim = hidden_states.shape
#         flat_hidden = hidden_states.view(-1, hidden_dim)  # [T, H], T=B*S

#         # --- Router forward ---
#         router_input = self.router_norm(flat_hidden)      # [T, H]
#         router_logits = self.router(router_input)         # [T, E]
#         routing_weights = F.softmax(router_logits, dim=-1)

#         # --- Top-K selection ---
#         topk_weights, topk_indices = torch.topk(
#             routing_weights, self.top_k, dim=-1
#         )  # [T, K], [T, K]
#         # normalize within top-k
#         topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

#         # --- Expert computation ---
#         student_output = torch.zeros_like(router_input)   # [T, H]

#         flat_indices = topk_indices.view(-1, self.top_k)  # [T, K]
#         flat_weights = topk_weights.view(-1, self.top_k)  # [T, K]

#         # 각 expert별로 한 번씩만 forward
#         for expert_idx, expert in enumerate(self.experts):
#             mask = (flat_indices == expert_idx)  # [T, K]
#             if not mask.any():
#                 continue

#             # (token_index, rank_index) 중 token_index만 사용
#             row_indices = torch.where(mask)[0]           # [N_used]
#             inp = flat_hidden[row_indices]               # [N_used, H]
#             out = expert(inp)                            # [N_used, H]

#             weights = flat_weights[mask].unsqueeze(-1)   # [N_used, 1]
#             weighted_out = out * weights.to(out.dtype)   # [N_used, H]

#             student_output.index_add_(
#                 0, row_indices, weighted_out.to(student_output.dtype)
#             )

#         # reshape + scaling
#         student_output = student_output.view(batch_size, seq_len, hidden_dim)
#         student_output = student_output * self.scaling_factor

#         # --- Aux (load-balancing) loss ---
#         aux_loss = torch.tensor(0.0, device=hidden_states.device)
#         if self.training:
#             routing_weights_32 = routing_weights.to(torch.float32)
#             T = batch_size * seq_len

#             # importance: expert마다 softmax 상에서 받은 확률 질량 평균
#             importance = routing_weights_32.sum(dim=0) / T   # [E]

#             # load: 실제로 top-k에 몇 번 들어갔는지 (비율)
#             expert_mask = F.one_hot(
#                 topk_indices, num_classes=self.num_experts
#             ).float()                    # [T, K, E]
#             load = expert_mask.sum(dim=(0, 1)) / (T * self.top_k)  # [E]

#             aux_loss = (importance * load).sum() * self.num_experts

#         # --- Distill loss ---
#         distill_loss = torch.tensor(0.0, device=hidden_states.device)
#         if teacher_output is not None:
#             distill_loss = F.mse_loss(
#                 student_output.to(teacher_output.dtype), teacher_output
#             )

#         return student_output, aux_loss, distill_loss


class StudentMoELayer(nn.Module):
    def __init__(self, config: DistillLlamaConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_selects
        self.hidden_size = config.hidden_size

        # Distill config
        self.distill_loss_type = getattr(config, "distill_loss_type", "mse")  # "mse" | "cka" | "cov"
        self.distill_eps = float(getattr(config, "distill_eps", 1e-8))
        self.distill_cov_use_correlation = bool(getattr(config, "distill_cov_use_correlation", False))

        # Output rescaling
        self.scaling_factor = self.num_experts / self.top_k

        # 1) Student Experts
        expert_config = DistillLlamaConfig(**config.to_dict())
        expert_config.intermediate_size = config.intermediate_size // config.num_experts
        self.experts = nn.ModuleList([LlamaMLP(expert_config) for _ in range(self.num_experts)])

        # 2) Router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.router_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _cka_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Linear CKA loss: 1 - CKA(X, Y)
        student/teacher: [B, S, H] or [T, H]
        """
        if student.dim() == 3:
            student = student.reshape(-1, student.size(-1))  # [T, Hs]
        if teacher.dim() == 3:
            teacher = teacher.reshape(-1, teacher.size(-1))  # [T, Ht]

        # 안정성: float32로 계산 (gradient는 유지됨)
        X = student.to(torch.float32)
        Y = teacher.to(torch.float32)

        # center along tokens
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        # CKA numerator/denominator
        XtY = X.t() @ Y                         # [Hs, Ht]
        num = (XtY * XtY).sum()                 # ||X^T Y||_F^2

        XtX = X.t() @ X                         # [Hs, Hs]
        YtY = Y.t() @ Y                         # [Ht, Ht]
        den = torch.sqrt((XtX * XtX).sum() * (YtY * YtY).sum() + self.distill_eps)

        cka = num / (den + self.distill_eps)
        return (1.0 - cka).to(student.dtype)

    def _covariance_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Covariance (or correlation) matching loss:
        ||Cov(X) - Cov(Y)||_F^2
        student/teacher: [B, S, H] or [T, H]
        """
        if student.dim() == 3:
            student = student.reshape(-1, student.size(-1))  # [T, H]
        if teacher.dim() == 3:
            teacher = teacher.reshape(-1, teacher.size(-1))  # [T, H]

        X = student.to(torch.float32)
        Y = teacher.to(torch.float32)

        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)

        n = X.size(0)
        denom = max(n - 1, 1)

        cov_x = (X.t() @ X) / denom             # [H, H]
        cov_y = (Y.t() @ Y) / denom             # [H, H]

        if self.distill_cov_use_correlation:
            # corr = cov / (std_i * std_j)
            std_x = torch.sqrt(torch.diag(cov_x).clamp_min(self.distill_eps))
            std_y = torch.sqrt(torch.diag(cov_y).clamp_min(self.distill_eps))
            cov_x = cov_x / (std_x[:, None] * std_x[None, :] + self.distill_eps)
            cov_y = cov_y / (std_y[:, None] * std_y[None, :] + self.distill_eps)

        diff = cov_x - cov_y
        loss = (diff * diff).mean()             # scale 안정화(= H^2로 나눔 효과)
        return loss.to(student.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,                          # [B, S, H]
        teacher_output: Optional[torch.Tensor] = None,        # [B, S, H]
    ):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)      # [T, H], T=B*S

        # --- Router forward ---
        router_input = self.router_norm(flat_hidden)          # [T, H]
        router_logits = self.router(router_input)             # [T, E]
        routing_weights = F.softmax(router_logits, dim=-1)

        # --- Top-K selection ---
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # [T, K], [T, K]
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- Expert computation ---
        student_output = torch.zeros_like(router_input)       # [T, H]
        flat_indices = topk_indices.view(-1, self.top_k)      # [T, K]
        flat_weights = topk_weights.view(-1, self.top_k)      # [T, K]

        for expert_idx, expert in enumerate(self.experts):
            mask = (flat_indices == expert_idx)
            if not mask.any():
                continue

            row_indices = torch.where(mask)[0]                # [N_used]
            inp = flat_hidden[row_indices]                    # [N_used, H]
            out = expert(inp)                                 # [N_used, H]

            weights = flat_weights[mask].unsqueeze(-1)        # [N_used, 1]
            weighted_out = out * weights.to(out.dtype)        # [N_used, H]

            student_output.index_add_(0, row_indices, weighted_out.to(student_output.dtype))

        student_output = student_output.view(batch_size, seq_len, hidden_dim)
        student_output = student_output * self.scaling_factor

        # --- Aux (load-balancing) loss ---
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            routing_weights_32 = routing_weights.to(torch.float32)
            T = batch_size * seq_len
            importance = routing_weights_32.sum(dim=0) / T

            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # [T, K, E]
            load = expert_mask.sum(dim=(0, 1)) / (T * self.top_k)

            aux_loss = (importance * load).sum() * self.num_experts

        # --- Distill loss ---
        distill_loss = torch.tensor(0.0, device=hidden_states.device)
        if teacher_output is not None:
            # dtype 맞춰서 비교(teacher dtype 기준)
            s = student_output.to(teacher_output.dtype)
            t = teacher_output

            if self.distill_loss_type == "mse":
                distill_loss = F.mse_loss(s, t)

            elif self.distill_loss_type == "cka":
                distill_loss = self._cka_loss(s, t)

            elif self.distill_loss_type == "cov":
                distill_loss = self._covariance_loss(s, t)

            else:
                raise ValueError(f"Unknown distill_loss_type: {self.distill_loss_type}")

        return student_output, aux_loss, distill_loss
    

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DistillLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DistillLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DistillLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-Attention (오리지널 동일)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # ✅ Teacher FFN: Dense LlamaMLP (DecoderLayer 수준에 위치)
        self.teacher_expert = LlamaMLP(config)
        # for p in self.teacher_expert.parameters():
        #     p.requires_grad = False

        # ✅ Student MoE: 기존 DistillationMoELayer 대신 StudentMoELayer 사용
        #    LLaMA-Factory에서 name_module_trainable="mlp" 를 주면
        #    이 모듈만 trainable로 잡을 수 있음.
        self.mlp = StudentMoELayer(config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack["TransformersKwargs"],
    ):
        """
        오리지널 DistillLlamaDecoderLayer와 동일한 역할:
        - self_attn + residual
        - post_attention_layernorm
        - FFN(teacher + student MoE) + residual
        - 반환: hidden_states, aux_loss, distill_loss
        """
        # ---- Self Attention block (원본과 동일) ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # ---- FFN block: Teacher + Student MoE ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # 1) Teacher FFN (gradient 없이)
        with torch.no_grad():
            teacher_output = self.teacher_expert(hidden_states)

        # 2) Student MoE (mlp)
        student_output, aux_loss, distill_loss = self.mlp(
            hidden_states, teacher_output=teacher_output
        )

        hidden_states = residual + student_output

        return hidden_states, aux_loss, distill_loss


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: DistillLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DistillLlamaDecoderLayer,
        "attentions": LlamaAttention,
    }

@dataclass
class DistillModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    aux_loss: Optional[torch.FloatTensor] = None
    distill_loss: Optional[torch.FloatTensor] = None


@auto_docstring
class DistillLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: DistillLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DistillLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


        
        

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DistillModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)


        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds


        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        aux_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        distill_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
        
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, l_aux, l_distill = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # simple_nan_check(hidden_states, f"hidden_after_layer_{layer_idx}")
            # simple_nan_check(l_aux,        f"aux_loss_layer_{layer_idx}")
            # simple_nan_check(l_distill,    f"distill_loss_layer_{layer_idx}")

            aux_loss = aux_loss + l_aux
            distill_loss = distill_loss + l_distill

            
        hidden_states = self.norm(hidden_states)

        return DistillModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            aux_loss=aux_loss,
            distill_loss=distill_loss,
        )


# @auto_docstring
# class DistillLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
#     _tied_weights_keys = ["lm_head.weight"]
#     _tp_plan = {"lm_head": "colwise_rep"}
#     _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

#     def __init__(self, config):
#         super().__init__(config)
#         self.model = DistillLlamaModel(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

        

#     @can_return_tuple
#     @auto_docstring
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         **kwargs: Unpack[TransformersKwargs],
#     ) -> CausalLMOutputWithPast:
#         r"""
#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
#         >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

#         >>> prompt = "Hey, are you conscious? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
#         ```"""
#         outputs: DistillModelOutputWithPast = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             **kwargs,
#         )
#         aux_loss = outputs.aux_loss          # 0-dim tensor
#         distill_loss = outputs.distill_loss  # 0-dim tensor

#         hidden_states = outputs.last_hidden_state
#         # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])

#         loss = None
#         if labels is not None:
#             loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
#             # aux / distill loss 결합 (텐서 연산)
#             alpha_aux = getattr(self.config, "alpha_aux", 0.0)
#             alpha_distill = getattr(self.config, "alpha_distill", 0.0)

#             # aux_loss, distill_loss는 이미 same device/dtype 스칼라 텐서라고 가정
#             extra_loss = alpha_aux * aux_loss + alpha_distill * distill_loss

#             # gradient는 extra_loss에만 흐르게 하고, scale은 loss + extra_loss 값 유지
#             total_loss = loss + (extra_loss - extra_loss.detach())

#             # total_loss = loss 
#         else:
#             extra_loss = torch.zeros(
#                 (), device=logits.device, dtype=logits.dtype
#             )
#             total_loss = None

#         # total_loss = loss + extra_loss
#         # total_loss = loss + (extra_loss - extra_loss.detach())

                   
#         return CausalLMOutputWithPast(
#             loss=total_loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
@auto_docstring
class DistillLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    ...

    def __init__(self, config):
        super().__init__(config)
        self.model = DistillLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 🔹 distill용 step 카운터 버퍼 (checkpoint에 같이 저장됨)
        self.register_buffer("current_step", torch.zeros((), dtype=torch.long))

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: DistillModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        aux_loss = outputs.aux_loss          # 0-dim tensor
        distill_loss = outputs.distill_loss  # 0-dim tensor

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        total_loss = None

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

            # ============================
            # 🔹 distill / aux 스케줄링
            # ============================
            # 1) step 카운터 증가 (학습 중 + label 있을 때만)
        
            # 스텝 증가
            self.current_step += 1

            # hyperparameter 로드
            # alpha_init = getattr(self.config, "alpha_distill", 0.1)
            # alpha_min  = getattr(self.config, "min_alpha_distill", 0.0)
            # warmup_steps = getattr(self.config, "distill_warmup_steps", 100)  # warmup
            # decay_steps = getattr(self.config, "distill_decay_steps", 1000)

            # # progress (0 → 1)
            # progress = (self.current_step.float() / decay_steps).clamp(0, 1)

            # # cosine decay
            # cosine = 0.5 * (1 + torch.cos(torch.pi * progress))
            # alpha_distill_dyn = alpha_min + (alpha_init - alpha_min) * cosine

            # extra_loss = getattr(self.config, "alpha_aux", 0.0) * aux_loss + alpha_distill_dyn * distill_loss

            # total_loss = loss + (extra_loss - extra_loss.detach())
            # hyperparameter 로드
            alpha_max = getattr(self.config, "alpha_distill", 0.1)          # peak
            alpha_min = getattr(self.config, "min_alpha_distill", 0.0)      # floor
            warmup_steps = getattr(self.config, "distill_warmup_steps", 100)  # warmup
            total_steps  = getattr(self.config, "distill_decay_steps", 1000)  # 전체 스케줄 길이

            # 안전장치
            warmup_steps = max(int(warmup_steps), 1)
            total_steps  = max(int(total_steps), warmup_steps + 1)

            step_f = self.current_step.float()

            # 1) warmup: 0 -> alpha_max
            warmup_progress = (step_f / warmup_steps).clamp(0, 1)
            alpha_warm = alpha_max * warmup_progress

            # 2) decay: alpha_max -> alpha_min (warmup 이후부터)
            decay_progress = ((step_f - warmup_steps) / (total_steps - warmup_steps)).clamp(0, 1)
            cosine = 0.5 * (1 + torch.cos(torch.pi * decay_progress))
            alpha_decay = alpha_min + (alpha_max - alpha_min) * cosine

            # 3) piecewise combine
            alpha_distill_dyn = torch.where(step_f <= warmup_steps, alpha_warm, alpha_decay)
            extra_loss = getattr(self.config, "alpha_aux", 0.0) * aux_loss + alpha_distill_dyn * distill_loss
            total_loss = loss + (extra_loss - extra_loss.detach())



        
        # 디버깅용으로 가끔 찍어볼 수 있음
            # if self.current_step % 100 == 0 and labels is not None:
            #     print(f"[DEBUG] step={int(self.current_step)} alpha_distill={float(alpha_distill_dyn):.5f}")
                
        else:
            
            total_loss = None

        return CausalLMOutputWithPast(
            loss=total_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "DistillLlamaForCausalLM",
    "DistillLlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
