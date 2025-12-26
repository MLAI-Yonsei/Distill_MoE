import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Callable, Optional, Union
from transformers.generation import GenerationMixin
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRotaryEmbedding,
    LlamaRMSNorm,
)
from transformers.masking_utils import create_causal_mask
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils.generic import check_model_inputs
# ---------------------------------------------------------
# 0. RMSNorm (LLaMA 스타일)
# ---------------------------------------------------------
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
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


# ---------------------------------------------------------
# 1. Student MoE Layer (experts + router)
#    - teacher는 여기서 가지지 않는다!
#    - teacher_output을 인자로 받아서 distill_loss만 계산
# ---------------------------------------------------------
class StudentMoELayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_selects
        self.hidden_size = config.hidden_size

        # Output rescaling (전통적인 MoE에서 자주 쓰는 스케일링)
        self.scaling_factor = self.num_experts / self.top_k

        # 1) Student Experts
        expert_config = LlamaConfig(**config.to_dict())
        expert_config.intermediate_size = config.intermediate_size // config.num_experts
        self.experts = nn.ModuleList(
            [LlamaMLP(expert_config) for _ in range(self.num_experts)]
        )

        # 2) Router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.router_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, S, H]
        teacher_output: Optional[torch.Tensor] = None,  # [B, S, H]
    ):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)  # [T, H], T=B*S

        # --- Router forward ---
        router_input = self.router_norm(flat_hidden)      # [T, H]
        router_logits = self.router(router_input)         # [T, E]
        routing_weights = F.softmax(router_logits, dim=-1)

        # --- Top-K selection ---
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # [T, K], [T, K]
        # normalize within top-k
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- Expert computation ---
        student_output = torch.zeros_like(router_input)   # [T, H]

        flat_indices = topk_indices.view(-1, self.top_k)  # [T, K]
        flat_weights = topk_weights.view(-1, self.top_k)  # [T, K]

        # 각 expert별로 한 번씩만 forward
        for expert_idx, expert in enumerate(self.experts):
            mask = (flat_indices == expert_idx)  # [T, K]
            if not mask.any():
                continue

            # (token_index, rank_index) 중 token_index만 사용
            row_indices = torch.where(mask)[0]           # [N_used]
            inp = flat_hidden[row_indices]               # [N_used, H]
            out = expert(inp)                            # [N_used, H]

            weights = flat_weights[mask].unsqueeze(-1)   # [N_used, 1]
            weighted_out = out * weights.to(out.dtype)   # [N_used, H]

            student_output.index_add_(
                0, row_indices, weighted_out.to(student_output.dtype)
            )

        # reshape + scaling
        student_output = student_output.view(batch_size, seq_len, hidden_dim)
        student_output = student_output * self.scaling_factor

        # --- Aux (load-balancing) loss ---
        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.training:
            routing_weights_32 = routing_weights.to(torch.float32)
            T = batch_size * seq_len

            # importance: expert마다 softmax 상에서 받은 확률 질량 평균
            importance = routing_weights_32.sum(dim=0) / T   # [E]

            # load: 실제로 top-k에 몇 번 들어갔는지 (비율)
            expert_mask = F.one_hot(
                topk_indices, num_classes=self.num_experts
            ).float()                    # [T, K, E]
            load = expert_mask.sum(dim=(0, 1)) / (T * self.top_k)  # [E]

            aux_loss = (importance * load).sum() * self.num_experts

        # --- Distill loss ---
        distill_loss = None
        if teacher_output is not None:
            distill_loss = F.l1_loss(
                student_output.to(teacher_output.dtype), teacher_output
            )

        return student_output, aux_loss, distill_loss


# ---------------------------------------------------------
# 2. DistillLlamaDecoderLayer
#    - LlamaDecoderLayer를 상속
#    - self.teacher_expert : Dense LlamaMLP
#    - self.mlp : StudentMoELayer
#    - forward는 DistillLlamaForCausalLM에서 수동으로 layer loop
# ---------------------------------------------------------
class DistillLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-Attention (오리지널 동일)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        # ✅ Teacher FFN: Dense LlamaMLP (DecoderLayer 수준에 위치)
        self.teacher_expert = LlamaMLP(config)

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


class DistillLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
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

        # Initialize weights and apply final processing
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
    ) -> BaseModelOutputWithPast:
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

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, l_aux, l_distill = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            aux_loss = aux_loss + l_aux
            distill_loss = distill_loss + l_distill

            
        hidden_states = self.norm(hidden_states)
        return DistillModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            aux_loss=aux_loss,
            distill_loss=distill_loss,
        )
# ---------------------------------------------------------
# 3. DistillLlamaForCausalLM
#    - TinyLlama 등 pretrained dense 모델 config를 받아서
#      모든 Decoder Layer를 DistillLlamaDecoderLayer로 교체
#    - forward에서 layer loop를 직접 구현하며 각 layer의 aux/distill loss를 합산
# ---------------------------------------------------------
class DistillLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DistillLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            # aux / distill loss 결합 (텐서 연산)
            alpha_aux = getattr(self.config, "alpha_aux", 0.0)
            alpha_distill = getattr(self.config, "alpha_distill", 0.0)

            # aux_loss, distill_loss는 이미 same device/dtype 스칼라 텐서라고 가정
            extra_loss = alpha_aux * aux_loss + alpha_distill * distill_loss

            # gradient는 extra_loss에만 흐르게 하고, scale은 loss + extra_loss 값 유지
            total_loss = loss + (extra_loss - extra_loss.detach())
            # total_loss = loss 
        else:
            extra_loss = torch.zeros(
                (), device=logits.device, dtype=logits.dtype
            )
            total_loss = None

        # total_loss = loss + extra_loss
        # total_loss = loss + (extra_loss - extra_loss.detach())

                   
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------
# 4. Dense → Distill-MoE weight transfer
#    - attention / norm / embed / lm_head: 그대로 복사
#    - dense FFN: teacher_expert로 100% 복사
#    - dense FFN: student experts로 chunk해서 복사
# ---------------------------------------------------------
def transfer_weights_for_distillation(
    dense_model: LlamaForCausalLM,
    moe_model: DistillLlamaForCausalLM,
    num_experts: int,
):
    """
    dense_model : 원본 LLaMA / TinyLlama dense 모델 (LlamaForCausalLM)
    moe_model   : DistillLlamaForCausalLM (내부에 DistillLlamaModel을 가짐)
    """
    print("Weight Transfer Start...")

    num_layers = len(dense_model.model.layers)

    for i in range(num_layers):
        print(f"Processing Layer {i+1}/{num_layers}...")

        dense_layer = dense_model.model.layers[i]          # LlamaDecoderLayer
        moe_layer = moe_model.model.layers[i]              # DistillLlamaDecoderLayer

        # 1) Attention & LayerNorm 복사
        #    - 구조는 동일하므로 모듈 단위로 state_dict를 바로 복사하는 게 가장 안전함
        moe_layer.self_attn.load_state_dict(dense_layer.self_attn.state_dict())
        moe_layer.input_layernorm.load_state_dict(dense_layer.input_layernorm.state_dict())
        moe_layer.post_attention_layernorm.load_state_dict(dense_layer.post_attention_layernorm.state_dict())

        # 2) MLP (FFN) -> teacher_expert & student experts
        dense_mlp = dense_layer.mlp  # LlamaMLP(dense)

        # (A) Teacher: dense FFN 전체를 그대로 복사
        moe_layer.teacher_expert.load_state_dict(dense_mlp.state_dict())

        # (B) Student Experts: gate/up/down 쪼개서 복사
        w_gate = dense_mlp.gate_proj.weight.data   # [intermediate_size, hidden_size]
        w_up   = dense_mlp.up_proj.weight.data     # [intermediate_size, hidden_size]
        w_down = dense_mlp.down_proj.weight.data   # [hidden_size, intermediate_size]

        # intermediate dimension을 expert 수만큼 균등 분할
        gate_chunks = torch.chunk(w_gate, num_experts, dim=0)  # [inter/num_experts, H] * num_experts
        up_chunks   = torch.chunk(w_up,   num_experts, dim=0)
        down_chunks = torch.chunk(w_down, num_experts, dim=1)  # [H, inter/num_experts] * num_experts

        for e_idx in range(num_experts):
            student_expert = moe_layer.mlp.experts[e_idx]  # StudentMoELayer 안의 LlamaMLP

            student_expert.gate_proj.weight.data.copy_(gate_chunks[e_idx])
            student_expert.up_proj.weight.data.copy_(up_chunks[e_idx])
            student_expert.down_proj.weight.data.copy_(down_chunks[e_idx])

    # 3) Embeddings & final norm & LM head 복사
    moe_model.model.embed_tokens.load_state_dict(
        dense_model.model.embed_tokens.state_dict()
    )
    moe_model.model.norm.load_state_dict(
        dense_model.model.norm.state_dict()
    )
    moe_model.lm_head.load_state_dict(
        dense_model.lm_head.state_dict()
    )

    print("✅ All weights transferred successfully!")
    return moe_model
def debug_check_layer(dense_model, moe_model, layer_idx=0, device="cuda"):
    dense_layer = dense_model.model.layers[layer_idx]
    moe_layer = moe_model.model.layers[layer_idx]

    # 1) self_attn weight 비교
    for name, p_dense in dense_layer.self_attn.state_dict().items():
        p_moe = moe_layer.self_attn.state_dict()[name]
        print("[ATTN]", name, (p_dense - p_moe).abs().max().item())

    # 2) layernorm weight 비교
    for name, p_dense in dense_layer.input_layernorm.state_dict().items():
        p_moe = moe_layer.input_layernorm.state_dict()[name]
        print("[IN LN]", name, (p_dense - p_moe).abs().max().item())
    for name, p_dense in dense_layer.post_attention_layernorm.state_dict().items():
        p_moe = moe_layer.post_attention_layernorm.state_dict()[name]
        print("[POST LN]", name, (p_dense - p_moe).abs().max().item())

    # 3) teacher_expert vs dense_mlp
    for name, p_dense in dense_layer.mlp.state_dict().items():
        p_teacher = moe_layer.teacher_expert.state_dict()[name]
        print("[TEACHER MLP]", name, (p_dense - p_teacher).abs().max().item())

    # 4) dense_mlp forward vs teacher_expert forward
    dev = dense_layer.mlp.gate_proj.weight.device
    hidden_size = dense_model.config.hidden_size
    x = torch.randn(2, 10, hidden_size, device=dev, dtype=dense_layer.mlp.gate_proj.weight.dtype)

    dense_layer.to(dev)
    moe_layer.to(dev)

    with torch.no_grad():
        y_dense   = dense_layer.mlp(x)
        y_teacher = moe_layer.teacher_expert(x)

    print("MLP forward diff (max abs):", (y_dense - y_teacher).abs().max().item())

# ---------------------------------------------------------
# 5. main: TinyLlama → Distill-MoE 변환 + 저장
# ---------------------------------------------------------
def main():
    MODEL_ID = "TinyLlama/TinyLlama_v1.1"
    NUM_EXPERTS = 8
    NUM_SELECTS = 2
    SAVE_PATH = "./Distill_MoEv2"

    print(f"Loading Dense Model: {MODEL_ID}")
    dense_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Config 불러와서 MoE 관련 설정 추가
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_experts = NUM_EXPERTS
    config.num_selects = NUM_SELECTS
    config.alpha_aux = 0.01
    config.alpha_distill = 0.1
    config.architectures = ["DistillLlamaForCausalLM"]
    config.model_type = "distill_llama"

    print("Creating Empty Distill-MoE Model...")
    moe_model = DistillLlamaForCausalLM(config).to(torch.float16)

    # 가중치 이식
    moe_model = transfer_weights_for_distillation(
        dense_model, moe_model, NUM_EXPERTS
    )
    debug_check_layer(dense_model, moe_model, layer_idx=0, device="cuda")

    # 저장
    print(f"Saving converted model to {SAVE_PATH}...")
    moe_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("\n=== Simple Verification ===")
    moe_model.float()
    print("Layer 0 structure:")
    print(moe_model.model.layers[0])
    print("Teacher Expert exists:", hasattr(moe_model.model.layers[0], "teacher_expert"))
    print("Student Experts Count:", len(moe_model.model.layers[0].mlp.experts))


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    main()