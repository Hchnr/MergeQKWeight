import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
)  # Import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    eager_attention_forward,
)  # Add the missing import
from .modeling_internlm3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .configuration_internlm3 import InternLM3Config
from .modeling_internlm3 import InternLM3Model, InternLM3RotaryEmbedding

logger = logging.get_logger(__name__)


def trans(model: InternLM3Model):
    for layer_i, layer in enumerate(model.layers):
        if not hasattr(layer, "self_attn"):
            raise ValueError(
                "The model does not have the expected 'self_attn' attribute in its layers."
            )
        self_attn = layer.self_attn
        if not hasattr(self_attn, "q_proj") or not hasattr(self_attn, "k_proj"):
            raise ValueError(
                "The self_attn does not have the expected 'q_proj' and 'k_proj' attributes."
            )

        new_attn = InternLM3AttentionCustom(model.config, layer_idx=layer_i)

        # wq:[4096,4096], wk:[1024,4096], wqk:[1024,4096]
        # Linear:[out_features, in_features]
        wq = self_attn.q_proj.weight
        wk = self_attn.k_proj.weight
        wqk = torch.matmul(wk, wq.transpose(0, 1))

        new_attn_dict = new_attn.state_dict()
        old_attn_state = self_attn.state_dict()
        del old_attn_state["q_proj.weight"]
        del old_attn_state["k_proj.weight"]
        new_attn_dict.update(old_attn_state)
        new_attn_dict["qk_proj.weight"] = wqk
        new_attn.load_state_dict(new_attn_dict)
        model.layers[layer_i].self_attn = new_attn

    print(f"Attn transformed to Qwen3AttentionCustom")
    model.eval()
    return model


def custom_attention_forward(
    module: nn.Module,
    x_states: torch.Tensor,
    qk_states: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    x_states = repeat_kv(x_states, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    # x_states:     torch.Size([1, 32, 513, 128])
    # qk_states:    torch.Size([1, 32, 513, 128])
    # value_states: torch.Size([1, 32, 513, 128])

    attn_weights = torch.matmul(x_states, qk_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : qk_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        x_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class InternLM3AttentionCustom(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternLM3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.bias
        )

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = InternLM3RotaryEmbedding(config=self.config)

        # QK Weights Merge
        if config.qkv_bias:
            raise ValueError(
                "The `qkv_bias` argument is not supported in QK merge. "
                "Please set `qkv_bias=False` in the configuration."
            )
        self.qk_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        qk_states = self.qk_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        x_states = hidden_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        qk_states = qk_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        x_states, qk_states = apply_rotary_pos_emb(x_states, qk_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            x_states, qk_states = past_key_value.update(
                x_states, qk_states, self.layer_idx, cache_kwargs
            )

        x_states = repeat_kv(x_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(x_states, qk_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : qk_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(x_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


if __name__ == "__main__":
    # Example usage
    BATCH_SIZE = 1
    SEQ_LENGTH = 10
    MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-8B"
    config = InternLM3Config.from_json_file(f"{MODEL_PATH}/config.json")
    attention_layer = InternLM3AttentionCustom(config, layer_idx=0)
    input = torch.randn(BATCH_SIZE, 10, config.hidden_size)
    output = attention_layer(input)
    print(attention_layer)
    # You can add more tests or example usages here
