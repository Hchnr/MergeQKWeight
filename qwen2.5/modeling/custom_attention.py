from typing import Callable, Optional, Tuple
from transformers.processing_utils import Unpack
import torch
from transformers.models.qwen2.modeling_qwen2 import (
    FlashAttentionKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,  # Add the missing import
    ALL_ATTENTION_FUNCTIONS,  # Import ALL_ATTENTION_FUNCTIONS
    Qwen2Model,
    Qwen2Config,
)
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.cache_utils import Cache
from transformers.utils import logging

from transformers.models.qwen3.modeling_qwen3 import repeat_kv

logger = logging.get_logger(__name__)


def trans(model: Qwen2Model):
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

        new_attn = Qwen3AttentionCustom(model.config, layer_idx=layer_i)

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


class Qwen2AttentionCustom(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        # QK Weights Merge
        self.qk_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qk_states = self.qk_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        x_states = hidden_states.view(hidden_shape).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        x_states, qk_states = apply_rotary_pos_emb(x_states, qk_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            x_states, qk_states = past_key_value.update(
                x_states, qk_states, self.layer_idx, cache_kwargs
            )

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        attention_interface: Callable = custom_attention_forward
        """
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
        """
        attn_output, attn_weights = attention_interface(
            self,
            x_states,
            qk_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


if __name__ == "__main__":
    # Example usage
    BATCH_SIZE = 1
    SEQ_LENGTH = 10
    MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen2-7B-Instruct"
    config = Qwen2Config.from_json_file(f"{MODEL_PATH}/config.json")
    attention_layer = Qwen3AttentionCustom(config, layer_idx=0)
    input = torch.randn(BATCH_SIZE, 10, config.hidden_size)
    output = attention_layer(input)
    print(attention_layer)
    # You can add more tests or example usages here
