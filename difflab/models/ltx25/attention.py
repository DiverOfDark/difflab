"""comfy-kitchen's int8 attention (see `minimax_h3/attention.py`) for LTX's video self-attention, 2.5's and 2.3's.

Only `attn1` is swapped: it is the one attention quadratic in the clip's rows. Audio self-attention is 25 rows a
second, and the rest attend to the 1024 prompt tokens or to the audio.
"""

import torch
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2VideoTransformerBlock,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)

from difflab.models.minimax_h3.attention import Int8AttnProcessor


class Int8VideoAttnProcessor:
    """`LTX2PerturbedAttnProcessor` with the kernel in place of SDPA, for unmasked self-attention without STG."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        self.kernel = Int8AttnProcessor()

    def __call__(
        self,
        attn: LTX2Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or attention_mask is not None or perturbation_mask is not None:
            raise NotImplementedError("Int8VideoAttnProcessor is unmasked self-attention without STG")

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(hidden_states))
        value = attn.to_v(hidden_states)
        if query_rotary_emb is not None:
            rotate = apply_split_rotary_emb if attn.rope_type == "split" else apply_interleaved_rotary_emb
            query = rotate(query, query_rotary_emb)
            key = rotate(key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

        heads = (attn.heads, -1)
        out = self.kernel.attend(query.unflatten(2, heads), key.unflatten(2, heads), value.unflatten(2, heads))
        out = out.type_as(query)
        if gate_logits is not None:
            out = out * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = out.flatten(2, 3)
        result: torch.Tensor = attn.to_out[1](attn.to_out[0](hidden_states))
        return result


def use_int8_attention(model: torch.nn.Module) -> int:
    """Returns how many attentions were swapped."""
    processor = Int8VideoAttnProcessor()
    blocks = [module for module in model.modules() if isinstance(module, LTX2VideoTransformerBlock)]
    for block in blocks:
        block.attn1.processor = processor
    return len(blocks)
