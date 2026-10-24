"""comfy-kitchen's int8 attention for the H3 self-attention, replacing SDPA.

Attention is ~75% of a step at full length, so this is the kernel that matters most. Measured on the
3090 on real H3 Q/K/V (chunk 1 of a production 9 take at steps 1-3, 95849 packed rows, 56 heads of 128),
per block, error as relative L2 against bf16 SDPA:

    SDPA (flash, bf16)                  4321 ms   1.00x   -
    SageAttention 1.0.6                 2193 ms   1.97x   0.78-1.24%
    comfy_kitchen.int8_attention        1851 ms   2.33x   0.73-1.18%
    the same, built for sm_86           1585 ms   2.73x   bit-identical to the above

Sage quantizes only Q and K and runs P/V in fp16. Kitchen quantizes V and the softmax probabilities
too, and stays as accurate because it first rotates Q and K by the same block Hadamard, which leaves
their dot products exact and spreads the outliers int8 would otherwise clip. It is the kernel pack
that already runs every int8 linear (`convrot.py`), so this drops a dependency rather than adding one.

The last row is why `pyproject.toml` builds comfy-kitchen from its git tag instead of taking the PyPI
wheel: that wheel has no sm_86 cubin, so the 3090 falls back to the sm_80 one, where every int->float
in the softmax loop is an I2F on the quarter-rate special-function unit. Compiled for sm_86 the same
source emits I2FP on the full-rate ALU, which is worth 17% here at identical output.

The reference ComfyUI workflow patches every block onto SageAttention (KJNodes' *MiniMax H3 Memory
Efficient Sage Attention Patch*), so this is a deviation from it, of the same size as the int8
weights the model already runs on.
"""

import comfy_kitchen
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention, _apply_rotary_emb

from difflab.core import ui


def _int8_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """diffusers' `[batch, seq, heads, dim]` into the kernel's `[batch, heads, seq, dim]`. The transposes
    stay views: the kernel reads strided inputs, so the only copy is its output's."""
    out: torch.Tensor = comfy_kitchen.int8_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2)


class Int8AttnProcessor:
    """`MiniMaxH3AttnProcessor` with `comfy_kitchen.int8_attention` in place of its dispatched SDPA call.

    Not a subclass: diffusers asks a processor only to be callable with this signature, and its own
    processor is untyped.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        # Heads are independent and the kernel quantizes per head, so a group at a time is bit-identical
        # to the whole. One processor for all fifty blocks, and sticky: once one has had to back off the
        # rest start there rather than provoking the same failure again. A rescue path, not a knob.
        self.head_chunks = 1

    def attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        heads = query.shape[2]
        while True:
            try:
                if self.head_chunks <= 1:
                    return _int8_attention(query, key, value)
                out = torch.empty_like(query)
                start = 0
                for index in range(self.head_chunks):
                    stop = start + heads // self.head_chunks + (1 if index < heads % self.head_chunks else 0)
                    out[:, :, start:stop] = _int8_attention(
                        query[:, :, start:stop], key[:, :, start:stop], value[:, :, start:stop]
                    )
                    start = stop
                return out
            except torch.OutOfMemoryError:
                if self.head_chunks >= heads:
                    raise
                self.head_chunks = min(heads, max(2, self.head_chunks * 2))
                torch.cuda.empty_cache()
                ui.note(f"attention out of memory: retrying in {self.head_chunks} head groups")

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            # H3 packs one request into a single attention document and never passes a mask.
            raise NotImplementedError("Int8AttnProcessor does not support an attention mask")

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states = self.attend(query, key, value).flatten(2, 3).type_as(query)
        out: torch.Tensor = attn.to_out[1](attn.to_out[0](hidden_states))
        return out


def use_int8_attention(model: torch.nn.Module) -> int:
    """Returns how many attentions were swapped."""
    processor = Int8AttnProcessor()
    swapped = 0
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.processor = processor
            swapped += 1
    return swapped
