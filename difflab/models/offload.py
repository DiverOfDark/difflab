"""Stream large block stacks through the GPU one block at a time.

Neither big model fits beside the rest of a 24 GB card: the text encoder is 22.7 GB int8 plus 2.6 GB
bf16, and the DiT is 19.5 GB int8 with roughly 6 GB of activations at full size. The block lists stay in
CPU memory, and the CPU copy is kept and restored rather than freed, so nothing is reallocated per step.

Measured at 15 ms per 385 MB block (25.6 GB/s over PCIe), i.e. ~0.8 s per step for all 50 DiT blocks -
small next to the compute.
"""

from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class BlockStreamer:
    def __init__(self, block: nn.Module, device: torch.device) -> None:
        self.device = device
        self.slots: list[tuple[nn.Module, str, bool]] = []
        self.cpu: list[torch.Tensor] = []
        for module in block.modules():
            for name, parameter in module._parameters.items():
                if parameter is not None:
                    self.slots.append((module, name, True))
                    self.cpu.append(parameter.data)
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    self.slots.append((module, name, False))
                    self.cpu.append(buffer)

    def onload(self, module: nn.Module, args: tuple[object, ...]) -> None:
        for (owner, name, is_parameter), cpu in zip(self.slots, self.cpu, strict=True):
            resident = cpu.to(self.device, non_blocking=True)
            if is_parameter:
                parameter = owner._parameters[name]
                if parameter is not None:
                    parameter.data = resident
            else:
                owner._buffers[name] = resident

    def offload(self, module: nn.Module, args: tuple[object, ...], output: object) -> None:
        for (owner, name, is_parameter), cpu in zip(self.slots, self.cpu, strict=True):
            if is_parameter:
                parameter = owner._parameters[name]
                if parameter is not None:
                    parameter.data = cpu
            else:
                owner._buffers[name] = cpu


def stream_blocks(blocks: Iterable[nn.Module], device: str | torch.device) -> list[RemovableHandle]:
    target = torch.device(device)
    handles: list[RemovableHandle] = []
    for block in blocks:
        streamer = BlockStreamer(block, target)
        handles.append(block.register_forward_pre_hook(streamer.onload))
        handles.append(block.register_forward_hook(streamer.offload))
    return handles
