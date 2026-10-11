"""Keep as many blocks of a large stack on the GPU as fit, and stream the rest through it one at a time.

Neither big model fits beside the rest of a 24 GB card: the text encoder is 22.7 GB int8 plus 2.6 GB
bf16, and the DiT is 19.5 GB int8 with roughly 6 GB of activations at full size. Every block keeps its
CPU copy, resident or not, so releasing one is a pointer swap and nothing is reallocated per step.

Measured at 15 ms per 385 MB block (25.6 GB/s over PCIe), i.e. ~0.8 s per step for all 50 DiT blocks,
paid in full: the copy is not overlapped with compute.
"""

import weakref
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

_ATTRIBUTE = "_difflab_streamer"


class BlockStreamer:
    def __init__(self, block: nn.Module, device: torch.device) -> None:
        self.block = block
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
        self.size = sum(tensor.numel() * tensor.element_size() for tensor in self.cpu)
        self.handles: list[RemovableHandle] = []
        self.resident = False
        self._hook()

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

    def settle(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.onload(self.block, ())
        self.resident = True

    def unsettle(self) -> None:
        self.offload(self.block, (), None)
        self._hook()
        self.resident = False

    def _hook(self) -> None:
        self.handles = [
            self.block.register_forward_pre_hook(self.onload),
            self.block.register_forward_hook(self.offload),
        ]


_streamers: "weakref.WeakSet[BlockStreamer]" = weakref.WeakSet()


def stream_blocks(blocks: Iterable[nn.Module], device: str | torch.device) -> None:
    target = torch.device(device)
    for block in blocks:
        streamer = BlockStreamer(block, target)
        setattr(block, _ATTRIBUTE, streamer)
        _streamers.add(streamer)


def fit(blocks: Iterable[nn.Module], reserve: int) -> int:
    """Settle blocks in order while `reserve` bytes stay free, releasing from the end if they do not;
    returns how many are resident."""
    streamers: list[BlockStreamer] = [getattr(block, _ATTRIBUTE) for block in blocks]
    for streamer in reversed(streamers):
        if _room(streamer.device) >= reserve:
            break
        if streamer.resident:
            streamer.unsettle()
    for streamer in streamers:
        if not streamer.resident and _room(streamer.device) - streamer.size >= reserve:
            streamer.settle()
    return sum(streamer.resident for streamer in streamers)


def release() -> int:
    """Stream every resident block again, for a stage that needs the card; returns how many were."""
    resident = [streamer for streamer in _streamers if streamer.resident]
    for streamer in resident:
        streamer.unsettle()
    if resident:
        torch.cuda.empty_cache()
    return len(resident)


def _room(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
