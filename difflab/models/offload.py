"""Keep as many blocks of a large stack on the GPU as fit, and stream the rest through it one at a time.

Neither big model fits beside the rest of a 24 GB card: the text encoder is 22.7 GB int8 plus 2.6 GB
bf16, and the DiT is 19.5 GB int8 with roughly 6 GB of activations at full size. Every block keeps its
CPU copy, resident or not, so releasing one is a pointer swap and nothing is reallocated per step.

Measured at 15 ms per 385 MB block (25.6 GB/s over PCIe). From pageable memory that copy also held the
worker thread: py-spy found it in 73% of a 30-minute H3 render (2026-10-11), so a streamed block's CPU copy
is pinned, and the next streamed block is copied on a side stream while this one computes. Each block is
one exact-size buffer pinned in place: `pin_memory()` rounds every allocation up to a power of two, which
turns a 37 MB weight into 64 MB of locked RAM on a 58 GB host.

The prefetched block is allocated on the compute stream, not the copy stream: the CPU runs blocks ahead of
the GPU, and memory a side stream allocated cannot be reused until the compute stream is done with it, so
each block ahead would hold another 385 MB. On the compute stream the extra is one block, which `fit`
leaves room for.
"""

import weakref
from collections.abc import Iterable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

_ATTRIBUTE = "_difflab_streamer"
_ALIGNMENT = 256


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
        self.pinned = False
        self.following: BlockStreamer | None = None
        self.fetched: list[torch.Tensor | None] | None = None
        self._hook()

    def onload(self, module: nn.Module, args: tuple[object, ...]) -> None:
        if not self.pinned:
            self._pin()
        if self.fetched is None:
            self._place([None] * len(self.cpu))
        else:
            torch.cuda.current_stream(self.device).wait_stream(_copier(self.device))
            self._place(self.fetched)
            self.fetched = None
        following = self.following
        while following is not None and following.resident:
            following = following.following
        if following is not None and following.fetched is None:
            following.prefetch()

    def prefetch(self) -> None:
        if not self.pinned:
            self._pin()
        copier = _copier(self.device)
        copier.wait_stream(torch.cuda.current_stream(self.device))
        fetched = [torch.empty_like(cpu, device=self.device) if type(cpu) is torch.Tensor else None for cpu in self.cpu]
        with torch.cuda.stream(copier):
            for cpu, gpu in zip(self.cpu, fetched, strict=True):
                if gpu is not None:
                    gpu.copy_(cpu, non_blocking=True)
        self.fetched = fetched

    def drop(self) -> None:
        if self.fetched is not None:
            torch.cuda.current_stream(self.device).wait_stream(_copier(self.device))
            self.fetched = None

    def _place(self, fetched: list[torch.Tensor | None]) -> None:
        for (owner, name, is_parameter), cpu, gpu in zip(self.slots, self.cpu, fetched, strict=True):
            resident = cpu.to(self.device, non_blocking=True) if gpu is None else gpu
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
        self.drop()
        self._place([None] * len(self.cpu))
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

    def _pin(self) -> None:
        self.pinned = True
        # Subclasses (quantized wrappers) and strided views stay pageable: a flat copy would lose their layout.
        movable = [i for i, tensor in enumerate(self.cpu) if type(tensor) is torch.Tensor and tensor.is_contiguous()]
        offsets, total = [], 0
        for i in movable:
            offsets.append(total)
            total += -(-self.cpu[i].numel() * self.cpu[i].element_size() // _ALIGNMENT) * _ALIGNMENT
        if not total:
            return
        flat = torch.empty(total, dtype=torch.uint8)
        for i, offset in zip(movable, offsets, strict=True):
            tensor = self.cpu[i]
            view = flat[offset : offset + tensor.numel() * tensor.element_size()].view(tensor.dtype).view(tensor.shape)
            view.copy_(tensor)
            owner, name, is_parameter = self.slots[i]
            if is_parameter:
                parameter = owner._parameters[name]
                if parameter is not None:
                    parameter.data = view
            else:
                owner._buffers[name] = view
            self.cpu[i] = view
        cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
        torch.cuda.check_error(cudart.cudaHostRegister(flat.data_ptr(), total, 0))
        weakref.finalize(self, _unpin, flat)


_streamers: "weakref.WeakSet[BlockStreamer]" = weakref.WeakSet()
_copiers: dict[torch.device, torch.cuda.Stream] = {}


def stream_blocks(blocks: Iterable[nn.Module], device: str | torch.device) -> None:
    target = torch.device(device)
    previous: BlockStreamer | None = None
    for block in blocks:
        streamer = BlockStreamer(block, target)
        setattr(block, _ATTRIBUTE, streamer)
        _streamers.add(streamer)
        if previous is not None:
            previous.following = streamer
        previous = streamer


def fit(blocks: Iterable[nn.Module], reserve: int) -> int:
    """Settle blocks in order while `reserve` bytes stay free, releasing from the end if they do not;
    returns how many are resident."""
    streamers: list[BlockStreamer] = [getattr(block, _ATTRIBUTE) for block in blocks]
    reserve += max((streamer.size for streamer in streamers), default=0)
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
    for streamer in _streamers:
        streamer.drop()
    resident = [streamer for streamer in _streamers if streamer.resident]
    for streamer in resident:
        streamer.unsettle()
    if resident:
        torch.cuda.empty_cache()
    return len(resident)


def _copier(device: torch.device) -> torch.cuda.Stream:
    if device not in _copiers:
        _copiers[device] = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]
    return _copiers[device]


def _unpin(flat: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
    torch.cuda.check_error(cudart.cudaHostUnregister(flat.data_ptr()))


def _room(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
