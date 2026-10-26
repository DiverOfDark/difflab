"""Keep as many blocks of a large stack on the GPU as fit, and stream the rest through it one at a time.

Neither big model fits beside the rest of a 24 GB card: the text encoder is 22.7 GB int8 plus 2.6 GB
bf16, and the DiT is 19.5 GB int8 with roughly 6 GB of activations at full size. Every block keeps its
CPU copy, resident or not, so releasing one is a pointer swap and nothing is reallocated per step.

Measured at 15 ms per 385 MB block (25.6 GB/s over PCIe). From pageable memory that copy also held the
worker thread: py-spy found it in 73% of a 30-minute H3 render (2026-10-11). Pinning every streamed block
in place fixed that but locked 15-25 GB of a 58 GB host, and loading any other model then swapped the
desktop out (memory PSI full 18%, video stuttered). So the CPU copies stay pageable and swappable; a
background thread copies the next block into one of two pinned staging buffers per device, and the copy
stream uploads it from there while this block computes. Each staging buffer is an exact-size allocation
registered with cudaHostRegister: `pin_memory()` rounds up to a power of two, 385 MB into 512 MB.

The prefetched block is allocated on the compute stream, not the copy stream: the CPU runs blocks ahead of
the GPU, and memory a side stream allocated cannot be reused until the compute stream is done with it, so
each block ahead would hold another 385 MB. On the compute stream the extra is one block, which `fit`
leaves room for.
"""

import weakref
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor

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
        self.offsets: list[int] = []
        self.staged = 0
        for tensor in self.cpu:
            self.offsets.append(self.staged)
            self.staged += -(-tensor.numel() * tensor.element_size() // _ALIGNMENT) * _ALIGNMENT
        self.resident = False
        self.following: BlockStreamer | None = None
        self.fetched: list[torch.Tensor | None] | None = None
        self.pending: Future[torch.cuda.Event] | None = None
        self._hook()

    def onload(self, module: nn.Module, args: tuple[object, ...]) -> None:
        self._place(self._take())
        following = self.following
        while following is not None and following.resident:
            following = following.following
        if following is not None and following.fetched is None:
            following.prefetch()

    def prefetch(self) -> None:
        compute = torch.cuda.current_stream(self.device)
        # Subclasses (quantized wrappers) go through `_place` from pageable memory: a flat copy would lose their layout.
        fetched = [torch.empty_like(cpu, device=self.device) if type(cpu) is torch.Tensor else None for cpu in self.cpu]
        allocated = torch.cuda.Event()  # type: ignore[no-untyped-call]
        allocated.record(compute)
        self.fetched = fetched
        self.pending = _stager.submit(_stage, self, fetched, _slot(self.device), allocated)

    def drop(self) -> None:
        if self.fetched is not None:
            self._take()

    def _take(self) -> list[torch.Tensor | None]:
        if self.fetched is None:
            self.prefetch()
        assert self.fetched is not None and self.pending is not None
        torch.cuda.current_stream(self.device).wait_event(self.pending.result())
        fetched, self.fetched, self.pending = self.fetched, None, None
        return fetched

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
        self._place(self._take())
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


class _Staging:
    def __init__(self) -> None:
        self.buffer: torch.Tensor | None = None
        self.uploaded: torch.cuda.Event | None = None

    def reserve(self, size: int) -> torch.Tensor:
        if self.buffer is None or self.buffer.numel() < size:
            if self.buffer is not None:
                _unpin(self.buffer)
            self.buffer = torch.empty(size, dtype=torch.uint8)
            cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
            torch.cuda.check_error(cudart.cudaHostRegister(self.buffer.data_ptr(), size, 0))
        return self.buffer


def _stage(
    streamer: BlockStreamer, fetched: list[torch.Tensor | None], slot: _Staging, allocated: torch.cuda.Event
) -> torch.cuda.Event:
    # Inference mode is thread-local: blocks the text encoder prefetches under it are inference tensors, which this
    # thread may write only inside it. Writing a normal tensor inside it is allowed, so the DiT's go through too.
    with torch.inference_mode():
        return _upload(streamer, fetched, slot, allocated)


def _upload(
    streamer: BlockStreamer, fetched: list[torch.Tensor | None], slot: _Staging, allocated: torch.cuda.Event
) -> torch.cuda.Event:
    if slot.uploaded is not None:
        slot.uploaded.synchronize()
    flat = slot.reserve(streamer.staged)
    views = []
    for cpu, gpu, offset in zip(streamer.cpu, fetched, streamer.offsets, strict=True):
        if gpu is None:
            continue
        view = flat[offset : offset + cpu.numel() * cpu.element_size()].view(cpu.dtype).view(cpu.shape)
        view.copy_(cpu)
        views.append((view, gpu))
    copier = _copier(streamer.device)
    copier.wait_event(allocated)
    with torch.cuda.stream(copier):
        for view, gpu in views:
            gpu.copy_(view, non_blocking=True)
    uploaded = torch.cuda.Event()  # type: ignore[no-untyped-call]
    uploaded.record(copier)
    slot.uploaded = uploaded
    return uploaded


_streamers: "weakref.WeakSet[BlockStreamer]" = weakref.WeakSet()
_copiers: dict[torch.device, torch.cuda.Stream] = {}
_slots: dict[torch.device, tuple[_Staging, _Staging]] = {}
_turns: dict[torch.device, int] = {}
_stager = ThreadPoolExecutor(1, thread_name_prefix="difflab-stage")


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


def unstage() -> None:
    """Unpin the staging buffers, two blocks per device, for an idle process."""
    for pair in _slots.values():
        for slot in pair:
            if slot.uploaded is not None:
                slot.uploaded.synchronize()
            if slot.buffer is not None:
                _unpin(slot.buffer)
    _slots.clear()


def _copier(device: torch.device) -> torch.cuda.Stream:
    if device not in _copiers:
        _copiers[device] = torch.cuda.Stream(device)  # type: ignore[no-untyped-call]
    return _copiers[device]


def _slot(device: torch.device) -> _Staging:
    if device not in _slots:
        _slots[device] = (_Staging(), _Staging())
    _turns[device] = 1 - _turns.get(device, 0)
    return _slots[device][_turns[device]]


def _unpin(flat: torch.Tensor) -> None:
    cudart = torch.cuda.cudart()  # type: ignore[no-untyped-call]
    torch.cuda.check_error(cudart.cudaHostUnregister(flat.data_ptr()))


def _room(device: torch.device) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
