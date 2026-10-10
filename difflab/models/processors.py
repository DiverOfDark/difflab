"""Image-only Qwen-VL processor.

transformers requires a video processor, which needs torchvision (not a dependency: without it
`transformers.BaseVideoProcessor` is a dummy that can't be instantiated). Only still images are
ever passed here, so this subclass accepts `video_processor=None`.
"""

from typing import Any

from transformers import ProcessorMixin, Qwen2VLProcessor, Qwen3VLProcessor


class _AcceptNoVideo(ProcessorMixin):
    def check_argument_for_proper_class(self, argument_name: str, argument: Any) -> Any:
        if argument_name == "video_processor" and argument is None:
            return None
        return super().check_argument_for_proper_class(argument_name, argument)


class ImageOnlyQwen2VLProcessor(_AcceptNoVideo, Qwen2VLProcessor):
    """Editor text encoder (Qwen2.5-VL) processor."""


class ImageOnlyQwen3VLProcessor(_AcceptNoVideo, Qwen3VLProcessor):
    """Qwen-Image 2.1 text encoder (Qwen3-VL) processor."""
