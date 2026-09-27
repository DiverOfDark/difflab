"""Image-only Qwen-VL processors.

transformers requires a video processor, which needs torchvision (not a dependency: without it
`transformers.BaseVideoProcessor` is a dummy that can't be instantiated). Only still images are
ever passed here, so these subclasses accept `video_processor=None`.
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
    """Prompt-engineer VLM (Qwen3-VL) processor."""
