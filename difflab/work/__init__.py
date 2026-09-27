"""What runs on the card: a run is what was asked for, a job is one tool applied once, a task is a
phase of that tool. The worker is a thread beside the server that takes jobs off the store."""

from typing import Any


class Skipped(dict[str, Any]):
    """Outputs a job found already made: the artifact under its key was there, so nothing ran."""
