"""How long each tool takes on this machine, from the jobs it finished: what ETAs and estimates are made of."""

from __future__ import annotations

import statistics
from typing import Any

from difflab import store

RECENT = 50  # finished jobs per tool; older ones ran on models and settings since replaced
BY_LENGTH = ("render_shot",)  # whose time grows with the clip's `seconds`

# Medians over every finished job in output/factory.db on 2026-10-08 (3090, H3 at its default settings):
# the answer until this machine has run the tool itself.
FALLBACK = {
    "render_shot": 436.0,
    "judge": 8.5,
    "compose_keyframe": 66.0,
    "draw_reference": 55.0,
    "draw_look": 55.0,  # not yet run on 2026-10-08: draw_reference's, the same editor at the same size
    "plan_shots": 114.0,
    "write_shot": 5.8,
    "compile_shot": 0.0,
    "assemble": 17.0,
    "enlarge_shot": 380.0,
    "finish": 1029.0,
}
FALLBACK_PER_SECOND = {"render_shot": 46.7}


def measured(connection: Any) -> dict[str, dict[str, Any]]:
    """Per tool: median `seconds` a job, `per_second` of clip for BY_LENGTH, and `samples` (0: the fallback)."""
    rows = connection.execute(
        "SELECT tool, ended_at - started_at, json_extract(inputs, '$.seconds') FROM ("
        "  SELECT *, row_number() OVER (PARTITION BY tool ORDER BY id DESC) AS recent FROM jobs"
        "  WHERE state = ? AND started_at IS NOT NULL AND ended_at IS NOT NULL"
        ") WHERE recent <= ?",
        (store.DONE, RECENT),
    ).fetchall()
    took: dict[str, list[float]] = {}
    rates: dict[str, list[float]] = {}
    for tool, seconds, length in rows:
        took.setdefault(tool, []).append(float(seconds))
        if tool in BY_LENGTH and isinstance(length, int | float) and length > 0:
            rates.setdefault(tool, []).append(float(seconds) / float(length))
    out: dict[str, dict[str, Any]] = {
        tool: {"seconds": seconds, "per_second": FALLBACK_PER_SECOND.get(tool), "samples": 0}
        for tool, seconds in FALLBACK.items()
    }
    for tool, seconds in took.items():
        out[tool] = {
            "seconds": round(statistics.median(seconds), 1),
            "per_second": round(statistics.median(rates[tool]), 1) if tool in rates else None,
            "samples": len(seconds),
        }
    return out
