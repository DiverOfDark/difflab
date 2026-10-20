from __future__ import annotations

from typing import Any

from difflab import store

FINISHED = ("images/", "videos/")


def listing() -> list[dict[str, Any]]:
    """Newest first. A result is a finished file from the run's last job to deliver one; a `still` job's clip
    (an H3 edit's) and everything earlier in the run are its steps."""
    found = []
    for run in _runs():
        finals, steps = _split(run["made"])
        texts = run["texts"]
        for made in finals:
            poster = next((m["path"] for m in finals if m["job"] == made["job"] and m["kind"] == "image"), None)
            found.append(
                {
                    **{key: made[key] for key in ("id", "path", "kind", "seed", "created", "bytes")},
                    "run": run["id"],
                    "title": run["title"],
                    "production": run["production"],
                    "poster": poster if made["kind"] == "video" else None,
                    "steps": [{key: s[key] for key in ("path", "kind", "label")} for s in steps] + texts,
                }
            )
    found.sort(key=lambda one: (one["created"], one["id"]), reverse=True)
    return found


def doomed(artifact_id: int) -> list[int] | None:
    """The artifact, and once it is the run's last result its finished steps too, which no tile shows then.
    Blobs stay for the cleanup: a production may still read them."""
    for run in _runs():
        finals, steps = _split(run["made"])
        if any(made["id"] == artifact_id for made in finals):
            if len(finals) > 1:
                return [artifact_id]
            return [artifact_id, *(s["id"] for s in steps if s["path"].startswith(FINISHED))]
    return None


def _split(made: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    finished = [m for m in made if m["path"].startswith(FINISHED)]
    finished = [m for m in finished if not (m["still"] and m["kind"] == "video")] or finished
    if not finished:
        return [], made
    last = max(m["position"] for m in finished)
    finals = [m for m in finished if m["position"] == last]
    return finals, [m for m in made if m not in finals]


def _runs() -> list[dict[str, Any]]:
    """Every run that delivered a finished file, and each finished file whose run is gone as a run of its own."""
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT a.id, a.path, a.kind, a.seed, a.created_at, a.run_id, coalesce(j.position, 0) AS position, "
            "coalesce(j.label, j.tool, '') AS label, coalesce(json_extract(j.inputs, '$.still'), 0) AS still, "
            "r.title, r.production_id FROM artifacts a LEFT JOIN jobs j ON j.id = a.job_id "
            "LEFT JOIN runs r ON r.id = a.run_id "
            "WHERE a.run_id IN (SELECT run_id FROM artifacts WHERE path GLOB 'images/*' OR path GLOB 'videos/*') "
            "OR (a.run_id IS NULL AND (a.path GLOB 'images/*' OR a.path GLOB 'videos/*')) ORDER BY a.id"
        ).fetchall()
        written = connection.execute(
            "SELECT run_id, coalesce(label, tool) AS label, json_extract(outputs, '$.text') AS text FROM jobs "
            "WHERE json_type(outputs, '$.text') = 'text' AND run_id IN "
            "(SELECT run_id FROM artifacts WHERE path GLOB 'images/*' OR path GLOB 'videos/*') ORDER BY position"
        ).fetchall()
    runs: dict[str, dict[str, Any]] = {}
    for row in rows:
        file = store.under_output(row["path"])
        if not file.is_file():
            continue
        key = row["run_id"] or f"artifact:{row['id']}"
        run = runs.setdefault(
            key,
            {
                "id": row["run_id"],
                "title": row["title"] or "",
                "production": row["production_id"],
                "made": [],
                "texts": [],
            },
        )
        run["made"].append(
            {
                "id": row["id"],
                "path": row["path"],
                "kind": row["kind"],
                "seed": row["seed"],
                "created": row["created_at"],
                "bytes": file.stat().st_size,
                "job": (row["run_id"], row["position"]),
                "position": row["position"],
                "label": row["label"],
                "still": bool(row["still"]),
            }
        )
    for row in written:
        if row["run_id"] in runs:
            runs[row["run_id"]]["texts"].append(
                {"path": "", "kind": "text", "label": row["label"], "text": row["text"]}
            )
    return list(runs.values())
