from __future__ import annotations

from typing import Any

from difflab import store


def listing() -> list[dict[str, Any]]:
    """Every finished image and video still on disk, newest first."""
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT a.id, a.path, a.kind, a.seed, a.created_at, a.run_id, a.job_id, r.title, r.production_id, "
            "(SELECT p.path FROM artifacts p WHERE p.job_id = a.job_id AND p.kind = 'image' "
            "AND p.path GLOB 'images/*' ORDER BY p.id LIMIT 1) AS poster "
            "FROM artifacts a LEFT JOIN runs r ON r.id = a.run_id "
            "WHERE a.path GLOB 'images/*' OR a.path GLOB 'videos/*' ORDER BY a.created_at DESC, a.id DESC"
        ).fetchall()
    found = []
    for row in rows:
        file = store.under_output(row["path"])
        if not file.is_file():
            continue
        found.append(
            {
                "id": row["id"],
                "path": row["path"],
                "kind": row["kind"],
                "seed": row["seed"],
                "created": row["created_at"],
                "bytes": file.stat().st_size,
                "run": row["run_id"],
                "title": row["title"] or "",
                "production": row["production_id"],
                "poster": row["poster"] if row["kind"] == "video" else None,
            }
        )
    return found
