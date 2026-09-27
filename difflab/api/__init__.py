"""The web app: a page to write a job and watch the render it starts. The page itself is built from
`frontend/` into `static/`.

    pixi run python -m difflab.api      # serve, and nothing else

The routes import no model package; the worker thread beside them does, so one process holds the
page and the card.
"""

from difflab.api.serve import serve_forever, url

__all__ = ["serve_forever", "url"]
