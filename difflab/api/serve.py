"""Running the server: beside a render, or on its own.

Two ways in, and they are the same server. `serve_forever` is the one a pod runs, and it starts the
worker that takes jobs off the queue. `ensure_running` is for the other direction - a render started
from an editor brings a server up in a thread so the take can be watched, with no worker, since the
editor's own process is doing the rendering - and it does nothing when the port is already held.

Neither asks who is connecting. On a LAN that is a choice; on a pod's public URL it is an open card
and a public gallery, so binding to anything but localhost says so on every start.
"""

from __future__ import annotations

import socket
import threading

from difflab import config

BANNER = (
    "DiffLab is serving with no authentication: anyone who can reach this address can start renders "
    "on this card and watch everything it makes."
)


def url() -> str:
    host = "localhost" if config.DASHBOARD_HOST in ("0.0.0.0", "") else config.DASHBOARD_HOST
    return f"http://{host}:{config.DASHBOARD_PORT}"


def taken() -> bool:
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((config.DASHBOARD_HOST, config.DASHBOARD_PORT))
        except OSError:
            return True
    return False


def ensure_running() -> bool:
    if taken():
        return False
    threading.Thread(target=_run, kwargs={"quiet": True}, daemon=True, name="difflab-web").start()
    _warn()
    return True


def serve_forever() -> None:
    """What a container runs as its command: the page, and the worker that renders what it asks for."""
    from difflab.work.watchdog import Watchdog
    from difflab.work.worker import Worker

    print(f"DiffLab: {url()}  (runs from {config.RUNS_DIR})")
    _warn()
    Worker().start()
    if config.POD_ID:
        Watchdog().start()
    _run(quiet=False)


def _run(quiet: bool) -> None:
    import uvicorn

    from difflab.api.app import app

    uvicorn.run(
        app,
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        log_level="error" if quiet else "info",
        access_log=not quiet,
    )


def _warn() -> None:
    if config.DASHBOARD_HOST not in ("127.0.0.1", "localhost"):
        print(f"  !  {BANNER}")
