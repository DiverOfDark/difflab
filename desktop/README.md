# DiffLab desktop

Starts difflab on this PC (Docker, a 24 GB NVIDIA card, 64 GB RAM) or on a RunPod pod, opens its page,
and downloads the results. The pod deletes itself (`difflab/work/watchdog.py`), so closing the app never
leaves a GPU billing.

    npm install
    npm run tauri dev        # or: npm run tauri build

Linux needs the WebKitGTK 4.1 headers to build (`webkit2gtk4.1-devel` on Fedora). On an immutable
Fedora (Bazzite, Silverblue) build in a toolbox; the binary runs on the host. Installers for Windows,
macOS and Linux come from the `release` workflow, on every push to main.

What a RunPod user needs: an API key. The first pod creates a global volume named `difflab` (no
datacenter, grows as used, billed for storage and requests) and mounts it at `/app/output`; the first pod
downloads ~96 GB of weights and keeps a copy in `.model-store` there, which later pods copy onto their own disk. Results download from the running pod over HTTP, as RunPod's
S3 API reaches only network volumes. Closing the window leaves the app in the tray; Quit is its menu.
