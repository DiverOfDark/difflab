"""SAM 3D Body (Meta, SAM License; `models/sam3d_body`): the body under the clothes, measured as a tape would.

Read in its neutral pose, a body's girths do not move with the pose or the garment: on the 2026-10-17 probe a
clean edit (cat ears) moved bust, waist and hip by 1.3% (robust sigma, 6 photos), a dress swap by under 6%, and
the AIO's "heavier" by 27-42%. Silhouettes could not do this: a loose shirt widened them as much as a fault did.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from difflab import config
from difflab.core import ui
from difflab.models.sam3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine
from difflab.models.sam3d_body.data.utils.prepare_batch import prepare_batch
from difflab.models.sam3d_body.models.meta_arch import SAM3DBody
from difflab.models.sam3d_body.utils import recursive_to
from difflab.models.sam3d_body.utils.config import get_config

# MHR70's shoulders and hips. A close portrait shows neither, and the model then invents a body: the probe's
# earring edits of five portraits moved its girths past the cut three times.
FRAME = (5, 6, 9, 10)
# Heights as a fraction of the neutral body's, where a tape goes. The thigh is left out: a long dress hides the
# legs and the model then guesses them (-28% on the probe's dress swaps).
GIRTHS = {"bust": 0.72, "waist": 0.63, "hips": 0.53}


class BodyShape:
    def __init__(self) -> None:
        with ui.loading("Loading SAM 3D Body"):
            weights = config.BODY_CHECKPOINT.fetch()
            cfg = get_config(str(config.BODY_CONFIG.fetch()))
            cfg.defrost()
            cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(config.BODY_RIG.fetch())
            cfg.freeze()
            model = SAM3DBody(cfg)
            state = torch.load(weights, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("state_dict", state), strict=False)
            self.model = model.to("cuda").eval()
            self.cfg = cfg
            self.faces = model.head_pose.faces.cpu().numpy()

    @torch.inference_mode()
    def girths(self, image: Image.Image) -> dict[str, float] | None:
        """Each girth over the body's height, for the person the whole picture shows; None unless the picture shows
        the shoulders and hips."""
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        transform = Compose([
            GetBBoxCenterScale(), TopdownAffine(input_size=self.cfg.MODEL.IMAGE_SIZE, use_udp=False), _to_tensor,
        ])  # fmt: skip
        batch = recursive_to(prepare_batch(rgb, transform, np.array([[0, 0, width, height]], dtype=np.float32)), "cuda")
        self.model._initialize_batch(batch)
        found = self.model.run_inference(rgb, batch, inference_type="body")["mhr"]
        if not len(found["shape"]):
            return None
        joints = found["pred_keypoints_2d"][0][list(FRAME)].float().cpu().numpy()
        if not ((joints[:, 0] >= 0) & (joints[:, 0] < width) & (joints[:, 1] >= 0) & (joints[:, 1] < height)).all():
            return None
        vertices = self._neutral(found["shape"][:1], found["scale"][:1])
        low, high = vertices[:, 1].min(), vertices[:, 1].max()
        tall = float(high - low)
        return {name: _section(vertices, self.faces, low + level * tall) / tall for name, level in GIRTHS.items()}

    def _neutral(self, shape: torch.Tensor, scale: torch.Tensor) -> np.ndarray:
        """The body's vertices with every pose parameter at zero, in metres."""
        head: Any = self.model.head_pose

        def zeros(n: int) -> torch.Tensor:
            return torch.zeros(1, n, device="cuda")

        made = head.mhr_forward(zeros(3), zeros(3), zeros(133), None, scale.float(), shape.float(), zeros(72))
        vertices = made[0] if isinstance(made, (list, tuple)) else made
        return np.asarray(vertices[0].float().cpu().numpy())

    def unload(self) -> None:
        self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()


def _to_tensor(results: dict[str, Any]) -> dict[str, Any]:
    results["img"] = torch.from_numpy(np.ascontiguousarray(results["img"])).permute(2, 0, 1).float() / 255
    return results


def _section(vertices: np.ndarray, faces: np.ndarray, y: float) -> float:
    """The girth, as a tape takes it (the convex hull), of the closed loop the plane at height `y` cuts around the
    body's axis. The mesh has 18k vertices: a slice of points alone breaks into pieces."""
    tri = vertices[faces]
    above = tri[:, :, 1] > y
    crossing = above.any(1) & ~above.all(1)
    segments = []
    for corners, side in zip(tri[crossing], above[crossing], strict=True):
        flat = corners[:, [0, 2]]
        ends = [
            flat[i] + (y - corners[i, 1]) / (corners[j, 1] - corners[i, 1]) * (flat[j] - flat[i])
            for i, j in ((0, 1), (1, 2), (2, 0)) if side[i] != side[j]
        ]  # fmt: skip
        segments.append(ends)
    cut = np.array(segments)
    parent = list(range(len(cut)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[float, float], int] = {}
    for n, ends in enumerate(cut):
        for end in ends:
            point = (round(float(end[0]), 4), round(float(end[1]), 4))
            if point in seen:
                parent[root(n)] = root(seen[point])
            else:
                seen[point] = n
    loops: dict[int, list[int]] = {}
    for n in range(len(cut)):
        loops.setdefault(root(n), []).append(n)
    torso = min(loops.values(), key=lambda loop: abs(float(cut[loop][:, :, 0].mean())))
    return _hull_perimeter(cut[torso].reshape(-1, 2))


def _hull_perimeter(points: np.ndarray) -> float:
    ordered = sorted(map(tuple, points.tolist()))
    if len(ordered) < 3:
        return 0.0

    def half(sequence: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for p in sequence:
            while len(out) >= 2 and (
                (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0
            ):
                out.pop()
            out.append(p)
        return out

    hull = half(ordered)[:-1] + half(ordered[::-1])[:-1]
    ring = np.array([*hull, hull[0]])
    return float(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum())
