"""InsightFace's antelopev2 on onnxruntime, CPU: SCRFD finds the face and its five points, glintr100 (ArcFace)
says who it is, and 2d106det outlines the eyes for their colour. Non-commercial weights (InsightFace); difflab is
personal use.

On CPU because each picture costs about half a second and the card is busy with the editor and the VLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from difflab import config

# ArcFace's 112x112 template: eyes, nose tip, mouth corners, as InsightFace aligns its training faces.
TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
)
# 2d106det's own order, read off a labelled face: the outline of the eye on the picture's left, its centre, then
# the other eye's.
EYES = ((tuple(i for i in range(33, 43) if i != 38), 38), (tuple(i for i in range(87, 97) if i != 88), 88))


@dataclass(frozen=True)
class Eye:
    colour: tuple[float, float, float]  # median RGB of the iris
    open: float  # outline height over width; a shut or squinting eye shows no iris to read


@dataclass(frozen=True)
class Face:
    identity: np.ndarray  # unit ArcFace embedding
    eyes: tuple[Eye | None, Eye | None]  # the picture's left, then right
    skin: tuple[float, float, float]  # median RGB of the cheeks and nose: a colour cast scales it as it does the irises
    height: float  # the face box in pixels: below ~150 every reading is noisier


class FaceId:
    def __init__(self) -> None:
        cpu = ["CPUExecutionProvider"]
        self.detector = ort.InferenceSession(str(config.FACE_DETECTOR.fetch()), providers=cpu)
        self.recogniser = ort.InferenceSession(str(config.FACE_RECOGNISER.fetch()), providers=cpu)
        self.landmarker = ort.InferenceSession(str(config.FACE_LANDMARKS.fetch()), providers=cpu)

    def read(self, image: Image.Image) -> Face | None:
        rgb = np.asarray(image.convert("RGB"))
        found = self._detect(rgb)
        if found is None:
            return None
        box, five = found
        matrix, _ = cv2.estimateAffinePartial2D(five, TEMPLATE, method=cv2.LMEDS)
        aligned = cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0)
        blob = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        identity = self.recogniser.run(None, {self.recogniser.get_inputs()[0].name: blob})[0][0]
        points = self._landmarks(rgb, box)
        eyes = tuple(_eye(rgb, points[list(outline)], points[centre]) for outline, centre in EYES)
        x0, y0, x1, y1 = box.round().astype(int)
        cheeks = rgb[
            max(0, y0 + (y1 - y0) * 45 // 100) : y0 + (y1 - y0) * 70 // 100,
            max(0, x0 + (x1 - x0) // 4) : x0 + (x1 - x0) * 3 // 4,
        ]
        skin = _median(cheeks.reshape(-1, 3))
        return Face(identity / np.linalg.norm(identity), (eyes[0], eyes[1]), skin, float(box[3] - box[1]))

    def _detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The most confident face's box (x0, y0, x1, y1) and its five points, in pixels."""
        h, w, _ = rgb.shape
        scale = 640 / max(h, w)
        board = np.zeros((640, 640, 3), np.float32)
        board[: round(h * scale), : round(w * scale)] = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        outs = self.detector.run(
            None, {self.detector.get_inputs()[0].name: ((board - 127.5) / 128).transpose(2, 0, 1)[None]}
        )
        best, found = config.FACE_DETECT_SCORE, None
        for level, stride in enumerate((8, 16, 32)):
            scores, boxes, points = outs[level][:, 0], outs[level + 3] * stride, outs[level + 6] * stride
            k = int(scores.argmax())
            if scores[k] > best:
                n = 640 // stride
                cy, cx = np.mgrid[0:n, 0:n] * stride
                centre = np.repeat(np.stack([cx.ravel(), cy.ravel()], 1), 2, axis=0)[k]
                best = float(scores[k])
                found = (
                    np.concatenate([centre - boxes[k, :2], centre + boxes[k, 2:]]) / scale,
                    (centre + points[k].reshape(5, 2)) / scale,
                )
        return found

    def _landmarks(self, rgb: np.ndarray, box: np.ndarray) -> np.ndarray:
        centre = (box[:2] + box[2:]) / 2
        scale = 192 / (max(box[2] - box[0], box[3] - box[1]) * 1.5)
        matrix = np.array([[scale, 0, 96 - centre[0] * scale], [0, scale, 96 - centre[1] * scale]], np.float32)
        crop = cv2.warpAffine(rgb, matrix, (192, 192)).astype(np.float32).transpose(2, 0, 1)[None]
        points = (self.landmarker.run(None, {self.landmarker.get_inputs()[0].name: crop})[0].reshape(-1, 2) + 1) * 96
        return np.asarray((points - matrix[:, 2]) / scale)


def _eye(rgb: np.ndarray, outline: np.ndarray, centre: np.ndarray) -> Eye | None:
    width = float(np.ptp(outline[:, 0]))
    if width < config.FACE_EYE_PIXELS:
        return None
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillPoly(mask, [outline.round().astype(np.int32)], 1)
    disc = np.zeros_like(mask)
    cv2.circle(disc, tuple(int(v) for v in centre.round()), max(2, round(0.3 * width)), 1, -1)
    picked = rgb[(mask & disc).astype(bool)]
    value = picked.max(1)
    picked = picked[(value > 30) & (value < 235)]  # pupil, lashes and highlight
    if len(picked) < 10:
        return None
    return Eye(_median(picked), float(np.ptp(outline[:, 1])) / width)


def _median(pixels: np.ndarray) -> tuple[float, float, float]:
    r, g, b = np.median(pixels.reshape(-1, 3).astype(np.float32), 0) + 1
    return float(r), float(g), float(b)
