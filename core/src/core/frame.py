"""Frame dataclass shared across practice and live modes."""

from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Frame:
    """A single captured video frame."""

    image: Image.Image
    """The frame as a PIL Image (RGB)."""

    timestamp: datetime
    """When the frame was captured."""

    def person_only_image(
        self,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        model_name: str = "yolo11n-seg.pt",
        transparent: bool = True,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Return an image where only the detected person is kept.

        Segmentation strategy:
        1) Try YOLO segmentation (class=person) for best quality and speed.
        2) Fall back to OpenCV HOG + GrabCut if YOLO is unavailable.

        Args:
            conf: Minimum confidence for person detections.
            iou: IOU threshold used by YOLO NMS.
            model_name: YOLO segmentation model name/path.
            transparent: If True, output RGBA with transparent background.
                If False, output RGB with the background replaced by
                ``background_color``.
            background_color: RGB color used when ``transparent=False``.

        Returns:
            Person-only image. If no person is found, returns a fully
            transparent image (RGBA) or solid background image (RGB).
        """
        mask = _segment_person_mask(
            self.image,
            conf=conf,
            iou=iou,
            model_name=model_name,
        )

        if mask is None:
            if transparent:
                return Image.new("RGBA", self.image.size, (0, 0, 0, 0))
            return Image.new("RGB", self.image.size, background_color)

        base_rgba = self.image.convert("RGBA")
        base_rgba.putalpha(mask)

        if transparent:
            return base_rgba

        out = Image.new("RGB", self.image.size, background_color)
        out.paste(base_rgba.convert("RGB"), mask=mask)
        return out


def _segment_person_mask(
    image: Image.Image,
    *,
    conf: float,
    iou: float,
    model_name: str,
) -> Image.Image | None:
    """Build a single-channel person mask from an RGB image."""
    yolo_mask = _segment_person_mask_yolo(
        image,
        conf=conf,
        iou=iou,
        model_name=model_name,
    )
    if yolo_mask is not None:
        return yolo_mask

    return _segment_person_mask_grabcut(image)


@lru_cache(maxsize=2)
def _load_yolo_model(model_name: str) -> Any:
    """Lazy-load and cache YOLO model for repeated frame processing."""
    from ultralytics import YOLO

    return YOLO(model_name)


def _segment_person_mask_yolo(
    image: Image.Image,
    *,
    conf: float,
    iou: float,
    model_name: str,
) -> Image.Image | None:
    """Person segmentation using YOLO segmentation masks."""
    try:
        import numpy as np
    except Exception:
        return None

    try:
        model = _load_yolo_model(model_name)
    except Exception:
        return None

    rgb = np.array(image.convert("RGB"))
    try:
        results = model.predict(
            source=rgb,
            task="segment",
            classes=[0],  # person
            conf=conf,
            iou=iou,
            verbose=False,
            max_det=5,
        )
    except Exception:
        return None

    if not results:
        return None

    result = results[0]
    if result.masks is None or result.boxes is None:
        return None

    try:
        masks = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
    except Exception:
        return None

    if len(masks) == 0 or len(scores) == 0:
        return None

    h, w = rgb.shape[:2]
    merged = np.zeros((h, w), dtype=np.uint8)

    for mask, score in zip(masks, scores, strict=False):
        if float(score) < conf:
            continue
        merged = np.maximum(merged, (mask > 0.5).astype(np.uint8) * 255)

    if merged.max() == 0:
        return None

    pil_mask = Image.fromarray(merged, mode="L")
    # Mild post-filtering removes tiny mask speckles while keeping edges smooth.
    return pil_mask.filter(ImageFilter.MedianFilter(size=3)).filter(
        ImageFilter.GaussianBlur(radius=1)
    )


def _segment_person_mask_grabcut(image: Image.Image) -> Image.Image | None:
    """Fallback person extraction via OpenCV HOG detector + GrabCut."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _weights = hog.detectMultiScale(
        bgr,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    if len(rects) == 0:
        return None

    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])

    # Expand the detection window to avoid trimming arms/legs.
    pad_w = int(w * 0.1)
    pad_h = int(h * 0.1)
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(bgr.shape[1] - x, w + 2 * pad_w)
    h = min(bgr.shape[0] - y, h + 2 * pad_h)

    gc_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            bgr,
            gc_mask,
            (int(x), int(y), int(w), int(h)),
            bg_model,
            fg_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except Exception:
        return None

    fg = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    if fg.max() == 0:
        return None

    return Image.fromarray(fg, mode="L").filter(ImageFilter.GaussianBlur(radius=1))
