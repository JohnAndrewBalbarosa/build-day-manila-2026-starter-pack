"""Manual test harness for ``Frame.person_only_image``.

Run with:
    uv run python -m core.test_frame

By default this reads images from ``FolderImage/input`` at the repo root and
writes segmented PNGs into ``FolderImage/output``.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from core.frame import Frame

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_input_dir() -> Path:
    return _repo_root() / "FolderImage" / "input"


def _default_output_dir() -> Path:
    return _repo_root() / "FolderImage" / "output"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _available_backends() -> list[str]:
    backends: list[str] = []
    has_numpy = _module_available("numpy")

    if has_numpy and _module_available("ultralytics"):
        backends.append("YOLO")
    if has_numpy and _module_available("cv2"):
        backends.append("GrabCut")

    return backends


def _iter_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _build_output_path(source_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = source_path.relative_to(input_dir)
    return output_dir / relative_path.parent / f"{source_path.stem}_segmented.png"


def _has_non_empty_mask(image: Image.Image) -> bool:
    if image.mode == "RGBA":
        return image.getchannel("A").getbbox() is not None
    return image.getbbox() is not None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-test Frame.person_only_image() using FolderImage/input and "
            "FolderImage/output by default."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory where segmented PNGs will be written.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence for person detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold for YOLO NMS.",
    )
    parser.add_argument(
        "--model-name",
        default="yolo11n-seg.pt",
        help="YOLO segmentation model name or local path.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    backends = _available_backends()
    errors: list[str] = []

    if not input_dir.exists():
        errors.append(f"Input directory does not exist: {input_dir}")
        image_paths: list[Path] = []
    else:
        image_paths = _iter_image_paths(input_dir)
        if not image_paths:
            errors.append(f"No supported images found in: {input_dir}")

    if not backends:
        errors.append(
            "No segmentation backend is installed. If you use uv, run "
            "`uv add --package core opencv-python` for GrabCut fallback or "
            "`uv add --package core ultralytics` for YOLO segmentation, then "
            "rerun this script."
        )

    if errors:
        raise SystemExit("\n".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[test_frame] Reading images from: {input_dir}")
    print(f"[test_frame] Writing results to: {output_dir}")
    print(f"[test_frame] Available backends: {', '.join(backends)}")

    detected_count = 0

    for source_path in image_paths:
        with Image.open(source_path) as image_file:
            image = image_file.convert("RGB")

        frame = Frame(
            image=image,
            timestamp=datetime.now(timezone.utc),
        )
        segmented = frame.person_only_image(
            conf=args.conf,
            iou=args.iou,
            model_name=args.model_name,
        )

        target_path = _build_output_path(source_path, input_dir, output_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        segmented.save(target_path)

        detected = _has_non_empty_mask(segmented)
        if detected:
            detected_count += 1

        status = "person detected" if detected else "no person detected"
        print(f"[test_frame] {source_path.name} -> {target_path.name} ({status})")

    print(
        f"[test_frame] Processed {len(image_paths)} image(s); "
        f"{detected_count} produced a non-empty segmentation mask."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
