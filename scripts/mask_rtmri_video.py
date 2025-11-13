import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskPreset:
    """Simple container for predefined mask polygons."""

    name: str
    points: Tuple[Tuple[float, float], ...]
    base_size: Tuple[float, float] = (256.0, 256.0)

    def scaled(self, target_size: Tuple[int, int]) -> np.ndarray:
        """Scale source points (defined in base_size) to the desired size."""
        width, height = target_size
        base_w, base_h = self.base_size
        pts = np.array(self.points, dtype=np.float32)
        scale_x = width / base_w
        scale_y = height / base_h
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        return pts


# Preset taken from lip mask screenshot (approx. 256x256 coords)
LIP_MASK = MaskPreset(
    name="lip",
    points=(
        (8.0, 84.0),
        (43.0, 84.0),
        (45.0, 156.0),
        (8.0, 156.0),
    ),
)

TONGUE_MASK = MaskPreset(
    name="tongue",
    points=(
        (36.1, 102.7),
        (63.4, 90.9),
        (122.7, 111.5),
        (133.4, 172.2),
        (47.6, 155.0),
    ),
)


def build_mask(
    shape: Tuple[int, int],
    polygon: np.ndarray,
    alpha: float,
    blur_kernel: int,
) -> np.ndarray:
    """Create a soft mask with the given polygon and attenuation."""
    h, w = shape
    mask = np.ones((h, w), dtype=np.float32)
    poly_int = np.round(polygon).astype(np.int32)
    cv2.fillConvexPoly(mask, poly_int, alpha)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), sigmaX=0.0)
    return np.clip(mask, alpha, 1.0)


def apply_mask_to_video(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if mask.shape != (height, width):
        raise ValueError(f"Mask shape {mask.shape} != frame shape {(height, width)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            masked = (frame.astype(np.float32) * mask[..., None]).clip(0.0, 255.0).astype(
                np.uint8
            )
            writer.write(masked)
    finally:
        cap.release()
        writer.release()


def parse_args() -> argparse.Namespace:
    preset_names = ["lip", "tongue"]
    parser = argparse.ArgumentParser(description="Apply soft articulation mask to rtMRI video")
    parser.add_argument("--input", required=True, help="Input rtMRI video (mp4)")
    parser.add_argument("--output", required=True, help="Output masked video path")
    parser.add_argument(
        "--mask-type",
        default="lip",
        choices=preset_names,
        help="Which preset mask to apply",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Residual intensity inside the mask (0-1).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=11,
        help="Gaussian blur kernel size for soft edges (odd int).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    presets = {"lip": LIP_MASK, "tongue": TONGUE_MASK}
    preset = presets[args.mask_type]

    # Load first frame to determine size
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read first frame from video")
    height, width = frame.shape[:2]
    polygon = preset.scaled((width, height))
    mask = build_mask((height, width), polygon, alpha=args.alpha, blur_kernel=args.blur_kernel)
    apply_mask_to_video(input_path, output_path, mask)
    print(f"[INFO] Masked video written to {output_path}")


if __name__ == "__main__":
    main()
