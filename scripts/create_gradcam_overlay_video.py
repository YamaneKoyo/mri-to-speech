import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import subprocess
import tempfile


def load_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError("No frames loaded from video.")

    return frames, fps


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    min_val = h.min()
    max_val = h.max()
    if max_val > min_val:
        h = (h - min_val) / (max_val - min_val)
    else:
        h = np.zeros_like(h, dtype=np.float32)
    return np.clip(h, 0.0, 1.0)


def colorize_heatmap(h: np.ndarray, alpha: float) -> np.ndarray:
    heat_uint8 = (h * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    return colored * alpha


def overlay_heatmaps(
    frames: list[np.ndarray],
    heatmaps: np.ndarray,
    alpha: float,
    resize_to: Optional[tuple[int, int]] = None,
) -> list[np.ndarray]:
    overlays: list[np.ndarray] = []
    for frame, heat in zip(frames, heatmaps):
        if resize_to and (heat.shape[1], heat.shape[0]) != resize_to[::-1]:
            heat = cv2.resize(heat, resize_to[::-1], interpolation=cv2.INTER_LINEAR)
        heat_norm = normalize_heatmap(heat)
        color = colorize_heatmap(heat_norm, alpha)
        if resize_to:
            frame_resized = cv2.resize(frame, resize_to[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            frame_resized = frame
        base = np.repeat(frame_resized[:, :, None], 3, axis=2)
        composite = np.clip(base * (1.0 - alpha) + color, 0.0, 1.0)
        overlays.append((composite * 255.0).astype(np.uint8))
    return overlays


def combine_heatmaps(
    primary: np.ndarray,
    secondary: Optional[np.ndarray],
    mode: str,
) -> np.ndarray:
    if secondary is None:
        return primary
    if primary.shape != secondary.shape:
        raise ValueError("Heatmap arrays must have the same shape to combine.")
    if mode == "max":
        return np.maximum(primary, secondary)
    if mode == "mean":
        return 0.5 * (primary + secondary)
    raise ValueError(f"Unsupported combine mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Grad-CAM overlay video with audio.")
    parser.add_argument("--video", required=True, help="Input video (rtMRI).")
    parser.add_argument("--heatmap", required=True, help="Primary heatmap .npy (T,H,W).")
    parser.add_argument("--heatmap2", help="Optional secondary heatmap .npy (T,H,W).")
    parser.add_argument(
        "--combine-mode",
        choices=["max", "mean"],
        default="max",
        help="How to combine primary and secondary heatmaps (default: max).",
    )
    parser.add_argument("--audio", required=True, help="Audio file (.wav) to attach.")
    parser.add_argument("--output", required=True, help="Output video path (.mp4).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Heatmap overlay alpha (default: 0.6).")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (defaults to source video FPS).")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), help="Resize frames to WxH.")
    args = parser.parse_args()

    frames, src_fps = load_video_frames(Path(args.video))
    target_size = None
    if args.resize:
        target_size = (args.resize[1], args.resize[0])  # cv2 uses (H,W)

    heat1 = np.load(args.heatmap)
    heat2 = np.load(args.heatmap2) if args.heatmap2 else None
    heat_combined = combine_heatmaps(heat1, heat2, args.combine_mode)

    if heat_combined.shape[0] != len(frames):
        raise ValueError(f"Heatmap length {heat_combined.shape[0]} does not match video frames {len(frames)}")

    overlays = overlay_heatmaps(frames, heat_combined, args.alpha, target_size)

    fps = args.fps if args.fps else src_fps
    height, width = overlays[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video = Path(tmp.name)
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter.")
    for frame in overlays:
        writer.write(frame)
    writer.release()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        args.audio,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        args.output,
    ]
    subprocess.run(cmd, check=True)
    temp_video.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
