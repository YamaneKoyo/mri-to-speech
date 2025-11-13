import argparse
from pathlib import Path

import cv2
import numpy as np


def load_frames(video_path: Path, start_frame: int, count: int, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
        gray = gray.astype(np.float32)
        gray -= gray.min()
        if gray.max() > 0:
            gray /= gray.max()
        frames.append(gray)
    cap.release()
    if not frames:
        raise RuntimeError("No frames loaded from video segment.")
    return np.stack(frames, axis=0)


def heat_to_bgr(heat: np.ndarray, alpha: float) -> np.ndarray:
    heat_norm = np.clip(heat, 0.0, 1.0)
    heat_uint8 = (heat_norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return colored.astype(np.float32) / 255.0 * alpha


def overlay_frames(frames: np.ndarray, heats: np.ndarray, alpha: float) -> np.ndarray:
    if len(frames) != len(heats):
        raise ValueError("Number of frames and heatmaps must match.")

    overlays = []
    for frame, heat in zip(frames, heats):
        base = np.repeat(frame[:, :, None], 3, axis=2)
        colored = heat_to_bgr(heat, alpha)
        combined = np.clip(colored + base * (1.0 - alpha), 0.0, 1.0)
        overlays.append((combined * 255.0).astype(np.uint8))
    return overlays


def write_video(output_path: Path, frames: list[np.ndarray], fps: float, repeat: int) -> None:
    if not frames:
        raise RuntimeError("No frames to write.")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer for {output_path}")

    for frame in frames:
        for _ in range(max(repeat, 1)):
            writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create slow Grad-CAM overlay video from heatmap sequence.")
    parser.add_argument("--video", required=True, help="Input rtMRI video (.mp4).")
    parser.add_argument("--sequence", required=True, help="Grad-CAM sequence .npy file (shape [T,H,W]).")
    parser.add_argument("--start-frame", type=int, required=True, help="Frame index in the video where the sequence starts.")
    parser.add_argument("--output", required=True, help="Output video path (.mp4).")
    parser.add_argument("--fps", type=float, default=5.0, help="Output video FPS (default: 5).")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat each frame for slowing down (default: 3).")
    parser.add_argument("--alpha", type=float, default=0.6, help="Heatmap overlay alpha (default: 0.6).")
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256], metavar=("W", "H"), help="Target frame size (default: 256 256).")
    args = parser.parse_args()

    sequence = np.load(args.sequence)
    frames = load_frames(Path(args.video), args.start_frame, sequence.shape[0], (args.resize[0], args.resize[1]))
    overlays = overlay_frames(frames, sequence, args.alpha)
    write_video(Path(args.output), overlays, args.fps, args.repeat)


if __name__ == "__main__":
    main()

