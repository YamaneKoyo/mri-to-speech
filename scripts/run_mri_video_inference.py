
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from models import Generator
import cv2


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_sys_path(path: Path):
    if path and path.exists():
        sys.path.insert(0, str(path))


def _preprocess_frame(frame: np.ndarray, target_size=(256, 256)) -> np.ndarray:
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    if gray.shape[::-1] != target_size:
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    gray = gray.astype(np.float32)
    mean = gray.mean()
    std = gray.std()
    if std > 0:
        gray = (gray - mean) / std
    else:
        gray = gray - mean
    min_val = gray.min()
    max_val = gray.max()
    if max_val > min_val:
        gray = (gray - min_val) / (max_val - min_val)
    else:
        gray = np.zeros_like(gray)
    return gray


def load_video_frames(video_path: Path, target_size=(256, 256), max_frames=None) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(_preprocess_frame(frame, target_size))
    cap.release()
    if not frames:
        raise ValueError("No frames could be read from video")
    frames_array = np.asarray(frames, dtype=np.float32)
    return torch.from_numpy(frames_array)


def load_scaler(stats_path: Path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    if "mean" not in stats or "std" not in stats:
        raise KeyError("Scaler JSON must contain 'mean' and 'std' lists")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Scaler mean/std must be 1-D lists")
    return mean, std


def load_hifigan(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path, "r", encoding="utf-8") as f:
        h = AttrDict(json.load(f))
    generator = Generator(h).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "generator" not in ckpt:
        raise KeyError("HiFi-GAN checkpoint missing 'generator' state")
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    # remove weight norm for stability (best-effort)
    try:
        from torch.nn.utils import remove_weight_norm
    except Exception:
        remove_weight_norm = None

    if remove_weight_norm is not None:
        for module in list(generator.ups) + [generator.conv_post]:
            try:
                remove_weight_norm(module)
            except (ValueError, AttributeError):
                pass
        for res in generator.resblocks:
            try:
                res.remove_weight_norm()
            except (ValueError, AttributeError):
                pass
    return generator, h


def build_mri_model(args, device: torch.device):
    code_dir = Path(args.mri_code_dir) if args.mri_code_dir else None
    if code_dir is None:
        # default: assume script relative to project root
        code_dir = Path(args.mri_checkpoint).resolve().parent.parent / "mri2speech_code"
    _ensure_sys_path(code_dir)
    try:
        from mri_acoustic_model import build_acoustic_model
    except ImportError as exc:
        raise ImportError("Failed to import mri_acoustic_model. Use --mri-code-dir to point to the mri2speech_code directory.") from exc

    model_kwargs = {
        "n_mels": args.n_mels,
        "cnn_pretrained": False,
        "rnn_hidden": args.rnn_hidden,
        "dropout": args.dropout,
        "use_checkpoint": False,
        "ckpt_segments": 2,
        "use_reentrant": False,
    }
    model = build_acoustic_model(**model_kwargs).to(device)
    checkpoint = torch.load(args.mri_checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading MRI model: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading MRI model: {unexpected}")
    model.eval()
    return model


def frames_to_tensor(frames: torch.Tensor, use_channel: bool = True) -> torch.Tensor:
    if frames.dim() != 3:
        raise ValueError(f"Expected frames tensor of shape (T,H,W), got {tuple(frames.shape)}")
    frames = frames.unsqueeze(0)  # (1,T,H,W)
    if use_channel:
        frames = frames.unsqueeze(2)  # (1,T,1,H,W)
    return frames


def denormalize_mel(mel_normalized: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.from_numpy(mean).to(mel_normalized.device)
    std_t = torch.from_numpy(std).to(mel_normalized.device)
    return mel_normalized * std_t + mean_t


def save_outputs(audio: np.ndarray, mel: np.ndarray, output_dir: Path, sampling_rate: int, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{stem}_generated.wav"
    sf.write(audio_path, audio, sampling_rate)

    mel_path = output_dir / f"{stem}_mel.npy"
    np.save(mel_path, mel)

    plt.figure(figsize=(12, 4))
    plt.imshow(mel.T, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar()
    plt.title(f"Generated Mel Spectrogram - {stem}")
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    fig_path = output_dir / f"{stem}_mel.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return audio_path, mel_path, fig_path


def parse_args():
    parser = argparse.ArgumentParser(description="rtMRI ? Speech inference using OTN-like MRI model and HiFi-GAN vocoder")
    parser.add_argument("--video", required=True, help="Input rtMRI video (.mp4)")
    parser.add_argument("--mri-checkpoint", required=True, help="Path to OTN-like MRI checkpoint (.pt)")
    parser.add_argument("--scaler-json", required=True, help="Path to scaler.json (contains per-mel mean/std)")
    parser.add_argument("--hifigan-config", required=True, help="HiFi-GAN config JSON")
    parser.add_argument("--hifigan-checkpoint", required=True, help="HiFi-GAN generator checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated artifacts")
    parser.add_argument("--mri-code-dir", help="Directory containing mri_acoustic_model.py (defaults to sibling mri2speech_code)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max number of frames to process")
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--rnn-hidden", type=int, default=640)
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    scaler_path = Path(args.scaler_json)
    mean, std = load_scaler(scaler_path)
    if len(mean) != args.n_mels or len(std) != args.n_mels:
        raise ValueError("Scaler mean/std length does not match n_mels")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    frames = load_video_frames(video_path, target_size=(256, 256), max_frames=args.max_frames)
    frames_tensor = frames_to_tensor(frames, use_channel=True).to(device)

    mri_model = build_mri_model(args, device)
    with torch.no_grad():
        pred_norm = mri_model(frames_tensor)  # (1,T,n_mels)
    pred_norm = pred_norm.squeeze(0)
    print(f"[INFO] Predicted normalized mel shape: {tuple(pred_norm.shape)}")

    mel_denorm = denormalize_mel(pred_norm, mean, std)
    mel_denorm_np = mel_denorm.cpu().numpy().astype(np.float32)
    print(f"[INFO] Mel (denormalized dB) range: {mel_denorm_np.min():.3f} .. {mel_denorm_np.max():.3f}")

    # Convert dB-scaled mel to power, then apply HiFi-GAN's log compression
    mel_power = torch.pow(10.0, mel_denorm / 10.0)
    mel_log = torch.log(torch.clamp(mel_power, min=1e-5))
    mel_log_np = mel_log.cpu().numpy().astype(np.float32)
    print(f"[INFO] Mel (log-power) range: {mel_log_np.min():.3f} .. {mel_log_np.max():.3f}")

    generator, hifigan_config = load_hifigan(Path(args.hifigan_config), Path(args.hifigan_checkpoint), device)
    mel_for_hifigan = mel_log.transpose(0, 1).unsqueeze(0)  # (1, n_mels, T)
    mel_for_hifigan = mel_for_hifigan.float().to(device)

    with torch.no_grad():
        audio = generator(mel_for_hifigan).squeeze().cpu().numpy()
    print(f"[INFO] Generated audio length: {audio.shape[0]} samples")

    stem = video_path.stem
    output_dir = Path(args.output_dir)
    audio_path, mel_path, fig_path = save_outputs(audio, mel_denorm_np, output_dir, hifigan_config.sampling_rate, stem)
    log_mel_path = output_dir / f"{stem}_mel_log.npy"
    np.save(log_mel_path, mel_log_np)

    print("[DONE] Inference complete.")
    print(f"  Audio : {audio_path}")
    print(f"  Mel   : {mel_path}")
    print(f"  LogMel: {log_mel_path}")
    print(f"  Figure: {fig_path}")


if __name__ == "__main__":
    main()
