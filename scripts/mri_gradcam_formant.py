import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Ensure the scripts directory (this file) is importable together with the project root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Reuse preprocessing helpers from the inference script to avoid duplication.
    from run_mri_video_inference import (  # type: ignore
        build_mri_model,
        frames_to_tensor,
        load_scaler,
        load_video_frames,
    )
except ImportError as exc:  # pragma: no cover - safeguard for unexpected layouts
    raise ImportError(
        "Failed to import run_mri_video_inference helpers. Please run this script "
        "from the project root (e.g., python scripts/mri_gradcam_formant.py ...)."
    ) from exc


@dataclass
class GradCAMOutputs:
    """
    Container for Grad-CAM results.

    Attributes
    ----------
    heatmaps : torch.Tensor
        Heatmaps with shape (T, H, W) in [0, 1].
    per_frame : Dict[int, torch.Tensor]
        Optional per-frame heatmaps keyed by frame index.
    band_name : str
        Name of the frequency band this Grad-CAM corresponds to.
    """

    heatmaps: torch.Tensor
    per_frame: Dict[int, torch.Tensor]
    band_name: str


def hz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_bin_frequencies(
    n_mels: int, sampling_rate: int, fmin: float, fmax: Optional[float]
) -> np.ndarray:
    if fmax is None or fmax <= 0:
        fmax = sampling_rate / 2
    mel_min = float(hz_to_mel(np.array([fmin]))[0])
    mel_max = float(hz_to_mel(np.array([fmax]))[0])
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    mel_centers = 0.5 * (mels[:-1] + mels[1:])
    freqs = mel_to_hz(mel_centers)
    return freqs


def parse_band_arguments(
    band_args: Optional[Sequence[str]],
    n_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: Optional[float],
) -> Dict[str, np.ndarray]:
    """
    Parse --formant-band arguments such as "F1:300-900" into mel-bin indices.
    """
    default_bands = {"F1": (300.0, 900.0), "F2": (900.0, 2500.0)}
    bands = {}
    if not band_args:
        bands = default_bands
    else:
        for spec in band_args:
            if ":" not in spec or "-" not in spec:
                raise ValueError(f"Invalid band specification '{spec}'. Use NAME:LOW-HIGH.")
            name, rest = spec.split(":", 1)
            low_str, high_str = rest.split("-", 1)
            try:
                low = float(low_str)
                high = float(high_str)
            except ValueError as exc:
                raise ValueError(f"Band range must be numeric: '{spec}'.") from exc
            if high <= low:
                raise ValueError(f"Band upper bound must exceed lower bound: '{spec}'.")
            bands[name.strip()] = (low, high)

    freqs = mel_bin_frequencies(n_mels, sampling_rate, fmin, fmax)
    band_to_indices: Dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            raise ValueError(
                f"No mel bins fall inside {name} range ({low}-{high} Hz). "
                "Adjust the band or mel settings."
            )
        band_to_indices[name] = idx
    return band_to_indices


def denormalize_mel(pred_norm: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.from_numpy(mean).to(pred_norm.device)
    std_t = torch.from_numpy(std).to(pred_norm.device)
    return pred_norm * std_t + mean_t


def _forward_with_features(model: torch.nn.Module, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass that exposes the CNN feature maps prior to global pooling.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded OTN-like CNN-LSTM model.
    frames : torch.Tensor
        Input tensor of shape (B, T, 1, H, W) or (B, T, H, W).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (predicted normalized mels (B, T, n_mels), cnn feature maps (B*T, C, Hc, Wc))
    """

    if frames.dim() not in (4, 5):
        raise ValueError(f"Expected frames of shape (B,T,1,H,W) or (B,T,H,W), got {tuple(frames.shape)}")

    B, T = frames.shape[0], frames.shape[1]
    x = frames.reshape(B * T, *frames.shape[2:])
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)

    backbone = model.cnn.backbone  # type: ignore[attr-defined]
    feats = backbone(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    feats = feats.requires_grad_(True)
    feats.retain_grad()

    pooled = feats.mean(dim=(2, 3))
    seq = pooled.view(B, T, -1)
    lstm_out = model.rnn(seq)
    pred = model.head(lstm_out)
    return pred, feats


def _compute_cam_from_grads(
    feats: torch.Tensor,
    grads: torch.Tensor,
    batch: int,
    timesteps: int,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert gradients + feature maps into time-resolved heatmaps.
    """
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))  # (B*T,1,Hc,Wc)
    cam = cam.view(batch, timesteps, *cam.shape[-2:])
    cam = cam.detach()

    cams = []
    for t in range(timesteps):
        cam_t = cam[:, t, :, :]
        cam_t = F.interpolate(
            cam_t.unsqueeze(1),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        cam_t = cam_t - cam_t.amin(dim=(-2, -1), keepdim=True)
        denom = cam_t.amax(dim=(-2, -1), keepdim=True)
        cam_t = cam_t / (denom + 1e-6)
        cams.append(cam_t)
    cams = torch.stack(cams, dim=1)  # (B, T, H, W)
    if cams.shape[0] != 1:
        raise ValueError("Grad-CAM helper currently supports batch size 1.")
    return cams.squeeze(0)


def compute_gradcam(
    model: torch.nn.Module,
    frames: torch.Tensor,
    mel_mean: np.ndarray,
    mel_std: np.ndarray,
    band_indices: np.ndarray,
    reduction: str = "mean",
    frame_indices: Optional[Iterable[int]] = None,
) -> GradCAMOutputs:
    """
    Compute Grad-CAM heatmaps for a given frequency band.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("Reduction must be 'mean' or 'sum'.")

    device = frames.device
    was_training = model.training
    dropout_training_state = None
    if hasattr(model, "rnn") and hasattr(model.rnn, "dropout"):
        dropout_training_state = model.rnn.dropout.training
    model.train()
    if hasattr(model, "rnn") and hasattr(model.rnn, "dropout"):
        model.rnn.dropout.train(False)

    pred_norm, feats = _forward_with_features(model, frames)
    B, T, n_mels = pred_norm.shape

    mel_denorm = denormalize_mel(pred_norm, mel_mean, mel_std)
    mel_power = torch.pow(10.0, mel_denorm / 10.0)

    band_idx_t = torch.as_tensor(band_indices, device=device, dtype=torch.long)
    power_in_band = mel_power.index_select(dim=-1, index=band_idx_t).sum(dim=-1)  # (B, T)

    frame_list: List[int] = list(frame_indices) if frame_indices else []
    retain_graph = len(frame_list) > 0

    model.zero_grad(set_to_none=True)
    if feats.grad is not None:
        feats.grad.zero_()

    if reduction == "mean":
        scalar_target = power_in_band.mean()
    else:
        scalar_target = power_in_band.sum()
    scalar_target.backward(retain_graph=retain_graph)
    grads = feats.grad.detach().clone()

    target_hw = (frames.shape[-2], frames.shape[-1])
    cams = _compute_cam_from_grads(feats.detach(), grads, B, T, target_hw)

    per_frame_maps: Dict[int, torch.Tensor] = {}
    if frame_list:
        for idx, frame_idx in enumerate(frame_list):
            if not (0 <= frame_idx < T):
                raise IndexError(f"Frame index {frame_idx} out of range (0 <= idx < {T}).")
            model.zero_grad(set_to_none=True)
            if feats.grad is not None:
                feats.grad.zero_()
            target = power_in_band[:, frame_idx].mean()
            retain = idx < len(frame_list) - 1
            target.backward(retain_graph=retain)
            frame_grads = feats.grad.detach().clone()
            frame_cam = _compute_cam_from_grads(
                feats.detach(),
                frame_grads,
                B,
                T,
                target_hw,
            )[frame_idx]
            per_frame_maps[frame_idx] = frame_cam

    if hasattr(model, "rnn") and hasattr(model.rnn, "dropout") and dropout_training_state is not None:
        model.rnn.dropout.train(dropout_training_state)
    if not was_training:
        model.eval()

    return GradCAMOutputs(heatmaps=cams, per_frame=per_frame_maps, band_name="unknown")


def overlay_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    output_path: Path,
    cmap: str = "jet",
    alpha: float = 0.5,
) -> None:
    """
    Save an overlay of the Grad-CAM heatmap on top of an MRI frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(frame, cmap="gray", interpolation="nearest")
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_heatmap_sequence(
    cams: torch.Tensor,
    frames_np: np.ndarray,
    band_name: str,
    output_dir: Path,
    target_frames: Sequence[int],
) -> None:
    """
    Persist Grad-CAM results (numpy dump + optional frame overlays).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"gradcam_{band_name}_sequence.npy"
    np.save(npy_path, cams.cpu().numpy())

    # Save a time-averaged summary for a quick glance.
    avg_cam = cams.mean(dim=0).cpu().numpy()
    overlay_heatmap(frames_np.mean(axis=0), avg_cam, output_dir / f"gradcam_{band_name}_average.png")

    for frame_idx in target_frames:
        heat = cams[frame_idx].cpu().numpy()
        frame = frames_np[frame_idx]
        overlay_heatmap(frame, heat, output_dir / f"gradcam_{band_name}_frame{frame_idx:04d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for CNN-LSTM MRI -> mel model focusing on formant bands."
    )
    parser.add_argument("--video", required=True, help="Path to rtMRI video (.mp4).")
    parser.add_argument("--mri-checkpoint", required=True, help="Path to CNN-LSTM checkpoint (acoustic model).")
    parser.add_argument("--scaler-json", required=True, help="Path to scaler.json with mel mean/std.")
    parser.add_argument("--output-dir", required=True, help="Directory to store Grad-CAM outputs.")
    parser.add_argument(
        "--mri-code-dir",
        help="Directory containing mri_acoustic_model.py if different from default (Desktop/mri2speech_code).",
    )
    parser.add_argument("--n-mels", type=int, default=64, help="Number of mel bins (default: 64).")
    parser.add_argument("--sampling-rate", type=int, default=11413, help="Sampling rate used for mel extraction.")
    parser.add_argument("--fmin", type=float, default=0.0, help="Mel filter minimum frequency (Hz).")
    parser.add_argument("--fmax", type=float, default=8000.0, help="Mel filter maximum frequency (Hz).")
    parser.add_argument(
        "--formant-band",
        action="append",
        metavar="NAME:LOW-HIGH",
        help="Frequency band in Hz for Grad-CAM target (e.g., F1:300-900). "
        "Can be specified multiple times. Defaults to F1 and F2 ranges if omitted.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        nargs="*",
        default=[],
        help="Optional list of frame indices for which per-frame heatmaps should be saved.",
    )
    parser.add_argument(
        "--reduction",
        choices=["mean", "sum"],
        default="mean",
        help="Reduction over time when building the Grad-CAM scalar target.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run Grad-CAM on. 'auto' uses CUDA if available.",
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()

    device = resolve_device(args.device)
    print(f"[INFO] Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_scaler(Path(args.scaler_json))
    bands = parse_band_arguments(args.formant_band, args.n_mels, args.sampling_rate, args.fmin, args.fmax)

    frames = load_video_frames(Path(args.video))
    frames_tensor = frames_to_tensor(frames, use_channel=True).to(device)

    model_args = argparse.Namespace(
        mri_checkpoint=args.mri_checkpoint,
        n_mels=args.n_mels,
        cnn_pretrained=False,
        rnn_hidden=640,
        dropout=0.5,
        mri_code_dir=args.mri_code_dir,
    )
    model = build_mri_model(model_args, device)

    frames_np = frames.numpy()
    for band_name, band_idx in bands.items():
        print(f"[INFO] Computing Grad-CAM for {band_name} (bins={band_idx.tolist()}).")
        outputs = compute_gradcam(
            model,
            frames_tensor,
            mean,
            std,
            band_idx,
            reduction=args.reduction,
            frame_indices=args.target_frames,
        )
        outputs.band_name = band_name
        save_heatmap_sequence(outputs.heatmaps, frames_np, band_name, output_dir, args.target_frames)

        for frame_idx, heat in outputs.per_frame.items():
            overlay_heatmap(
                frames_np[frame_idx],
                heat.cpu().numpy(),
                output_dir / f"gradcam_{band_name}_frame{frame_idx:04d}_detail.png",
            )

    print("[DONE] Grad-CAM computation finished.")


if __name__ == "__main__":
    main()
