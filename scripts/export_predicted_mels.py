import argparse
import json
from pathlib import Path

import numpy as np
import sys
import torch
from tqdm import tqdm


def load_scaler(scaler_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with scaler_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32)
    return mean, std


def build_model(checkpoint_path: Path, n_mels: int, device: torch.device):
    from mri_acoustic_model import build_acoustic_model

    model = build_acoustic_model(
        n_mels=n_mels,
        cnn_pretrained=False,
        rnn_hidden=640,
        dropout=0.5,
        use_checkpoint=False,
        ckpt_segments=2,
        use_reentrant=False,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] missing keys when loading MRI model: {missing}")
    if unexpected:
        print(f"[WARN] unexpected keys when loading MRI model: {unexpected}")
    model.eval()
    return model


def export_mels(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir).resolve()
    samples_dir = processed_dir / "samples"
    if not samples_dir.is_dir():
        raise SystemExit(f"samples directory not found: {samples_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = Path(args.scaler_json).resolve()
    mean, std = load_scaler(scaler_path)
    if mean.numel() != std.numel():
        raise SystemExit("Scaler mean/std length mismatch")
    n_mels = mean.numel()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if args.cpu:
        print("[INFO] Forcing CPU execution")
    else:
        print(f"[INFO] Using device: {device}")

    if args.mri_code_dir:
        code_dir = Path(args.mri_code_dir).resolve()
        if code_dir.is_dir() and str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))

    model = build_model(Path(args.mri_checkpoint).resolve(), n_mels, device)
    mean = mean.to(device)
    std = std.to(device)

    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not sample_dirs:
        raise SystemExit(f"No sample folders found under {samples_dir}")

    with torch.no_grad():
        for sample_path in tqdm(sample_dirs, desc="Exporting mels"):
            stem = sample_path.name
            out_path = output_dir / f"{stem}.npy"
            if out_path.exists() and not args.overwrite:
                continue

            mri_path = sample_path / "mri.npy"
            if not mri_path.is_file():
                print(f"[WARN] MRI file missing for {stem}, skipping")
                continue

            mri = np.load(mri_path).astype(np.float32)
            frames = torch.from_numpy(mri).unsqueeze(0).unsqueeze(2).to(device)

            pred_norm = model(frames).squeeze(0)  # (T, n_mels)
            mel_db = pred_norm * std + mean

            mel_power = torch.pow(10.0, mel_db / 10.0).clamp_min(1e-5)
            mel_log = torch.log(mel_power)

            mel_log_np = mel_log.transpose(0, 1).cpu().numpy().astype(np.float32)
            np.save(out_path, mel_log_np)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export predicted log-mel features for HiFi-GAN fine-tuning.")
    parser.add_argument("--processed_dir", required=True, help="rtMRI processed dataset root (contains samples/).")
    parser.add_argument("--mri_checkpoint", required=True, help="Path to trained MRI->mel checkpoint (.pt).")
    parser.add_argument("--scaler_json", required=True, help="Path to scaler.json (mean/std for denormalization).")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to store generated log-mel numpy files (one per sample, shape [64, T]).",
    )
    parser.add_argument(
        "--mri_code_dir",
        help="Directory containing mri_acoustic_model.py (if not importable by default).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution even if CUDA is available.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_mels(args)


if __name__ == "__main__":
    main()
