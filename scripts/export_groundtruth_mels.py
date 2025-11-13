import argparse
from pathlib import Path

import numpy as np


def convert_mel_db_to_log_power(mel_db: np.ndarray) -> np.ndarray:
    """
    Convert mel spectrogram in dB scale (T, n_mels) to log-power (n_mels, T).
    """
    if mel_db.ndim != 2:
        raise ValueError(f"Expected 2-D mel array (T, n_mels), got shape {mel_db.shape}")
    mel_power = np.power(10.0, mel_db / 10.0).astype(np.float32, copy=False)
    np.maximum(mel_power, 1e-5, out=mel_power)
    mel_log = np.log(mel_power).astype(np.float32, copy=False)
    return mel_log.T


def export_groundtruth_mels(args: argparse.Namespace) -> None:
    processed_dir = Path(args.processed_dir).resolve()
    samples_dir = processed_dir / "samples"
    if not samples_dir.is_dir():
        raise SystemExit(f"samples directory not found: {samples_dir}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not sample_dirs:
        raise SystemExit(f"No subdirectories found in {samples_dir}")

    skipped = 0
    converted = 0
    for sample_path in sample_dirs:
        stem = sample_path.name
        src = sample_path / "mel_db.npy"
        if not src.is_file():
            print(f"[WARN] mel_db.npy missing in {sample_path}, skipping")
            skipped += 1
            continue

        dst = output_dir / f"{stem}.npy"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        mel_db = np.load(src).astype(np.float32)
        mel_log = convert_mel_db_to_log_power(mel_db)
        np.save(dst, mel_log)
        converted += 1

    print(f"[DONE] Converted {converted} mel files. Skipped {skipped}. Output dir: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ground-truth mel spectrograms (log-power, [n_mels, T]) from mel_db.npy files."
    )
    parser.add_argument("--processed_dir", required=True, help="rtMRI processed dataset root (contains samples/).")
    parser.add_argument("--output_dir", required=True, help="Directory to store exported mel numpy files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_groundtruth_mels(args)


if __name__ == "__main__":
    main()
