import argparse
import os
from pathlib import Path
import numpy as np


def convert_npz_to_npy(pairs_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(pairs_dir.glob("*.npz"))
    if not files:
        files = sorted(pairs_dir.glob("*/*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found in {pairs_dir}")

    for idx, npz_path in enumerate(files, 1):
        stem = npz_path.stem
        target_dir = output_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        mri_out = target_dir / "mri.npy"
        mel_out = target_dir / "mel.npy"
        mask_out = target_dir / "mask.npy"
        if not overwrite and mri_out.exists() and mel_out.exists() and mask_out.exists():
            print(f"[{idx}/{len(files)}] skip {stem} (already exists)")
            continue
        print(f"[{idx}/{len(files)}] convert {npz_path.name} -> {stem}/[mri, mel, mask].npy")
        with np.load(npz_path, allow_pickle=False) as data:
            np.save(mri_out, data["mri"], allow_pickle=False)
            np.save(mel_out, data["mel"], allow_pickle=False)
            np.save(mask_out, data["mask"], allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert compressed pair npz files to npy arrays for mmap loading.")
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--ref_frames", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pairs_dir = args.processed_dir / f"pairs_ref{args.ref_frames}"
    output_dir = args.processed_dir / f"pairs_ref{args.ref_frames}_npy"
    convert_npz_to_npy(pairs_dir, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
