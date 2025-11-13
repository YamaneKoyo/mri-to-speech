import random
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create HiFi-GAN training/validation filelists from wav directory")
    parser.add_argument("wav_dir", type=Path, help="Directory with wav files")
    parser.add_argument("output_dir", type=Path, help="Directory to write training.txt/validation.txt")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Fraction of files for validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    wav_paths = sorted([p for p in args.wav_dir.glob("*.wav") if p.is_file()])
    if not wav_paths:
        raise SystemExit(f"No wav files found in {args.wav_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(wav_paths)
    valid_count = max(1, int(len(wav_paths) * args.valid_ratio))
    valid_paths = wav_paths[:valid_count]
    train_paths = wav_paths[valid_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "training.txt"
    valid_file = args.output_dir / "validation.txt"
    def write(paths, dest):
        with dest.open("w", encoding="utf-8") as f:
            for path in paths:
                stem = path.stem
                f.write(f"{stem}|dummy|dummy\n")
    write(train_paths, train_file)
    write(valid_paths, valid_file)
    print(f"Wrote {len(train_paths)} training entries to {train_file}")
    print(f"Wrote {len(valid_paths)} validation entries to {valid_file}")

if __name__ == "__main__":
    main()
