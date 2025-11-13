"""Data preprocessing pipeline for rtMRI-to-speech experiments.

This script normalizes audio extracted from video files, computes mel spectrograms
synchronized with MRI frames, and produces fixed-length training pairs compatible
with the OTN-style training pipeline.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf

try:
    import soxr  # type: ignore

    HAS_SOXR = True
except Exception:  # pragma: no cover - optional dependency
    soxr = None
    HAS_SOXR = False

try:
    from moviepy.editor import VideoFileClip  # type: ignore

    HAS_MOVIEPY = True
except Exception:  # pragma: no cover - optional dependency
    VideoFileClip = None
    HAS_MOVIEPY = False


def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - coef * x[:-1]
    return y


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x
    if HAS_SOXR:
        return soxr.resample(x, sr_in, sr_out, quality="VHQ")  # type: ignore[arg-type]
    return librosa.resample(x, orig_sr=sr_in, target_sr=sr_out, res_type="kaiser_best")


def read_audio_from_video(video_path: str, target_sr: int = 11413) -> Tuple[np.ndarray, int]:
    if not HAS_MOVIEPY:
        raise RuntimeError("moviepy is required to extract audio from video files.")
    clip = VideoFileClip(video_path)  # type: ignore[operator]
    try:
        audio_clip = clip.audio
        if audio_clip is None:
            raise RuntimeError(f"audio track not found: {video_path}")
        fps = getattr(audio_clip, "fps", target_sr) or target_sr
        try:
            y = np.asarray(audio_clip.to_soundarray(fps=fps))
        except TypeError as exc:
            if "arrays to stack" not in str(exc):
                raise
            if getattr(audio_clip, "nframes", None) is None and getattr(audio_clip, "duration", None) is not None:
                audio_clip.nframes = int(round(audio_clip.duration * fps))
            chunk_size = int(max(fps, 1024))
            chunks = [
                np.asarray(chunk)
                for chunk in audio_clip.iter_chunks(
                    fps=fps, chunksize=chunk_size, quantize=False, nbytes=2, logger=None
                )
            ]
            if chunks:
                y = np.concatenate(chunks, axis=0)
            else:
                y = np.zeros((0,), dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = y.astype(np.float32, copy=False)
        y = resample(y, int(fps), target_sr)
        return y, target_sr
    finally:
        clip.close()


def read_audio_from_wav(path: str, target_sr: int = 11413) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    y = resample(y, sr, target_sr)
    return y, target_sr


def read_video_frames(path: str, resize_hw: Tuple[int, int] = (256, 256)) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = resize_hw
            if gray.shape[:2] != (h, w):
                gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(gray.astype(np.float32) / 255.0)
    finally:
        cap.release()
    if not frames:
        return np.zeros((0, resize_hw[0], resize_hw[1]), dtype=np.float32)
    return np.stack(frames, axis=0)


def compute_mel_db(
    y: np.ndarray,
    sr: int,
    *,
    n_mels: int = 64,
    n_fft: int = 2048,
    win_length: int = 2048,
    hop_length: int = 420,
    fmin: float = 0.0,
    fmax: float | None = None,
    preemph: float = 0.97,
) -> np.ndarray:
    y = pre_emphasis(y, coef=preemph)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=False,
        power=2.0,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=1.0)
    return mel_db.T.astype(np.float32)


def align_by_hop(mri_T: int, audio_len_samples: int, hop_length: int) -> int:
    mel_T = int(np.floor(audio_len_samples / hop_length))
    return min(mri_T, mel_T)


def save_sample(out_dir: Path, stem: str, mri: np.ndarray, mel_db: np.ndarray) -> int:
    sample_dir = out_dir / "samples" / stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    T = min(mri.shape[0], mel_db.shape[0])
    mri = mri[:T]
    mel_db = mel_db[:T]
    mask = np.ones((T,), dtype=np.float32)
    np.save(sample_dir / "mri.npy", mri)
    np.save(sample_dir / "mel_db.npy", mel_db)
    np.save(sample_dir / "mask.npy", mask)
    return T


def pass2_compute_stats(out_dir: Path, stems: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    sum_vec: np.ndarray | None = None
    sumsq_vec: np.ndarray | None = None
    count = 0
    for stem in stems:
        mel_path = out_dir / "samples" / stem / "mel_db.npy"
        if not mel_path.exists():
            continue
        mel = np.load(mel_path)
        if mel.size == 0:
            continue
        mel64 = mel.astype(np.float64)
        if sum_vec is None:
            sum_vec = mel64.sum(axis=0)
            sumsq_vec = (mel64 ** 2).sum(axis=0)
        else:
            sum_vec += mel64.sum(axis=0)
            sumsq_vec += (mel64 ** 2).sum(axis=0)
        count += mel.shape[0]
    if count == 0 or sum_vec is None or sumsq_vec is None:
        raise RuntimeError("no mel frames collected; check the input data set")
    mean = sum_vec / count
    var = sumsq_vec / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    stats = {"mean": mean.tolist(), "std": std.tolist(), "count_frames": int(count)}
    with open(out_dir / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return mean.astype(np.float32), std.astype(np.float32)


def pass3_save_pairs(
    out_dir: Path,
    stems: Iterable[str],
    ref_frames: int,
    *,
    add_channel_dim: bool = True,
) -> Tuple[Path, int]:
    with open(out_dir / "scaler.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, -1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, -1)

    pairs_dir = out_dir / f"pairs_ref{ref_frames}"
    if pairs_dir.exists():
        shutil.rmtree(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    for stem in stems:
        base = out_dir / "samples" / stem
        mri_path = base / "mri.npy"
        mel_path = base / "mel_db.npy"
        mask_path = base / "mask.npy"
        if not (mri_path.exists() and mel_path.exists() and mask_path.exists()):
            continue
        mri = np.load(mri_path)
        mel = np.load(mel_path)
        mask = np.load(mask_path)
        T = min(len(mri), len(mel))
        if T < ref_frames:
            continue
        mri = mri[:T]
        mel = mel[:T]
        mask = mask[:T]
        mel_std = (mel - mean) / std
        n_pairs = T - ref_frames + 1
        if add_channel_dim:
            H, W = mri.shape[1:3]
            mri_pairs = np.empty((n_pairs, ref_frames, 1, H, W), dtype=np.float32)
        else:
            H, W = mri.shape[1:3]
            mri_pairs = np.empty((n_pairs, ref_frames, H, W), dtype=np.float32)
        mel_pairs = np.empty((n_pairs, ref_frames, mel_std.shape[1]), dtype=np.float32)
        mask_pairs = np.empty((n_pairs, ref_frames), dtype=np.float32)
        for i in range(n_pairs):
            mri_seg = mri[i : i + ref_frames]
            mel_seg = mel_std[i : i + ref_frames]
            mask_seg = mask[i : i + ref_frames]
            if add_channel_dim:
                mri_pairs[i] = mri_seg[:, None, :, :]
            else:
                mri_pairs[i] = mri_seg
            mel_pairs[i] = mel_seg
            mask_pairs[i] = mask_seg
        np.savez_compressed(
            pairs_dir / f"{stem}.npz",
            mri=mri_pairs,
            mel=mel_pairs,
            mask=mask_pairs,
        )
        total_pairs += int(n_pairs)
    return pairs_dir, total_pairs


def build_file_index(data_dir: Path, patterns: Iterable[str]) -> Dict[str, str]:
    files: Dict[str, str] = {}
    if not data_dir.exists():
        return files
    for ext in patterns:
        for path in data_dir.glob(f"**/*{ext}"):
            stem = path.stem
            files[stem] = str(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="rtMRI -> mel preprocessing pipeline")
    parser.add_argument("--data_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--sr", type=int, default=11413)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=420)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--preemph", type=float, default=0.97)
    parser.add_argument("--resize_h", type=int, default=256)
    parser.add_argument("--resize_w", type=int, default=256)
    parser.add_argument("--ref_frames", type=int, default=4)
    parser.add_argument("--audio_dir", type=Path, default=None, help="Optional separate directory for audio files")
    parser.add_argument("--video_exts", nargs="+", default=[".mp4", ".avi", ".mov"])
    parser.add_argument("--audio_exts", nargs="+", default=[".wav"])
    parser.add_argument("--prefer_wav", action="store_true", default=True)
    parser.add_argument("--no_prefer_wav", dest="prefer_wav", action="store_false")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.out_dir / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old_pairs in args.out_dir.glob("pairs_ref*"):
        shutil.rmtree(old_pairs)
    for stale_file in ("scaler.json", "meta.json"):
        target = args.out_dir / stale_file
        if target.exists():
            target.unlink()

    videos = build_file_index(args.data_dir, args.video_exts)
    audio_root = args.audio_dir if args.audio_dir is not None else args.data_dir
    audios = build_file_index(audio_root, args.audio_exts)
    stems = sorted(set(videos) | set(audios))
    if not stems:
        raise RuntimeError("no video or audio files found in data_dir")

    audio_required = args.audio_dir is not None

    print("Pass1: compute global audio peak")
    global_absmax = 0.0
    for stem in stems:
        y = None
        if args.prefer_wav and stem in audios:
            y, _ = read_audio_from_wav(audios[stem], target_sr=args.sr)
        elif stem in audios:
            y, _ = read_audio_from_wav(audios[stem], target_sr=args.sr)
        elif not audio_required and stem in videos:
            y, _ = read_audio_from_video(videos[stem], target_sr=args.sr)
        else:
            if audio_required:
                print(f"  [WARN] audio file missing for {stem}; skipping in peak computation")
            continue
        if y.size == 0:
            continue
        global_absmax = max(global_absmax, float(np.max(np.abs(y))))
    if global_absmax <= 0:
        global_absmax = 1.0
    print(f"  global_absmax = {global_absmax:.6f}")

    print("Pass2: extract samples (MRI frames + mel)")
    saved_stems: List[str] = []
    resize_hw = (args.resize_h, args.resize_w)
    for stem in stems:
        if stem not in videos:
            continue
        mri = read_video_frames(videos[stem], resize_hw=resize_hw)
        if mri.size == 0:
            continue
        y = None
        if args.prefer_wav and stem in audios:
            y, sr = read_audio_from_wav(audios[stem], target_sr=args.sr)
        elif stem in audios:
            y, sr = read_audio_from_wav(audios[stem], target_sr=args.sr)
        elif not audio_required:
            y, sr = read_audio_from_video(videos[stem], target_sr=args.sr)
        else:
            print(f"  [WARN] audio file missing for {stem}; skipping sample")
            continue
        if y.size == 0:
            continue
        y = y / global_absmax
        mel_db = compute_mel_db(
            y,
            sr=sr,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            preemph=args.preemph,
        )
        T = align_by_hop(mri.shape[0], len(y), args.hop_length)
        if T <= 0:
            continue
        mri = mri[:T]
        mel_db = mel_db[:T]
        save_sample(args.out_dir, stem, mri, mel_db)
        saved_stems.append(stem)
    if not saved_stems:
        raise RuntimeError("no samples were generated; verify input alignment")

    print("Pass3: compute global mel statistics")
    mean, std = pass2_compute_stats(args.out_dir, saved_stems)
    print("  saved scaler.json")

    print(f"Pass4: build fixed {args.ref_frames}-frame pairs")
    pairs_dir, total_pairs = pass3_save_pairs(
        args.out_dir,
        saved_stems,
        ref_frames=args.ref_frames,
        add_channel_dim=True,
    )
    print(f"  saved {total_pairs} pairs to {pairs_dir}")

    meta = {
        "sr": args.sr,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "win_length": args.win_length,
        "hop_length": args.hop_length,
        "preemph": args.preemph,
        "resize_h": args.resize_h,
        "resize_w": args.resize_w,
        "ref_frames": args.ref_frames,
        "stems": saved_stems,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(args.out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
