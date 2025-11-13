"""dataset_fixedlen.py - fixed-length pair dataset loader with mmap npy support."""
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _natural_key(s: str) -> List[object]:
    tokens: List[object] = []
    start = 0
    for idx, ch in enumerate(s):
        if ch.isdigit():
            if start < idx:
                tokens.append(s[start:idx])
            j = idx
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(int(s[idx:j]))
            start = j
    if start < len(s):
        tokens.append(s[start:])
    return tokens


def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    mri = torch.stack([b["mri"] for b in batch], dim=0)
    mel = torch.stack([b["mel"] for b in batch], dim=0)
    mask = torch.stack([b["mask"] for b in batch], dim=0)
    return {"mri": mri, "mel": mel, "mask": mask}


class FixedLenPairDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        ref_frames: int,
        strict_T: bool = True,
        allow_broken_skip: bool = True,
        debug_print: bool = True,
        cache_index: bool = True,
        force_reindex: bool = False,
        validate_first_n: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.processed_dir = processed_dir
        self.ref_frames = int(ref_frames)
        self.strict_T = bool(strict_T)
        self.allow_broken_skip = bool(allow_broken_skip)
        self.debug_print = bool(debug_print)
        self.cache_index = bool(cache_index)
        self.force_reindex = bool(force_reindex)
        self.validate_first_n = int(validate_first_n)
        self.rng = random.Random(seed)

        base = Path(processed_dir)
        self.pairs_dir = base / f"pairs_ref{self.ref_frames}"
        self.npy_dir = base / f"pairs_ref{self.ref_frames}_npy"
        if not self.pairs_dir.is_dir() and not self.npy_dir.is_dir():
            raise FileNotFoundError(f"{self.pairs_dir} not found")

        self._mem_cache: Dict[int, Dict[str, np.memmap]] = {}
        if self.npy_dir.is_dir():
            self.mode = "npy"
            self._init_from_npy()
        else:
            self.mode = "npz"
            self._init_from_npz()

        if self.total_pairs == 0:
            raise RuntimeError(f"No pairs available in {self.processed_dir}")

    # ------------------------------------------------------------------
    def _init_from_npz(self) -> None:
        pairs_dir = self.pairs_dir
        cache_txt = pairs_dir / f"index_ref{self.ref_frames}.txt"
        cache_json = pairs_dir / f"index_ref{self.ref_frames}.json"

        files: List[str] = []
        used_cache = False
        if self.cache_index and not self.force_reindex:
            if cache_txt.is_file():
                try:
                    files = [ln.strip() for ln in cache_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    used_cache = True
                except Exception:
                    files = []
            elif cache_json.is_file():
                try:
                    files = json.loads(cache_json.read_text(encoding="utf-8"))
                    used_cache = True
                except Exception:
                    files = []

        if not files:
            candidates: List[str] = []
            candidates.extend(str(p) for p in pairs_dir.glob("*.npz"))
            candidates.extend(str(p) for p in pairs_dir.glob("*/*.npz"))
            if not candidates:
                candidates.extend(str(p) for p in pairs_dir.rglob("*.npz"))
            files = sorted(set(candidates), key=_natural_key)
            if self.cache_index and files:
                try:
                    cache_txt.write_text("\n".join(files), encoding="utf-8")
                except Exception:
                    pass
                try:
                    cache_json.write_text(json.dumps(files), encoding="utf-8")
                except Exception:
                    pass

        if not files:
            raise RuntimeError(f"No .npz pairs found in {pairs_dir}")

        self.files: List[str] = []
        self.counts: List[int] = []
        self.cumsum: List[int] = [0]
        total = 0
        for fp in files:
            try:
                with np.load(fp, mmap_mode="r", allow_pickle=False) as data:
                    mri = data["mri"]
                    if mri.ndim != 5:
                        raise ValueError(f"Unexpected ndim {mri.ndim}")
                    n_pairs, T = int(mri.shape[0]), int(mri.shape[1])
                    if self.strict_T and T != self.ref_frames:
                        raise ValueError(f"T mismatch {T}")
            except Exception as exc:
                if self.allow_broken_skip:
                    if self.debug_print:
                        print(f"[DATASET] skip (index): {fp} reason={exc}")
                    continue
                raise
            if n_pairs <= 0:
                continue
            self.files.append(fp)
            self.counts.append(n_pairs)
            total += n_pairs
            self.cumsum.append(total)

        self.total_pairs = total
        self.mode_records = None
        if self.debug_print and self.files:
            src = "cache" if used_cache else "scan"
            print(f"[DATASET] pairs_dir={pairs_dir} (from {src})")
            print(f"[DATASET] files={len(self.files)}, total_pairs={self.total_pairs}")
            print(f"[DATASET] sample file: {self.files[0]} (pairs={self.counts[0]})")

    # ------------------------------------------------------------------
    def _init_from_npy(self) -> None:
        dirs = sorted([p for p in self.npy_dir.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name))
        if not dirs:
            raise RuntimeError(f"No directories in {self.npy_dir}")

        records = []
        counts: List[int] = []
        files: List[str] = []
        total = 0
        for folder in dirs:
            mri_path = folder / "mri.npy"
            mel_path = folder / "mel.npy"
            mask_path = folder / "mask.npy"
            if not (mri_path.is_file() and mel_path.is_file() and mask_path.is_file()):
                if self.debug_print:
                    print(f"[DATASET] skip (missing npy): {folder}")
                continue
            try:
                mri = np.load(mri_path, mmap_mode="r", allow_pickle=False)
                if mri.ndim != 5:
                    raise ValueError("unexpected ndim")
                n_pairs, T = int(mri.shape[0]), int(mri.shape[1])
                del mri
            except Exception as exc:
                if self.allow_broken_skip:
                    if self.debug_print:
                        print(f"[DATASET] skip (load error): {folder} reason={exc}")
                    continue
                raise
            if n_pairs == 0:
                continue
            if self.strict_T and T != self.ref_frames:
                if self.allow_broken_skip:
                    if self.debug_print:
                        print(f"[DATASET] skip (T mismatch): {folder}")
                    continue
                raise RuntimeError(f"T mismatch in {folder}")
            records.append({
                "mri": str(mri_path),
                "mel": str(mel_path),
                "mask": str(mask_path),
            })
            counts.append(n_pairs)
            files.append(str(folder))
            total += n_pairs
        self.mode_records = records
        self.counts = counts
        self.files = files
        self.cumsum = [0]
        for cnt in counts:
            self.cumsum.append(self.cumsum[-1] + cnt)
        self.total_pairs = self.cumsum[-1]
        if self.debug_print and self.files:
            print(f"[DATASET] pairs_dir={self.npy_dir} (npy mmap mode)")
            print(f"[DATASET] files={len(self.files)}, total_pairs={self.total_pairs}")
            print(f"[DATASET] sample file: {self.files[0]} (pairs={self.counts[0]})")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.total_pairs

    def _map_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += self.total_pairs
        if not (0 <= idx < self.total_pairs):
            raise IndexError(idx)
        import bisect
        file_idx = bisect.bisect_right(self.cumsum, idx) - 1
        local = idx - self.cumsum[file_idx]
        return file_idx, local

    def _get_memmap(self, file_idx: int, key: str) -> np.memmap:
        cache = self._mem_cache.setdefault(file_idx, {})
        arr = cache.get(key)
        if arr is None:
            record = self.mode_records[file_idx]
            arr = np.load(record[key], mmap_mode="r", allow_pickle=False)
            cache[key] = arr
        return arr

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_idx, local_idx = self._map_index(index)
        if self.mode == "npy" and self.mode_records is not None:
            mri_np = np.asarray(self._get_memmap(file_idx, "mri")[local_idx])
            mel_np = np.asarray(self._get_memmap(file_idx, "mel")[local_idx])
            mask_np = np.asarray(self._get_memmap(file_idx, "mask")[local_idx])
        else:
            fp = self.files[file_idx]
            try:
                with np.load(fp, mmap_mode="r", allow_pickle=False) as data:
                    mri_np = data["mri"][local_idx]
                    mel_np = data["mel"][local_idx]
                    mask_np = data["mask"][local_idx]
            except Exception:
                if self.allow_broken_skip:
                    return self.__getitem__((index + 1) % self.total_pairs)
                raise

        if self.strict_T:
            T = self.ref_frames
            if not (mri_np.shape[0] == mel_np.shape[0] == mask_np.shape[0] == T):
                if self.allow_broken_skip:
                    return self.__getitem__((index + 1) % self.total_pairs)
                raise RuntimeError("Time dimension mismatch")
        H, W = int(mri_np.shape[-2]), int(mri_np.shape[-1])
        if (H, W) != (256, 256):
            if self.allow_broken_skip:
                return self.__getitem__((index + 1) % self.total_pairs)
            raise RuntimeError("Unexpected spatial size")

        mri_t = torch.from_numpy(np.array(mri_np, copy=True))
        mel_t = torch.from_numpy(np.array(mel_np, copy=True))
        mask_t = torch.from_numpy(np.array(mask_np, copy=True))
        return {"mri": mri_t, "mel": mel_t, "mask": mask_t}
