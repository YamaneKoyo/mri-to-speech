# dataset.py
# Dataset for synchronized rtMRI (video) and Mel-spectrogram pairs produced by preprocess_synced.py
# - Pairs *_video.npy with *_audio.npy (and optional *_meta.json)
# - Returns (T,1,H,W) MRI, (T,n_mels) Mel, (T,) mask
# - Supports fixed sequence_length with last-frame replication padding
# - Provides a collate function for variable-length batches

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

def _gather_items(processed_dir: str):
    """Collect triplets by stem: *_video.npy, *_audio.npy, optional *_meta.json."""
    by_stem = {}
    for f in os.listdir(processed_dir):
        path = os.path.join(processed_dir, f)
        stem, ext = os.path.splitext(f)
        if stem.endswith("_video"):
            s = stem[:-6]
            by_stem.setdefault(s, {})["video"] = path
        elif stem.endswith("_audio"):
            s = stem[:-6]
            by_stem.setdefault(s, {})["audio"] = path
        elif stem.endswith("_meta"):
            s = stem[:-5]
            by_stem.setdefault(s, {})["meta"] = path
    items = []
    for s, v in by_stem.items():
        if "video" in v and "audio" in v:
            items.append(v)
    items.sort(key=lambda d: os.path.basename(d["video"]))
    return items

class MRIMelDataset(Dataset):
    def __init__(self, processed_dir: str, sequence_length: int | None = None, use_mask: bool = True):
        """
        Args:
            processed_dir: directory containing *_video.npy, *_audio.npy, *_meta.json
            sequence_length: if set, trim/pad every sample to this T; else keep variable length
            use_mask: return (T,) mask where valid=1.0, padded=0.0
        """
        self.processed_dir = processed_dir
        self.sequence_length = sequence_length
        self.use_mask = use_mask
        self.items = _gather_items(processed_dir)
        if not self.items:
            raise RuntimeError(f"No matched pairs found in {processed_dir}. "
                               f"Expected files like <stem>_video.npy and <stem>_audio.npy")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        video = np.load(it["video"]).astype(np.float32)   # (T,H,W) z-score per frame
        mel   = np.load(it["audio"]).astype(np.float32)   # (T,n_mels)
        meta  = None
        if "meta" in it and os.path.exists(it["meta"]):
            with open(it["meta"], "r") as f:
                meta = json.load(f)

        # Basic consistency
        T_v, T_m = video.shape[0], mel.shape[0]
        if T_v != T_m:
            T = min(T_v, T_m)
            video = video[:T]
            mel   = mel[:T]

        T = video.shape[0]
        if self.sequence_length is not None:
            target_T = self.sequence_length
            if T >= target_T:
                video = video[:target_T]
                mel   = mel[:target_T]
                mask = np.ones((target_T,), dtype=np.float32)
            else:
                pad = target_T - T
                # last-frame replication padding (more stable than zeros for log-mel)
                video_pad = np.repeat(video[-1:,...], pad, axis=0)
                mel_pad   = np.repeat(mel[-1: ,...], pad, axis=0)
                video = np.concatenate([video, video_pad], axis=0)
                mel   = np.concatenate([mel,   mel_pad],   axis=0)
                mask  = np.concatenate([np.ones((T,), np.float32),
                                        np.zeros((pad,), np.float32)], axis=0)
        else:
            mask = np.ones((T,), dtype=np.float32)

        # Add channel dim for CNNs: (T,1,H,W)
        video = video[:, None, :, :]

        sample = {
            "mri": torch.from_numpy(video),  # (T,1,H,W)
            "mel": torch.from_numpy(mel),    # (T,n_mels)
            "mask": torch.from_numpy(mask) if self.use_mask else None,
            "meta": meta
        }
        return sample

def collate_pad(batch: list[dict]):
    """Collate variable-length samples by padding to max T with last-frame replication.
    Returns:
        {
            "mri":  (B,T,1,H,W),
            "mel":  (B,T,n_mels),
            "mask": (B,T),
            "meta": list[Any]
        }
    """
    # Determine max length
    T_max = max(x["mri"].shape[0] for x in batch)
    B = len(batch)
    # Infer shapes
    _, _, H, W = batch[0]["mri"].shape
    n_mels = batch[0]["mel"].shape[1]

    mri_out = torch.empty((B, T_max, 1, H, W), dtype=batch[0]["mri"].dtype)
    mel_out = torch.empty((B, T_max, n_mels),  dtype=batch[0]["mel"].dtype)
    mask_out = torch.zeros((B, T_max), dtype=torch.float32)
    metas = []

    for i, x in enumerate(batch):
        T = x["mri"].shape[0]
        mri_out[i, :T] = x["mri"]
        mel_out[i, :T] = x["mel"]
        if x.get("mask") is not None:
            mask_out[i, :T] = x["mask"]
        else:
            mask_out[i, :T] = 1.0
        # replicate last frame if padding is needed
        if T < T_max:
            mri_out[i, T:T_max] = x["mri"][-1:].repeat(T_max - T, 1, 1, 1)
            mel_out[i, T:T_max] = x["mel"][-1:].repeat(T_max - T, 1)
        metas.append(x.get("meta"))

    return {"mri": mri_out, "mel": mel_out, "mask": mask_out, "meta": metas}
