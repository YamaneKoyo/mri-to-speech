# train_mri_acoustic_model.py
# -------------------------------------------------------------
# OTN貅匁侠繝｢繝・Ν逕ｨ繝医Ξ繝ｼ繝奇ｼ亥崋螳夐聞 ref_frames / 蜍ｾ驟垢KPT / 繝槭う繧ｯ繝ｭ繝舌ャ繝・ｼ・
# 萓晏ｭ・ mri_acoustic_model.py, dataset.py (MRIMelDataset, collate_pad)
# 菴ｿ縺・婿・井ｾ具ｼ・
#   python3 train_mri_acoustic_model.py \
#     --out_ckpt "/content/drive/MyDrive/MRI_to_Speech/best_mri_acoustic_model.pth" \
#     --ref_frames 4 --epochs 80 --batch_size 8 --micro_batch_size 2 \
# -------------------------------------------------------------
import os, gc, json, argparse, traceback
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset_fixedlen import FixedLenPairDataset, collate_pad
from mri_acoustic_model import build_acoustic_model

from packaging import version

# ==== perf/env knobs (threads, TF32) ====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
except Exception:
    pass
print(f"[INIT] CUDA available={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
# ========================================

# ----------------- 菴弱Γ繝｢繝ｪ險ｭ螳・-----------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Optional: improve matmul perf on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def _using_cuda():
    return torch.cuda.is_available()

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

# ----------------- 繝槭せ繧ｯ莉倥″謳榊､ｱ -----------------
class MaskedMSEMAE(nn.Module):
    """
    Frequency/temporal weighted MSE with delta regularisation.
    - 蠑ｷ隱ｿ: 菴主沺(0-7)縺ｨ鬮伜沺(48-63)
    - 譎る俣: 蜈磯ｭ繝輔Ξ繝ｼ繝驥阪∩莉倥￠縲・℃貂｡(ﾎ・ﾎ・)縺ｮ貊代ｉ縺九＆繧呈鋸譚・
    """

    def __init__(self, num_mels: int = 64, max_frames: int = 128, ramp_steps: int = 120000):
        super().__init__()
        freq_base = torch.ones(num_mels, dtype=torch.float32)
        freq_target = freq_base.clone()
        f0_range = (0, min(6, num_mels))
        f1_range = (6, min(16, num_mels))
        f2_range = (16, min(32, num_mels))
        upper_mid_range = (32, min(48, num_mels))
        high_range = (max(num_mels - 16, 0), num_mels)

        def _apply_weight(target, rng, weight):
            start, end = rng
            if end > start:
                target[start:end] = weight

        _apply_weight(freq_target, f0_range, 2.0)
        _apply_weight(freq_target, f1_range, 3.0)
        _apply_weight(freq_target, f2_range, 2.4)
        _apply_weight(freq_target, upper_mid_range, 1.6)
        _apply_weight(freq_target, high_range, 1.8)
        self.register_buffer("freq_base", freq_base.view(1, 1, num_mels))
        self.register_buffer("freq_target", freq_target.view(1, 1, num_mels))

        time_base = torch.ones(max_frames, dtype=torch.float32)
        time_target = time_base.clone()
        focus_weights = [1.6, 1.45, 1.3, 1.2, 1.15, 1.1, 1.05, 1.02]
        for idx, val in enumerate(focus_weights):
            if idx < max_frames:
                time_target[idx] = val
        self.register_buffer("time_base", time_base)
        self.register_buffer("time_target", time_target)

        self.ramp_steps = float(ramp_steps)
        self.current_step = 0
        self.band_ranges = {
            "f0": f0_range,
            "f1": f1_range,
            "f2": f2_range,
            "high": high_range,
        }

    def _apply_mask(self, tensor, mask):
        if mask is None:
            return tensor, tensor.numel()
        masked = tensor * mask
        denom = mask.sum().clamp_min(1.0)
        return masked, denom

    def forward(self, pred, target, mask=None):
        """
        pred/target: (B, T, M), mask: (B, T)
        """
        B, T, M = pred.shape
        ramp = min(1.0, float(self.current_step) / self.ramp_steps) if self.ramp_steps > 0 else 1.0

        freq_w = ((1 - ramp) * self.freq_base + ramp * self.freq_target)[..., :M].to(pred.device)
        time_vec = ((1 - ramp) * self.time_base + ramp * self.time_target)[:T].to(pred.device)
        time_w = time_vec.view(1, T, 1)
        weights = (freq_w * time_w).expand(B, T, M)  # (B,T,M)

        diff = pred - target
        if mask is not None:
            mask = mask.unsqueeze(-1)
            weights = weights * mask
        denom_base = weights.sum().clamp_min(1e-6)
        mse = torch.sum((diff ** 2) * weights) / denom_base
        mae = torch.sum(torch.abs(diff) * weights) / denom_base

        # 1st-order temporal smoothness
        if T > 1:
            delta = diff[:, 1:, :] - diff[:, :-1, :]
            delta_w = freq_w[:, :T - 1, :] * time_w[:, 1:, :]
            delta_w = delta_w.expand(B, T - 1, M)
            if mask is not None:
                delta_w = delta_w * mask[:, 1:, :] * mask[:, :-1, :]
            denom_delta = delta_w.sum().clamp_min(1e-6)
            delta_loss = torch.sum((delta ** 2) * delta_w) / denom_delta
        else:
            delta_loss = diff.new_tensor(0.0)

        # 2nd-order (acceleration) smoothness
        if T > 2:
            accel = diff[:, 2:, :] - 2 * diff[:, 1:-1, :] + diff[:, :-2, :]
            accel_w = freq_w[:, :T - 2, :] * time_w[:, 1:T - 1, :]
            accel_w = accel_w.expand(B, T - 2, M)
            if mask is not None:
                accel_w = accel_w * mask[:, 2:, :] * mask[:, 1:-1, :] * mask[:, :-2, :]
            denom_accel = accel_w.sum().clamp_min(1e-6)
            accel_loss = torch.sum((accel ** 2) * accel_w) / denom_accel
        else:
            accel_loss = diff.new_tensor(0.0)

        # 譛譁ｰ繝輔Ξ繝ｼ繝縺ｮ陬懷勧謳榊､ｱ・医Γ繝ｫ 1 繝輔Ξ繝ｼ繝 MSE・・
        latest_diff = diff[:, -1, :]
        latest_w = freq_w[:, -1:, :].expand(B, 1, M)
        latest_denom = latest_w.sum().clamp_min(1e-6)
        latest_loss = torch.sum((latest_diff ** 2) * latest_w[:, 0, :]) / latest_denom

        delta_coeff = 0.3 + 0.15 * ramp  # 0.30 -> 0.45
        accel_coeff = 0.1 + 0.05 * ramp
        latest_coeff = 0.2 + 0.2 * ramp  # 0.20 -> 0.40

        loss = mse + delta_coeff * delta_loss + accel_coeff * accel_loss + latest_coeff * latest_loss
        return loss, mse.detach(), mae.detach()

    def set_step(self, step: int):
        self.current_step = step

# ----------------- DataLoader -----------------
def make_loaders(processed_dir, ref_frames, batch_size=8, val_bs=8, num_workers=4, prefetch_factor=4, seed=42, device=None):
    ds = FixedLenPairDataset(processed_dir, ref_frames=ref_frames)
    if device is None:
        device = "cuda" if _using_cuda() else "cpu"
    pin_mem = bool(device == "cuda")
    n = len(ds)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(ds, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(
        tr, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_pad, drop_last=True
    )
    val_loader = DataLoader(
        va, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_pad
    )
    test_loader = DataLoader(
        te, batch_size=val_bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        collate_fn=collate_pad
    )
    return train_loader, val_loader, test_loader

# ----------------- Trainer -----------------
class OTNLikeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device="cuda",
        lr=1e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        micro_batch_size=2,
        scaler_device="cuda",
        max_train_steps=None,
        max_val_steps=None,
        log_dir=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.micro_batch_size = micro_batch_size
        self.max_train_steps = max_train_steps if max_train_steps and max_train_steps > 0 else None
        self.max_val_steps = max_val_steps if max_val_steps and max_val_steps > 0 else None
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        # Choose autocast dtype: prefer bf16 on Ampere+ (A100), else fp16
        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.autocast_dtype = torch.bfloat16 if bf16_ok else torch.float16
        self.scaler = torch.amp.GradScaler(
            device=scaler_device,
            enabled=not bf16_ok,              # bf16 縺ｪ繧峨せ繧ｱ繝ｼ繝ｩ荳崎ｦ・
            init_scale=2**10, growth_factor=1.5, backoff_factor=0.5, growth_interval=100
        )
        self.crit = MaskedMSEMAE()
        self.best_val = float('inf')
        self.patience = 0
        self.early_stop_patience = 20
        self.hist = {"train": [], "val": []}
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.writer_last_logged_epoch = 0
        self.resume_global_step = 0
        self.start_epoch = 1
        self.global_step = 0

    def _micro_batches(self, batch):
        """ (B, ...) 繧・micro_batch_size 縺ｧ蛻・牡縺励※ generator 縺ｧ霑斐☆ """
        B = batch["mri"].size(0)
        mbs = self.micro_batch_size
        for s in range(0, B, mbs):
            e = min(B, s + mbs)
            yield {k: (v[s:e] if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _compute_band_mae(self, pred, target):
        metrics = {}
        if not hasattr(self.crit, "band_ranges"):
            return metrics
        with torch.no_grad():
            pred_f = pred.detach().float()
            target_f = target.detach().float()
            for name, (start, end) in self.crit.band_ranges.items():
                end = min(end, pred_f.shape[-1])
                if end <= start:
                    continue
                sl = slice(start, end)
                mae = torch.mean(torch.abs(pred_f[..., sl] - target_f[..., sl]))
                metrics[name] = float(mae.item())
        return metrics

    def train_epoch(self, epoch_idx=1, use_checkpoint=False):
        self.model.train()
        limit = self.max_train_steps or len(self.train_loader)
        est_total = min(limit, len(self.train_loader)) if len(self.train_loader) else limit
        tot = mse_tot = mae_tot = 0.0
        band_tot = {k: 0.0 for k in getattr(self.crit, "band_ranges", {})}
        band_count = 0
        pbar = tqdm(self.train_loader, desc=f"Train[{epoch_idx}]", total=est_total if est_total else None)
        self.opt.zero_grad(set_to_none=True)

        autocast_device = "cuda" if _using_cuda() else "cpu"
        step_count = 0

        for batch in pbar:
            step_count += 1
            try:
                self.opt.zero_grad(set_to_none=True)
                step_loss = step_mse = step_mae = 0.0
                self.crit.set_step(self.global_step)

                for mb in self._micro_batches(batch):
                    mri = mb["mri"].to(self.device, non_blocking=True).contiguous()
                    assert mri.dim() == 5, f"Expected (B,T,1,H,W), got {tuple(mri.shape)}"
                    mel = mb["mel"].to(self.device, non_blocking=True)
                    mask = mb["mask"].to(self.device, non_blocking=True)

                    with torch.amp.autocast(device_type=autocast_device, dtype=self.autocast_dtype, enabled=True):
                        pred = self.model(mri)
                        loss, mse, mae = self.crit(pred, mel, mask)
                        div = max(1, (batch["mri"].size(0) + self.micro_batch_size - 1) // self.micro_batch_size)
                        loss = loss / div

                    band_metrics = self._compute_band_mae(pred, mel)
                    self.scaler.scale(loss).backward()
                    step_loss += loss.detach().float().item()
                    step_mse += mse.detach().float().item()
                    step_mae += mae.detach().float().item()
                    for name, val in band_metrics.items():
                        band_tot[name] = band_tot.get(name, 0.0) + val
                    band_count += 1

                self.scaler.unscale_(self.opt)
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

                tot += step_loss
                mse_tot += step_mse
                mae_tot += step_mae
                pbar.set_postfix(loss=f"{step_loss:.2f}", mse=f"{step_mse:.2f}")
                self.global_step += 1

                if self.max_train_steps and step_count >= self.max_train_steps:
                    break

            except RuntimeError as e:
                print(f"[Train] RuntimeError: {e}")
                traceback.print_exc()
                clean_memory()
                continue

            if self.max_train_steps and step_count >= self.max_train_steps:
                break

        n = max(1, step_count)
        band_avg = {k: (v / max(1, band_count)) for k, v in band_tot.items()}
        return tot/n, mse_tot/n, mae_tot/n, band_avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        autocast_device = "cuda" if _using_cuda() else "cpu"
        tot = mse_tot = mae_tot = 0.0
        cnt = 0
        band_tot = {k: 0.0 for k in getattr(self.crit, "band_ranges", {})}
        band_count = 0
        limit = self.max_val_steps or len(self.val_loader)
        est_total = min(limit, len(self.val_loader)) if len(self.val_loader) else limit
        iterator = tqdm(self.val_loader, desc="Val", total=est_total if est_total else None)
        for step, batch in enumerate(iterator, start=1):
            try:
                mri = batch["mri"].to(self.device, non_blocking=True).contiguous()
                assert mri.dim() == 5, f"Expected (B,T,1,H,W), got {tuple(mri.shape)}"
                mel = batch["mel"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                self.crit.set_step(self.global_step)
                with torch.amp.autocast(device_type=autocast_device, dtype=self.autocast_dtype, enabled=True):
                    pred = self.model(mri)
                    loss, mse, mae = self.crit(pred, mel, mask)
                tot += loss.item(); mse_tot += mse.item(); mae_tot += mae.item()
                for name, val in self._compute_band_mae(pred, mel).items():
                    band_tot[name] = band_tot.get(name, 0.0) + val
                band_count += 1
                cnt += 1
            except RuntimeError as e:
                print(f"[Val] RuntimeError: {e}")
                traceback.print_exc()
                continue
            if self.max_val_steps and step >= self.max_val_steps:
                break
        if cnt == 0:
            return float('inf'), float('inf'), float('inf'), {}
        band_avg = {k: (v / max(1, band_count)) for k, v in band_tot.items()}
        return tot/cnt, mse_tot/cnt, mae_tot/cnt, band_avg

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def resume_from_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} not found.")
        print(f"[RESUME] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt.get("model_state_dict")
        if model_state is None:
            raise KeyError("model_state_dict missing in checkpoint")
        self.model.load_state_dict(model_state)
        self.model.to(self.device)

        optim_state = ckpt.get("optimizer_state_dict")
        if optim_state is not None:
            self.opt.load_state_dict(optim_state)
            self._move_optimizer_state_to_device(self.opt, self.device)
        else:
            print("[RESUME] Optimizer state missing; optimizer reset.")

        sched_state = ckpt.get("scheduler_state_dict")
        if sched_state is not None:
            self.sched.load_state_dict(sched_state)
        else:
            print("[RESUME] Scheduler state missing; scheduler reset.")

        train_loss = ckpt.get("train_loss")
        if train_loss is not None and not np.isnan(train_loss):
            self.hist["train"].append(float(train_loss))

        val_loss = ckpt.get("val_loss")
        if val_loss is not None and not np.isnan(val_loss):
            self.best_val = float(val_loss)
            self.hist["val"].append(float(val_loss))

        if self.writer:
            logged_epochs = min(len(self.hist["train"]), len(self.hist["val"]))
            if logged_epochs > 0 and self.writer_last_logged_epoch < logged_epochs:
                for idx in range(self.writer_last_logged_epoch, logged_epochs):
                    ep_idx = idx + 1
                    self.writer.add_scalar("loss/train", self.hist["train"][idx], ep_idx)
                    self.writer.add_scalar("loss/val", self.hist["val"][idx], ep_idx)
                self.writer_last_logged_epoch = logged_epochs
                self.writer.flush()

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1

        steps = []
        if optim_state is not None:
            for state in optim_state.get("state", {}).values():
                if isinstance(state, dict):
                    step = state.get("step")
                    if isinstance(step, torch.Tensor):
                        step = step.item()
                    if step is not None:
                        steps.append(int(step))
        self.resume_global_step = max(steps) if steps else 0
        self.global_step = self.resume_global_step

        scaler_state = ckpt.get("scaler_state_dict")
        if scaler_state is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"[RESUME] Failed to load GradScaler state: {exc}")
        else:
            print("[RESUME] GradScaler state not found; using fresh scaler state.")

        self.patience = 0
        msg_val = f"{self.best_val:.6f}" if np.isfinite(self.best_val) else str(self.best_val)
        print(f"[RESUME] Resuming from epoch {self.start_epoch} (global step {self.resume_global_step}) with best_val={msg_val}")
        return ckpt

    def fit(self, epochs=80, save_path="best_mri_acoustic_model.pth"):
        print(f"[TRAIN] Start training: start_epoch={self.start_epoch}, target_epochs={epochs}, micro_batch={self.micro_batch_size}")
        if self.start_epoch > epochs:
            print(f"[TRAIN] start_epoch {self.start_epoch} exceeds target_epochs {epochs}; nothing to train.")
            return self.model
        for ep in range(self.start_epoch, epochs+1):
            tr_loss, tr_mse, tr_mae, tr_band = self.train_epoch(epoch_idx=ep)
            va_loss, va_mse, va_mae, va_band = self.validate()

            self.hist["train"].append(tr_loss)
            self.hist["val"].append(va_loss)

            if self.writer:
                self.writer.add_scalar("loss/train", tr_loss, ep)
                self.writer.add_scalar("loss/val", va_loss, ep)
                self.writer.add_scalar("metrics/train_mse", tr_mse, ep)
                self.writer.add_scalar("metrics/val_mse", va_mse, ep)
                self.writer.add_scalar("metrics/train_mae", tr_mae, ep)
                self.writer.add_scalar("metrics/val_mae", va_mae, ep)
                for name, value in tr_band.items():
                    self.writer.add_scalar(f"band/train_{name}", value, ep)
                for name, value in va_band.items():
                    self.writer.add_scalar(f"band/val_{name}", value, ep)
                self.writer.add_scalar("lr", self.opt.param_groups[0]['lr'], ep)
                self.writer_last_logged_epoch = ep
                self.writer.flush()

            print(f"\nEpoch {ep}/{epochs}")
            print(f"Train: loss={tr_loss:.6f} mse={tr_mse:.6f} mae={tr_mae:.6f}")
            if tr_band:
                parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(tr_band.items()))
                print(f"    band train: {parts}")
            print(f"Val  : loss={va_loss:.6f} mse={va_mse:.6f} mae={va_mae:.6f}")
            if va_band:
                parts = ", ".join(f"{k}={v:.4f}" for k, v in sorted(va_band.items()))
                print(f"    band val  : {parts}")
            print(f"LR: {self.opt.param_groups[0]['lr']:.2e}")

            old_lr = self.opt.param_groups[0]['lr']
            self.sched.step(va_loss)
            new_lr = self.opt.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"[SCHEDULER] LR reduced: {old_lr:.6e} -> {new_lr:.6e}")

            if va_loss < self.best_val and not np.isnan(va_loss):
                self.best_val = va_loss
                self.patience = 0
                torch.save({
                    "epoch": ep,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.opt.state_dict(),
                    "scheduler_state_dict": self.sched.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict(),
                    "val_loss": va_loss,
                    "val_mse": va_mse,
                    "train_loss": tr_loss
                }, save_path)
                print("[BEST] New best model saved.")
            else:
                self.patience += 1

            if self.patience >= 20:
                print("[STOP] Early stopping."); break
            sched_min = getattr(self.sched, 'min_lr', None)
            if sched_min is None:
                min_list = getattr(self.sched, 'min_lrs', [0.0])
                sched_min = min_list[0] if min_list else 0.0
            if self.opt.param_groups[0]['lr'] <= sched_min + 1e-12:
                print("[STOP] LR reached min."); break

        # 繝吶せ繝医・繝ｭ繝ｼ繝会ｼ井ｻｻ諢擾ｼ・
        try:
            ckpt = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[CHECKPOINT] Loaded best model (val_loss={ckpt['val_loss']:.6f})")
        except Exception as e:
            print(f"[WARN] could not load best: {e}")
        return self.model

# ----------------- 繝ｩ繝ｳ繝√Ε -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True, help="Processed dataset root (contains pairs_ref{ref_frames} directories).")
    ap.add_argument("--out_ckpt", type=str, default="best_mri_acoustic_model.pth")
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume training from.")
    ap.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory.")
    ap.add_argument("--ref_frames", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_batch_size", type=int, default=8)
    ap.add_argument("--micro_batch_size", type=int, default=2, help="繝溘ル繝舌ャ繝√ｒ縺薙・繧ｵ繧､繧ｺ縺ｧ蛻・牡縺励※騾先ｬ｡ backward")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps per epoch (for profiling).")
    ap.add_argument("--max_val_steps", type=int, default=None, help="Maximum validation steps per epoch (for profiling).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # 繝｢繝・Ν髢｢騾｣
    ap.add_argument("--cnn_pretrained", action="store_true", help="Enable EfficientNetV2-B2 pretrained weights for the CNN backbone.")
    ap.add_argument("--use_checkpoint", action="store_true", help="Enable gradient checkpointing inside the CNN encoder.")
    ap.add_argument("--ckpt_segments", type=int, default=2, help="Number of segments for checkpoint_sequential.")
    ap.add_argument("--use_reentrant", action="store_true", help="Set torch.utils.checkpoint to use reentrant checkpoints.")

    args = ap.parse_args()

    if not os.path.isdir(args.processed_dir):
        raise FileNotFoundError(f"{args.processed_dir} not found. Run preprocess first.")

    clean_memory()

    # Data
    train_loader, val_loader, _ = make_loaders(
        args.processed_dir, args.ref_frames,
        batch_size=args.batch_size, val_bs=args.val_batch_size, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=args.device
    )
    print(f"[DATALOADER] workers={args.num_workers} prefetch={args.prefetch_factor} pin_memory={(args.device=='cuda')} persistent={(args.num_workers>0)}")

    # Model
    model = build_acoustic_model(
        n_mels=64,
        cnn_pretrained=args.cnn_pretrained,
        rnn_hidden=640,
        dropout=0.5,
        use_checkpoint=args.use_checkpoint,
        ckpt_segments=args.ckpt_segments,
        use_reentrant=args.use_reentrant
    ).to(args.device)

    log_dir = os.path.abspath(args.log_dir) if args.log_dir else None

    # Trainer
    trainer = OTNLikeTrainer(
        model, train_loader, val_loader, device=args.device,
        lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, micro_batch_size=args.micro_batch_size,
        scaler_device=args.device,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
        log_dir=log_dir
    )

    resume_path = None
    if args.resume_ckpt:
        resume_path = os.path.abspath(args.resume_ckpt)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint {resume_path} not found.")
        trainer.resume_from_checkpoint(resume_path)
        if args.out_ckpt == "best_mri_acoustic_model.pth":
            args.out_ckpt = resume_path

    # Fit
    best = trainer.fit(epochs=args.epochs, save_path=args.out_ckpt)
    if trainer.writer:
        trainer.writer.close()
    print("[DONE] Done. Saved:", args.out_ckpt)

if __name__ == "__main__":
    main()





