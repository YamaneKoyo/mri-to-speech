# eval_mel.py
# ---------------------------------------------------------
# Colab でそのまま動く / あなたの model.py・dataset.py を利用
# 評価: masked MSE / MAE（必須） + オプション MCD-like（statsがある場合）
# ---------------------------------------------------------
import os, json, math, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# 既存コードを利用
from model import RevisedCNNBiLSTM
from dataset import MRIMelDataset, collate_pad

# ====== マスク付き損失（学習と揃える） ======
class MaskedMSEMAE(nn.Module):
    def __init__(self, w_mse=0.8, w_mae=0.2):
        super().__init__()
        self.w_mse = w_mse
        self.w_mae = w_mae
    def forward(self, pred, target, mask):
        # pred/target: (B,T,n_mels), mask: (B,T)
        mask = mask.unsqueeze(-1)  # (B,T,1)
        diff = (pred - target) * mask
        denom = mask.sum().clamp_min(1.0)
        mse = (diff ** 2).sum() / denom
        mae = diff.abs().sum() / denom
        total = self.w_mse * mse + self.w_mae * mae
        return total, mse, mae

# ====== MCD-like (オプション) ======
# - 前処理で使った mean/std が手元にある場合のみ有効化
# - 手順: 標準化解除 -> dB -> power -> MFCC -> MCD 近似
import librosa

def _inv_standardize(mel_stdzed, mean, std):
    return mel_stdzed * std + mean

def _db_to_power(db):
    # librosa.power_to_db の逆（ref=1.0想定）
    return librosa.db_to_power(db)

def _meldb_to_mfcc(mel_db, sr=11866, n_mfcc=13, n_mels=64, fmin=0.0, fmax=None):
    """
    mel_db: (T, n_mels) [dB]
    1) dB->power
    2) mel-power を librosa.feature.mfcc の入力に合わせるため、一旦 linear スペクトルに戻すのが本来だが、
       ここでは 'mel-power' をそのまま log-energy 系特徴の近似として MFCC 計算に流用（実用上の近似）。
    """
    # 近似: mel-power から直接 MFCC を計算（実務上の proxy）
    mel_power = _db_to_power(mel_db.T)  # (n_mels, T)
    # librosa の mfcc は通常 linear スペクトログラムから計算されるが、
    # 実務的には log-mel から DCT しても相関の高い指標になるため、ここでは擬似 MFCC として扱う。
    # mel_power -> log -> DCT (librosa.feature.mfcc は内部で行う)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_power), sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (T, n_mfcc)

def mcd_like(mel_pred, mel_gt, mean=None, std=None, sr=11866, n_mfcc=13):
    """
    mel_pred/mel_gt: (T, n_mels) 標準化後の dB スケール前提（あなたの前処理に準拠）
    mean/std が渡された場合のみ、標準化解除して dB スケールに戻す（≒論文仕様の比較に近づく）
    """
    if mean is None or std is None:
        return None  # スキップ

    mel_pred_db = _inv_standardize(mel_pred, mean, std)
    mel_gt_db   = _inv_standardize(mel_gt,   mean, std)

    mfcc_pred = _meldb_to_mfcc(mel_pred_db, sr=sr, n_mfcc=n_mfcc)
    mfcc_gt   = _meldb_to_mfcc(mel_gt_db,   sr=sr, n_mfcc=n_mfcc)

    # 長さ合わせ（min 長で切る）
    T = min(mfcc_pred.shape[0], mfcc_gt.shape[0])
    D = mfcc_pred[:T] - mfcc_gt[:T]
    # MCD 定義に近い係数： (10 / ln10) * sqrt(2) * RMSE
    # ただしここは「擬似MFCC」なので "MCD-like" とする
    const = (10.0 / math.log(10.0)) * math.sqrt(2.0)
    rmse = np.sqrt((D**2).sum(axis=1)).mean()
    return const * rmse

# ====== データローダ（Val/Test 作成） ======
def make_loader(processed_dir, split="val", batch_size=4, seq_len=None, num_workers=2):
    ds = MRIMelDataset(processed_dir, sequence_length=seq_len, use_mask=True)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    n_test  = n - n_train - n_val
    g = torch.Generator().manual_seed(42)
    tr, va, te = random_split(ds, [n_train, n_val, n_test], generator=g)
    if split == "val":
        subset = va
    elif split == "test":
        subset = te
    else:
        raise ValueError("--split は val か test を指定してください。")
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_pad)

# ====== 評価ループ ======
@torch.no_grad()
def evaluate(model, loader, device, stats=None, sr=11866):
    crit = MaskedMSEMAE()
    tot_loss = tot_mse = tot_mae = 0.0
    n_batches = 0

    mcd_vals = []  # MCD-like の収集（stats があれば）

    model.eval()
    for batch in tqdm(loader, desc="Eval"):
        mri  = batch["mri"].to(device)     # (B,T,1,H,W)
        mel  = batch["mel"].to(device)     # (B,T,n_mels)
        mask = batch["mask"].to(device)    # (B,T)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            pred = model(mri)              # (B,T,n_mels)
            loss, mse, mae = crit(pred, mel, mask)

        tot_loss += loss.item()
        tot_mse  += mse.item()
        tot_mae  += mae.item()
        n_batches += 1

        # ----- MCD-like（任意）-----
        if stats is not None:
            mean = np.array(stats.get("mean")).reshape(1, -1)   # (1, n_mels)
            std  = np.array(stats.get("std")).reshape(1, -1)    # (1, n_mels)
            # CPU へ移して numpy に
            pred_np = pred.detach().float().cpu().numpy()
            mel_np  = mel.detach().float().cpu().numpy()
            mask_np = mask.detach().float().cpu().numpy().astype(bool)

            B = pred_np.shape[0]
            for b in range(B):
                # マスク True の部分だけ取り出し
                tb = mask_np[b]
                if not tb.any():
                    continue
                p = pred_np[b, tb, :]  # (T, n_mels)
                g = mel_np[b,  tb, :]  # (T, n_mels)
                mcd_val = mcd_like(p, g, mean=mean, std=std, sr=sr, n_mfcc=13)
                if mcd_val is not None and np.isfinite(mcd_val):
                    mcd_vals.append(float(mcd_val))

    out = {
        "loss": tot_loss / max(1, n_batches),
        "mse":  tot_mse  / max(1, n_batches),
        "mae":  tot_mae  / max(1, n_batches),
    }
    if len(mcd_vals) > 0:
        out["mcd_like"] = float(np.mean(mcd_vals))
    return out

# ====== メイン ======
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=str, required=True,
                   help="preprocess（または preprocess_synced）の出力ディレクトリ")
    p.add_argument("--ckpt", type=str, default="best_model_stable.pth",
                   help="学習済みモデルのチェックポイント（.pth）")
    p.add_argument("--split", type=str, default="val", choices=["val","test"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stats_json", type=str, default=None,
                   help="前処理で保存した mean/std（JSON: {mean:[...], std:[...]})。指定が無ければMCD-likeはスキップ")
    p.add_argument("--sr", type=int, default=11866)
    args = p.parse_args()

    # ロード
    print(f"[INFO] loading dataset from: {args.processed_dir}")
    loader = make_loader(args.processed_dir, split=args.split,
                         batch_size=args.batch_size, seq_len=args.seq_len,
                         num_workers=args.num_workers)

    print(f"[INFO] loading model from: {args.ckpt}")
    device = args.device
    model = RevisedCNNBiLSTM(n_mels=64, lstm_hidden_size=512, dropout_rate=0.3,
                             pretrained_cnn=True, chunk_size=12).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # 直ロードと state_dict の両対応
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    # stats（任意）
    stats = None
    if args.stats_json and os.path.isfile(args.stats_json):
        with open(args.stats_json, "r") as f:
            stats = json.load(f)
        # 形式チェック
        if "mean" not in stats or "std" not in stats:
            print("[WARN] stats_json に mean/std が見つかりません。MCD-like はスキップします。")
            stats = None
        else:
            print("[INFO] stats loaded for MCD-like.")

    # 評価
    res = evaluate(model, loader, device=device, stats=stats, sr=args.sr)

    # 出力
    print("\n=== Evaluation (split: {}) ===".format(args.split))
    print("masked loss: {:.6f}".format(res["loss"]))
    print("masked mse : {:.6f}".format(res["mse"]))
    print("masked mae : {:.6f}".format(res["mae"]))
    if "mcd_like" in res:
        print("MCD-like   : {:.4f}".format(res["mcd_like"]))
    else:
        print("MCD-like   : (stats未指定のためスキップ)")

if __name__ == "__main__":
    main()