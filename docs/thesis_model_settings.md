# 卒論向けモデル設定メモ

本ドキュメントは、研究テーマ「rtMRI を用いた日本語母音音声合成と調音器官解析」で使用した主要モジュール（CNN-LSTM 音響モデル / HiFi-GAN / Grad-CAM 可視化）の設定をまとめた備忘録です。前処理の共通ルールから学習ハイパーパラメータ、可視化ツールの使い方までを網羅しているため、卒論執筆時の参照資料として利用してください。

---

## 1. データ前処理の共通設定
- **動画正規化**
  - すべての動画を 256×256 のグレースケールへ変換し、フレーム単位で平均 0 / 標準偏差 1 に正規化後、[0, 1] に再スケーリング。
  - まとまったデータを処理する際は `preprocess_rtmri_data.py` を使用する。
    ```powershell
    python preprocess_rtmri_data.py `
      --data_dir <video_root> `
      --audio_dir <audio_root> `
      --out_dir dataset/rtmri_normalized_processed `
      --sr 11413 --n_mels 64 --n_fft 2048 --hop_length 420 `
      --win_length 2048 --preemph 0.97 --ref_frames 4
    ```
- **音声処理**
  - サンプリングレート 11,413 Hz へ揃え、プリエンファシス 0.97、64 メルのパワーメルを計算後 dB 変換。
  - `scaler.json` にメル全体の平均・標準偏差を保存し、CNN-LSTM 学習での正規化と逆変換に利用する。
- **データセット構造**
  - `dataset/rtmri_normalized_processed/samples/<ID>/{mri.npy, mel_db.npy, mask.npy}`
    - `mask.npy` は現在オール 1。後日、口腔マスクを掛ける予定がある場合はここに保存。
  - 参照フレーム数 4 の固定長ペアを `pairs_ref4` (npz) と `pairs_ref4_npy` (mmap 用) に格納。
  - `scaler.json`、`meta.json`、`hifigan_filelists/{training,validation}.txt` を必ず同時に生成する。

---

## 2. CNN-LSTM ベース メル生成器
- **モデル構造** （`mri2speech_code/mri_acoustic_model.py`）
  - EfficientNetV2-B2 (timm, `features_only=True`) で各フレームの特徴量を抽出。rtMRI の 4 フレームを 3ch×4 のテンソルに畳み、1 サンプルとして投入。
  - Global Average Pooling → BiLSTM (Hidden 640, 双方向の和) → Dropout 0.5 → Linear (出力 64mel)。
- **学習スクリプト**: `mri2speech_code/train_mri_acoustic_model.py`
  - DataLoader: `FixedLenPairDataset` + `collate_pad`。データは 80/10/10% にランダム分割 (`torch.utils.data.random_split`, seed=42)。
  - ハイパーパラメータ: `batch_size=16`、`micro_batch_size=4`（勾配蓄積）、`num_workers=4`、`prefetch_factor=4`、`pin_memory=True`。
  - Optimizer: `AdamW` (lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)。
  - Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=5, min_lr=1e-6)。
  - Mixed Precision: bf16（GPU が非対応のときは fp16 + GradScaler）。
  - 勾配クリップ: `clip_grad_norm_=1.0`。
  - CNN 部分には `torch.utils.checkpoint` を適用可能 (`--use_checkpoint --ckpt_segments 2`)。
- **損失関数 `MaskedMSEMAE` の調整**
  - 周波数帯域ごとの重み:
    - F0 (mel bin 0〜5): 2.0
    - F1 (6〜15): 3.0
    - F2 (16〜31): 2.4
    - F3 付近 (32〜47): 1.6
    - 高域 (48 以上): 1.8
  - 時間方向の重み付け: 先頭 8 フレームは 1.6 から 1.02 まで逓減させて強調。
  - Ramp 設定: `ramp_steps=120000`。序盤はベース重み、以降ターゲット重みに移行。
  - 付加損失: Δ（一次差分）、Δ²（二次差分）、最新フレーム MSE を加重。係数は `delta_coeff=0.3→0.45`、`accel_coeff=0.1→0.15`、`latest_coeff=0.2→0.4`。
  - TensorBoard にはバンド別 MAE を `band/train_*` / `band/val_*` として記録。
- **学習運用**
  - 目標エポック 4,500。`EarlyStopping`: Val loss が 20 回連続で改善しない、または学習率が最小学習率まで下がったら停止。
  - ログ出力: `checkpoints/mri_acoustic_model_retrain/logs`（TensorBoard + stdout）。
  - 最良モデルは `--out_ckpt` で指定したパスに保存。

---

## 3. HiFi-GAN ボコーダ
- **初期値**: `checkpoints/jvs_11413_2048_scratch/g_00055000` をコピーしてファインチューニングを開始。
- **主な設定** (`config_custom.json`)
  - バッチサイズ 16、学習率 5e-5、Optimizer: Adam (β1=0.8, β2=0.99)、`lr_decay=0.999`。
  - Segment size 8400、メル条件は CNN-LSTM と同一 (n_mels=64, hop=420)。
  - Upsampling rates [10,7,3,2]、ResBlock kernel [3,7,11]、dilation [[1,3,5], …]。
- **データ**
  - 音声リストは `dataset/rtmri_normalized_processed/hifigan_filelists/{training,validation}.txt`。前処理スクリプトで生成 (seed=42, valid 10%)。
  - メル入力:
    - CNN-LSTM 予測メル `mels_ft_log_normalized`
    - Ground-truth メル `mels_gt_log`（`scripts/export_groundtruth_mels.py` で `samples/<ID>/mel_db.npy` から生成）
  - Fine-tuning 時は `--extra_mels_dir mels_gt_log --extra_mels_weight 0.8` で Ground-truth 80%, 予測 20% を混合サンプリング。
- **実行例**
  ```powershell
  C:\Users\Yamane\hifigan-env\Scripts\python.exe train.py ^
    --config config_custom.json ^
    --input_wavs_dir "C:\Users\Yamane\Desktop\山根研究用\audio_wav" ^
    --input_training_file "dataset\rtmri_normalized_processed\hifigan_filelists\training.txt" ^
    --input_validation_file "dataset\rtmri_normalized_processed\hifigan_filelists\validation.txt" ^
    --input_mels_dir "dataset\rtmri_normalized_processed\mels_ft_log_normalized" ^
    --extra_mels_dir "dataset\rtmri_normalized_processed\mels_gt_log" ^
    --extra_mels_weight 0.8 ^
    --checkpoint_path "checkpoints\jvs_11413_2048_ft_mri_mix_gt08" ^
    --fine_tuning 1
  ```
- **ログ/評価**
  - TensorBoard: `checkpoints/jvs_11413_2048_ft_mri_mix_gt08/logs`。
  - 各チェックポイントの音質比較は `scripts/run_mri_video_inference.py` を用いて `g_*.pt` から推論し、`output/mri_infer_mix_gt08/g_0006xxxx` などに保存する。

---

## 4. Grad-CAM 可視化

- **基本ツール**: `scripts/mri_gradcam_formant.py`
  - EfficientNet の最終特徴マップを取得し、フォルマント帯域（例: F1=300-900 Hz, F2=900-2500 Hz）のエネルギーをターゲットに逆伝播。
  - GPU がある場合は `--device cuda` を指定。cuDNN RNN 逆伝播の制約に合わせ、推論モード中に LSTM を一時的に train 状態へ切替。
  - 出力: `gradcam_<band>_sequence.npy` (T×256×256)、`gradcam_<band>_average.png`、指定フレームのオーバーレイ PNG。
  - 実行例:
    ```powershell
    python scripts/mri_gradcam_formant.py `
      --video normalized_videos/000.mp4 `
      --mri-checkpoint checkpoints/mri_acoustic_model.pt `
      --scaler-json dataset/rtmri_normalized_processed/scaler.json `
      --output-dir output/gradcam_formant/000_mix_gt08_60k `
      --formant-band F1:300-900 --formant-band F2:900-2500 `
      --target-frames 60 90 120 --device cuda
    ```
- **区間抽出 / スロー再生**
  - NumPy で任意区間 (例: 0.8〜1.2 s) を抽出し `gradcam_F*_aSegment_sequence.npy` として保存。
  - `scripts/create_gradcam_video.py` でヒートマップのみのスロー動画を生成。
    ```powershell
    python scripts/create_gradcam_video.py `
      --video normalized_videos/000.mp4 `
      --sequence output/gradcam_formant/000_mix_gt08_60k/gradcam_F1_sequence.npy `
      --start-frame 0 `
      --output output/gradcam_formant/F1_slow.mp4 `
      --fps 5 --repeat 4 --alpha 0.7
    ```
- **映像＋音声オーバーレイ**
  - `scripts/create_gradcam_overlay_video.py` で元動画と生成音声（例: `output/mri_infer_latest_ft/000_generated.wav`）を組み合わせ、F1/F2 のヒートマップを重畳。
    ```powershell
    python scripts/create_gradcam_overlay_video.py `
      --video normalized_videos/000.mp4 `
      --heatmap output/gradcam_formant/000_mix_gt08_60k/gradcam_F1_sequence.npy `
      --heatmap2 output/gradcam_formant/000_mix_gt08_60k/gradcam_F2_sequence.npy `
      --audio output/mri_infer_latest_ft/000_generated.wav `
      --output output/gradcam_formant/000_overlay.mp4 `
      --alpha 0.7 --resize 256 256
    ```
  - `--heatmap2` を省略すれば単一帯域のみの可視化となる。

---

## 5. メモ

- 新しいデータセットを使う場合も、上記の「前処理 → CNN-LSTM 再学習 → HiFi-GAN 微調整 → Grad-CAM 可視化」を踏めば既存環境を流用して再現できる。
- 本メモは `docs/thesis_model_settings.md` として追跡しているため、論文執筆時はこのファイルを参照しながら設定値を記述すること。
