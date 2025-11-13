# 卒論向けモチE��設定メモ

本ドキュメントでは、本研究で用ぁE��吁E��ジュール�E�ENN-LSTM、HiFi-GAN、Grad-CAM 可視化�E��E詳細設定をまとめる。前処琁E�EチE�Eタ刁E��・モチE��構造・損失関数・学習ハイパ�Eパラメータ・可視化手頁E��ど、卒業論文記述時に参�Eすることを想定してぁE��、E
---

## 1. チE�Eタ前�E琁E���E通設定！E
- **動画正規化**
  - 允E��画は 256ÁE56 のグレースケールに変換し、フレームごとに平坁E0 / 標準偏差 1 正規化後、E、E にスケーリング、E  - 褁E��動画を扱ぁE��合�E `preprocess_rtmri_data.py` を使用して一括処琁E��E 
    侁E  
    ```powershell
    python preprocess_rtmri_data.py `
      --data_dir <video_root> `
      --audio_dir <audio_root> `
      --out_dir dataset/rtmri_normalized_processed `
      --sr 11413 --n_mels 64 --n_fft 2048 --hop_length 420 `
      --win_length 2048 --preemph 0.97 --ref_frames 4
    ```
- **音声処琁E*
  - サンプリングレーチE11,413 Hz にリサンプリング、�Eリエンファシス係数 0.97、パワーメル (64 メル) を算�E征EdB 変換、E  - `scaler.json` にメルの全体平坁E標準偏差を保存し、CNN-LSTM 学習時に正規化・再標準化へ使用、E- **チE�EタセチE��構造**
  - `dataset/rtmri_normalized_processed/samples/<ID>/{mri.npy, mel_db.npy, mask.npy}`  
    `mask.npy` は現状 1 のみ�E�封E��皁E��口腔�Eスクを乗算予定）、E  - 4 フレーム参�Eの固定長ペアめE`pairs_ref4` (npz)、`pairs_ref4_npy` (mmap 用) に保存、E  - `scaler.json`、`meta.json`、`hifigan_filelists/{training,validation}.txt` を併せて生�E、E
---

## 2. CNN-LSTM�E�メル生�E器�E�E
- **モチE��構造**�E�Emri2speech_code/mri_acoustic_model.py`�E�E  - EfficientNetV2-B2�E�Eimm, `features_only=True`�E�で吁E��レームの特徴抽出、Ech MRI めE3ch に褁E��して入力、E  - Global Average Pooling ↁEBiLSTM�E�Eidden 640、双方向�E和）�E Dropout 0.5 ↁELinear (n_mels=64)、E- **学習スクリプト**: `mri2speech_code/train_mri_acoustic_model.py`
  - DataLoader: `FixedLenPairDataset` + `collate_pad`、E0/10/10 のランダム刁E�� (`torch.utils.data.random_split`, seed=42)、E  - バッチE `batch_size=16`、`micro_batch_size=4`�E�勾配蓄積）、`num_workers=4`、`prefetch_factor=4`、`pin_memory=True`、E  - Optimizer: `AdamW` (lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4)、E  - Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=5, min_lr=1e-6)、E  - Mixed Precision: bf16 (GPU が対忁E / fp16、�E勁EGradScaler、E  - 勾配クリチE�E: `clip_grad_norm_=1.0`、E  - CNN 部刁E�E `torch.utils.checkpoint` を使用可能 (`--use_checkpoint --ckpt_segments 2`)、E- **損失関数 `MaskedMSEMAE` 改訂�E容**
  - 周波数帯域ごとの重み:
    - F0 (mel bin 0 E): 2.0
    - F1 (6 E5): 3.0
    - F2 (16 E1): 2.4
    - F3 付迁E(32 E7): 1.6
    - 高域 (上佁E16 bin): 1.8
  - 時間方向�E重み: 先頭 8 フレームに 1.6 ↁE1.02 まで段階的に強調、E  - Ramp 設宁E `ramp_steps=120000`。�E期�Eベ�Eス重み、以降ターゲチE��重みに遷移、E  - 付加損失: ΁E(一次差刁E・Δ΁E(二次差刁E・最新フレーム MSE を加重。係数は ramp に応じて `delta_coeff=0.3ↁE.45`、`accel_coeff=0.1ↁE.15`、`latest_coeff=0.2ↁE.4`、E  - 損失冁E��バンド別 MAE を計測し、`band/train_*` / `band/val_*` として TensorBoard に記録、E- **学習設宁E*
  - 目標エポック 4,500。`EarlyStopping` 皁E��挙動: val loss 改喁E�� 20 回連続で得られなぁE��また�E LR が最小学習率以下になると停止、E  - ログ: `checkpoints/mri_acoustic_model_retrain/logs`�E�EensorBoard�E�と stdout、E  - 最良モチE��は持E��Eckpt (`--out_ckpt`) に保存、E
---

## 3. HiFi-GAN�E��Eコーダ�E�E
- **初期値**: `checkpoints/jvs_11413_2048_scratch/g_00055000` をコピ�Eしてスタート、E- **設宁E* (`config_custom.json`)
  - バッチサイズ 16、学習率 5e-5、Adam (β1=0.8, β2=0.99)、lr_decay=0.999、E  - Segment size 8400、メル設定�E CNN-LSTM と同一 (n_mels=64, hop=420)、E  - Upsampling rates [10,7,3,2]、ResBlock kernel [3,7,11]、dilation [[1,3,5], …]、E- **チE�Eタ**
  - 音声リスチE `dataset/rtmri_normalized_processed/hifigan_filelists/{training,validation}.txt`�E�Ereprocess 時に作�E、seed=42, valid 10%�E�、E  - メル入劁E
    - CNN-LSTM 予測メル `mels_ft_log_normalized`
    - ground-truth メル `mels_gt_log`�E�Escripts/export_groundtruth_mels.py` で `samples/<ID>/mel_db.npy` から生�E�E�E  - Fine-tuning 時�E `--extra_mels_dir mels_gt_log --extra_mels_weight 0.8` で 80% めEground-truth、E0% を予測メルからサンプリング、E- **実行侁E*
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
  - TensorBoard: `checkpoints/jvs_11413_2048_ft_mri_mix_gt08/logs`
  - チェチE��ポイント間の音質比輁E `scripts/run_mri_video_inference.py` を用ぁE��吁E`g_*.pt` で推論し、`output/mri_infer_mix_gt08/g_0006xxxx` などに保存、E
---

## 4. Grad-CAM 可視化

- **基本チE�Eル**: `scripts/mri_gradcam_formant.py`
  - EfficientNet バックボ�Eンの最終特徴マップを取得し、フォルマント帯埁E(侁E F1=300-900 Hz, F2=900-2500 Hz) のエネルギーをターゲチE��として送E��播、E  - GPU 利用 (`--device cuda`) が可能、EuDNN RNN 送E��播の制紁E��対応するため、推論モードでも一時的に LSTM めEtrain 状態に刁E��替えてぁE��、E  - 出劁E `gradcam_<band>_sequence.npy` (TÁE56ÁE56)、`gradcam_<band>_average.png`、指定フレームのオーバ�Eレイ PNG、E  - 実行侁E
    ```powershell
    python scripts/mri_gradcam_formant.py `
      --video normalized_videos/000.mp4 `
      --mri-checkpoint checkpoints/mri_acoustic_model.pt `
      --scaler-json dataset/rtmri_normalized_processed/scaler.json `
      --output-dir output/gradcam_formant/000_mix_gt08_60k `
      --formant-band F1:300-900 --formant-band F2:900-2500 `
      --target-frames 60 90 120 --device cuda
    ```
- **区間抽出・スロー再生**
  - NumPy で任意区閁E(侁E 0.8、E.2 s) を抽出ぁE`gradcam_F*_aSegment_sequence.npy` として保存、E  - `scripts/create_gradcam_video.py` でヒ�Eト�EチE�Eのみのスロー動画を生成、E    ```powershell
    python scripts/create_gradcam_video.py `
      --video normalized_videos/000.mp4 `
      --sequence output/gradcam_formant/000_mix_gt08_60k/gradcam_F1_sequence.npy `
      --start-frame 0 `
      --output output/gradcam_formant/F1_slow.mp4 `
      --fps 5 --repeat 4 --alpha 0.7
    ```
- **映像＋音声オーバ�Eレイ**
  - `scripts/create_gradcam_overlay_video.py` で允E��画�E�生成音声�E�Eoutput/mri_infer_latest_ft/000_generated.wav` 等）を結合し、F1/F2 のヒ�Eト�EチE�Eを重畳、E    ```powershell
    python scripts/create_gradcam_overlay_video.py `
      --video normalized_videos/000.mp4 `
      --heatmap output/gradcam_formant/000_mix_gt08_60k/gradcam_F1_sequence.npy `
      --heatmap2 output/gradcam_formant/000_mix_gt08_60k/gradcam_F2_sequence.npy `
      --audio output/mri_infer_latest_ft/000_generated.wav `
      --output output/gradcam_formant/000_overlay.mp4 `
      --alpha 0.7 --resize 256 256
    ```
  - `--heatmap2` を省略すれば単一帯域�Eみの可視化となる、E
---

## 5. メモ

- 新たなチE�EタセチE��を用ぁE��場合も、上記�E前�E琁EↁECNN-LSTM 再学翁EↁEHiFi-GAN 微調整 ↁEGrad-CAM 可視化の頁E��手頁E��踏めば、既存環墁E��流用して再現できる、E- 本ドキュメント�E `docs/thesis_model_settings.md` として保存してぁE��ため、論文執筁E��にはこ�Eファイルを参照しながら設定値を記述すること、E
