# rtMRI ベース音声合成ワークフロー引き継ぎメモ

2025 年 10 月時点の `C:\Users\Yamane\hifi-gan` 環境で、rtMRI 動画から音声を生成する際の作業手順と依存関係の整理メモです。CNN-LSTM（メル生成器）と HiFi-GAN（ボコーダ）の両方を更新する運用を前提としています。

---

## 1. データ前処理
### 1.1 正規化動画 + 音声からサンプル生成

```powershell
# 必要に応じて仮想環境を有効化
C:\Users\Yamane\hifigan-env\Scripts\activate

cd C:\Users\Yamane\Desktop\山根研究用\mri2speech_code

# 正規化動画 + 音声からサンプル作成（既存 wav を利用）
python preprocess_rtmri_data.py `
  --data_dir   C:\Users\Yamane\Desktop\山根研究用\normalized_videos `
  --audio_dir  C:\Users\Yamane\Desktop\山根研究用\audio_wav `
  --out_dir    C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed `
  --sr 11413 --n_mels 64 --n_fft 2048 --hop_length 420 --ref_frames 4

# pairs_ref4 を npy に変換（PyTorch Dataset で mmap 利用）
cd C:\Users\Yamane\hifi-gan
python scripts/convert_pairs_to_npy.py `
  --processed_dir C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed `
  --ref_frames 4 --overwrite

# HiFi-GAN 用の wav リスト生成
python scripts/create_rtmri_filelists.py `
  C:\Users\Yamane\Desktop\山根研究用\audio_wav `
  C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed\hifigan_filelists `
  --valid-ratio 0.1 --seed 42
```

生成される主なファイル・フォルダ:

- `dataset/rtmri_normalized_processed/samples/<ID>/{mri.npy, mel_db.npy, mask.npy}`
- `dataset/rtmri_normalized_processed/pairs_ref4(_npy)/...`
- `dataset/rtmri_normalized_processed/hifigan_filelists/{training,validation}.txt`
- `dataset/rtmri_normalized_processed/scaler.json`
- `dataset/rtmri_normalized_processed/meta.json`

---

## 2. CNN-LSTM（メル生成器）の学習
### 2.1 学習コマンド

```powershell
C:\Users\Yamane\hifigan-env\Scripts\python.exe "C:\Users\Yamane\Desktop\山根研究用\mri2speech_code\train_mri_acoustic_model.py" `
  --processed_dir "C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed" `
  --out_ckpt "C:\Users\Yamane\hifi-gan\checkpoints\mri_acoustic_model.pt" `
  --resume_ckpt "C:\Users\Yamane\hifi-gan\checkpoints\mri_acoustic_model.pt" `
  --log_dir "C:\Users\Yamane\hifi-gan\checkpoints\mri_acoustic_model_logs" `
  --epochs 4500 `
  --batch_size 8 `
  --val_batch_size 8 `
  --micro_batch_size 2 `
  --num_workers 4 `
  --prefetch_factor 4 `
  --cnn_pretrained `
  --use_checkpoint `
  --ckpt_segments 2
```

### 2.2 損失関数の要点
- `train_mri_acoustic_model.py` 内の `MaskedMSEMAE` を参照。周波数／時間方向の重み付けを `ramp_steps` で段階的に強化
- Δ（一次差分）および最新フレームの補助 MSE を併用。係数調整後もチェックポイントから再開可能
- `ReduceLROnPlateau` で学習率が下がり切った場合は帯域重みや補助損失係数の調整を検討

### 2.3 TensorBoard の確認

```powershell
C:\Users\Yamane\hifigan-env\Scripts\activate
tensorboard --logdir "C:\Users\Yamane\hifi-gan\checkpoints\mri_acoustic_model_logs" --port 6007 --host 0.0.0.0
```

---

## 3. HiFi-GAN のファインチューニング
### 3.1 CNN-LSTM 出力メルの一括生成

```powershell
cd C:\Users\Yamane\hifi-gan
python scripts/export_predicted_mels.py `
  --processed_dir C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed `
  --mri_checkpoint C:\Users\Yamane\hifi-gan\checkpoints\mri_acoustic_model.pt `
  --scaler_json C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed\scaler.json `
  --output_dir C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed\mels_ft_log_normalized `
  --mri_code_dir C:\Users\Yamane\Desktop\山根研究用\mri2speech_code `
  --overwrite
```

### 3.2 HiFi-GAN 再学習コマンド

```powershell
C:\Users\Yamane\hifigan-env\Scripts\python.exe train.py ^
  --config config_custom.json ^
  --input_wavs_dir "C:\Users\Yamane\Desktop\山根研究用\audio_wav" ^
  --input_training_file "dataset\rtmri_normalized_processed\hifigan_filelists\training.txt" ^
  --input_validation_file "dataset\rtmri_normalized_processed\hifigan_filelists\validation.txt" ^
  --input_mels_dir "C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed\mels_ft_log_normalized" ^
  --checkpoint_path "checkpoints\jvs_11413_2048_ft_mri" ^
  --fine_tuning 1 ^
  --training_epochs 10000
```

TensorBoard ログは `checkpoints\jvs_11413_2048_ft_mri\logs` に出力。ポート衝突時は `tensorboard --logdir ... --port 6006` などで調整。

---

## 4. 推論

最終的に CNN-LSTM (`mri_acoustic_model.pt`) と HiFi-GAN (`jvs_11413_2048_ft_mri\g_xxxxxx`) を組み合わせて音声を生成します。

```powershell
cd C:\Users\Yamane\hifi-gan

python scripts/run_mri_video_inference.py ^
  --video "C:\Users\Yamane\Desktop\山根研究用\normalized_videos\000.mp4" ^
  --mri-checkpoint "checkpoints\mri_acoustic_model.pt" ^
  --scaler-json "dataset\rtmri_normalized_processed\scaler.json" ^
  --hifigan-config "checkpoints\jvs_11413_2048_ft_mri\config.json" ^
  --hifigan-checkpoint "checkpoints\jvs_11413_2048_ft_mri\g_00330000" ^
  --output-dir "output\mri_infer_latest_ft" ^
  --mri-code-dir "C:\Users\Yamane\Desktop\山根研究用\mri2speech_code"
```

複数ファイルを処理する場合はループやバッチスクリプトで `--video` 引数を切り替える。生成音 (`*_generated.wav`) と元音 (`*_original.wav`) を同一ディレクトリに保存すると比較しやすい。

---

## 5. 主なディレクトリとモジュール

| パス | 用途 |
|------|------|
| `checkpoints\mri_acoustic_model.pt` | 最新 CNN-LSTM (メル生成器) |
| `checkpoints\mri_acoustic_model_logs` | CNN-LSTM 学習ログ (TensorBoard) |
| `checkpoints\jvs_11413_2048_ft_mri\g_*.pt` | HiFi-GAN ファインチューニング済み重み (最新は g_00330000 など) |
| `dataset\rtmri_normalized_processed\mels_ft_log_normalized` | HiFi-GAN 学習用メルログパワー |
| `output\mri_infer_latest_ft` | 最新推論結果 (generated/original wav + メル画像) |
| `logs\tensorboard` | 過去ジョブの TensorBoard ログ。必要に応じ整理 |

---

## 6. 注意事項
- CNN-LSTM の損失係数は段階的に強化されるため、Loss が一時的に上昇しても安定するまで観察する
- `ReduceLROnPlateau` で学習率が下がり続ける場合は、帯域ウェイトや補助損失係数を調整してから再開
- HiFi-GAN の TensorBoard に出力される audio サンプルは `audio_wav` の元データが基準。rtMRI 由来音声は推論コマンドで確認する
- 大容量チェックポイント (`jvs_11413_2048_*`) はストレージを圧迫するため、不要な世代は退避・整理する

---

## 7. モデル構造とコード依存関係
### 7.1 CNN-LSTM (メル生成器)
- 実装ファイル: `C:\Users\Yamane\Desktop\山根研究用\mri2speech_code\mri_acoustic_model.py`
  - EfficientNetV2-B2 (timm) 特徴マップ → GlobalAvgPool → BiLSTM (双方向和, hidden=640) → Dropout(0.5) → Linear(n_mels=64)
- 学習スクリプト: `train_mri_acoustic_model.py` (`dataset/*.py` と連携)。`MaskedMSEMAE` と Δフレーム補助 MSE を使用
- 推論スクリプト: `scripts/run_mri_video_inference.py` / `scripts/export_predicted_mels.py` が `build_mri_model()` を経由してモデルをロードし、`frames_to_tensor()` で入力整形

### 7.2 HiFi-GAN (ボコーダ)
- 実装ファイル: ルート直下の `models.py`, `meldataset.py`, `env.py`, `utils.py`, `train.py`。構成値は `config_custom.json`
- 学習フロー: `scripts/export_predicted_mels.py` で生成した `dataset\rtmri_normalized_processed\mels_ft_log_normalized` を `train.py --fine_tuning 1` で読み込み
- チェックポイント運用: `checkpoints\jvs_11413_2048_scratch` の `g_00055000` / `do_00055000` を基点にコピーし、`checkpoints\jvs_11413_2048_ft_mri_YYYYMMDD` を作成して再学習

### 7.3 データ前処理と補助スクリプト
- 前処理: `mri2speech_code/preprocess_rtmri_data.py` → `scripts/convert_pairs_to_npy.py` → `scripts/create_rtmri_filelists.py`
- 学習・推論: `scripts/export_predicted_mels.py` → `train.py` → `scripts/run_mri_video_inference.py` の順で依存
- 成果物配置: `dataset/`, `checkpoints/`, `output/` に格納。旧ユーティリティは `archive/legacy_scripts` などへ移動予定

### 7.4 Grad-CAM 可視化
    - 使用コマンド例:
      `python scripts/mri_gradcam_formant.py --video normalized_videos/000.mp4 --mri-checkpoint checkpoints/mri_acoustic_model.pt --scaler-json dataset/rtmri_normalized_processed/scaler.json --output-dir output/gradcam_formant/000_mix_gt08_60k --formant-band F1:300-900 --formant-band F2:900-2500 --target-frames 60 90 120 --device cuda`
    - 出力: `gradcam_<band>_sequence.npy` (T×256×256), `gradcam_<band>_average.png`、指定フレームのオーバーレイPNG
    - 区間抽出: `gradcam_F*_first2s_sequence.npy`（0.8?1.2s など狙い区間を NumPy で切り出し）
    - 動画化: `python scripts/create_gradcam_video.py --video ... --sequence ... --start-frame ... --output ... --fps 5 --repeat 4 --alpha 0.7`
    - 元映像＋音声合成: `python scripts/create_gradcam_overlay_video.py --video ... --heatmap gradcam_F1_sequence.npy --heatmap2 gradcam_F2_sequence.npy --audio <generated.wav> --output ... --alpha 0.7 --resize 256 256`
