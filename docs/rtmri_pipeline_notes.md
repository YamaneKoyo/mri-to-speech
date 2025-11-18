# rtMRI ベース研究パイプライン作業メモ

2025 年 10 月時点で `C:\Users\Yamane\hifi-gan` に構築した rtMRI からの音声再合成環境の実行手順をまとめたノートです。CNN-LSTM（rtMRI → メル）と HiFi-GAN（メル → 波形）の更新、Grad-CAM 可視化までの一連の流れを PowerShell コマンドとともに記録しています。

---

## 1. データの準備
### 1.1 環境アクティベート + 前処理スクリプト

```powershell
# 事前に作成した仮想環境を起動
C:\Users\Yamane\hifigan-env\Scripts\activate

cd C:\Users\Yamane\Desktop\山根研究用\mri2speech_code

# rtMRI 動画 + 音声の前処理（wav も生成）
python preprocess_rtmri_data.py `
  --data_dir   C:\Users\Yamane\Desktop\山根研究用\normalized_videos `
  --audio_dir  C:\Users\Yamane\Desktop\山根研究用\audio_wav `
  --out_dir    C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed `
  --sr 11413 --n_mels 64 --n_fft 2048 --hop_length 420 --ref_frames 4

# pairs_ref4 を npy（mmap 用）に変換
cd C:\Users\Yamane\hifi-gan
python scripts/convert_pairs_to_npy.py `
  --processed_dir C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed `
  --ref_frames 4 --overwrite

# HiFi-GAN 用の wav リスト作成
python scripts/create_rtmri_filelists.py `
  C:\Users\Yamane\Desktop\山根研究用\audio_wav `
  C:\Users\Yamane\hifi-gan\dataset\rtmri_normalized_processed\hifigan_filelists `
  --valid-ratio 0.1 --seed 42
```

生成される主なファイル/フォルダ:

- `dataset/rtmri_normalized_processed/samples/<ID>/{mri.npy, mel_db.npy, mask.npy}`
- `dataset/rtmri_normalized_processed/pairs_ref4(_npy)/...`
- `dataset/rtmri_normalized_processed/hifigan_filelists/{training,validation}.txt`
- `dataset/rtmri_normalized_processed/{scaler.json, meta.json}`

---

## 2. CNN-LSTM（rtMRI → メル） 学習
### 2.1 実行コマンド

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
  --pin_memory `
  --cnn_pretrained --use_checkpoint --ckpt_segments 2
```

### 2.2 チェックポイント運用
- `checkpoints/mri_acoustic_model.pt` に最新/最良モデルを上書き保存。
- ログは `checkpoints/mri_acoustic_model_logs` に TensorBoard 形式で出力。`tensorboard --logdir <path>` で確認。
- 途中で再開する場合は `--resume_ckpt` を同じパスに向ける。

### 2.3 精度チェック
- `scripts/run_mri_video_inference.py` を使い、数本の動画でメル + 波形を生成。
- `output/mri_infer_latest_ft/<ID>/` に `*_mel_pred.npy`、`*_generated.wav` などが出力されるため、GT と比較しながら評価。

---

## 3. メルの書き出し

CNN-LSTM 学習後に下記を実行し、HiFi-GAN の条件付けに使うメルを保存する。

```powershell
python scripts/export_predicted_mels.py `
  --processed_dir dataset/rtmri_normalized_processed `
  --mri_checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler_json dataset/rtmri_normalized_processed/scaler.json `
  --output_dir dataset/rtmri_normalized_processed/mels_ft_log_normalized `
  --mri_code_dir C:\Users\Yamane\Desktop\山根研究用\mri2speech_code `
  --overwrite

python scripts/export_groundtruth_mels.py `
  --processed_dir dataset/rtmri_normalized_processed `
  --output_dir dataset/rtmri_normalized_processed/mels_gt_log `
  --overwrite
```

- 結果ディレクトリに `*_mel_pred_log_norm.npy`、`*_mel_gt_log.npy` が生成される。HiFi-GAN の `--input_mels_dir` / `--extra_mels_dir` に指定。

---

## 4. HiFi-GAN ファインチューニング
### 4.1 学習コマンド

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

- 学習途中で `Ctrl+C` してもチェックポイント (`g_*.pt`, `do_*.pt`) は `checkpoint_path` 以下に保存される。
- TensorBoard: `checkpoints/jvs_11413_2048_ft_mri_mix_gt08/logs`。

### 4.2 推論比較

```powershell
python scripts/run_mri_video_inference.py `
  --video normalized_videos/000.mp4 `
  --mri-checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler-json dataset/rtmri_normalized_processed/scaler.json `
  --hifigan-config checkpoints/jvs_11413_2048_ft_mri_mix_gt08/config.json `
  --hifigan-checkpoint checkpoints/jvs_11413_2048_ft_mri_mix_gt08/g_00065000 `
  --output-dir output/mri_infer_mix_gt08 `
  --mri-code-dir C:\Users\Yamane\Desktop\山根研究用\mri2speech_code
```

- `output/mri_infer_mix_gt08/g_00065000/` 以下に生成物がまとまる。
- 同じ ID を複数の checkpoint で推論し、聴感評価を実施。

---

## 5. マスキング実験

`scripts/mask_rtmri_video.py` で唇・舌領域を暗くし、動きの寄与を調べる。

```powershell
python scripts/mask_rtmri_video.py `
  --input normalized_videos/000.mp4 `
  --output temp/000_lip_alpha030.mp4 `
  --mask-type lip `
  --alpha 0.3
```

- `mask-type` は `lip` / `tongue` / `custom` を用意。`custom` の場合は JSON でポリゴン座標を指定。
- 出力動画をそのまま推論スクリプトに渡し、生成音声の劣化量を比較。

---

## 6. Grad-CAM 可視化

```powershell
python scripts/mri_gradcam_formant.py `
  --video normalized_videos/000.mp4 `
  --mri-checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler-json dataset/rtmri_normalized_processed/scaler.json `
  --output-dir output/gradcam_formant/000 `
  --formant-band F1:300-900 --formant-band F2:900-2500 `
  --target-frames 60 90 120 `
  --mri-code-dir C:\Users\Yamane\Desktop\山根研究用\mri2speech_code
```

- 出力: `gradcam_F*_sequence.npy`, `gradcam_F*_average.png`, `frame*_overlay.png`。
- `scripts/create_gradcam_overlay_video.py` でヒートマップと生成音声を 1 本の動画にまとめると発表資料に使いやすい。

---

## 7. 追加メモ

- HiFi-GAN の TensorBoard に記録される audio サンプルは `audio_wav` の元データを参照。rtMRI 側の動画は別コマンドで確認する。
- 大きな実験を始める前に `dataset/`, `checkpoints/`, `output/` を外付けドライブへバックアップすること。
- 研究室メンバーが環境を再現する場合は本ノートと `docs/thesis_model_settings.md` を参照すれば十分なはず。
