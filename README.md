# MRI-to-speech 研究パイプライン

「rtMRIを用いた日本語母音音声合成における調音器官の影響分析」で使用しているスクリプト一式をまとめたものです。HiFi-GANの学習、rtMRI 動画からメルスペクトログラムを推定する CNN-BiLSTMの学習、rtMRI動画から音声の推論、grad-camによるモデル内部の可視化、マスキングのための補助ツール、再現手順付きドキュメントを収録しています。



---

## リポジトリ構成

```
.
├─ docs/
│  ├─ rtmri_pipeline_notes.md        # 逐次作業メモ（PowerShell 例付き）
│  └─ thesis_model_settings.md       # 学習ハイパーパラメータ一覧
├─ mri2speech_code/                  # rtMRI→mel 学習・前処理コード
├─ scripts/                          # 推論・可視化など補助ツール
├─ config_custom.json                # HiFi-GAN 設定
├─ env.py / inference*.py / train.py # 実験用エントリポイント
├─ meldataset.py / models.py / utils.py ...
├─ requirements.txt                  # 旧環境用依存
└─ requirements.lab.txt              # 研究室現行環境（PyTorch 2.7, OpenCV など）
```

※ `dataset/` や `checkpoints/` などのディレクトリは Git 追跡対象外です。各自の環境で必要に応じて作成し、`.gitignore` で除外したまま運用してください。

rtMRI→mel の学習コード本体 (`mri_acoustic_model.py`, `train_mri_acoustic_model.py` …) は、このリポジトリ直下の `mri2speech_code/` ディレクトリに格納しています。スクリプトに `--mri-code-dir` を渡さない場合は `./mri2speech_code` が既定で読み込まれます。

---

## 環境構築

1. Python 3.10 以上（研究室では **Python 3.13.3** で検証）。
2. 依存パッケージをインストール:

   ```bash
   python -m pip install -r requirements.lab.txt
   ```

   検証済みバージョン例:

   | Package           | Version   |
   |-------------------|-----------|
   | torch             | 2.7.0+cpu |
   | torchvision       | 0.22.0+cpu |
   | torchaudio        | 2.7.0+cpu |
   | numpy             | 2.2.5 |
   | scipy             | 1.15.2 |
   | librosa           | 0.11.0 |
   | soundfile         | 0.13.1 |
   | matplotlib        | 3.10.1 |
   | tqdm              | 4.67.1 |
   | opencv-python     | 4.12.0.88 |
   | timm              | 1.0.21 |
   | huggingface-hub   | 1.0.1 |
   | pdfminer.six      | 20250506 |
   | pypdf             | 6.1.3 |

   GPU マシンでは PyTorch の CUDA 版に置き換えても問題ありません。

---

## パイプライン全体像

### 1. 元データの配置

```
<raw_root>/
├─ normalized_videos/   # 256x256 グレースケールで正規化済み rtMRI mp4
└─ audio_wav/           # 11,413 Hz に整列させた発話 wav
```



### 2. rtMRI + 音声の前処理

リポジトリ直下の `mri2speech_code/` で以下を実行します:

```powershell
python preprocess_rtmri_data.py `
  --data_dir <raw_root>/normalized_videos `
  --audio_dir <raw_root>/audio_wav `
  --out_dir dataset/rtmri_normalized_processed `
  --sr 11413 --n_mels 64 --n_fft 2048 --hop_length 420 `
  --win_length 2048 --preemph 0.97 --ref_frames 4

python scripts/convert_pairs_to_npy.py `
  --processed_dir dataset/rtmri_normalized_processed `
  --ref_frames 4 --overwrite

python scripts/create_rtmri_filelists.py `
  <raw_root>/audio_wav `
  dataset/rtmri_normalized_processed/hifigan_filelists `
  --valid-ratio 0.1 --seed 42
```

ここで生成される `samples/`, `pairs_ref4/`, `scaler.json`, filelists は以降すべてで使用します。

### 3. CNN-BiLSTM 音響モデルの学習

```
python train_mri_acoustic_model.py \
  --processed_dir dataset/rtmri_normalized_processed \
  --out_ckpt checkpoints/mri_acoustic_model.pt \
  --epochs 4500 --batch_size 8 --cnn_pretrained --use_checkpoint --ckpt_segments 2 \
  ...（詳細ハイパーパラメータは docs/thesis_model_settings.md を参照）
```

損失関数 `MaskedMSEMAE` は F0/F1/F2 の帯域重みと Δ/Δ² のペナルティを含みます。

### 4. メル書き出しと HiFi-GAN 微調整

```powershell
python scripts/export_predicted_mels.py `
  --processed_dir dataset/rtmri_normalized_processed `
  --mri_checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler_json dataset/rtmri_normalized_processed/scaler.json `
  --output_dir dataset/rtmri_normalized_processed/mels_ft_log_normalized `
  --mri_code_dir mri2speech_code `
  --overwrite

python train.py ^
  --config config_custom.json ^
  --input_wavs_dir <raw_root>\audio_wav ^
  --input_training_file dataset\rtmri_normalized_processed\hifigan_filelists\training.txt ^
  --input_validation_file dataset\rtmri_normalized_processed\hifigan_filelists\validation.txt ^
  --input_mels_dir dataset\rtmri_normalized_processed\mels_ft_log_normalized ^
  --extra_mels_dir dataset\rtmri_normalized_processed\mels_gt_log ^
  --extra_mels_weight 0.8 ^
  --checkpoint_path checkpoints\jvs_11413_2048_ft_mri_mix_gt08 ^
  --fine_tuning 1
```

Ground-truth メルと予測メルを混合しながら学習し、チェックポイントは `checkpoints/jvs_11413_2048_ft_mri_mix_gt08/` 以下に保存されます。

### 5. 推論（動画 → 音声）

```powershell
python scripts/run_mri_video_inference.py `
  --video <path>/000.mp4 `
  --mri-checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler-json dataset/rtmri_normalized_processed/scaler.json `
  --hifigan-config checkpoints/jvs_11413_2048_ft_mri_mix_gt08/config.json `
  --hifigan-checkpoint checkpoints/jvs_11413_2048_ft_mri_mix_gt08/g_00065000 `
  --output-dir output/mri_infer `
  --mri-code-dir mri2speech_code
```

`*_generated.wav`, `*_mel_pred.npy`, 可視化用 PNG などが `output/mri_infer/<ID>/` に出力されます。

### 6. 調音器官マスキング実験

`scripts/mask_rtmri_video.py` で一部領域を暗くして影響を評価します。

```
python scripts/mask_rtmri_video.py \
  --input normalized_videos/000.mp4 \
  --output temp/000_lip_alpha030.mp4 \
  --mask-type lip \
  --alpha 0.3      # 0.0=完全マスク, 1.0=マスク無効
```

`lip`/`tongue`/`custom` から選択し、生成された動画をそのまま推論スクリプトへ投入します。

### 7. Grad-CAM（F1/F2 帯域の解釈）

```
python scripts/mri_gradcam_formant.py \
  --video normalized_videos/000.mp4 \
  --mri-checkpoint checkpoints/mri_acoustic_model.pt \
  --scaler-json dataset/rtmri_normalized_processed/scaler.json \
  --output-dir output/gradcam_formant/000 \
  --formant-band F1:300-900 --formant-band F2:900-2500 \
  --target-frames 60 90 120 \
  --mri-code-dir mri2speech_code
```

得られたヒートマップは NumPy/PNG 形式で保存され、`scripts/create_gradcam_overlay_video.py` を使うと音声付きの説明動画を作れます。

---

## ドキュメントと運用メモ

- `docs/rtmri_pipeline_notes.md` … Windows/PowerShell 前提の詳細手順。
- `docs/thesis_model_settings.md` … 卒論で参照したハイパーパラメータ・損失設定・可視化条件。
- 生データ (`dataset/*`, `checkpoints/*`, `output/*`) は Git 管理外に置くことを推奨。`.gitignore` で除外済みです。
- 実験結果を共有する際は、パスをハードコーディングせず CLI 引数で指定できる形を保ってください（全スクリプトで `--mri-code-dir` やパス引数に対応済み）。
