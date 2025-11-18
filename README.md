# MRI-to-Audio HiFi-GAN Research Pipeline

This repository hosts the scripts we used in the laboratory project *“rtMRI を用いた日本語母音音声合成と調音器官解析”*.  
It extends the original HiFi-GAN implementation with:

- A CNN-BiLSTM acoustic model that predicts mel features directly from rtMRI videos (`mri2speech_code`).
- Utilities for data preprocessing, inference, articulator masking, and Grad-CAM based visualization.
- Documentation so that lab members can reproduce the full pipeline after cloning from GitHub.

**Important:** rtMRI videos / aligned audio / MRI checkpoints are **not** distributed here.  
Only scripts and configs are versioned; you must supply your own data that matches the directory structure described below.

---

## Repository Layout

```
.
├─ docs/
│  ├─ rtmri_pipeline_notes.md        # step-by-step runbook
│  └─ thesis_model_settings.md       # hyper-parameter reference
├─ scripts/
│  ├─ mask_rtmri_video.py            # lip / tongue masking helper
│  ├─ run_mri_video_inference.py     # rtMRI → mel → HiFi-GAN audio
│  ├─ export_predicted_mels.py       # dumps mel features for HiFi-GAN fine-tuning
│  └─ mri_gradcam_formant.py         # Grad-CAM visualizer for F1/F2 bands
├─ checkpoints/                      # place acoustic + HiFi-GAN checkpoints here
├─ dataset/                          # preprocessed npy, scaler.json, filelists, etc.
├─ requirements.txt                  # legacy HiFi-GAN deps
└─ requirements.lab.txt              # current lab environment (PyTorch 2.7, OpenCV, timm, ...)
```

The acoustic model code (`mri_acoustic_model.py`, `train_mri_acoustic_model.py` …) lives in a **separate** repository (`mri2speech_code`).  
Keep it as a sibling folder or pass `--mri-code-dir` whenever a script needs it.

---

## Environment Setup

1. Install Python 3.10+ (we validated on **Python 3.13.3**).
2. Clone this repository and (optionally) clone `mri2speech_code` next to it.
3. Install Python packages:

   ```bash
   python -m pip install -r requirements.lab.txt
   ```

   Tested versions:

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

   Feel free to install GPU builds of PyTorch if hardware is available.

---

## End-to-End Workflow

### 1. Prepare raw data

```
<raw_root>/
├─ normalized_videos/   # rtMRI mp4 files (256x256 grayscale, normalized)
└─ audio_wav/           # aligned speech at 11,413 Hz
```

Statistical masks or metadata live elsewhere and are not stored in Git.

### 2. Preprocess rtMRI + audio

From the `mri2speech_code` repo:

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

Artifacts (mel/db npy, `scaler.json`, filelists, …) are required for every later step.

### 3. Train the CNN-BiLSTM acoustic model

```
python train_mri_acoustic_model.py \
  --processed_dir dataset/rtmri_normalized_processed \
  --out_ckpt checkpoints/mri_acoustic_model.pt \
  --epochs 4500 --batch_size 8 --cnn_pretrained --use_checkpoint --ckpt_segments 2 \
  ... (see docs/thesis_model_settings.md for the complete hyper-parameter set)
```

The loss (`MaskedMSEMAE`) weights F0/F1/F2 bands and adds Δ / Δ² penalties as described in `docs/thesis_model_settings.md`.

### 4. Export predicted mels and fine-tune HiFi-GAN

```powershell
python scripts/export_predicted_mels.py `
  --processed_dir dataset/rtmri_normalized_processed `
  --mri_checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler_json dataset/rtmri_normalized_processed/scaler.json `
  --output_dir dataset/rtmri_normalized_processed/mels_ft_log_normalized `
  --mri_code_dir <path-to-mri2speech_code> `
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

### 5. Run inference (video → speech)

```powershell
python scripts/run_mri_video_inference.py `
  --video <path>/000.mp4 `
  --mri-checkpoint checkpoints/mri_acoustic_model.pt `
  --scaler-json dataset/rtmri_normalized_processed/scaler.json `
  --hifigan-config checkpoints/jvs_11413_2048_ft_mri_mix_gt08/config.json `
  --hifigan-checkpoint checkpoints/jvs_11413_2048_ft_mri_mix_gt08/g_00065000 `
  --output-dir output/mri_infer `
  --mri-code-dir <path-to-mri2speech_code>
```

Outputs: waveform (`*_generated.wav`), linear/log mels (`*.npy`), and PNG previews.

### 6. Apply articulator masks

`scripts/mask_rtmri_video.py` darkens a polygonal region before inference.

```
python scripts/mask_rtmri_video.py \
  --input normalized_videos/000.mp4 \
  --output temp/000_lip_alpha030.mp4 \
  --mask-type lip \
  --alpha 0.3      # residual intensity (0.0=hidden, 1.0=no mask)
```

- `--mask-type lip` uses a rectangular ROI around the lips.
- `--mask-type tongue` uses a 5-point polygon covering tongue dorsum + pharynx.
- Change `alpha` per experiment (e.g., 0.1 to 0.9).  
  Afterwards run `run_mri_video_inference.py` on the masked video.

### 7. Grad-CAM for F1 / F2

```
python scripts/mri_gradcam_formant.py \
  --video normalized_videos/000.mp4 \
  --mri-checkpoint checkpoints/mri_acoustic_model.pt \
  --scaler-json dataset/rtmri_normalized_processed/scaler.json \
  --output-dir output/gradcam_formant/000 \
  --formant-band F1:300-900 --formant-band F2:900-2500 \
  --target-frames 60 90 120 \
  --mri-code-dir <path-to-mri2speech_code>
```

The script denormalizes mel predictions, converts them to linear power, sums the power inside each band, and backpropagates it to EfficientNet feature maps. Resulting heatmaps are saved as numpy arrays and overlays.

---

## Documentation & Tips

- `docs/rtmri_pipeline_notes.md` … chronological runbook with concrete Windows/PowerShell commands.
- `docs/thesis_model_settings.md` … reference of all knobs used in the thesis (loss weights, Grad-CAM settings, etc.).
- Store raw data/checkpoints outside the repo. Use `.gitignore` to keep `dataset/*` and `checkpoints/*` private if needed.
- When publishing to GitHub, remove private paths from scripts or replace them with CLI flags (already supported by `--mri-code-dir` and path arguments above).

With these instructions and the packaged scripts, anyone in the lab can rebuild the full pipeline—from preprocessing through Grad-CAM visualization—after cloning the repository. Feel free to open issues or PRs to document additional datasets or masking patterns.
