import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    # ファイルパスの修正: .wav.wav を .wav に置き換え
    if full_path.endswith('.wav.wav'):
        full_path = full_path[:-4]  # .wav を1つ削除
    
    # ファイルの存在確認
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")
    
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):
    """
    メルスペクトログラムを計算
    """
    # librosaバージョンを確認（デバッグ用）

    import librosa
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        # 修正: キーワード引数を使用
        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax
        )
        mel_basis[fmax] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # return_complex=False パラメータを追加
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                  center=False, pad_mode='constant', normalized=False, onesided=True, return_complex=False)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[fmax], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None,
                 mel_dirs=None, mel_weights=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.mel_dirs = None
        self.mel_weights = None
        if mel_dirs is not None:
            if mel_weights is None:
                mel_weights = [1.0] * len(mel_dirs)
            if len(mel_dirs) != len(mel_weights):
                raise ValueError("mel_dirs and mel_weights must be the same length")
            filtered = [(d, w) for d, w in zip(mel_dirs, mel_weights) if d and w > 0]
            if not filtered:
                filtered = [(mel_dirs[0], 1.0)]
            dirs, weights = zip(*filtered)
            total = sum(weights)
            if total <= 0:
                raise ValueError("mel_weights sum must be > 0")
            self.mel_dirs = list(dirs)
            self.mel_weights = [w / total for w in weights]
        elif base_mels_path is not None:
            self.mel_dirs = [base_mels_path]
            self.mel_weights = [1.0]

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if 'jvs031' in filename and 'VOICEACTRESS100_004' in filename:
            print(f"問題のファイル: {filename}")
            print(f"存在するか: {os.path.exists(filename)}")
        # 修正を試みる
            if filename.endswith('.wav.wav'):
                print(f"拡張子修正を試みます")
                alt_filename = filename[:-4]
                print(f"代替ファイル: {alt_filename}")
                print(f"代替が存在するか: {os.path.exists(alt_filename)}")
                if os.path.exists(alt_filename):
                    filename = alt_filename
    
        audio, sampling_rate = load_wav(filename)
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            if not self.mel_dirs:
                raise ValueError("Fine-tuning requires mel directories to be specified.")
            stem = os.path.splitext(os.path.split(filename)[-1])[0]
            mel_dir = random.choices(self.mel_dirs, weights=self.mel_weights, k=1)[0]
            mel_path = os.path.join(mel_dir, stem + '.npy')
            if not os.path.exists(mel_path):
                raise FileNotFoundError(f"Mel file not found: {mel_path}")
            mel = np.load(mel_path)
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
