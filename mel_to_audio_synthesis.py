# mel_to_audio_synthesis.py
import torch
import numpy as np
import os
import json
import soundfile as sf
import matplotlib.pyplot as plt
from models import Generator
import argparse
import gc

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'...")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def safe_remove_weight_norm(module):
    """weight_normが適用されている場合のみ削除する"""
    try:
        from torch.nn.utils import remove_weight_norm
        remove_weight_norm(module)
        return True
    except ValueError:
        return False
    except AttributeError:
        return False

def load_mel_spectrogram(mel_path):
    """メルスペクトログラムファイルを読み込み"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Mel file not found: {mel_path}")
    
    mel_np = np.load(mel_path)
    print(f"Loaded mel spectrogram from: {mel_path}")
    print(f"Mel shape: {mel_np.shape}")
    print(f"Mel range: {mel_np.min():.3f} to {mel_np.max():.3f}")
    
    return mel_np

def process_mel_file(mel_path, h, generator, device, output_dir):
    """単一のメルスペクトログラムファイルを処理"""
    basename = os.path.splitext(os.path.basename(mel_path))[0]
    if basename.endswith('_mel'):
        basename = basename[:-4]  # _mel サフィックスを削除
    
    print(f"\n=== Processing: {mel_path} ===")
    
    try:
        # メルスペクトログラムを読み込み
        mel_np = load_mel_spectrogram(mel_path)
        
        # Tensorに変換
        mel_tensor = torch.FloatTensor(mel_np).to(device)
        
        # 次元の調整
        if mel_tensor.dim() == 2:
            # [mels, time] -> [1, mels, time]
            mel_tensor = mel_tensor.unsqueeze(0)
        elif mel_tensor.dim() == 3:
            # [1, mels, time] or [batch, mels, time]
            if mel_tensor.size(0) != 1:
                print(f"Warning: Batch size is {mel_tensor.size(0)}, using first sample")
                mel_tensor = mel_tensor[0:1]
        else:
            raise ValueError(f"Invalid mel spectrogram dimensions: {mel_tensor.shape}")
        
        print(f"Mel tensor shape for synthesis: {mel_tensor.shape}")
        
        # 設定値との整合性チェック
        expected_mels = h.num_mels
        actual_mels = mel_tensor.size(1)
        if actual_mels != expected_mels:
            print(f"Warning: Mel bins mismatch. Expected: {expected_mels}, Got: {actual_mels}")
            if actual_mels > expected_mels:
                print(f"Truncating to {expected_mels} mel bins")
                mel_tensor = mel_tensor[:, :expected_mels, :]
            else:
                print(f"Padding to {expected_mels} mel bins")
                padding = expected_mels - actual_mels
                mel_tensor = torch.nn.functional.pad(mel_tensor, (0, 0, 0, padding), 'constant', 0)
        
        with torch.no_grad():
            # 音声合成
            print("Generating audio from mel spectrogram...")
            audio_generated = generator(mel_tensor)
            audio_output = audio_generated.squeeze().cpu().numpy()
            
            print(f"Generated audio shape: {audio_output.shape}")
            print(f"Generated audio range: {audio_output.min():.3f} to {audio_output.max():.3f}")
            print(f"Generated audio duration: {len(audio_output)/h.sampling_rate:.2f} seconds")
        
        # 音声を保存
        output_path = os.path.join(output_dir, f"{basename}_from_mel.wav")
        sf.write(output_path, audio_output, h.sampling_rate)
        print(f"Generated audio saved to: {output_path}")
        
        # メルスペクトログラムを可視化
        plt.figure(figsize=(12, 4))
        plt.imshow(mel_tensor.squeeze().cpu().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"Input Mel Spectrogram - {basename}")
        plt.xlabel("Time")
        plt.ylabel("Mel Bins")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{basename}_input_mel.png"), dpi=150)
        plt.close()
        
        # 統計情報を保存
        stats = {
            'input_file': mel_path,
            'mel_shape': mel_tensor.shape,
            'mel_range': [float(mel_tensor.min()), float(mel_tensor.max())],
            'audio_shape': audio_output.shape,
            'audio_range': [float(audio_output.min()), float(audio_output.max())],
            'duration_seconds': len(audio_output) / h.sampling_rate,
            'sampling_rate': h.sampling_rate
        }
        
        stats_path = os.path.join(output_dir, f"{basename}_synthesis_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return True, basename, stats
        
    except Exception as e:
        print(f"Error processing {mel_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input .npy mel file or directory with .npy files')
    parser.add_argument('--checkpoint_file', required=True, help='Generator checkpoint file')
    parser.add_argument('--config', default='config_custom.json', help='HiFi-GAN config file')
    parser.add_argument('--output_dir', default='mel_synthesis_result', help='Output directory')
    parser.add_argument('--max_files', default=20, type=int, help='Maximum number of files to process (if directory)')
    args = parser.parse_args()
    
    # 設定ファイルを読み込む
    with open(args.config) as f:
        data = f.read()
    h = AttrDict(json.loads(data))
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # GPU使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    
    # 入力ファイル/ディレクトリの処理
    if os.path.isfile(args.input) and args.input.endswith('.npy'):
        # 単一ファイル
        mel_files = [args.input]
        print(f"Processing single mel file: {args.input}")
    elif os.path.isdir(args.input):
        # ディレクトリ
        mel_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                     if f.lower().endswith('.npy')]
        
        if not mel_files:
            print(f"No .npy files found in {args.input}")
            return
        
        # ファイル数を制限
        if len(mel_files) > args.max_files:
            print(f"Found {len(mel_files)} files, processing {args.max_files} files")
            mel_files = mel_files[:args.max_files]
        
        print(f"Processing {len(mel_files)} mel files from directory")
    else:
        print(f"Invalid input: {args.input} (must be .npy file or directory)")
        return
    
    with torch.no_grad():
        # モデルを初期化
        generator = Generator(h).to(device)
        
        # チェックポイントを読み込む
        state_dict_g = load_checkpoint(args.checkpoint_file, device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        
        # weight_norm を安全に削除
        print("Removing weight norm...")
        print("Skipping conv_pre (no weight_norm applied)")
        
        removed_count = 0
        for i, up_layer in enumerate(generator.ups):
            if safe_remove_weight_norm(up_layer):
                removed_count += 1
        
        for i, resblock in enumerate(generator.resblocks):
            try:
                resblock.remove_weight_norm()
            except (ValueError, AttributeError):
                pass
        
        if safe_remove_weight_norm(generator.conv_post):
            removed_count += 1
        
        print(f"Weight norm removal completed. Removed from {removed_count} layers.")
        
        # 各ファイルを処理
        success_count = 0
        processed_files = []
        all_stats = []
        
        for mel_file in mel_files:
            success, basename, stats = process_mel_file(mel_file, h, generator, device, args.output_dir)
            if success:
                success_count += 1
                processed_files.append(basename)
                all_stats.append(stats)
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {success_count}/{len(mel_files)} files")
        
        # 結果HTMLを生成
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HiFi-GAN Mel-to-Audio Synthesis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .file-section {{ 
                    border: 1px solid #ddd; 
                    margin: 20px 0; 
                    padding: 15px; 
                    border-radius: 5px; 
                }}
                .audio-container {{ display: flex; gap: 10px; align-items: center; margin: 10px 0; }}
                .audio-label {{ min-width: 150px; font-weight: bold; }}
                audio {{ width: 100%; }}
                img {{ max-width: 100%; height: auto; margin-top: 10px; }}
                .info {{ 
                    background-color: #e8f5e8; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    border-left: 4px solid #4CAF50;
                }}
                .stats {{ 
                    background-color: #f0f0f0; 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                    font-family: monospace;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <h1>HiFi-GAN Mel-to-Audio Synthesis</h1>
            
            <div class="info">
                <strong>Mel Spectrogram to Audio Synthesis</strong>
                <br>• Direct synthesis from .npy mel spectrograms
                <br>• No audio preprocessing required
                <br>• Processed {success_count} files successfully
                <br>• Model config: {h.num_mels} mels, {h.sampling_rate}Hz sampling rate
            </div>
        """
        
        # 各ファイルの結果を追加
        for i, (basename, stats) in enumerate(zip(processed_files, all_stats)):
            html_content += f"""
            <div class="file-section">
                <h2>File {i+1}: {basename}</h2>
                
                <div class="stats">
                    Input mel shape: {stats['mel_shape']}<br>
                    Mel range: {stats['mel_range'][0]:.3f} to {stats['mel_range'][1]:.3f}<br>
                    Generated audio duration: {stats['duration_seconds']:.2f} seconds<br>
                    Audio range: {stats['audio_range'][0]:.3f} to {stats['audio_range'][1]:.3f}
                </div>
                
                <div class="audio-container">
                    <div class="audio-label">Generated Audio:</div>
                    <audio controls>
                        <source src="{basename}_from_mel.wav" type="audio/wav">
                    </audio>
                </div>
                
                <img src="{basename}_input_mel.png" alt="Input Mel Spectrogram - {basename}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(os.path.join(args.output_dir, "mel_synthesis_results.html"), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Results saved to: {args.output_dir}")
        print("Open 'mel_synthesis_results.html' to view all results")
        
        # 全体の統計を保存
        overall_stats = {
            'total_files': len(mel_files),
            'successful_syntheses': success_count,
            'model_config': {
                'num_mels': h.num_mels,
                'sampling_rate': h.sampling_rate,
                'n_fft': h.n_fft,
                'hop_size': h.hop_size,
                'win_size': h.win_size
            },
            'individual_stats': all_stats
        }
        
        with open(os.path.join(args.output_dir, "overall_synthesis_stats.json"), 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        # GPUメモリを解放
        del generator, state_dict_g
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()