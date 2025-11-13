# mri_acoustic_model.py
# -------------------------------------------------------------
# OTN/論文設定に寄せた PyTorch 実装:
#   - EfficientNetV2-B2 (tf_* from timm), include_top=False, global avg pooling
#   - BiLSTM 1層 (hidden=640), bidirectional 出力は "加算" (sum merge)
#   - Dropout(0.5) → Linear(n_mels)
#   - 勾配チェックポイント(オプション) / 低メモリで安定学習
# -------------------------------------------------------------
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm import create_model

class GlobalAvgPool(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C)
        return torch.mean(x, dim=(2, 3))

class EffNetV2B2Backbone(nn.Module):
    """
    EfficientNetV2-B2 backbone (features only, no classifier), global avg pool -> (B, Cfeat)
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # timm の tf_efficientnetv2_b2: TensorFlow weights 互換の実装
        # include_top=False 相当で features / forward_head を使わない
        self.backbone = create_model(
            "tf_efficientnetv2_b2",
            pretrained=pretrained,
            features_only=True,   # ステージ出力を得る
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        # 最終ステージのチャネル数を取得
        self.out_channels = self.backbone.feature_info.channels()[-1]
        self.gap = GlobalAvgPool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) または (B, H, W) → RGB 3ch へブロードキャスト
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # (B,3,H,W)

        feats = self.backbone(x)[-1]  # 最終ステージ (B, C, H', W')
        f = self.gap(feats)           # (B, C)
        return f  # (B, C), C = self.out_channels

class BiLSTMSumMerge(nn.Module):
    """
    1層 BiLSTM (hidden=640)。双方向は "加算" で融合（concatではない）。
    入力: (B, T, C) -> 出力: (B, T, H), H=hidden_size
    """
    def __init__(self, in_dim: int, hidden_size: int = 640, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y, _ = self.lstm(x)                  # (B, T, 2*H)
        y_fwd, y_bwd = y.chunk(2, dim=-1)    # (B, T, H), (B, T, H)
        y_sum = y_fwd + y_bwd                # (B, T, H) ← "加算" merge
        return self.dropout(y_sum)

class OTNLikeCNNBiLSTM(nn.Module):
    """
    OTN/論文に寄せた最終モデル:
      - frame-wise CNN 特徴抽出 (EffNetV2-B2)
      - 時系列 BiLSTM (1層, H=640, 双方向加算)
      - Dropout(0.5) → Linear(n_mels)
    期待する入力形状:
      - (B, T, 1, H, W) または (B, T, H, W)
    出力:
      - (B, T, n_mels)
    """
    def __init__(
        self,
        n_mels: int = 64,
        cnn_pretrained: bool = False,
        rnn_hidden: int = 640,
        dropout: float = 0.5,
        use_checkpoint: bool = False,
        ckpt_segments: int = 2,  # checkpoint_sequential の分割数
        use_reentrant: bool = False  # PyTorch 2.5 警告対策: 明示
    ):
        super().__init__()
        self.n_mels = n_mels
        self.use_checkpoint = use_checkpoint
        self.ckpt_segments = ckpt_segments
        self.use_reentrant = use_reentrant

        self.cnn = EffNetV2B2Backbone(pretrained=cnn_pretrained)
        self.rnn = BiLSTMSumMerge(in_dim=self.cnn.out_channels, hidden_size=rnn_hidden, dropout=dropout)
        self.head = nn.Linear(rnn_hidden, n_mels)

    def _cnn_time_distributed(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 1, H, W) or (B, T, H, W)
        -> 時間次元を結合して CNN 適用 -> (B, T, Cfeat)
        """
        B, T = x.shape[0], x.shape[1]
        x_ = x.reshape(B * T, *x.shape[2:])           # (B*T, 1, H, W)
        f = self.cnn(x_)                              # (B*T, C)
        f = f.view(B, T, -1)                          # (B, T, C)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W) or (B, T, H, W)
        if self.use_checkpoint and self.training:
            # 時間次元を分割して逐次 CNN を checkpoint で処理（メモリ節約）
            B, T = x.shape[:2]
            # 分割ユニットを list[Module] に格納するため、Closure でラップ
            chunks = torch.chunk(x, self.ckpt_segments, dim=1)  # T を segment 個に分割
            feats_list = []
            for seg in chunks:
                def run_seg(inp):
                    return self._cnn_time_distributed(inp)
                # checkpoint: use_reentrant を明示（2.5以降の警告回避）
                seg_f = cp.checkpoint(run_seg, seg, use_reentrant=self.use_reentrant) if seg.requires_grad else run_seg(seg)
                feats_list.append(seg_f)
            f = torch.cat(feats_list, dim=1)          # (B, T, C)
        else:
            f = self._cnn_time_distributed(x)         # (B, T, C)

        y = self.rnn(f)                                # (B, T, H=640)
        out = self.head(y)                             # (B, T, n_mels)
        return out

# 便利関数: モデル生成
def build_acoustic_model(
    n_mels: int = 64,
    cnn_pretrained: bool = False,
    rnn_hidden: int = 640,
    dropout: float = 0.5,
    use_checkpoint: bool = False,
    ckpt_segments: int = 2,
    use_reentrant: bool = False
) -> nn.Module:
    return OTNLikeCNNBiLSTM(
        n_mels=n_mels,
        cnn_pretrained=cnn_pretrained,
        rnn_hidden=rnn_hidden,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        ckpt_segments=ckpt_segments,
        use_reentrant=use_reentrant
    )
