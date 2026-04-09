import torch
import torch.nn as nn
from ts_benchmark.baselines.NPMoETSF.layers.mamba_simple import MambaExtractor


class LinearPatternExtractor(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=32):
        super(LinearPatternExtractor, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.seasonal_mamba = MambaExtractor(seq_len, pred_len + seq_len, d_model=d_model)
        self.trend_mamba = MambaExtractor(seq_len, pred_len + seq_len, d_model=d_model)

        self.relu = nn.ReLU()

    def forward(self, x):

        kernel_size = 25
        pad_l = (kernel_size - 1) // 2
        pad_r = kernel_size - 1 - pad_l

        front = x[:, 0:1, :].repeat(1, pad_l, 1)
        end = x[:, -1:, :].repeat(1, pad_r, 1)
        x_pad = torch.cat([front, x, end], dim=1)

        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        trend_part = avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal_part = x - trend_part
        trend_out = self.trend_mamba(trend_part)
        seasonal_out = self.seasonal_mamba(seasonal_part)
        total_out = trend_out + seasonal_out
        pred = total_out[:, self.seq_len:, :]  # [Batch, Pred_Len, 1]
        return pred