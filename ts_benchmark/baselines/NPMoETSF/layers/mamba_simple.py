import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DWT1D_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('dec_lo', torch.tensor([1 / sqrt(2), 1 / sqrt(2)]).view(1, 1, 2))
        self.register_buffer('dec_hi', torch.tensor([1 / sqrt(2), -1 / sqrt(2)]).view(1, 1, 2))

    def forward(self, x):
        # x: [Batch, 1, Seq_Len]
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1), mode='reflect')
        lo = F.conv1d(x, self.dec_lo, stride=2)
        hi = F.conv1d(x, self.dec_hi, stride=2)
        return lo, hi


class HybridDecompExtractor(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = 25
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
        self.dwt = DWT1D_Extractor()
        self.half_len = (seq_len + 1) // 2
        self.lin_trend = nn.Linear(seq_len, pred_len)
        self.lin_res_lo = nn.Linear(self.half_len, pred_len)
        self.lin_res_hi = nn.Linear(self.half_len, pred_len)
        nn.init.zeros_(self.lin_trend.weight)
        nn.init.zeros_(self.lin_trend.bias)
        nn.init.zeros_(self.lin_res_lo.weight)
        nn.init.zeros_(self.lin_res_lo.bias)
        nn.init.zeros_(self.lin_res_hi.weight)
        nn.init.zeros_(self.lin_res_hi.bias)

    def forward(self, x):
        B, L, _ = x.shape
        if B == 0:
            return torch.zeros(0, self.pred_len, 1, device=x.device)
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size - 1 - pad_l
        x_trans = x.permute(0, 2, 1)  # [B, 1, L]

        front = x_trans[:, :, 0:1].repeat(1, 1, pad_l)
        end = x_trans[:, :, -1:].repeat(1, 1, pad_r)
        x_pad = torch.cat([front, x_trans, end], dim=-1)

        trend = self.avg(x_pad)  # [B, 1, L]

        res = x_trans - trend
        res_mean = torch.mean(res, dim=-1, keepdim=True)
        res_std = torch.std(res, dim=-1, keepdim=True) + 1e-5
        res_norm = (res - res_mean) / res_std

        res_lo, res_hi = self.dwt(res_norm)

        pred_trend = self.lin_trend(trend.squeeze(1))

        pred_res_lo = self.lin_res_lo(res_lo.squeeze(1))
        pred_res_hi = self.lin_res_hi(res_hi.squeeze(1))
        pred_res_norm = pred_res_lo + pred_res_hi
        pred_res = pred_res_norm * res_std.squeeze(1)  # + res_mean? 通常残差均值接近0，只恢复方差更稳

        # ==========================
        # 4. Final Fusion
        # ==========================
        total_out = pred_trend + pred_res

        return total_out.unsqueeze(-1)

MultiScaleExtractor = HybridDecompExtractor