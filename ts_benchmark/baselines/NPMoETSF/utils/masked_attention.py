import torch.nn as nn
import torch
from math import sqrt
import math
import torch.fft
from einops import rearrange
import torch.nn.functional as F
from torch.nn.functional import gumbel_softmax


class DWT1D(nn.Module):
    def __init__(self, wave='haar', max_levels=3):
        super(DWT1D, self).__init__()
        self.register_buffer('dec_lo', torch.tensor([1 / sqrt(2), 1 / sqrt(2)]).view(1, 1, 2))
        self.register_buffer('dec_hi', torch.tensor([1 / sqrt(2), -1 / sqrt(2)]).view(1, 1, 2))
        self.max_levels = max_levels

    def forward(self, x):
        coeffs_list = []
        current_x = x
        for _ in range(self.max_levels):
            if current_x.shape[-1] % 2 != 0:
                current_x = F.pad(current_x, (0, 1), mode='reflect')
            lo = F.conv1d(current_x, self.dec_lo, stride=2)
            hi = F.conv1d(current_x, self.dec_hi, stride=2)
            coeffs_list.append(hi)
            current_x = lo
        coeffs_list.append(current_x)
        return coeffs_list


class Wavelet_Mahalanobis_mask(nn.Module):
    def __init__(self, input_size, levels=3):
        super(Wavelet_Mahalanobis_mask, self).__init__()
        self.dwt = DWT1D(max_levels=levels)
        dummy_input = torch.zeros(1, 1, input_size)
        with torch.no_grad():
            coeffs = self.dwt(dummy_input)
            total_dim = sum(c.shape[-1] for c in coeffs)

        self.norm = nn.InstanceNorm1d(total_dim, affine=False)
        self.A = nn.Parameter(torch.randn(total_dim, total_dim), requires_grad=True)
        nn.init.orthogonal_(self.A)

    def calculate_prob_distance(self, X):
        B, N, L = X.shape
        X_flat = rearrange(X, 'b n l -> (b n) 1 l')
        coeffs_list = self.dwt(X_flat)
        X_wave_flat = torch.cat(coeffs_list, dim=-1).squeeze(1)

        X_wave_flat = self.norm(X_wave_flat.unsqueeze(1)).squeeze(1)

        X_wave = rearrange(X_wave_flat, '(b n) d -> b n d', b=B)
        X_wave = torch.abs(X_wave)

        X1 = X_wave.unsqueeze(2);
        X2 = X_wave.unsqueeze(1)
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        exp_dist = 1 / (dist + 1e-5)
        identity_matrices = 1 - torch.eye(N, device=X.device).unsqueeze(0)
        exp_dist = exp_dist * identity_matrices
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        p = exp_dist / (exp_max.detach() + 1e-5)
        p = (p * identity_matrices + torch.eye(N, device=X.device).unsqueeze(0)) * 0.99
        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape
        flatten_matrix = rearrange(distribution_matrix, 'b c d -> (b c d) 1')
        r_flatten_matrix = 1 - flatten_matrix + 1e-10
        log_flatten_matrix = torch.log(flatten_matrix + 1e-10)
        log_r_flatten_matrix = torch.log(r_flatten_matrix)
        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)
        resample_matrix = rearrange(resample_matrix[..., 0], '(b c d) -> b c d', b=b, c=c, d=d)
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        sample = self.bernoulli_gumbel_rsample(p)
        return sample.unsqueeze(1)


class Mahalanobis_mask(nn.Module):
    def __init__(self, input_size): super().__init__()

    def forward(self, X): pass


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model);
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout);
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        if self.norm is not None: x = self.norm(x)
        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale;
        self.mask_flag = mask_flag;
        self.output_attention = output_attention;
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            large_negative = -1e9
            attention_mask = torch.where(attn_mask > 0.5, torch.zeros_like(scores),
                                         torch.ones_like(scores) * large_negative)
            scores = scores + attention_mask
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->bhld", A, values)
        return (V.contiguous(), A) if self.output_attention else (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads);
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads  # ============ 修复点：保存 n_heads ============

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        H = self.n_heads  # ============ 修复点：直接调用 self.n_heads ============

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, keys.shape[1], H, -1)
        values = self.value_projection(values).view(B, values.shape[1], H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.out_projection(out), attn