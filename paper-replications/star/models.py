"""
STAR: Spatio-Temporal Attention-based Regression model for C-MAPSS RUL prediction.

Architecture overview:
  DimensionWisePatchEmbed -> [TwoStageAttentionEncoder + PatchMerging] x (n_scales-1)
  -> TwoStageAttentionEncoder (deepest) -> Decoder with cross-attention at each scale
  -> Per-scale heads -> Final MLP -> scalar in [0, 1]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormTransformerLayer(nn.Module):
    """Single Transformer encoder layer with Pre-LN: LN->MHA->residual->LN->FFN->residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # x: (N, seq, d_model)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class PreNormCrossAttnLayer(nn.Module):
    """Cross-attention layer: LN->MHA(Q=x, K=V=context)->residual->LN->FFN->residual."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        # x: (N, seq, d_model), context: (N, seq_ctx, d_model)
        normed_x = self.norm1(x)
        normed_ctx = self.norm_ctx(context)
        attn_out, _ = self.cross_attn(normed_x, normed_ctx, normed_ctx)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class DimensionWisePatchEmbed(nn.Module):
    """
    Embed temporal patches per sensor dimension.

    Input: (B, T, D)
    Output: (B, K, D, d_model) where K = T // patch_length
    """

    def __init__(self, T: int, D: int, patch_length: int, d_model: int):
        super().__init__()
        assert T % patch_length == 0, f"T={T} must be divisible by patch_length={patch_length}"
        self.K = T // patch_length
        self.D = D
        self.patch_length = patch_length
        self.d_model = d_model

        self.proj = nn.Linear(patch_length, d_model)
        # Learnable positional embedding: (K, D, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(self.K, D, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        # -> (B, D, T)
        x = x.transpose(1, 2)
        # -> (B, D, K, patch_length)
        x = x.reshape(B, D, self.K, self.patch_length)
        # -> (B, K, D, patch_length)
        x = x.permute(0, 2, 1, 3)
        # -> (B, K, D, d_model)
        x = self.proj(x)
        x = x + self.pos_embed.unsqueeze(0)
        return x


class TwoStageAttentionEncoderBlock(nn.Module):
    """
    Two-stage attention encoder block.
    Stage 1: temporal attention over K patches (per sensor).
    Stage 2: sensor-wise attention over D sensors (per patch).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.temporal_layer = PreNormTransformerLayer(d_model, n_heads, dropout)
        self.sensor_layer = PreNormTransformerLayer(d_model, n_heads, dropout)

    def forward(self, x):
        # x: (B, K, D, d_model)
        B, K, D, dm = x.shape

        # Stage 1: temporal (across K patches, for each sensor independently)
        x_t = x.reshape(B * D, K, dm)
        x_t = self.temporal_layer(x_t)
        x = x_t.reshape(B, K, D, dm)

        # Stage 2: sensor-wise (across D sensors, for each patch independently)
        x_s = x.permute(0, 1, 2, 3).reshape(B * K, D, dm)
        x_s = self.sensor_layer(x_s)
        x = x_s.reshape(B, K, D, dm)

        return x


class PatchMerging(nn.Module):
    """
    Merge adjacent patch pairs: K -> K//2 (drops last if K is odd).
    (B, K, D, d_model) -> (B, K//2, D, d_model)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x):
        # x: (B, K, D, d_model)
        K = x.shape[1]
        if K % 2 == 1:
            x = x[:, :-1]  # drop last patch if K is odd
        x0 = x[:, 0::2]   # (B, K//2, D, d_model)
        x1 = x[:, 1::2]   # (B, K//2, D, d_model)
        merged = torch.cat([x0, x1], dim=-1)  # (B, K//2, D, 2*d_model)
        return self.proj(merged)


class TwoStageAttentionDecoderBlock(nn.Module):
    """
    Two-stage attention decoder block.
    Self-attention (two-stage: temporal then sensor-wise).
    Cross-attention to encoder features (two-stage).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention stages
        self.self_temporal = PreNormTransformerLayer(d_model, n_heads, dropout)
        self.self_sensor = PreNormTransformerLayer(d_model, n_heads, dropout)
        # Cross-attention stages
        self.cross_temporal = PreNormCrossAttnLayer(d_model, n_heads, dropout)
        self.cross_sensor = PreNormCrossAttnLayer(d_model, n_heads, dropout)

    def forward(self, x, enc_feats):
        """
        x: (B, K_dec, D, d_model) - decoder sequence
        enc_feats: (B, K_enc, D, d_model) - encoder features at same scale
        """
        B, K_dec, D, dm = x.shape
        B, K_enc, D, dm = enc_feats.shape

        # Self-attention: temporal
        x_t = x.reshape(B * D, K_dec, dm)
        x_t = self.self_temporal(x_t)
        x = x_t.reshape(B, K_dec, D, dm)

        # Self-attention: sensor-wise
        x_s = x.reshape(B * K_dec, D, dm)
        x_s = self.self_sensor(x_s)
        x = x_s.reshape(B, K_dec, D, dm)

        # Cross-attention: temporal (each sensor attends to encoder patches)
        x_t = x.reshape(B * D, K_dec, dm)
        enc_t = enc_feats.permute(0, 2, 1, 3).reshape(B * D, K_enc, dm)
        x_t = self.cross_temporal(x_t, enc_t)
        x = x_t.reshape(B, K_dec, D, dm)

        # Cross-attention: sensor-wise (each patch position attends to encoder sensors)
        # Use minimum K to handle potential K mismatch
        K_min = min(K_dec, K_enc)
        x_s = x[:, :K_min].reshape(B * K_min, D, dm)
        enc_s = enc_feats[:, :K_min].reshape(B * K_min, D, dm)
        x_s = self.cross_sensor(x_s, enc_s)
        if K_dec > K_min:
            x = torch.cat([x_s.reshape(B, K_min, D, dm), x[:, K_min:]], dim=1)
        else:
            x = x_s.reshape(B, K_dec, D, dm)

        return x


def make_sinusoidal_pe(K: int, D: int, d_model: int, device: torch.device) -> torch.Tensor:
    """
    Create sinusoidal positional encoding of shape (1, K, D, d_model).
    Uses standard Vaswani-style sin/cos along the d_model dimension,
    applied identically across D sensors.
    """
    pe_1d = torch.zeros(K, d_model, device=device)
    position = torch.arange(0, K, device=device).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe_1d[:, 0::2] = torch.sin(position * div_term)
    pe_1d[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
    # Expand to (1, K, D, d_model)
    pe = pe_1d.unsqueeze(1).expand(K, D, d_model).unsqueeze(0)
    return pe


class ScalePredHead(nn.Module):
    """Per-scale prediction head: mean-pool over K and D, then MLP -> scalar."""

    def __init__(self, K: int, D: int, d_model: int):
        super().__init__()
        # Pool over K and D dims, then project from d_model to scalar
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        # x: (B, K, D, d_model)
        # Mean pool over K and D dims -> (B, d_model)
        x_pooled = x.mean(dim=(1, 2))
        return self.head(x_pooled)  # (B, 1)


class STAR(nn.Module):
    """
    STAR: Spatio-Temporal Attention-based Regression.

    Parameters
    ----------
    T : int - window length (temporal)
    D : int - number of sensors (14)
    patch_length : int - patch size (4)
    n_scales : int - number of encoder/decoder scales
    d_model : int - embedding dimension
    n_heads : int - attention heads
    dropout : float - dropout rate
    """

    def __init__(
        self,
        T: int,
        D: int = 14,
        patch_length: int = 4,
        n_scales: int = 3,
        d_model: int = 128,
        n_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.T = T
        self.D = D
        self.patch_length = patch_length
        self.n_scales = n_scales
        self.d_model = d_model

        K0 = T // patch_length  # patches at scale 0

        # Compute K at each scale (accounting for odd-K truncation in PatchMerging)
        self.K_at_scale = [K0]
        for _ in range(n_scales - 1):
            self.K_at_scale.append(self.K_at_scale[-1] // 2)

        # Patch embedding
        self.patch_embed = DimensionWisePatchEmbed(T, D, patch_length, d_model)

        # Encoder blocks (one per scale)
        self.encoder_blocks = nn.ModuleList([
            TwoStageAttentionEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_scales)
        ])

        # Patch merging blocks (n_scales - 1)
        self.patch_merging = nn.ModuleList([
            PatchMerging(d_model)
            for _ in range(n_scales - 1)
        ])

        # Decoder blocks (one per scale)
        self.decoder_blocks = nn.ModuleList([
            TwoStageAttentionDecoderBlock(d_model, n_heads, dropout)
            for _ in range(n_scales)
        ])

        # Per-scale prediction heads
        self.scale_heads = nn.ModuleList([
            ScalePredHead(self.K_at_scale[s], D, d_model)
            for s in range(n_scales)
        ])

        # Final fusion MLP: n_scales scalars -> 1 scalar
        self.final_mlp = nn.Sequential(
            nn.Linear(n_scales, max(8, n_scales * 4)),
            nn.GELU(),
            nn.Linear(max(8, n_scales * 4), 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: (B, T, D) - input sensor windows
        returns: (B,) - predicted RUL in [0, 1] (normalized by RUL_CAP)
        """
        B = x.shape[0]
        device = x.device

        # Patch embedding
        h = self.patch_embed(x)  # (B, K0, D, d_model)

        # Encoder pass - save features at each scale
        enc_features = []
        for s in range(self.n_scales):
            h = self.encoder_blocks[s](h)
            enc_features.append(h)  # save before merging
            if s < self.n_scales - 1:
                h = self.patch_merging[s](h)

        # Decoder pass - start from sinusoidal PE at deepest scale
        deepest_scale = self.n_scales - 1
        K_deep = self.K_at_scale[deepest_scale]
        dec = make_sinusoidal_pe(K_deep, self.D, self.d_model, device)
        dec = dec.expand(B, -1, -1, -1)  # (B, K_deep, D, d_model)

        scale_preds = []
        for s in range(self.n_scales - 1, -1, -1):
            dec = self.decoder_blocks[s](dec, enc_features[s])
            # Collect per-scale prediction
            scale_preds.append(self.scale_heads[s](dec))  # (B, 1)
            # Upsample to next scale (repeat-interleave by 2 along K dim)
            if s > 0:
                dec = dec.repeat_interleave(2, dim=1)
                # Trim to match expected K at scale s-1
                K_target = self.K_at_scale[s - 1]
                dec = dec[:, :K_target]

        # scale_preds collected from deepest to shallowest
        # concat along feature dim: (B, n_scales)
        preds_cat = torch.cat(scale_preds, dim=-1)  # (B, n_scales)
        out = self.final_mlp(preds_cat).squeeze(-1)  # (B,)
        return out


def build_model(subset: str, device: torch.device) -> STAR:
    """Build STAR model for a given C-MAPSS subset using paper hyperparameters."""
    configs = {
        "FD001": dict(T=32, n_scales=3, d_model=128, n_heads=1),
        "FD002": dict(T=64, n_scales=4, d_model=64, n_heads=4),
        "FD003": dict(T=48, n_scales=1, d_model=128, n_heads=1),
        "FD004": dict(T=64, n_scales=4, d_model=256, n_heads=4),
    }
    cfg = configs[subset]
    model = STAR(
        T=cfg["T"],
        D=14,
        patch_length=4,
        n_scales=cfg["n_scales"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        dropout=0.1,
    ).to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        model = build_model(subset, device)
        n_params = count_parameters(model)
        print(f"\n{subset}: {n_params:,} parameters")

        configs = {
            "FD001": dict(T=32, B=2),
            "FD002": dict(T=64, B=2),
            "FD003": dict(T=48, B=2),
            "FD004": dict(T=64, B=2),
        }
        cfg = configs[subset]
        x = torch.randn(cfg["B"], cfg["T"], 14, device=device)
        out = model(x)
        print(f"  Input: {x.shape} -> Output: {out.shape}")
        print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        assert not torch.isnan(out).any(), "NaN in output!"
        print(f"  No NaN: OK")
