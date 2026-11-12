"""The PAN 2 diffusion transformer, ported from ComfyUI comfy/ldm/pan2/model.py (speedyrulz/ComfyUI), which follows
diffusers' PAN2Transformer3DModel and keeps its parameter names: the released checkpoint loads as it is.

A stack of joint-attention blocks over [video; text]. Each stream has its own projections and feed-forward network;
the video stream is modulated by the timestep, the text stream by a learned table, and only the video stream is
rotated. The text first goes through four self-attention refiner blocks.

Inference only. Linear, Conv3d and RMSNorm are plain torch.nn names, which Forge Neo's loader swaps for its own
operations while it builds the model (dtype, quantization, offload).
"""

import math

import torch
from backend.attention import attention_function
from backend.memory_management import cast_to
from torch import nn

# The reference runs the rotary embedding, the norms with their shift/scale and the gated residuals in float32 on the
# whole sequence. Each of them works on every token by itself, so they run over slices of the token axis: the same
# numbers, without several float32 copies of a long video sequence at once (ComfyUI does the same).
TOKEN_CHUNK = 8192


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    # diffusers Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0): [cos, sin]
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope_freqs(sizes, axes_dim, theta: float, temporal_scale: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin), each [T * H * W, sum(axes_dim)]; only the frame positions are rescaled."""
    grids = torch.meshgrid(*[torch.arange(s, device=device, dtype=torch.float32) for s in sizes], indexing="ij")
    cos, sin = [], []
    for i, dim in enumerate(axes_dim):
        pos = grids[i].reshape(-1)
        if i == 0:
            pos = pos * temporal_scale
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim))
        freqs = torch.outer(pos, freqs)
        cos.append(freqs.cos().repeat_interleave(2, dim=1))
        sin.append(freqs.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, dim=1), torch.cat(sin, dim=1)


def sliced(fn, x: torch.Tensor, *seq: torch.Tensor) -> torch.Tensor:
    """fn(x, *seq), a tensor of x's shape and dtype, over slices of the token axis (dim 1) that x and seq share."""
    if x.shape[1] <= TOKEN_CHUNK:
        return fn(x, *seq)
    out = torch.empty_like(x)
    for a in range(0, x.shape[1], TOKEN_CHUNK):
        b = a + TOKEN_CHUNK
        out[:, a:b] = fn(x[:, a:b], *[t[:, a:b] for t in seq])
    return out


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def apply_rope(x: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # x: [B, S, heads, head_dim]
    return sliced(_rope, x, pe[0][None, :, None, :], pe[1][None, :, None, :])


def layer_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, x.shape[-1:], None, None, eps)


def modulated_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return sliced(lambda t: (layer_norm(t.float(), eps) * (1 + scale) + shift).to(t.dtype), x)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return sliced(lambda t, u: (t.float() + u * gate).to(t.dtype), x, y)


class FeedForward(nn.Module):
    # diffusers FeedForward: net.0.proj, activation, net.2
    def __init__(self, dim: int, inner_dim: int, swiglu: bool = False, bias: bool = True):
        super().__init__()
        self.swiglu = swiglu
        proj = nn.Module()
        proj.proj = nn.Linear(dim, inner_dim * 2 if swiglu else inner_dim, bias=bias)
        self.net = nn.ModuleList([proj, nn.Identity(), nn.Linear(inner_dim, dim, bias=bias)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the wide intermediate (8x the model width for video) is only ever held for one slice of the sequence
        return sliced(self._forward, x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0].proj(x)
        if self.swiglu:
            x, gate = x.chunk(2, dim=-1)
            x = x * torch.nn.functional.silu(gate)
        else:
            x = torch.nn.functional.gelu(x, approximate="tanh")
        return self.net[2](x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float, bias: bool = True, joint: bool = False):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=bias)])
        if joint:
            self.add_q_proj = nn.Linear(dim, dim, bias=bias)
            self.add_k_proj = nn.Linear(dim, dim, bias=bias)
            self.add_v_proj = nn.Linear(dim, dim, bias=bias)
            self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)
            self.to_add_out = nn.Linear(dim, dim, bias=bias)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        return t.unflatten(-1, (self.heads, -1))

    def forward(self, x, context=None, pe=None, transformer_options={}):
        q = self.norm_q(self._heads(self.to_q(x)))
        if pe is not None:  # only the video stream is rotated
            q = apply_rope(q, pe)
        if context is not None:
            q = torch.cat([q, self.norm_added_q(self._heads(self.add_q_proj(context)))], dim=1)
        k = self.norm_k(self._heads(self.to_k(x)))
        if pe is not None:
            k = apply_rope(k, pe)
        if context is not None:
            k = torch.cat([k, self.norm_added_k(self._heads(self.add_k_proj(context)))], dim=1)
        v = self._heads(self.to_v(x))
        if context is not None:
            v = torch.cat([v, self._heads(self.add_v_proj(context))], dim=1)
        # no mask: every prompt is its own sequence without padding
        out = attention_function(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.heads, skip_reshape=True,
                                 transformer_options=transformer_options)
        del q, k, v
        if context is None:
            return self.to_out[0](out)
        n = x.shape[1]
        return self.to_out[0](out[:, :n]), self.to_add_out(out[:, n:])


class TokenRefinerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, dim_head, eps, bias=False)
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim, int(dim * mlp_ratio), swiglu=True, bias=False)

    def forward(self, x, transformer_options={}):
        x = x + self.attn(self.norm1(x), transformer_options=transformer_options)
        return x + self.ff(self.norm2(x))


class AdaLayerNorm(nn.Module):
    # holds the `linear` of diffusers' AdaLayerNormZero / AdaLayerNormContinuous
    def __init__(self, dim: int, chunks: int):
        super().__init__()
        self.linear = nn.Linear(dim, chunks * dim)

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.nn.functional.silu(temb))


class TransformerBlock(nn.Module):
    # As in the reference, the norms, their shift/scale and the gated residuals run in float32.
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, context_mlp_ratio: float, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm1 = AdaLayerNorm(dim, 6)
        self.scale_shift_table_context = nn.Parameter(torch.empty(6 * dim))
        self.attn = Attention(dim, heads, dim_head, eps, joint=True)
        self.ff = FeedForward(dim, int(dim * mlp_ratio))
        self.ff_context = FeedForward(dim, int(dim * context_mlp_ratio))

    def forward(self, x, context, temb, pe, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(temb).float().unsqueeze(1).chunk(6, dim=-1)
        table = cast_to(self.scale_shift_table_context, dtype=torch.float32, device=x.device)
        c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = table.chunk(6, dim=-1)

        norm_x = modulated_norm(x, scale_msa, shift_msa, self.eps)
        norm_c = modulated_norm(context, c_scale_msa, c_shift_msa, self.eps)
        attn_x, attn_c = self.attn(norm_x, norm_c, pe, transformer_options=transformer_options)
        del norm_x, norm_c
        x = gated_residual(x, attn_x, gate_msa)
        context = gated_residual(context, attn_c, c_gate_msa)
        del attn_x, attn_c

        norm_x = modulated_norm(x, scale_mlp, shift_mlp, self.eps)
        x = gated_residual(x, self.ff(norm_x), gate_mlp)
        del norm_x
        norm_c = modulated_norm(context, c_scale_mlp, c_shift_mlp, self.eps)
        context = gated_residual(context, self.ff_context(norm_c), c_gate_mlp)
        return x, context


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(torch.nn.functional.silu(self.linear_1(x)))


class PAN2Transformer3DModel(nn.Module):
    def __init__(self, patch_size=(1, 2, 2), in_channels=97, out_channels=48, num_attention_heads=32, attention_head_dim=128,
                 num_layers=44, num_refiner_layers=4, mlp_ratio=8.0, context_mlp_ratio=4.0, refiner_mlp_ratio=4.0,
                 text_embed_dim=4096, rope_axes_dim=(16, 56, 56), rope_theta=256.0, rope_temporal_scale=16 / 24,
                 image_model=None, **kwargs):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rope_axes_dim = tuple(rope_axes_dim)
        self.rope_theta = rope_theta
        self.rope_temporal_scale = rope_temporal_scale
        dim = num_attention_heads * attention_head_dim

        self.x_embedder = nn.Conv3d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.time_embedder = TimestepEmbedding(256, dim)
        self.context_embedder = nn.Sequential(nn.RMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, dim))
        self.context_refiner = nn.ModuleList([TokenRefinerBlock(dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5)
                                              for _ in range(num_refiner_layers)])
        self.context_type_embedding = nn.Parameter(torch.empty(dim))
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio)
                                                 for _ in range(num_layers)])
        self.norm_out = AdaLayerNorm(dim, 2)
        self.proj_out = nn.Linear(dim, math.prod(self.patch_size) * out_channels)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, transformer_options={}, **kwargs) -> torch.Tensor:
        """x: [B, in_channels, F, H, W] (noisy latent, conditioning latent and mask), timestep in [0, 1000],
        context: the text encoder hidden states [B, S, text_embed_dim]. Returns the velocity [B, out_channels, F, H, W]."""
        batch, _, frames, height, width = x.shape
        p_t, p_h, p_w = self.patch_size
        x = _pad_to_patch(x, self.patch_size)
        t_len, h_len, w_len = x.shape[2] // p_t, x.shape[3] // p_h, x.shape[4] // p_w
        device, dtype = x.device, x.dtype

        temb = self.time_embedder(timestep_embedding(timestep.reshape(-1)).to(device))  # the modulation runs in float32

        pe = rope_freqs((t_len, h_len, w_len), self.rope_axes_dim, self.rope_theta, self.rope_temporal_scale, device)
        x = self.x_embedder(x).flatten(2).transpose(1, 2).contiguous()

        context = self.context_embedder(context.to(dtype))
        for block in self.context_refiner:
            context = block(context, transformer_options=transformer_options)
        context = context + cast_to(self.context_type_embedding, dtype=context.dtype, device=device)

        for block in self.transformer_blocks:
            x, context = block(x, context, temb, pe, transformer_options=transformer_options)

        scale, shift = self.norm_out(temb).float().unsqueeze(1).chunk(2, dim=-1)
        x = self.proj_out(modulated_norm(x, scale, shift))
        x = x.reshape(batch, t_len, h_len, w_len, self.out_channels, p_t, p_h, p_w)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return x[:, :, :frames, :height, :width]


def _pad_to_patch(x: torch.Tensor, patch_size) -> torch.Tensor:
    # ComfyUI common_dit.pad_to_patch_size, circular padding of the trailing axes
    pad = []
    for size, p in zip(reversed(x.shape[2:]), reversed(patch_size)):
        pad += [0, (p - size % p) % p]
    if any(pad):
        x = torch.nn.functional.pad(x, pad, mode="circular")
    return x
