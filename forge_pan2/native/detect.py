"""PAN 2 checkpoints, told apart by their keys and read into the transformer's settings (ComfyUI model_detection.py).

Only the shapes are read, so the same code works on a loaded state dict and on a file's header. Quantized files keep the
output features of a weight in shape[0], so the ratios come from biases or from that axis.
"""

import math
import re

PREFIX = "model.diffusion_model."
MARKERS = ("transformer_blocks.0.scale_shift_table_context", "context_refiner.0.attn.to_q.weight")


def is_pan2(keys, prefix: str = "") -> bool:
    return all(prefix + marker in keys for marker in MARKERS)


def _count(keys, prefix: str) -> int:
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    return 1 + max((int(m.group(1)) for k in keys if (m := pattern.match(k))), default=-1)


def dit_config(shapes: dict, prefix: str = "") -> dict:
    """The PAN2Transformer3DModel settings for a DiT given as {key: shape}."""
    x_embedder = shapes[prefix + "x_embedder.weight"]
    dim = x_embedder[0]
    head_dim = shapes[prefix + "transformer_blocks.0.attn.norm_q.weight"][0]
    patch = list(x_embedder[2:])
    return {
        "in_channels": x_embedder[1],
        "patch_size": patch,
        "out_channels": shapes[prefix + "proj_out.bias"][0] // math.prod(patch),
        "attention_head_dim": head_dim,
        "num_attention_heads": dim // head_dim,
        "num_layers": _count(shapes, prefix + "transformer_blocks."),
        "num_refiner_layers": _count(shapes, prefix + "context_refiner."),
        "mlp_ratio": shapes[prefix + "transformer_blocks.0.ff.net.0.proj.bias"][0] / dim,
        "context_mlp_ratio": shapes[prefix + "transformer_blocks.0.ff_context.net.0.proj.bias"][0] / dim,
        "refiner_mlp_ratio": shapes[prefix + "context_refiner.0.ff.net.0.proj.weight"][0] / (2 * dim),
        "text_embed_dim": shapes[prefix + "context_embedder.0.weight"][0],
    }
