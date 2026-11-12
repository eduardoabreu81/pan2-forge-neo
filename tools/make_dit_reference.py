"""Builds tests/fixtures/pan2_dit_tiny_reference.pt with diffusers' own PAN2Transformer3DModel: a tiny model in the
PAN 2 layout, its random weights, inputs and outputs, as the reference for forge_pan2.native.transformer.

Needs diffusers with PAN 2 (main, 0.42.dev) in a separate environment (CPU is enough):

    D:/pan2-forge-dev/ref-venv/Scripts/python.exe tools/make_dit_reference.py
"""

from pathlib import Path

import torch
from diffusers import PAN2Transformer3DModel

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pan2_dit_tiny_reference.pt"

# the released configuration, scaled down: 4 heads x 16, rotary axes 4 / 6 / 6, latent 4 channels (+4 +1 mask)
CONFIG = dict(patch_size=(1, 2, 2), in_channels=9, out_channels=4, num_attention_heads=4, attention_head_dim=16,
              num_layers=2, num_refiner_layers=2, mlp_ratio=8.0, context_mlp_ratio=4.0, refiner_mlp_ratio=4.0,
              text_embed_dim=24, rope_axes_dim=(4, 6, 6), rope_theta=256.0, rope_temporal_scale=16 / 24)
F, H, W = 3, 4, 6


def main():
    torch.manual_seed(0)
    model = PAN2Transformer3DModel(**CONFIG).eval()
    with torch.no_grad():
        # the zero-initialized tables would hide the text modulation and the type embedding
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.ndim == 1 else 0.2))
    g = torch.Generator().manual_seed(1)
    cases = []
    for name, tokens, timestep in (("t2v", 5, 700.0), ("long_text", 9, 250.0)):
        x = torch.randn(1, CONFIG["in_channels"], F, H, W, generator=g)
        context = torch.randn(1, tokens, CONFIG["text_embed_dim"], generator=g)
        t = torch.tensor([timestep])
        with torch.no_grad():
            out = model(hidden_states=x, timestep=t, encoder_hidden_states=context, return_dict=False)[0]
        cases.append({"name": name, "x": x, "timestep": t, "context": context, "out": out})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": model.state_dict(), "cases": cases}, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
