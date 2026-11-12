"""Builds tests/fixtures/pan2_vae_tiny_reference.pt with diffusers' own AutoencoderKLWan: a tiny VAE in the layout of
the Wan 2.2 TI2V-5B one that PAN 2 uses (residual blocks, 2x2 patches, time downsampled after the first stage), its
random weights in the diffusers layout, a short clip, its latent and the decoded clip, as the reference for Forge Neo's
Wan22VAE built by forge_pan2.native.vae.

    D:/pan2-forge-dev/ref-venv/Scripts/python.exe tools/make_vae_reference.py
"""

from pathlib import Path

import torch
from diffusers import AutoencoderKLWan

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pan2_vae_tiny_reference.pt"

# the released configuration, scaled down
CONFIG = dict(base_dim=16, decoder_base_dim=24, z_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=2, attn_scales=[],
              temperal_downsample=[False, True, True], dropout=0.0, is_residual=True, in_channels=12, out_channels=12,
              patch_size=2, scale_factor_spatial=16, scale_factor_temporal=4,
              latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
FRAMES, HEIGHT, WIDTH = 9, 32, 48


def main():
    torch.manual_seed(0)
    model = AutoencoderKLWan(**CONFIG).eval()
    with torch.no_grad():
        # weights exactly representable in fp16, so the fixture can store them at half the size
        for p in model.parameters():
            p.copy_(p.half().float())
    g = torch.Generator().manual_seed(1)
    video = torch.rand(1, 3, FRAMES, HEIGHT, WIDTH, generator=g) * 2 - 1
    image = video[:, :, :1]
    with torch.no_grad():
        latent = model.encode(video).latent_dist.mode()
        image_latent = model.encode(image).latent_dist.mode()
        decoded = model.decode(latent).sample
    print(f"latent {tuple(latent.shape)}, image latent {tuple(image_latent.shape)}, decoded {tuple(decoded.shape)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": {k: v.half() for k, v in model.state_dict().items()}, "video": video, "latent": latent,
                "image_latent": image_latent, "decoded": decoded}, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
