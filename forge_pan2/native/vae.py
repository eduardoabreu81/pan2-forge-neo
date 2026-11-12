"""The PAN 2 VAE: the Wan 2.2 TI2V-5B VAE (48 latent channels, 16x in space, 4x in time), released unchanged.

Forge Neo already carries its code, ComfyUI's vae2_2.py, as Wan22VAE (backend/nn/qwen21_vae.py, which Qwen-Image 2.1
uses with 64 channels); here it is built with the TI2V-5B settings. Files come in two layouts: diffusers'
AutoencoderKLWan (IFM/PAN2, and its fp16 copy) and the original one (Comfy-Org's wan2.2_vae.safetensors). The diffusers
names are mapped to the original ones, as ComfyUI's PAN 2 support does (vae2_2.convert_wan_diffusers_vae).
"""

import json
import os

import torch

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "PAN2", "vae", "config.json")
LATENT_CHANNELS = 48
SPATIAL = 16
TEMPORAL = 4

MISSING_VAE = "PAN 2 needs its VAE: select the Wan 2.2 VAE (wan2.2_vae, 48 channels) under VAE / Text Encoder"
WRONG_VAE = ("This VAE has {channels} latent channels: PAN 2 uses the Wan 2.2 TI2V-5B VAE with 48 (wan2.2_vae), not the "
             "Wan 2.1 one")

with open(CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = json.load(f)


def wan22_settings() -> dict:
    """Wan22VAE's arguments for the PAN 2 VAE."""
    return {"dim": CONFIG["base_dim"], "dec_dim": CONFIG["decoder_base_dim"], "z_dim": CONFIG["z_dim"],
            "dim_mult": CONFIG["dim_mult"], "num_res_blocks": CONFIG["num_res_blocks"], "attn_scales": CONFIG["attn_scales"],
            "temporal_downsample": CONFIG["temperal_downsample"], "dropout": 0.0, "image_channels": 3,
            "patch_size": CONFIG["patch_size"], "temporal_kernel": 3}


def is_diffusers_layout(sd) -> bool:
    return "decoder.mid_block.resnets.0.norm1.gamma" in sd


def is_original_layout(sd) -> bool:
    return "decoder.head.0.gamma" in sd and "conv1.weight" in sd


def is_wan_vae(sd) -> bool:
    return is_diffusers_layout(sd) or is_original_layout(sd)


def latent_channels(sd) -> int:
    # the encoder's last convolution gives mean and log-variance: twice the latent channels
    weight = sd.get("encoder.conv_out.weight") if is_diffusers_layout(sd) else sd.get("encoder.head.2.weight")
    return int(weight.shape[0]) // 2


def convert_diffusers(sd: dict) -> dict:
    """diffusers AutoencoderKLWan (Wan 2.2 residual layout) key names -> the original ones (ComfyUI vae2_2.py)."""
    resnet = {"norm1": "residual.0", "conv1": "residual.2", "norm2": "residual.3", "conv2": "residual.6", "conv_shortcut": "shortcut"}
    counts = {}
    for k in sd:
        p = k.split(".")
        if len(p) > 4 and p[1] in ("down_blocks", "up_blocks") and p[3] == "resnets":
            counts[(p[0], p[2])] = max(counts.get((p[0], p[2]), 0), int(p[4]) + 1)
    out = {}
    for k, v in sd.items():
        p = k.split(".")
        if p[0] == "quant_conv":
            n = ["conv1"] + p[1:]
        elif p[0] == "post_quant_conv":
            n = ["conv2"] + p[1:]
        elif p[1] == "conv_in":
            n = [p[0], "conv1"] + p[2:]
        elif p[1] == "conv_out":
            n = [p[0], "head", "2"] + p[2:]
        elif p[1] == "norm_out":
            n = [p[0], "head", "0"] + p[2:]
        elif p[1] == "mid_block":
            if p[2] == "resnets":
                n = [p[0], "middle", str(int(p[3]) * 2), resnet[p[4]]] + p[5:]
            else:
                n = [p[0], "middle", "1"] + p[4:]
        else:
            group = "downsamples" if p[1] == "down_blocks" else "upsamples"
            if p[3] == "resnets":
                n = [p[0], group, p[2], group, p[4], resnet[p[5]]] + p[6:]
            else:  # downsampler / upsampler: it follows the resnets of its block
                n = [p[0], group, p[2], group, str(counts[(p[0], p[2])])] + p[4:]
        out[".".join(n)] = v
    return out


def to_original(sd: dict) -> dict:
    """A PAN 2 VAE state dict in the original layout, checked to be the 48-channel Wan 2.2 one."""
    if not isinstance(sd, dict) or not is_wan_vae(sd):
        raise ValueError(MISSING_VAE)
    channels = latent_channels(sd)
    if channels != LATENT_CHANNELS:
        raise ValueError(WRONG_VAE.format(channels=channels))
    return convert_diffusers(sd) if is_diffusers_layout(sd) else dict(sd)


def latent_statistics() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, LATENT_CHANNELS, 1, 1, 1)
    return torch.tensor(CONFIG["latents_mean"]).view(shape), torch.tensor(CONFIG["latents_std"]).view(shape)
