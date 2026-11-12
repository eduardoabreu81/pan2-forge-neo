"""PAN 2 registration for huggingface_guess, ported from ComfyUI (supported_models.PAN2, latent_formats.Wan22).

Where the parts sit in Forge Neo's state dict (see patches.py, which puts the extra files there):
- the DiT under model.diffusion_model. (the checkpoint itself)
- the Qwen3.5-9B text encoder under text_encoders.qwen35_9b.transformer.model.
- the Wan 2.2 VAE under vae., in its original layout
"""

import json
import os

import torch
from huggingface_guess.latent import Wan21
from huggingface_guess.model_list import BASE, ModelType

from . import detect, vae

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "PAN2")
TEXT_ENCODER = "qwen35_9b"
TEXT_PREFIX = f"text_encoders.{TEXT_ENCODER}.transformer."

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent_rgb.json"), encoding="utf-8") as f:
    _RGB = json.load(f)


class PAN2Latent(Wan21):
    # Wan 2.2 TI2V-5B: 48 channels, normalized by the VAE's own mean and std (Forge's Wan21 does the math)
    def __init__(self):
        super().__init__()
        self.latent_channels = vae.LATENT_CHANNELS
        self.latents_mean, self.latents_std = vae.latent_statistics()
        self.latent_rgb_factors = _RGB["factors"]
        self.latent_rgb_factors_bias = _RGB["bias"]
        self.taesd_decoder_name = None


class PAN2(BASE):
    # absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "pan2"}

    # flow matching, timestep = sigma * 1000, shifted by 9 (the reference sampler)
    sampling_settings = {"shift": 9.0}

    # ComfyUI's 2.8 under its 0.01 MB/element estimate; Forge Neo's KModel uses 0.02
    memory_usage_factor = 1.4
    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    unet_extra_config = {}
    latent_format = PAN2Latent

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {f"{TEXT_ENCODER}.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLOW


def detect_config(state_dict: dict, key_prefix: str) -> dict | None:
    """The PAN2Transformer3DModel settings for a DiT state dict, or None when it is not PAN 2."""
    if not detect.is_pan2(state_dict, key_prefix):
        return None
    shapes = {k: tuple(v.shape) for k, v in state_dict.items() if k.startswith(key_prefix)}
    return {"image_model": "pan2", **detect.dit_config(shapes, key_prefix)}
