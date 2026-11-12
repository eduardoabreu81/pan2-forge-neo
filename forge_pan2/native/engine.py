"""PAN 2 diffusion engine for Forge Neo: text-to-video, and image-to-video from a first frame.

PAN 2 rides Forge Neo's own video path, the one Wan uses: the engine says it is a video model (is_wan, and the loader's
`wan` flag), so Frames becomes the length of the clip (4n + 1), the latent is [1, 48, F, H/16, W/16] and Forge decodes
and saves the video itself. What PAN 2 adds is the first frame: in img2img, Forge hands encode_first_stage the input
image repeated over the frames; its latent becomes the conditioning that the DiT stacks onto the noisy latent
(forge_dit.py), and the generation starts from noise, as ComfyUI's PAN2ImageToVideo does.
"""

import math

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from ..contracts import raise_pending_error
from . import vae as pan2_vae
from .model import PAN2, TEXT_ENCODER
from .text_engine import MAX_SEQUENCE_LENGTH, PAN2TextEngine

# ComfyUI's estimate of a Qwen3.5 text encoder's working memory, per token, in 16-bit
TEXT_MEMORY_PER_TOKEN = 2 * 1024 * 1024


def video_vae_patcher(model) -> VAE:
    """Forge's VAE object around the Wan 2.2 VAE: Forge's Wan settings assume 8x in space, this one is 16x."""
    dtype = memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    vae = VAE(model=model, dtype=dtype, is_wan=True)
    s, t = pan2_vae.SPATIAL, pan2_vae.TEMPORAL
    vae.latent_channels = pan2_vae.LATENT_CHANNELS
    vae.upscale_ratio = (lambda a: max(0, a * t - (t - 1)), s, s)
    vae.upscale_index_formula = (t, s, s)
    vae.downscale_ratio = (lambda a: max(0, math.floor((a + t - 1) / t)), s, s)
    vae.downscale_index_formula = (t, s, s)
    # Forge's Wan estimates, per latent pixel of a 16x VAE instead of an 8x one
    vae.memory_used_decode = lambda shape, dtype: ((600 if shape[2] == 1 else 2200 if shape[2] <= 4 else 7000) * shape[3]
                                                   * shape[4] * (s * s) * memory_management.dtype_size(dtype))
    return vae


def first_frame_unguided(args):
    """The given first latent frame follows the positive prediction, without guidance (the reference sampler)."""
    denoised = args["denoised"]
    cond = args["cond_denoised"]
    if denoised is not cond and denoised.ndim == 5:
        denoised[:, :, :1] = cond[:, :, :1]
    return denoised


class PAN2Engine(ForgeDiffusionEngine):
    matched_guesses = [PAN2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        text_encoder = huggingface_components["text_encoder"]
        tokenizer = huggingface_components["tokenizer"]
        clip = CLIP(model_dict={TEXT_ENCODER: text_encoder}, tokenizer_dict={TEXT_ENCODER: tokenizer})
        vae = video_vae_patcher(huggingface_components["vae"])
        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None,
                                      k_predictor=self._get_predictor(), config=estimated_config)

        self.text_processing_engine_pan2 = PAN2TextEngine(getattr(clip.cond_stage_model, TEXT_ENCODER), getattr(clip.tokenizer, TEXT_ENCODER))

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True
        self.is_wan = True
        self.is_pan2 = True
        # Forge's video path keys on this flag, which the loader only sets for Wan repositories
        dynamic_args.wan = True
        del self.ini_latent
        del self.ref_latents

    @property
    def diffusion_model(self):
        return self.forge_objects.unet.model.diffusion_model

    def clear_first_frame(self) -> None:
        self.diffusion_model.condition = None

    @property
    def has_first_frame(self) -> bool:
        return self.diffusion_model.condition is not None

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise_pending_error()
        tokens = max(len(self.text_processing_engine_pan2.tokenize(text)) for text in prompt)
        memory_management.load_models_gpu([self.forge_objects.clip.patcher], memory_required=tokens * TEXT_MEMORY_PER_TOKEN)
        return self.text_processing_engine_pan2(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        return self.text_processing_engine_pan2.condition_length(prompt), MAX_SEQUENCE_LENGTH

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        """img2img: x is the input image [B, 3, H, W] in [-1, 1], repeated over the B frames. Its first copy becomes the
        first frame of the clip; the sampler starts from noise, so the latent returned is only the right shape."""
        raise_pending_error()
        frames, _, height, width = x.shape
        vae = self.forge_objects.vae
        image = x[:1].movedim(1, -1).mul(0.5).add(0.5)  # [1, H, W, 3] in [0, 1]
        first = vae.first_stage_model.process_in(vae.encode(image).to(torch.float32))  # [1, 48, 1, h, w]
        length = (frames - 1) // pan2_vae.TEMPORAL + 1
        shape = (1, pan2_vae.LATENT_CHANNELS, length, height // pan2_vae.SPATIAL, width // pan2_vae.SPATIAL)
        condition = torch.zeros((1, pan2_vae.LATENT_CHANNELS + 1, *shape[2:]), dtype=torch.float32)
        condition[:, :pan2_vae.LATENT_CHANNELS, :1] = first[:, :, :1].cpu()
        condition[:, pan2_vae.LATENT_CHANNELS:, :1] = 1.0
        self.diffusion_model.condition = condition
        return torch.zeros(shape, device=x.device, dtype=x.dtype)
