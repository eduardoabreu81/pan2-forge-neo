"""Runtime hooks into Forge Neo's loader and video writer.

Every hook hands anything that is not PAN 2 to the original Forge Neo function untouched.
"""

import contextlib
import functools
import logging

import torch
from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list
from transformers.modeling_utils import no_init_weights

from . import model, tokenizer, vae
from .engine import PAN2Engine
from .forge_dit import ForgePAN2
from .text_engine import PAN2TextModel

logger = logging.getLogger("forge_pan2")

MISSING_TE = "PAN 2 needs its Qwen3.5-9B text encoder: select one under VAE / Text Encoder"
# a Qwen3.5 text model on its own (transformers Qwen3_5TextModel: IFM/PAN2, speedyrulz/PAN2-ComfyUI) or inside the
# vision-language model (Qwen3_5Model / ...ForConditionalGeneration), whose vision tower and heads PAN 2 does not use
BARE_TE_KEY = "layers.0.linear_attn.A_log"
VL_TE_KEY = "model.language_model.layers.0.linear_attn.A_log"
UNUSED_TE_PREFIXES = ("model.visual.", "visual.", "lm_head.", "mtp.", "model.mtp.")
FPS = 24

_applied = False


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def is_text_encoder(asd: dict) -> bool:
    return BARE_TE_KEY in asd or VL_TE_KEY in asd


def text_encoder_keys(asd: dict) -> dict:
    """A Qwen3.5 text encoder file under the names Forge's Qwen3.5-9B module uses (model.layers..., model.norm...)."""
    out = {}
    for k, v in asd.items():
        if k.startswith(UNUSED_TE_PREFIXES):
            continue
        if k.startswith("model.language_model."):
            out["model." + k[len("model.language_model."):]] = v
        elif k.startswith("model."):
            out[k] = v
        else:
            out["model." + k] = v
    return out


def _replace(sd: dict, prefix: str, asd: dict) -> None:
    for k in [k for k in sd if k.startswith(prefix)]:
        del sd[k]
    for k, v in asd.items():
        sd[prefix + k] = v


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        config = model.detect_config(state_dict, key_prefix)
        if config is not None:
            return config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, *rest):
        # Forge Neo calls (sd, asd, guess, path), and (sd, asd, metadata, guess, path) once its GGUF update lands
        guess = rest[-2]
        if not isinstance(guess, model.PAN2):
            return original(sd, asd, *rest)
        if is_text_encoder(asd):
            _replace(sd, model.TEXT_PREFIX, text_encoder_keys(asd))
        elif vae.is_wan_vae(asd):
            _replace(sd, "vae.", vae.to_original(asd))
        else:
            return original(sd, asd, *rest)
        return sd

    loader.replace_state_dict = replace_state_dict


def _load_vae(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(vae.MISSING_VAE)
    from backend.nn.qwen21_vae import Wan22VAE

    dtype = memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=dtype, extra_dtype="vae"):
            module = Wan22VAE(**vae.wan22_settings())
    load_state_dict(module, vae.to_original(state_dict), log_name="PAN 2 VAE")
    return module


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.PAN2):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)
        if cls_name == "PAN2Tokenizer":
            return tokenizer.build()
        if cls_name == "PAN2VAE":
            return _load_vae(state_dict)
        if cls_name == "PAN2Transformer3DModel":
            # the Krea 2 branch is a generic DiT load: dtype, quantization (Comfy int8 / ConvRot, NVFP4, fp8, GGUF)
            # and device handling
            with _swapped(krea, "SingleStreamDiT", ForgePAN2):
                dit = original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            print(f"[PAN 2] diffusion model, {len(dit.transformer_blocks)} blocks")
            return dit
        if cls_name == "PAN2TextModel":
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError(MISSING_TE)
            # Forge Neo's own Qwen3.5 branch (as for MOSS-Image), with the model built without its vision tower
            with _swapped(llama, "Qwen35_9B", PAN2TextModel):
                return original(guess, component_name, lib_name, "Qwen3_5Model", repo_path, state_dict)
        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


_video_hooked = False


def hook_save_video() -> bool:
    """Forge's video writer runs at Wan's 16 fps; PAN 2 clips are 24 fps. Idempotent; False while modules.images
    cannot be imported yet (the script also calls it again before a generation)."""
    global _video_hooked
    if _video_hooked:
        return True
    try:
        from modules import images, shared
    except ImportError:
        return False

    original = images.save_video

    @functools.wraps(original)
    def save_video(p, frames, fps=16, *args, **kwargs):
        if getattr(shared.sd_model, "is_pan2", False) and fps == 16:
            fps = FPS
        return original(p, frames, fps, *args, **kwargs)

    images.save_video = save_video
    _video_hooked = True
    return True


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.PAN2 not in model_list.models:
        model_list.models.append(model.PAN2)
    if PAN2Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, PAN2Engine)
    _hook_replace_state_dict()
    _hook_components()
    hook_save_video()

    from . import presets

    try:
        presets.register()
    except Exception as e:
        # the model still works without the preset: Euler, Simple, 50 steps, CFG 2.5, Shift 9
        logger.warning(f"[PAN 2] could not add the pan2 UI preset: {e}")

    _applied = True
    print(f"[PAN 2] native backend enabled (torch {torch.__version__})")
