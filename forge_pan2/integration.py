"""PAN 2 inside Forge's own txt2img and img2img flows, through script callbacks.

Forge loads the model (native/patches.py registers it), samples, decodes and saves the video as it does for Wan. The
callbacks here check the request, fill in the released negative prompt when none is given, and, for image-to-video,
make the given first latent frame follow the positive prediction without guidance (the reference sampler).
"""

import json
import os
import struct
from functools import lru_cache

from .contracts import FPS, MAX_FRAMES, SIZE_STEP, PAN2Error, set_pending_error
from .native import detect
from .negative import default_negative

MAX_HEADER_BYTES = 32 * 1024 * 1024


@lru_cache(maxsize=32)
def _header_keys(path: str, size: int, mtime: float) -> frozenset:
    with open(path, "rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > MAX_HEADER_BYTES:
            return frozenset()
        return frozenset(json.loads(stream.read(length)))


def is_pan2_file(path: str) -> bool:
    """Whether a checkpoint is a PAN 2 DiT, from its safetensors header (before Forge loads it)."""
    if not path or not str(path).endswith(".safetensors") or not os.path.isfile(path):
        return False
    try:
        stat = os.stat(path)
        keys = _header_keys(str(path), stat.st_size, stat.st_mtime)
    except (OSError, ValueError, struct.error):
        return False
    return detect.is_pan2(keys) or detect.is_pan2(keys, detect.PREFIX)


def selected_checkpoint(p):
    from modules import sd_models, shared
    overrides = getattr(p, "override_settings", {})
    return sd_models.get_closet_checkpoint_match(overrides.get("sd_model_checkpoint", shared.opts.sd_model_checkpoint))


def selects_pan2(p) -> bool:
    info = selected_checkpoint(p)
    return info is not None and is_pan2_file(info.filename)


def validate(p) -> None:
    from modules import processing

    if getattr(p, "enable_hr", False) or getattr(p, "txt2img_upscale", False):
        raise PAN2Error("Disable Hires. fix for PAN 2.")
    if getattr(p, "batch_size", 1) > MAX_FRAMES:
        raise PAN2Error(f"PAN 2 makes clips of up to {MAX_FRAMES} frames (20 s at {FPS} fps).")
    if p.width % SIZE_STEP or p.height % SIZE_STEP:
        raise PAN2Error(f"PAN 2 needs a width and height divisible by {SIZE_STEP} (it was trained at 1248 x 704).")
    if isinstance(p, processing.StableDiffusionProcessingImg2Img):
        if getattr(p, "image_mask", None) is not None:
            raise PAN2Error("PAN 2 has no inpainting: use the img2img image as the first frame, without a mask.")
        if getattr(p, "denoising_strength", 1.0) < 1.0:
            raise PAN2Error("PAN 2 image-to-video starts from noise with the image as the first frame: set Denoising "
                            "strength to 1.")


def before_process(p) -> None:
    """Before Forge loads the model: check a PAN 2 request."""
    set_pending_error(None)
    if not selects_pan2(p):
        return
    try:
        validate(p)
    except PAN2Error as error:
        set_pending_error(error)
        raise


def script_before_process(p, *_):
    """before_process as Forge's script runner calls it: a rejected request prints one line instead of the traceback
    Forge logs for any exception; the error stays pending and stops the generation when the model is first called."""
    try:
        before_process(p)
    except PAN2Error as error:
        print(f"[PAN 2] {error}")


def _engine():
    from modules import shared
    model = shared.sd_model
    return model if getattr(model, "is_pan2", False) else None


def process(p) -> None:
    """After the prompts are set up, before img2img's init encodes the first frame."""
    engine = _engine()
    if engine is None:
        return
    from .native import patches

    if not patches.hook_save_video():
        print("[PAN 2] could not set the video frame rate: this clip is saved at 16 fps")
    engine.clear_first_frame()
    if any(not (n or "").strip() for n in p.all_negative_prompts):
        negative = default_negative(p.batch_size, FPS)
        p.all_negative_prompts = [n if (n or "").strip() else negative for n in p.all_negative_prompts]
        p.extra_generation_params["PAN 2 negative prompt"] = "released default"


def is_caption(text) -> bool:
    """Whether a prompt starts with a PAN 2 world-state caption (a JSON object with its events)."""
    text = text.lstrip() if isinstance(text, str) else ""
    if not text.startswith("{"):
        return False
    try:
        caption, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return False
    return isinstance(caption, dict) and isinstance(caption.get("events"), list) and "video_style" in caption


def _restore(original, batch: list, everything: list) -> bool:
    if not is_caption(original) or all(text == original for text in batch):
        return False
    batch[:] = [original] * len(batch)
    everything[:] = [original] * len(everything)
    return True


def restore_captions(p) -> None:
    """sd-dynamic-prompts reads the { } and the | of a PAN 2 caption (its subject IDs are <|Subject1|>) as variants and
    keeps one piece of each, which cannot be put back together; it leaves the prompt as written in p.prompt. Before the
    batch is encoded (and its LoRA tags parsed), a broken caption is replaced by the one that was written."""
    restored = _restore(getattr(p, "prompt", None), p.prompts, p.all_prompts)
    restored |= _restore(getattr(p, "negative_prompt", None), p.negative_prompts, p.all_negative_prompts)
    if restored:
        print("[PAN 2] Dynamic Prompts changed the PAN 2 description: the prompt as written is used")


def before_batch(p) -> None:
    if _engine() is not None:
        restore_captions(p)


def before_sampling(p) -> None:
    engine = _engine()
    if engine is None or not engine.has_first_frame:
        return
    from .native.engine import first_frame_unguided

    unet = p.sd_model.forge_objects.unet.clone()
    unet.set_model_sampler_post_cfg_function(first_frame_unguided)
    p.sd_model.forge_objects.unet = unet


def postprocess(p) -> None:
    engine = _engine()
    if engine is not None:
        engine.clear_first_frame()
