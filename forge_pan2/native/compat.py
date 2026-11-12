"""Checks that the installed Forge Neo has every piece the native PAN 2 backend builds on, before anything is patched."""

import importlib
import inspect
import os

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.attention": ("attention_function",),
    "backend.memory_management": ("load_models_gpu", "vae_dtype", "dtype_size", "cast_to", "cpu"),
    "backend.operations": ("using_forge_operations",),
    "backend.state_dict": ("load_state_dict",),
    "backend.loader": ("load_huggingface_component", "replace_state_dict", "possible_models", "HF"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.llm.llama": ("Qwen35_9B", "Qwen35_9BConfig", "Qwen35Transformer"),
    "backend.nn.qwen21_vae": ("Wan22VAE",),
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "backend.patcher.clip": ("CLIP",),
    "backend.patcher.unet": ("UnetPatcher",),
    "backend.patcher.vae": ("VAE",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "backend.args": ("dynamic_args",),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("Wan21",),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "SHIFT", "FRAMES", "register"),
    "modules.images": ("save_video",),
}

# the DiT and the text encoder go through these branches of the loader
LOADER_BRANCHES = ("Krea2Transformer2DModel", "Qwen3_5Model")


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader

        from . import tokenizer

        # the whole module, not load_huggingface_component: an extension that loaded first has already put its own wrapper
        # there, and a wrapper's source holds none of Forge's branches
        source = inspect.getsource(loader)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"backend.loader has no {', '.join(missing)} branch")
        vocabulary = tokenizer.forge_vocabulary_dir()
        missing = [f for f in ("vocab.json", "merges.txt") if not os.path.isfile(os.path.join(vocabulary, f))]
        if missing:
            problems.append(f"the Qwen3.5 tokenizer files are missing from {vocabulary}: {', '.join(missing)}")

    return problems


def report(problems: list[str]) -> str:
    lines = [
        "[PAN 2] This Forge Neo is too old for the extension, so nothing was changed and PAN 2 is disabled.",
        "[PAN 2] Update Forge Neo (git pull on the neo branch) and restart. Details:",
    ]
    lines += [f"[PAN 2]   - {p}" for p in problems]
    message = "\n".join(lines)
    print(message)
    return message
