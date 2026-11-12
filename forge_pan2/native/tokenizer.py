"""The PAN 2 tokenizer, rebuilt from what Forge Neo already ships.

PAN 2 uses the Qwen3.5 tokenizer. Its vocabulary and merges are the ones Forge Neo keeps for MOSS-Image
(backend/huggingface/OpenMOSS-Team/MOSS-Image-SFT/tokenizer), but those files alone miss the special tokens of the chat
template (<|im_start|>, <|im_end|>, <think>...) and the Qwen3.5 pre-tokenizer. tokenizer_skeleton.json is the PAN 2
tokenizer.json without its vocabulary and merges (a few KB instead of 20 MB); the two are joined here, which gives the
released tokenizer exactly.
"""

import json
import os

TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "PAN2", "tokenizer")
SKELETON = os.path.join(TOKENIZER_DIR, "tokenizer_skeleton.json")
MOSS_TOKENIZER = ("OpenMOSS-Team", "MOSS-Image-SFT", "tokenizer")


def forge_vocabulary_dir() -> str:
    from backend import loader
    return os.path.join(loader.HF, *MOSS_TOKENIZER)


def build(vocabulary_dir: str | None = None):
    """A PreTrainedTokenizerFast identical to the released PAN 2 tokenizer."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    vocabulary_dir = vocabulary_dir or forge_vocabulary_dir()
    with open(SKELETON, encoding="utf-8") as f:
        data = json.load(f)
    with open(os.path.join(vocabulary_dir, "vocab.json"), encoding="utf-8") as f:
        data["model"]["vocab"] = json.load(f)
    with open(os.path.join(vocabulary_dir, "merges.txt"), encoding="utf-8") as f:
        data["model"]["merges"] = [line.split(" ") for line in f.read().split("\n") if line and not line.startswith("#version")]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_str(json.dumps(data)), eos_token="<|im_end|>",
                                        pad_token="<|endoftext|>", model_max_length=262144)
    # Forge's own tokenizers get this too: long prompts are cut by the engine, not warned about
    tokenizer._eventual_warn_about_too_long_sequence = lambda *args, **kwargs: None
    return tokenizer
