"""PAN 2 prompt encoding, ported from ComfyUI comfy/text_encoders/pan2.py (speedyrulz/ComfyUI) and checked against
diffusers' get_pan2_prompt_embeds.

PAN 2 conditions on hidden_states[-3] of a stock Qwen3.5-9B (its last two layers skipped, no final norm), on the
tokens after the `<|im_start|>user\\n` header of the chat template: the prompt and the assistant header, with its open
think block, that the template adds after it. At most 1536 of them. Every prompt is its own sequence: no padding and no
attention mask. The prompt is read exactly as written: PAN 2 prompts are JSON, so there is no emphasis syntax.

The text encoder is Forge Neo's own Qwen3.5-9B, built without its vision tower, which PAN 2 does not ship or use.
"""

import numbers

import torch
from backend.args import dynamic_args
from backend.nn.llm import llama
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

SYSTEM_PROMPT = "You are a helpful assistant. Generate the video with the following instructions."
TEMPLATE = ("<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n")
MAX_SEQUENCE_LENGTH = 1536
# the token budget of the reference prompt rewriter, for the whole caption with the template suffix
SOFT_LIMIT = 1480
LAYER = -3
IM_START = 248045
USER_HEADER_LENGTH = 3  # <|im_start|> user \n
PAD = 248044  # <|endoftext|>, never in a PAN 2 prompt


class PAN2TextModel(llama.Qwen35_9B):
    """Forge Neo's Qwen3.5-9B text model without the vision tower."""

    def __init__(self, config_dict):
        torch.nn.Module.__init__(self)
        config = llama.Qwen35_9BConfig()
        self.num_layers = config.num_hidden_layers
        self.model = llama.Qwen35Transformer(config)


class PAN2TextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=LAYER, special_tokens={"pad": PAD},
                                        layer_norm_hidden_state=False, enable_attention_masks=False, return_attention_masks=False)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    @staticmethod
    def wrap(text: str) -> str:
        # the chat template trims the user turn: the reference's single space for an empty prompt ends up empty
        return TEMPLATE.format(text.strip())

    def tokenize(self, text: str) -> list[int]:
        """The ids of the whole templated prompt."""
        return self.tokenizer.tokenizer(self.wrap(text))["input_ids"]

    def condition_length(self, text: str) -> int:
        """How many tokens PAN 2 conditions on for this prompt, before the 1536 cut."""
        return len(self.tokenize(text)) - user_start(self.tokenize(text))

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"
        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        for text in texts:
            if text not in cache:
                tokens = self.tokenizer.tokenize_with_weights(self.wrap(text), disable_weights=True)
                out = self.text_encoder.encode_token_weights(tokens)[0]
                ids = [t[0] for t in tokens[0]]
                start = user_start(ids)
                cache[text] = out[:, start:start + MAX_SEQUENCE_LENGTH]
            zs.extend(cache[text])
        return zs


def user_start(ids) -> int:
    """Index of the first token after the user header: the user turn is the second of the template."""
    starts = [i for i, t in enumerate(ids) if isinstance(t, numbers.Integral) and int(t) == IM_START]
    if len(starts) < 2:
        raise ValueError("[PAN 2] the prompt template lost its user turn")
    return starts[1] + USER_HEADER_LENGTH
