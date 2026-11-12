"""Builds tests/fixtures/pan2_text_tiny_reference.pt with diffusers' own get_pan2_prompt_embeds and transformers'
Qwen3_5TextModel: a tiny Qwen3.5 in the layout of the released text encoder (full vocabulary, so the real token ids
reach it), its random weights, the prompts and their embeddings, as the reference for forge_pan2.native.text_engine.

Needs diffusers with PAN 2 and transformers 5.x in a separate environment, and the PAN 2 tokenizer files:

    D:/pan2-forge-dev/ref-venv/Scripts/python.exe tools/make_text_reference.py [--tokenizer D:/pan2-forge-dev/hf/PAN2/tokenizer]
"""

import argparse
from pathlib import Path

import torch
from diffusers.modular_pipelines.pan2.encoders import (
    get_pan2_default_negative_prompt,
    get_pan2_prompt_embeds,
)
from transformers import Qwen2Tokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pan2_text_tiny_reference.pt"
SYSTEM = "You are a helpful assistant. Generate the video with the following instructions."

# the released layout (three linear-attention layers, then a full-attention one), scaled down
LAYERS = 8
CONFIG = dict(vocab_size=248320, hidden_size=64, intermediate_size=96, num_hidden_layers=LAYERS, num_attention_heads=4,
              num_key_value_heads=2, head_dim=32, linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
              linear_num_key_heads=4, linear_num_value_heads=8, attn_output_gate=True, rms_norm_eps=1e-6,
              partial_rotary_factor=0.25, full_attention_interval=4,
              layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(LAYERS)],
              rope_parameters={"rope_type": "default", "rope_theta": 10000000, "mrope_section": [2, 1, 1],
                               "mrope_interleaved": True, "partial_rotary_factor": 0.25},
              tie_word_embeddings=False)

PROMPTS = [
    "",
    "A red fox trots across fresh snow at dawn.",
    '{"video_style":"realistic handheld documentary footage, one continuous shot","lighting":"soft overcast daylight",'
    '"perspective_view":"eye-level medium shot","subjects_lifecycle":[{"subject_id":"<|Subject1|>","display_name":"golden '
    'retriever","subject_attribute":"an adult golden retriever with a red collar"}],"init_frame_visual_description":"A golden '
    'retriever sits on green grass in a park (trees behind).","events":[{"event_index":0,"time_range_sec":"0 to 5",'
    '"event_description":"The dog stands up and walks toward the camera.","camera_motion":[{"time_range_sec":"0 to 5",'
    '"camera_motion":["static"]}],"contained_actions":[{"time_range_sec":"0 to 5","action":"<|Subject1|> stands up and '
    'walks forward"}]}],"quality":{"video_quality":"sharp natural detail","text_logos_overlays":null,"artifacts":null}}',
    get_pan2_default_negative_prompt(121, 24.0),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="D:/pan2-forge-dev/hf/PAN2/tokenizer")
    args = parser.parse_args()

    tokenizer = Qwen2Tokenizer.from_pretrained(args.tokenizer)
    torch.manual_seed(0)
    model = Qwen3_5TextModel(Qwen3_5TextConfig(**CONFIG)).eval()
    with torch.no_grad():
        # at the default initialization (std 0.02) the queries and keys of the linear attention are so small that the
        # epsilon of their L2 normalization, which Forge and transformers place differently, shows in the result;
        # trained weights are far from that, so the reference uses weights of a realistic scale
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * (0.2 if p.ndim > 1 else 0.1))
    cases = []
    for prompt in PROMPTS:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt or " "}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True).input_ids
        with torch.no_grad():
            embeds = get_pan2_prompt_embeds(model, tokenizer, prompt, SYSTEM, 2, 1536, torch.device("cpu"))
        cases.append({"prompt": prompt, "ids": list(ids), "embeds": embeds})
        print(f"{len(ids)} tokens -> {tuple(embeds.shape)}")
    # only the embedding rows the prompts use: the full table would be 60 MiB
    state_dict = model.state_dict()
    table = state_dict.pop("embed_tokens.weight")
    used = sorted({i for case in cases for i in case["ids"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": state_dict, "embed_ids": torch.tensor(used), "embed_rows": table[used].clone(),
                "cases": cases}, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
