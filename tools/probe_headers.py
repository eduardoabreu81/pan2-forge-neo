"""Read the safetensors headers of the PAN 2 files on the Hugging Face Hub, without downloading the weights.

Writes tests/fixtures/pan2_headers.json.gz: for each file, its size, metadata and every tensor's dtype and shape.
The official release is split into shards; each folder (transformer, text_encoder, vae) is stored as one entry. The CPU
tests build the modules on the meta device and compare them with these headers.

    python tools/probe_headers.py [--out PATH]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

# (entry name, repo, file or sharded folder)
FILES = [
    ("IFM/PAN2 transformer", "IFM/PAN2", "transformer/"),
    ("IFM/PAN2 text_encoder", "IFM/PAN2", "text_encoder/"),
    ("IFM/PAN2 vae", "IFM/PAN2", "vae/diffusion_pytorch_model.safetensors"),
    ("PAN2_int8_convrot.safetensors", "speedyrulz/PAN2-ComfyUI", "diffusion_models/PAN2_int8_convrot.safetensors"),
    ("PAN2_int8_convrot_full.safetensors", "speedyrulz/PAN2-ComfyUI", "diffusion_models/PAN2_int8_convrot_full.safetensors"),
    ("PAN2_nvfp4.safetensors", "speedyrulz/PAN2-ComfyUI", "diffusion_models/PAN2_nvfp4.safetensors"),
    ("PAN2_qwen3_5_int8_convrot.safetensors", "speedyrulz/PAN2-ComfyUI", "text_encoders/PAN2_qwen3_5_int8_convrot.safetensors"),
    ("PAN2_qwen3_5_nvfp4.safetensors", "speedyrulz/PAN2-ComfyUI", "text_encoders/PAN2_qwen3_5_nvfp4.safetensors"),
    ("PAN2_vae_fp16.safetensors", "speedyrulz/PAN2-ComfyUI", "vae/PAN2_vae_fp16.safetensors"),
    ("wan2.2_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan2.2_vae.safetensors"),
]

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pan2_headers.json.gz"


def probe(api, repo, path):
    if path.endswith("/"):
        names = sorted(f for f in api.list_repo_files(repo) if f.startswith(path) and f.endswith(".safetensors"))
    else:
        names = [path]
    sizes = {s.path: s.size for s in api.get_paths_info(repo, names)}
    entry = {"repo": repo, "files": names, "size": sum(sizes.get(n) or 0 for n in names), "metadata": {}, "tensors": {}}
    for name in names:
        info = api.parse_safetensors_file_metadata(repo, name)
        entry["metadata"].update(info.metadata or {})
        entry["tensors"].update({k: [t.dtype, list(t.shape)] for k, t in info.tensors.items()})
    entry["tensors"] = dict(sorted(entry["tensors"].items()))
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    api = HfApi()
    result, failed = {}, []
    for name, repo, path in FILES:
        try:
            entry = probe(api, repo, path)
        except Exception as error:  # missing or renamed: report and go on
            failed.append(name)
            print(f"FAIL {name}: {type(error).__name__}: {str(error).splitlines()[0]}")
            continue
        result[name] = entry
        print(f"ok   {name}: {len(entry['tensors'])} tensors")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"), sort_keys=True)
    print(f"{len(result)} files in {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
