# 🌍 PAN 2 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/pan2-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate **videos of up to 20 seconds** with **[PAN 2](https://huggingface.co/IFM/PAN2)**, the world model by the Institute of Foundation Models (MBZUAI), inside Forge Neo - describe a scene and what happens in it, or start from a picture, and get an MP4 made with the checkpoint, preset and Generate button you already know.

> [!Important]
> **Work in progress.** Text-to-video and image-to-video are in this first version, and more is on the way - see [What is not here yet](#-what-is-not-here-yet). PAN 2 is a **29-billion-parameter** model: plan for a large graphics card and plenty of system RAM.
>
> This extension requires an up-to-date **Forge Neo** (the `neo` branch, version 2.32 or later). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [What is not here yet](#-what-is-not-here-yet)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Public Preview

- **PAN 2 text-to-video** - on Forge Neo's own loader, samplers and video output
- **Image-to-video** - your img2img picture becomes the first frame of the clip
- **The released negative prompt** - used on its own when the negative prompt is empty
- **pan2 UI preset** - the official sampling recipe in one click

---

## 🎯 Features

### 🎬 Video from Text or from a Picture

- **txt2img** - describe the scene and what happens, and PAN 2 makes the clip
- **img2img** - your input image becomes the first frame, and PAN 2 continues from it
- Clips at **24 FPS**, up to **20 seconds**, saved as an MP4 in Forge's usual result area
- **Frames** takes the place of Batch Size: 241 frames are 10 seconds

### 🧭 Made for PAN 2's Prompts

- PAN 2 was trained on **structured descriptions** of a scene: style, lighting, the people and objects in it, and what happens second by second. The prompt box takes them exactly as written
- An **empty negative prompt** becomes the one PAN 2 was released with, timed to the length of your clip
- The **token counter** shows how much of PAN 2's limit of 1536 tokens your prompt uses

### 🎛️ Familiar Forge Controls

- Runs inside txt2img and img2img - no separate tab, no extra program
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- The **pan2** UI preset sets the official recipe: Euler, Simple, 50 steps, CFG 2.5, Shift 9
- Community files in **INT8 ConvRot** and **NVFP4** load as they are

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Checks the VAE you select, and explains clearly when it does not fit PAN 2
- Other models are not affected

---

## 🚧 What is not here yet

- **A visual editor for PAN 2's descriptions** - draw the events, camera moves and actions on a timeline, in the [Regional Boxes](https://github.com/eduardoabreu81/sd-regional-boxes-neo) panel
- **Prompt rewriter** - the official workflow turns a short idea into PAN 2's structured description with a language model. Until it is here, write the description yourself or with any chat assistant (see [Tips](#-tips))
- **Full-precision single files** - the official release is split into many files; for now use the single files in INT8 or NVFP4 listed below
- **GGUF** files
- **The PAN 2 guardrail** - IFM recommends running PAN 2 behind its separate [safety filter](https://github.com/ifm-ai/pan2-guardrail), which is not part of this extension
- **Faster sampling** options, such as step forecasting
- **Clips longer than 15 seconds from the slider** - the Frames slider stops at 361 frames; the model goes up to 481 frames through the API

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/pan2-forge-neo`
4. Click **Install** and restart the WebUI
5. Place the files in the usual folders:

| Part | File | Folder |
| :--- | :--- | :--- |
| Model | [PAN2_int8_convrot_full](https://huggingface.co/speedyrulz/PAN2-ComfyUI/blob/main/diffusion_models/PAN2_int8_convrot_full.safetensors) or [PAN2_nvfp4](https://huggingface.co/speedyrulz/PAN2-ComfyUI/blob/main/diffusion_models/PAN2_nvfp4.safetensors) | `models/Stable-diffusion` |
| Text encoder | [PAN2_qwen3_5_int8_convrot](https://huggingface.co/speedyrulz/PAN2-ComfyUI/blob/main/text_encoders/PAN2_qwen3_5_int8_convrot.safetensors) or [PAN2_qwen3_5_nvfp4](https://huggingface.co/speedyrulz/PAN2-ComfyUI/blob/main/text_encoders/PAN2_qwen3_5_nvfp4.safetensors) | `models/text_encoder` |
| VAE | [PAN2_vae_fp16](https://huggingface.co/speedyrulz/PAN2-ComfyUI/blob/main/vae/PAN2_vae_fp16.safetensors) or [wan2.2_vae](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/vae/wan2.2_vae.safetensors) | `models/VAE` |

NVFP4 runs natively on RTX 50 series cards; INT8 is closer to the original. The VAE is the Wan 2.2 one (48 channels) - the Wan 2.1 VAE does not work with PAN 2.

6. Make sure **FFmpeg** is installed, as for any video model in Forge Neo
7. Pick the **pan2** UI preset and the model, and under **VAE / Text Encoder** select the text encoder and the VAE

---

## ⚙️ Recommended Settings

| Sampler | Schedule type | Steps | CFG | Shift |
| :--- | :--- | :--- | :--- | :--- |
| Euler | Simple | 50 | 2.5 | 9 |

The **pan2** preset sets these.

**Width and Height** must be multiples of 32. PAN 2 was trained at **1248×704** (landscape); **704×1248** works for portrait.

**Frames** sets the length and follows the pattern 4 × n + 1. The slider moves one second at a time:

| Frames | Length at 24 FPS |
| :--- | :--- |
| 121 | 5 seconds |
| 241 | 10 seconds (the default) |
| 361 | 15 seconds |

**img2img** - set **Denoising strength** to 1 (the clip starts from noise, with your picture as its first frame) and **Resize mode** to **Crop and resize**, which is how PAN 2 was trained.

---

## 💡 Tips

- **Write structured descriptions.** PAN 2 was trained on descriptions of the scene in JSON, and follows them much better than a plain sentence - see the example below
- **Dynamic Prompts can stay on** - it would read the braces and the `|` of the JSON as its own syntax, so the extension gives PAN 2 the description exactly as you wrote it
- **Leave the negative prompt empty** to use the one PAN 2 was released with
- Keep the description **under about 1480 tokens**: the token counter shows the count, and PAN 2 ignores what goes beyond 1536
- Set the times in the description to the length of your clip - 241 frames are 0 to 10 seconds
- From a picture, describe the picture in **init_frame_visual_description** and what happens next in **events**
- Start with a short clip at a small size to check your setup, then go longer
- PAN 2 is released under the [Apache 2.0 license](https://huggingface.co/IFM/PAN2). Read its intended uses and limits on the model page, and say that a video is AI-generated wherever it could be mistaken for real footage

<details>
<summary><b>✍️ How a PAN 2 description is written</b></summary>

A short example of 5 seconds:

```json
{"video_style":"realistic handheld footage, one continuous shot","lighting":"soft overcast daylight","perspective_view":"eye-level medium shot","subjects_lifecycle":[{"subject_id":"<|Subject1|>","display_name":"potter","subject_attribute":"a woman in a clay-stained apron"}],"init_frame_visual_description":"A potter sits at a spinning wheel with a wet lump of clay, shelves of bowls behind her.","events":[{"event_index":0,"time_range_sec":"0 to 5","event_description":"She presses her thumbs into the clay and draws it up into a tall cylinder.","camera_motion":[{"time_range_sec":"0 to 5","camera_motion":["static"]}],"contained_actions":[{"time_range_sec":"0 to 5","action":"<|Subject1|> shapes the clay into a cylinder"}]}],"quality":{"video_quality":"sharp natural detail","text_logos_overlays":null,"artifacts":null}}
```

- **video_style**, **lighting** and **perspective_view** describe the look and the shot
- **subjects_lifecycle** lists the people and moving objects, each with an ID such as `<|Subject1|>`
- **init_frame_visual_description** is the first frame, as a still picture
- **events** follow each other from 0 to the end of the clip, each with its camera moves and the actions of each subject
- Camera moves use fixed words: `static`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `track_left`, `track_right`, `dolly_forward`, `dolly_backward`, `dolly_up`, `dolly_down`, `zoom_in`, `zoom_out`, `roll_clockwise`, `roll_counterclockwise`, `unstable`

The full rules are in PAN 2's official [prompt guide](https://github.com/ifm-ai/PAN2/blob/main/inference/pan-prompt-rewriter/system_prompt.md). Giving that guide and your idea to a chat assistant is a quick way to get a good description.

</details>

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[PAN 2](https://huggingface.co/IFM/PAN2)** by the Institute of Foundation Models (IFM), MBZUAI
- **[Diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pan2)** - official implementation of the model
- **[speedyrulz](https://huggingface.co/speedyrulz/PAN2-ComfyUI)** - PAN 2 support for ComfyUI and the INT8 and NVFP4 files
- **[Wan-AI](https://huggingface.co/Wan-AI)** - the Wan 2.2 VAE
- **[Qwen](https://huggingface.co/Qwen)** - the Qwen3.5 text encoder

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE). PAN 2 and its parts have their own licenses.

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/pan2-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/pan2-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
