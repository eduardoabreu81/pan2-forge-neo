"""PAN 2 through Forge Neo's own loader on CPU: detection, the DiT loaded by the loader's branch, the text encoder built
without the vision tower, the extra files put where the model expects them, the default negative prompt, and the
first frame stacked onto the latent with its guidance rule."""

import unittest

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import huggingface_guess
    import torch
    from backend import loader
    from backend.operations import using_forge_operations

    from forge_pan2 import negative
    from forge_pan2.native import engine, model, patches, text_engine, vae
    from forge_pan2.native.forge_dit import ForgePAN2

    patches.apply()


def tiny_checkpoint():
    reference = torch.load(FIXTURES / "pan2_dit_tiny_reference.pt", weights_only=False)
    return reference, {f"model.diffusion_model.{k}": v.clone() for k, v in reference["state_dict"].items()}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class LoaderTests(unittest.TestCase):
    def test_a_pan2_dit_is_guessed_as_pan2(self):
        reference, sd = tiny_checkpoint()
        guess = huggingface_guess.guess(sd)
        self.assertIsInstance(guess, model.PAN2)
        self.assertEqual(guess.unet_config["num_layers"], reference["config"]["num_layers"])
        self.assertEqual(guess.unet_config["in_channels"], reference["config"]["in_channels"])

    def test_the_loader_builds_the_dit_and_it_runs(self):
        reference, sd = tiny_checkpoint()
        guess = huggingface_guess.guess(sd)
        guess.unet_config.update({"rope_axes_dim": reference["config"]["rope_axes_dim"]})
        dit_sd = {k[len("model.diffusion_model."):]: v for k, v in sd.items()}
        dit = loader.load_huggingface_component(guess, "transformer", "diffusers", "PAN2Transformer3DModel", model.CONFIG_DIR, dit_sd)
        self.assertIsInstance(dit, ForgePAN2)
        case = reference["cases"][0]
        latent, condition = case["x"][:, :4], case["x"][:, 4:]
        dit.condition = condition
        with torch.no_grad():
            out = dit(latent.to(dit.computation_dtype), case["timestep"], case["context"].to(dit.computation_dtype))
        torch.testing.assert_close(out.float(), case["out"], rtol=1e-4, atol=1e-4)

    def test_text_to_video_stacks_zeros(self):
        reference, _ = tiny_checkpoint()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            dit = ForgePAN2(**reference["config"])
        x = torch.randn(2, 4, 3, 4, 6)
        cond = dit.conditioning(x)
        self.assertEqual(tuple(cond.shape), (2, 5, 3, 4, 6))
        self.assertEqual(cond.abs().sum().item(), 0.0)

    def test_a_first_frame_of_another_size_is_refused(self):
        reference, _ = tiny_checkpoint()
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            dit = ForgePAN2(**reference["config"])
        dit.condition = torch.zeros(1, 5, 3, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "first frame"):
            dit.conditioning(torch.zeros(1, 4, 3, 4, 6))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class TextEncoderTests(unittest.TestCase):
    def test_published_text_encoders_fill_the_model_without_vision(self):
        h = headers()
        with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
            te = text_engine.PAN2TextModel({})
        self.assertFalse(hasattr(te, "visual"))
        for name in ("IFM/PAN2 text_encoder", "PAN2_qwen3_5_int8_convrot.safetensors", "PAN2_qwen3_5_nvfp4.safetensors"):
            with self.subTest(name):
                tensors = h[name]["tensors"]
                placed = patches.text_encoder_keys({k: torch.empty(v[1], device="meta") for k, v in tensors.items()})
                described = {k: [tensors[k[len("model."):]][0], list(v.shape)] for k, v in placed.items()}
                self.assertEqual(layout_problems(te, described), [])

    def test_vision_language_layout_is_read_too(self):
        asd = {"model.language_model.layers.0.linear_attn.A_log": 1, "model.language_model.norm.weight": 2,
               "model.visual.blocks.0.attn.qkv.weight": 3, "lm_head.weight": 4, "mtp.fc.weight": 5}
        self.assertTrue(patches.is_text_encoder(asd))
        self.assertEqual(patches.text_encoder_keys(asd), {"model.layers.0.linear_attn.A_log": 1, "model.norm.weight": 2})


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class ExtraFileTests(unittest.TestCase):
    def guess(self):
        _, sd = tiny_checkpoint()
        return huggingface_guess.guess(sd), sd

    def test_text_encoder_file_goes_under_the_text_prefix(self):
        guess, sd = self.guess()
        asd = {"layers.0.linear_attn.A_log": torch.zeros(1), "embed_tokens.weight": torch.zeros(2, 2)}
        sd = loader.replace_state_dict(sd, asd, guess, "te.safetensors")
        self.assertIn(model.TEXT_PREFIX + "model.layers.0.linear_attn.A_log", sd)
        self.assertIn(model.TEXT_PREFIX + "model.embed_tokens.weight", sd)

    def test_diffusers_vae_file_is_converted_under_vae(self):
        guess, sd = self.guess()
        reference = torch.load(FIXTURES / "pan2_vae_tiny_reference.pt", weights_only=False)
        asd = dict(reference["state_dict"])
        # the released VAE has 48 latent channels; the tiny one 4
        asd["encoder.conv_out.weight"] = torch.zeros(96, 1, 1, 1, 1)
        sd = loader.replace_state_dict(sd, asd, guess, "vae.safetensors")
        self.assertIn("vae.decoder.head.0.gamma", sd)
        self.assertIn("vae.conv1.weight", sd)
        self.assertFalse(any(k.startswith("vae.decoder.mid_block") for k in sd))

    def test_other_models_are_left_to_forge(self):
        sd = {"model.diffusion_model.x": torch.zeros(1)}
        self.assertIsNone(model.detect_config(sd, "model.diffusion_model."))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class GenerationRuleTests(unittest.TestCase):
    def test_default_negative_is_the_released_one(self):
        reference = torch.load(FIXTURES / "pan2_text_tiny_reference.pt", weights_only=False)
        self.assertEqual(negative.default_negative(121, 24), reference["cases"][-1]["prompt"])

    def test_first_frame_follows_the_positive_prediction(self):
        cond = torch.randn(1, 4, 3, 2, 2)
        uncond = torch.randn(1, 4, 3, 2, 2)
        denoised = uncond + 2.5 * (cond - uncond)
        out = engine.first_frame_unguided({"denoised": denoised.clone(), "cond_denoised": cond, "uncond_denoised": uncond})
        torch.testing.assert_close(out[:, :, :1], cond[:, :, :1])
        torch.testing.assert_close(out[:, :, 1:], denoised[:, :, 1:])

    def test_vae_settings_match_the_release(self):
        settings = vae.wan22_settings()
        self.assertEqual((settings["dim"], settings["dec_dim"], settings["z_dim"], settings["patch_size"]), (160, 256, 48, 2))
        self.assertEqual(settings["temporal_downsample"], [False, True, True])


if __name__ == "__main__":
    unittest.main()
