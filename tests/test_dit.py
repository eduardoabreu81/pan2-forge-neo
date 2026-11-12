"""The PAN 2 DiT on CPU: settings from the published headers, every published layout, and the forward pass against
diffusers' own PAN2Transformer3DModel on a tiny model (tools/make_dit_reference.py)."""

import unittest

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import torch
    from backend.operations import using_forge_operations

    from forge_pan2.native import detect
    from forge_pan2.native.transformer import PAN2Transformer3DModel

DIT_FILES = ("IFM/PAN2 transformer", "PAN2_int8_convrot.safetensors", "PAN2_int8_convrot_full.safetensors",
             "PAN2_nvfp4.safetensors")
RELEASED = dict(in_channels=97, patch_size=[1, 2, 2], out_channels=48, attention_head_dim=128, num_attention_heads=32,
                num_layers=44, num_refiner_layers=4, mlp_ratio=8.0, context_mlp_ratio=4.0, refiner_mlp_ratio=4.0,
                text_embed_dim=4096)


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def shapes(self, name):
        return {k: v[1] for k, v in self.headers[name]["tensors"].items()}

    def test_every_published_dit_is_told_apart_and_read_alike(self):
        for name in DIT_FILES:
            with self.subTest(name):
                self.assertTrue(detect.is_pan2(self.shapes(name)))
                self.assertEqual(detect.dit_config(self.shapes(name)), RELEASED)

    def test_other_files_are_not_pan2(self):
        for name in ("IFM/PAN2 text_encoder", "IFM/PAN2 vae", "PAN2_qwen3_5_nvfp4.safetensors"):
            with self.subTest(name):
                self.assertFalse(detect.is_pan2(self.shapes(name)))

    def test_every_published_dit_loads_into_the_model(self):
        for name in DIT_FILES:
            with self.subTest(name):
                with using_forge_operations(device=torch.device("meta"), dtype=torch.bfloat16):
                    model = PAN2Transformer3DModel(**detect.dit_config(self.shapes(name)))
                self.assertEqual(layout_problems(model, self.headers[name]["tensors"]), [])


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class ForwardTests(unittest.TestCase):
    def test_matches_diffusers(self):
        reference = torch.load(FIXTURES / "pan2_dit_tiny_reference.pt", weights_only=False)
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = PAN2Transformer3DModel(**reference["config"])
        model.load_state_dict(reference["state_dict"], strict=True)
        model.eval()
        for case in reference["cases"]:
            with self.subTest(case["name"]), torch.no_grad():
                out = model(case["x"], case["timestep"], case["context"])
                torch.testing.assert_close(out, case["out"], rtol=1e-4, atol=1e-4)

    def test_long_sequences_run_in_slices_with_the_same_result(self):
        from forge_pan2.native import transformer

        reference = torch.load(FIXTURES / "pan2_dit_tiny_reference.pt", weights_only=False)
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            model = PAN2Transformer3DModel(**reference["config"])
        model.load_state_dict(reference["state_dict"], strict=True)
        case = reference["cases"][0]
        chunk = transformer.TOKEN_CHUNK
        try:
            transformer.TOKEN_CHUNK = 5  # 18 video tokens: four slices
            with torch.no_grad():
                out = model(case["x"], case["timestep"], case["context"])
        finally:
            transformer.TOKEN_CHUNK = chunk
        torch.testing.assert_close(out, case["out"], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
