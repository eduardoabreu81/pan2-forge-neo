"""The PAN 2 VAE on CPU: every published VAE file in Forge Neo's Wan22VAE, and encode / decode against diffusers'
own AutoencoderKLWan on a tiny model (tools/make_vae_reference.py)."""

import unittest

from forge_env import FIXTURES, FORGE, headers, layout_problems

if FORGE:
    import torch
    from backend.nn.qwen21_vae import Wan22VAE
    from backend.operations import using_forge_operations

    from forge_pan2.native import vae

VAE_FILES = ("IFM/PAN2 vae", "PAN2_vae_fp16.safetensors", "wan2.2_vae.safetensors")


def meta_tensors(entry) -> dict:
    return {k: torch.empty(v[1], device="meta") for k, v in entry["tensors"].items()}


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headers = headers()

    def test_every_published_vae_fits_wan22vae(self):
        with using_forge_operations(device=torch.device("meta"), dtype=torch.float32):
            model = Wan22VAE(**vae.wan22_settings())
        for name in VAE_FILES:
            with self.subTest(name):
                converted = vae.to_original(meta_tensors(self.headers[name]))
                tensors = {k: ["F32", list(v.shape)] for k, v in converted.items()}
                self.assertEqual(layout_problems(model, tensors), [])

    def test_a_wan21_vae_is_refused(self):
        sd = {"decoder.head.0.gamma": torch.empty(96, 1, 1, 1), "conv1.weight": torch.empty(32, 32, 1, 1, 1),
              "encoder.head.2.weight": torch.empty(32, 384, 3, 3, 3)}
        with self.assertRaisesRegex(ValueError, "16 latent channels"):
            vae.to_original(sd)

    def test_other_files_are_not_a_vae(self):
        for name in ("IFM/PAN2 transformer", "IFM/PAN2 text_encoder"):
            with self.subTest(name):
                self.assertFalse(vae.is_wan_vae(self.headers[name]["tensors"]))


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = torch.load(FIXTURES / "pan2_vae_tiny_reference.pt", weights_only=False)
        c = cls.reference["config"]
        with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
            cls.model = Wan22VAE(dim=c["base_dim"], dec_dim=c["decoder_base_dim"], z_dim=c["z_dim"], dim_mult=c["dim_mult"],
                                 num_res_blocks=c["num_res_blocks"], attn_scales=c["attn_scales"],
                                 temporal_downsample=c["temperal_downsample"], dropout=0.0, image_channels=3,
                                 patch_size=c["patch_size"], temporal_kernel=3)
        state_dict = vae.convert_diffusers({k: v.float() for k, v in cls.reference["state_dict"].items()})
        cls.model.load_state_dict(state_dict, strict=True)
        cls.model.eval()

    def test_encode_matches_diffusers(self):
        with torch.no_grad():
            latent = self.model.encode(self.reference["video"])
        torch.testing.assert_close(latent, self.reference["latent"], rtol=1e-4, atol=1e-4)

    def test_one_image_encodes_as_the_first_latent_frame(self):
        with torch.no_grad():
            latent = self.model.encode(self.reference["video"][:, :, :1])
        torch.testing.assert_close(latent, self.reference["image_latent"], rtol=1e-4, atol=1e-4)

    def test_decode_matches_diffusers(self):
        with torch.no_grad():
            decoded = self.model.decode(self.reference["latent"])
        # diffusers clips the decoded clip to [-1, 1]; Forge's VAE object does it after the model
        torch.testing.assert_close(decoded.clamp(-1.0, 1.0), self.reference["decoded"], rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
