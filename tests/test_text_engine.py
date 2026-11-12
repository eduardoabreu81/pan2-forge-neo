"""PAN 2 prompt encoding on CPU: the template and the token ids against the PAN 2 tokenizer's own chat template, and the
embeddings against diffusers' get_pan2_prompt_embeds on a tiny Qwen3.5 (tools/make_text_reference.py)."""

import unittest

from forge_env import FIXTURES, FORGE, FORGE_DIR

if FORGE:
    import torch
    from backend.nn.llm import llama
    from backend.operations import using_forge_operations

    from forge_pan2.native import text_engine, tokenizer

# the released PAN 2 tokenizer, for the comparison (hf download IFM/PAN2 --include "tokenizer/*")
RELEASED_TOKENIZER = FORGE_DIR.parent / "hf" / "PAN2" / "tokenizer"
SAMPLES = ["", "plain words", "Café naïve résumé — “quotes” 東京 (parentheses) [brackets] {braces}",
           '{"a":"<|Subject1|> walks","b":null}', "<think>\n</think>\n<|im_start|><|im_end|>", "tabs\tand\nnew lines\r\n", "a" * 300]


def tiny_text_model(config: dict):
    rope = config["rope_parameters"]
    forge_config = llama.Qwen35_9BConfig(
        vocab_size=config["vocab_size"], hidden_size=config["hidden_size"], intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"], num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"], partial_rotary_factor=config["partial_rotary_factor"],
        linear_num_key_heads=config["linear_num_key_heads"], linear_num_value_heads=config["linear_num_value_heads"],
        linear_key_head_dim=config["linear_key_head_dim"], linear_value_head_dim=config["linear_value_head_dim"],
        conv_kernel_size=config["linear_conv_kernel_dim"], rope_theta=float(rope["rope_theta"]),
        mrope_section=rope["mrope_section"], layer_types=config["layer_types"], rms_norm_eps=config["rms_norm_eps"])
    forge_config.head_dim = config["head_dim"]

    class TinyTextModel(text_engine.PAN2TextModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.num_layers = forge_config.num_hidden_layers
            self.model = llama.Qwen35Transformer(forge_config)

    with using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        return TinyTextModel()


@unittest.skipUnless(FORGE, "needs a Forge Neo checkout (PAN2_FORGE_DIR)")
class TextEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = torch.load(FIXTURES / "pan2_text_tiny_reference.pt", weights_only=False)
        cls.tokenizer = tokenizer.build(str(FORGE_DIR / "backend" / "huggingface" / "OpenMOSS-Team" / "MOSS-Image-SFT" / "tokenizer"))

    def engine(self):
        reference = self.reference
        model = tiny_text_model(reference["config"])
        state_dict = dict(reference["state_dict"])
        table = torch.zeros(reference["config"]["vocab_size"], reference["config"]["hidden_size"])
        table[reference["embed_ids"]] = reference["embed_rows"]
        state_dict["embed_tokens.weight"] = table
        model.model.load_state_dict(state_dict, strict=True)
        return text_engine.PAN2TextEngine(model.eval(), self.tokenizer)

    @unittest.skipUnless(RELEASED_TOKENIZER.is_dir(), "needs the released PAN 2 tokenizer files")
    def test_rebuilt_tokenizer_is_the_released_one(self):
        from transformers import PreTrainedTokenizerFast

        released = PreTrainedTokenizerFast(tokenizer_file=str(RELEASED_TOKENIZER / "tokenizer.json"))
        for text in SAMPLES + [case["prompt"] for case in self.reference["cases"]]:
            with self.subTest(text[:40]):
                self.assertEqual(self.tokenizer(text)["input_ids"], released(text)["input_ids"])

    def test_template_gives_the_reference_token_ids(self):
        engine = text_engine.PAN2TextEngine.__new__(text_engine.PAN2TextEngine)
        engine.tokenizer = type("T", (), {"tokenizer": self.tokenizer})()
        for case in self.reference["cases"]:
            with self.subTest(case["prompt"][:40]):
                self.assertEqual(engine.tokenize(case["prompt"]), case["ids"])

    def test_embeddings_match_diffusers(self):
        engine = self.engine()
        out = engine([case["prompt"] for case in self.reference["cases"]])
        self.assertEqual(len(out), len(self.reference["cases"]))
        for embeds, case in zip(out, self.reference["cases"]):
            with self.subTest(case["prompt"][:40]):
                self.assertEqual(tuple(embeds.shape), tuple(case["embeds"].shape[1:]))
                torch.testing.assert_close(embeds.float(), case["embeds"][0], rtol=1e-4, atol=1e-4)

    def test_condition_length_is_what_the_model_sees(self):
        engine = self.engine()
        for case in self.reference["cases"]:
            with self.subTest(case["prompt"][:40]):
                self.assertEqual(engine.condition_length(case["prompt"]), case["embeds"].shape[1])

    def test_prompts_keep_their_parentheses(self):
        # JSON prose with parentheses: no emphasis syntax, the text reaches the model as written
        engine = self.engine()
        text = '{"lighting":"warm (late afternoon) sun"}'
        tokens = engine.tokenizer.tokenize_with_weights(engine.wrap(text), disable_weights=True)
        self.assertEqual([t[0] for t in tokens[0]], engine.tokenize(text))
        self.assertTrue(all(t[1] == 1.0 for t in tokens[0]))


if __name__ == "__main__":
    unittest.main()
