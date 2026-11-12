"""The caption sd-dynamic-prompts breaks goes back to the one that was written (no Forge needed)."""

import unittest
from types import SimpleNamespace

from forge_pan2.integration import is_caption, restore_captions

CAPTION = ('{"video_style":"one continuous shot","lighting":"dusk","perspective_view":"medium shot","subjects_lifecycle":'
           '[{"subject_id":"<|Subject1|>","display_name":"potter","subject_attribute":"a woman in an apron"}],'
           '"init_frame_visual_description":"A potter at a wheel.","events":[{"event_index":0,"time_range_sec":"0 to 5",'
           '"event_description":"She shapes the clay.","camera_motion":[{"time_range_sec":"0 to 5","camera_motion":["static"]}],'
           '"contained_actions":[{"time_range_sec":"0 to 5","action":"<|Subject1|> shapes the clay"}]}],'
           '"quality":{"video_quality":"sharp","text_logos_overlays":null,"artifacts":null}} <lora:motion:0.8>')
# what sd-dynamic-prompts made of it (dynamicprompts' RandomPromptGenerator, seed 1)
BROKEN = '"video_style":"one continuous shot",...,"subjects_lifecycle":[Subject1],"events":[...,"action":"<]] <lora:motion:0.8>'


def processing(prompt, prompts, negative="", negatives=None):
    negatives = negatives if negatives is not None else [negative] * len(prompts)
    return SimpleNamespace(prompt=prompt, prompts=list(prompts), all_prompts=list(prompts) * 2, negative_prompt=negative,
                           negative_prompts=list(negatives), all_negative_prompts=list(negatives) * 2)


class RestoreTests(unittest.TestCase):
    def test_a_caption_is_told_apart(self):
        self.assertTrue(is_caption(CAPTION))
        self.assertFalse(is_caption(BROKEN))
        self.assertFalse(is_caption("a potter shapes clay"))
        self.assertFalse(is_caption('{"summary": {"caption": "x"}, "items": []}'))

    def test_a_broken_caption_goes_back_to_the_written_one(self):
        p = processing(CAPTION, [BROKEN, BROKEN])
        batch = p.prompts
        restore_captions(p)
        self.assertIs(p.prompts, batch)  # the list Forge encodes from, changed in place
        self.assertEqual(p.prompts, [CAPTION, CAPTION])
        self.assertEqual(p.all_prompts, [CAPTION] * 4)

    def test_other_prompts_are_left_alone(self):
        p = processing("a {red|blue} car", ["a red car", "a blue car"])
        restore_captions(p)
        self.assertEqual(p.prompts, ["a red car", "a blue car"])

    def test_a_caption_dynamic_prompts_left_alone_stays(self):
        p = processing(CAPTION, [CAPTION])
        restore_captions(p)
        self.assertEqual(p.all_prompts, [CAPTION, CAPTION])

    def test_a_caption_as_the_negative_prompt_goes_back_too(self):
        p = processing("x", ["x"], negative=CAPTION, negatives=[BROKEN])
        restore_captions(p)
        self.assertEqual(p.negative_prompts, [CAPTION])
        self.assertEqual(p.prompts, ["x"])


if __name__ == "__main__":
    unittest.main()
