"""Forge Neo discovers this always-on script when the extension is installed."""

import sys
import traceback
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import scripts

from forge_pan2 import integration
from forge_pan2.native import compat

ENABLED = False

_problems = compat.check()

if _problems:
    compat.report(_problems)
else:
    try:
        from forge_pan2.native import patches

        patches.apply()
        ENABLED = True
    except Exception:
        print("[PAN 2] failed to enable, Forge Neo is unchanged:")
        traceback.print_exc()


class Script(scripts.Script):
    sorting_priority = 10
    create_group = False

    def title(self):
        return "PAN 2"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        integration.script_before_process(p)

    def process(self, p, *args):
        integration.process(p)

    def before_process_batch(self, p, *args, **kwargs):
        integration.before_batch(p)

    def process_before_every_sampling(self, p, *args, **kwargs):
        integration.before_sampling(p)

    def postprocess(self, p, processed, *args):
        integration.postprocess(p)
