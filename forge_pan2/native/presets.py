"""A "pan2" UI preset, added at runtime: PresetArch gains a member and the preset's options are registered through
Forge Neo's own presets.register, so they match every other preset (as in the LTX and H3 extensions).

PAN 2 is a video preset like Wan's: the batch slider becomes Frames, stepping by the frame rate (24), which keeps every
value on PAN 2's 4n + 1 grid (241 frames is 10 s).
"""

from enum import Enum

from modules_forge import presets

PRESET = "pan2"

# the reference sampler: 50 steps of Euler on uniform sigmas shifted by 9, CFG 2.5
SAMPLER = "Euler"
SCHEDULER = "Simple"
STEPS = 50
CFG = 2.5
SHIFT = 9.0
FPS = 24
FRAMES = 241


def _add_enum_member(enum_cls: type[Enum], name: str) -> Enum:
    if name in enum_cls.__members__:
        return enum_cls[name]

    value = max(m.value for m in enum_cls) + 1
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    member.__objclass__ = enum_cls
    member._sort_order_ = len(enum_cls._member_names_)

    # EnumType.__setattr__ refuses new members, so go through type
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member
    if hasattr(enum_cls, "_hashable_values_"):
        enum_cls._hashable_values_.append(value)
    return member


def _is_preset_option(key: str) -> bool:
    return key.startswith(f"{PRESET}_") or key.endswith(f"_{PRESET}")


def register() -> None:
    arch = _add_enum_member(presets.PresetArch, PRESET)

    presets.SAMPLERS[arch] = SAMPLER
    presets.SCHEDULERS[arch] = SCHEDULER
    presets.STEPS[arch] = STEPS
    presets.CFG[arch] = CFG
    presets.SHIFT[arch] = SHIFT
    presets.FRAMES[arch.name] = FPS

    from modules import shared

    templates: dict = {}
    presets.register(templates)
    # Forge's video presets start at 1 frame; PAN 2's reference length is 241 (10 s)
    for key in (f"{PRESET}_t2i_batch_size", f"{PRESET}_i2i_batch_size"):
        if key in templates:
            templates[key].default = FRAMES
    for key, info in templates.items():
        if _is_preset_option(key) and key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
