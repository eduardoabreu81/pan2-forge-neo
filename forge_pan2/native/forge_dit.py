"""The PAN 2 DiT as Forge Neo's sampler calls it.

Forge hands the model the noisy video latent [B, 48, F, H, W]; PAN 2 also takes 49 conditioning channels: the
normalized latent of the given frames and a mask that is 1 where a latent frame is given (ComfyUI model_base.PAN2
concat_cond). For text-to-video both are zeros; for image-to-video the engine stores the first frame here
(engine.encode_first_stage), and every call stacks it onto the latent.
"""

import torch

from .transformer import PAN2Transformer3DModel


class ForgePAN2(PAN2Transformer3DModel):
    # [1, 49, F, H, W] for the generation being sampled, set by the engine; None: text-to-video
    condition: torch.Tensor | None = None

    def conditioning(self, x: torch.Tensor) -> torch.Tensor:
        channels = self.in_channels - x.shape[1]
        condition = self.condition
        if condition is None or tuple(condition.shape[2:]) != tuple(x.shape[2:]):
            if condition is not None:
                raise RuntimeError(f"[PAN 2] the first frame was prepared for a latent of {tuple(condition.shape[2:])}, "
                                   f"not {tuple(x.shape[2:])}")
            return x.new_zeros((x.shape[0], channels, *x.shape[2:]))
        return condition.to(x.device, x.dtype).expand(x.shape[0], -1, -1, -1, -1)

    def forward(self, x, timestep, context, transformer_options={}, **kwargs):
        x = torch.cat([x, self.conditioning(x)], dim=1)
        return super().forward(x, timestep, context, transformer_options=transformer_options, **kwargs)
