from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from .perceptual_loss import PerceptualLoss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    perceptual_loss: bool


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        if self.cfg.perceptual_loss:
            self.lpips = PerceptualLoss()
        else:
            self.lpips = LPIPS(net="vgg")

        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
        valid_depth_mask: Tensor | None,
        loss_on_input_views: bool = False,
        half_res_lpips: bool = False,
    ) -> Float[Tensor, ""]:
        if loss_on_input_views:
            image = batch["context"]["image"]
        else:
            image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        
        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5:
            prediction.color[valid_depth_mask] = 0
            image[valid_depth_mask] = 0

        if self.cfg.perceptual_loss:
            pred = rearrange(prediction.color, "b v c h w -> (b v) c h w")
            gt = rearrange(image, "b v c h w -> (b v) c h w")

            if half_res_lpips:
                pred = torch.nn.functional.interpolate(
                    pred, scale_factor=0.5, mode="bilinear", align_corners=True
                )
                gt = torch.nn.functional.interpolate(
                    gt, scale_factor=0.5, mode="bilinear", align_corners=True
                )

            loss = self.lpips.forward(
                pred,
                gt,
            )
        else:
            loss = self.lpips.forward(
                rearrange(prediction.color, "b v c h w -> (b v) c h w"),
                rearrange(image, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
        return self.cfg.weight * loss.mean()
