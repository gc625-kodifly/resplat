from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians | None,
        global_step: int,
        clamp_large_error: float,
        valid_depth_mask: Tensor | None,
        loss_on_input_views: bool = False
    ) -> Float[Tensor, ""]:
        if loss_on_input_views:
            delta = prediction.color - batch["context"]["image"]
        else:
            delta = prediction.color - batch["target"]["image"]

        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5 and valid_depth_mask.min() < 0.5:
            delta = delta[~valid_depth_mask]

        if clamp_large_error > 0:
            valid_mask = delta.abs() < clamp_large_error
            delta = delta[valid_mask]

        return self.cfg.weight * (delta.abs()).mean()
