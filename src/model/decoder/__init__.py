from ...dataset import DatasetCfg
from .decoder import Decoder
from .gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg, GSplatDecoderSplattingCUDA

DECODERS = {
    "gsplat": GSplatDecoderSplattingCUDA,
}

DecoderCfg = GSplatDecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
