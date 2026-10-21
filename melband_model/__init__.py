# Vendored model code from kijai/ComfyUI-MelBandRoFormer
# (ver 7b12f8c6105666552ac4e4f08b73e0f96eb05c64, model/mel_band_roformer.py & model/mel_converter.py)
# Kept self-contained inside ComfyUI-PainterNodes so the wrapper node works
# regardless of whether the original plugin is enabled.
from .mel_band_roformer import MelBandRoformer

__all__ = ["MelBandRoformer"]
