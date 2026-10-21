"""
Painter Mel-Band RoFormer
将 ComfyUI-MelBandRoFormer 的 "ModelLoader + Sampler" 子工作流封装为单一节点：
    输入: audio (AUDIO) + model_name (diffusion_models 中的 safetensors)
    输出: vocals (AUDIO) + instruments (AUDIO)

模型代码 vendor 自 kijai/ComfyUI-MelBandRoFormer (ver 7b12f8c)，
位于本插件 melband_model/ 子包，不依赖原插件是否启用。
依赖: rotary_embedding_torch, einops
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.functional as TAF

import folder_paths

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar

from .melband_model import MelBandRoformer

# 与 kijai/ComfyUI-MelBandRoFormer 的 MelBandRoFormerModelLoader 保持一致
MODEL_CONFIG = {
    "dim": 384,
    "depth": 6,
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,  # needed for mel filter bank
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


class PainterMelBandRoFormer:
    """人声/伴奏分离（Mel-Band RoFormer），单节点封装 Loader + Sampler。"""

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Mel-Band RoFormer 模型 (.safetensors)，位于 ComfyUI/models/diffusion_models 目录"},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "instruments")
    FUNCTION = "process"
    CATEGORY = "audio/processing"
    DESCRIPTION = (
        "Mel-Band RoFormer 人声分离：输入音频，输出 vocals（人声）与 instruments（伴奏）。"
        "模型放在 models/diffusion_models，首次加载后按文件名缓存。"
    )

    @classmethod
    def _load_model(cls, model_name):
        model = cls._model_cache.get(model_name)
        if model is not None:
            return model
        model = MelBandRoformer(**MODEL_CONFIG).eval()
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model.load_state_dict(load_torch_file(model_path), strict=True)
        cls._model_cache[model_name] = model
        return model

    def process(self, audio, model_name):
        model = self._load_model(model_name)
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]

        B, audio_channels, audio_length = audio_input.shape

        sr = 44100

        if audio_channels == 1:
            # Convert mono to stereo by duplicating the channel
            audio_input = audio_input.repeat(1, 2, 1)
            audio_channels = 2
            print("[PainterNodes] Mel-Band RoFormer: converted mono input to stereo.")

        if sample_rate != sr:
            print(f"[PainterNodes] Mel-Band RoFormer: resampling input {sample_rate} to {sr}")
            audio_input = TAF.resample(audio_input, orig_freq=sample_rate, new_freq=sr)
        audio_input = original_audio = audio_input[0]

        C = 352800
        N = 2
        step = C // N
        fade_size = C // 10
        border = C - step

        if audio_length > 2 * border and border > 0:
            audio_input = F.pad(audio_input, (border, border), mode='reflect')

        windowing_array = get_windowing_array(C, fade_size, device)

        audio_input = audio_input.to(device)
        vocals = torch.zeros_like(audio_input, dtype=torch.float32).to(device)
        counter = torch.zeros_like(audio_input, dtype=torch.float32).to(device)

        total_length = audio_input.shape[1]
        num_chunks = (total_length + step - 1) // step

        model.to(device)

        comfy_pbar = ProgressBar(num_chunks)

        try:
            for i in tqdm(range(0, total_length, step), desc="Mel-Band RoFormer chunks"):
                part = audio_input[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = F.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = F.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

                x = model(part.unsqueeze(0))[0]

                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                elif i + C >= total_length:
                    window[-fade_size:] = 1

                vocals[..., i:i + length] += x[..., :length] * window[..., :length]
                counter[..., i:i + length] += window[..., :length]
                comfy_pbar.update(1)
        finally:
            model.to(offload_device)

        estimated_sources = vocals / counter

        if audio_length > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]

        vocals_out = {
            "waveform": estimated_sources.unsqueeze(0).cpu(),
            "sample_rate": sr,
        }
        instruments_out = {
            "waveform": (original_audio.to(device) - estimated_sources).unsqueeze(0).cpu(),
            "sample_rate": sr,
        }

        return (vocals_out, instruments_out)


NODE_CLASS_MAPPINGS = {
    "PainterMelBandRoFormer": PainterMelBandRoFormer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterMelBandRoFormer": "Mel-Band RoFormer",
}
