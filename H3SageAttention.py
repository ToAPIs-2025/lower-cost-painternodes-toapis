"""H3SageAttention: standalone memory-efficient SageAttention patch for MiniMax H3.

Self-contained port of the int8-QK / fp8-V SageAttention path. It calls sageattention's
low-level quantization + attention kernels directly instead of the top-level ``sageattn``
API, so the float q/k/v can be released *before* the kernel runs -- attention is the VRAM
peak in these models, and the top-level API keeps extra float copies alive across the call.

No dependency on ComfyUI-KJNodes or any other custom node pack. The only requirements are:
  * a recent ``sageattention`` build with CUDA kernels compiled for this GPU
  * ComfyUI's native MiniMax H3 support (comfy/ldm/minimax/model.py + comfy_kitchen)

``head_chunks`` splits the attention call into head groups. Heads are independent, so the
result is mathematically unchanged while the kernel's internal transients (int8 q/k copies,
fp32 accumulator) shrink by the chunk count.

Supported GPU archs: sm75, sm80, sm86, sm89, sm90, sm120, sm121.
"""

import importlib
import logging
import types

import torch

import comfy.model_management as mm

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# --------------------------------------------------------------------------- #
# sageattention capability probing
# --------------------------------------------------------------------------- #

def get_cuda_version():
    try:
        version = torch.version.cuda
        if version is not None:
            major, minor = version.split('.')
            return int(major), int(minor)
        return 0, 0
    except Exception:
        return 0, 0


sageplus_sm89_available = False
_cuda_archs = None
try:
    # These are sageattention's *internal* symbols; importing them is the version gate.
    from sageattention.core import (
        per_thread_int8_triton,
        per_warp_int8_cuda,
        per_block_int8_triton,
        per_channel_fp8,
        get_cuda_arch_versions,
        attn_false,
    )
    _cuda_archs = get_cuda_arch_versions()
except Exception:
    per_thread_int8_triton = None
    per_warp_int8_cuda = None
    per_block_int8_triton = None
    per_channel_fp8 = None
    attn_false = None

_QATTN_PROBE = {
    "sm80": "qk_int8_sv_f16_accum_f32_attn",
    "sm89": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
    "sm90": "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf",
}


def _resolve_qattn(arch):
    """Locate sageattention's compiled attention module across its layout changes."""
    try:
        core = importlib.import_module("sageattention.core")
    except Exception:
        return None
    candidates = [getattr(core, f"_qattn_{arch}", None), getattr(core, f"{arch}_compile", None)]
    try:
        mod = importlib.import_module(f"sageattention.{arch}_compile")
        candidates += [mod, getattr(mod, f"_qattn_{arch}", None)]
    except Exception:
        pass
    for obj in candidates:
        if obj is not None and hasattr(obj, _QATTN_PROBE[arch]):
            return obj
    return None


_qattn_sm80 = _resolve_qattn("sm80")
_qattn_sm89 = _resolve_qattn("sm89")
_qattn_sm90 = _resolve_qattn("sm90")

if _qattn_sm89 is not None:
    sageplus_sm89_available = (
        hasattr(_qattn_sm89, 'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf')
        and get_cuda_version() >= (12, 8)
    )

try:
    from comfy.ldm.minimax.model import MiniMaxH3Model as _MiniMaxH3Model
except ImportError:
    _MiniMaxH3Model = None

try:
    from comfy.quant_ops import ck as _ck
except ImportError:
    _ck = None

SUPPORTED_ARCHS = {"sm75", "sm80", "sm86", "sm89", "sm90", "sm120", "sm121"}


# --------------------------------------------------------------------------- #
# int64-safe per-thread int8 quantization
#
# Vendored from sageattention's quant_per_thread.py with the row offsets promoted to
# int64 to avoid index overflow on long sequences (H3 packs text + video tokens, so the
# sequence length gets large enough to trip the upstream int32 offsets).
# --------------------------------------------------------------------------- #

if HAS_TRITON:
    @triton.jit
    def _painter_quant_query_per_thread_int8_i64_kernel(Input, Output, Scale, L,
                                                        stride_iz, stride_ih, stride_in,
                                                        stride_oz, stride_oh, stride_on,
                                                        stride_sz, stride_sh,
                                                        C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 8
        off_tld = tl.program_id(0) % 8
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld
        offs_k = tl.arange(0, C)

        input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 8 + off_tld

        x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
        x = x.to(tl.float32)
        scale = tl.max(tl.abs(x)) / 127. + 0.0000001
        x_int8 = x / scale
        x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
        x_int8 = x_int8.to(tl.int8)
        tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
        tl.store(scale_ptrs, scale)

    @triton.jit
    def _painter_quant_key_per_thread_int8_i64_kernel(Input, Output, Scale, L,
                                                      stride_iz, stride_ih, stride_in,
                                                      stride_oz, stride_oh, stride_on,
                                                      stride_sz, stride_sh,
                                                      C: tl.constexpr, BLK: tl.constexpr):
        off_blk = tl.program_id(0) // 4
        off_tld = tl.program_id(0) % 4
        off_h = tl.program_id(1)
        off_b = tl.program_id(2)

        offs_n0 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2
        offs_n1 = off_blk * BLK + tl.arange(0, BLK // 8) * 8 + off_tld * 2 + 1
        offs_k = tl.arange(0, C)

        input_ptrs0 = Input + off_b * stride_iz + off_h * stride_ih + offs_n0[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        input_ptrs1 = Input + off_b * stride_iz + off_h * stride_ih + offs_n1[:, None].to(tl.int64) * stride_in + offs_k[None, :]
        output_ptrs0 = Output + off_b * stride_oz + off_h * stride_oh + offs_n0[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        output_ptrs1 = Output + off_b * stride_oz + off_h * stride_oh + offs_n1[:, None].to(tl.int64) * stride_on + offs_k[None, :]
        scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk * 4 + off_tld

        x0 = tl.load(input_ptrs0, mask=offs_n0[:, None] < L)
        x1 = tl.load(input_ptrs1, mask=offs_n1[:, None] < L)
        x0 = x0.to(tl.float32)
        x1 = x1.to(tl.float32)
        scale = max(tl.max(tl.abs(x0)), tl.max(tl.abs(x1))) / 127. + 0.0000001
        x0_int8 = x0 / scale
        x1_int8 = x1 / scale
        x0_int8 += 0.5 * tl.where(x0_int8 >= 0, 1, -1)
        x1_int8 += 0.5 * tl.where(x1_int8 >= 0, 1, -1)
        x0_int8 = x0_int8.to(tl.int8)
        x1_int8 = x1_int8.to(tl.int8)
        tl.store(output_ptrs0, x0_int8, mask=offs_n0[:, None] < L)
        tl.store(output_ptrs1, x1_int8, mask=offs_n1[:, None] < L)
        tl.store(scale_ptrs, scale)


def _per_thread_int8_i64(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, tensor_layout="NHD"):
    """Drop-in for sageattention's per_thread_int8_triton using the int64-safe kernels."""
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    _painter_quant_query_per_thread_int8_i64_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    _painter_quant_key_per_thread_int8_i64_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK
    )

    return q_int8, q_scale, k_int8, k_scale


# --------------------------------------------------------------------------- #
# the attention kernel dispatch
# --------------------------------------------------------------------------- #

def _sageattn_int8_fp8_nhd(qkv, dtype):
    """qkv: [q, k, v], each [batch, seq_len, num_heads, head_dim] in NHD layout.

    Returns o in the same layout. The list is consumed so the only references to the float
    q/k/v are the locals below, letting ``del`` actually free them before the kernel runs.
    """
    q, k, v = qkv
    qkv.clear()
    head_dim_og = q.shape[-1]

    tensor_layout = "NHD"
    _tensor_layout = 0  # NHD
    _is_caual = 0
    _qk_quant_gran = 3
    _return_lse = 0
    sm_scale = head_dim_og**-0.5
    quant_v_scale_max = 448.0

    arch = _cuda_archs[0]

    if arch in {"sm80", "sm86"}:
        if _qattn_sm80 is None:
            raise RuntimeError("H3SageAttention: sageattention's sm80 attention kernel is not available.")
        # mean-sub in-place: passing km= would make the triton quant materialize k - km as a full float copy
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        v_fp16 = v.to(torch.float16)
        del v
        _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v_fp16, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif arch == "sm75":
        if per_block_int8_triton is None or attn_false is None:
            raise RuntimeError("H3SageAttention: sageattention's sm75 path is not available.")
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout)
        del q, k
        o, _ = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, attn_mask=None, return_lse=False)
        del v
    elif arch == "sm89":
        if _qattn_sm89 is None:
            raise RuntimeError("H3SageAttention: sageattention's sm89 attention kernel is not available.")
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32+fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif arch == "sm90":
        if _qattn_sm90 is None:
            raise RuntimeError("H3SageAttention: sageattention's sm90 attention kernel is not available.")
        k.sub_(k.mean(dim=1, keepdim=True))
        q_int8, q_scale, k_int8, k_scale = _per_thread_int8_i64(q, k, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)
        del q, k
        # sageattention's sm90 kernel requires kv_len padded to CTA_K=128, but
        # per_channel_fp8 only pads to 64 (upstream TODO in sageattention/quant.py).
        # sageattention.core.sageattn_qk_int8_pv_fp8_cuda_sm90 handles this at the
        # top-level API; mirror that pad here so kv_len % 128 != 0 (e.g. MiniMax H3 with
        # long text prompts) does not trigger a C++ assertion abort.
        seq_dim = 1  # NHD layout
        kv_len = v.size(seq_dim)
        v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
        if v_pad_len > 0:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=seq_dim)
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    elif arch in {"sm120", "sm121"}:
        if _qattn_sm89 is None:
            raise RuntimeError("H3SageAttention: sageattention's sm120 attention kernel is not available.")
        if not sageplus_sm89_available:
            pv_accum_dtype = "fp32"
        else:
            pv_accum_dtype = "fp32+fp16"
            quant_v_scale_max = 2.25
        _qk_quant_gran = 2  # per warp
        # km kept here: the CUDA quant fuses the mean-sub with no temp copy
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
        del q, k
        v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=False)
        del v
        o = torch.empty(q_int8.size(), dtype=dtype, device=q_int8.device)
        if pv_accum_dtype == "fp32":
            _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        elif pv_accum_dtype == "fp32+fp16":
            _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        del v_fp8, v_scale
    else:
        raise RuntimeError(f"H3SageAttention: unsupported CUDA arch '{arch}'. Supported: {sorted(SUPPORTED_ARCHS)}")

    del q_int8, q_scale, k_int8, k_scale
    return o


# --------------------------------------------------------------------------- #
# replacement forward for MiniMax H3 Attention
# --------------------------------------------------------------------------- #

def h3_sageattn_forward(self, x, rope_freqs=None, transformer_options={}, head_chunks=None):
    """Drop-in replacement for comfy.ldm.minimax.model.Attention.forward.

    x: [S, hidden], unbatched packed sequence; q/k/v are NHD views into the fused qkv buffer.
    A single-item list hands over the sole reference to the block's normed h so it can be
    freed right after the qkv GEMM (matches MiniMax H3's block-level memory handling).
    """
    if isinstance(x, list):
        x = x.pop()
    dtype = x.dtype
    device = x.device
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    del x
    q = q.view(1, s, self.heads, self.head_dim)
    k = k.view(1, s, self.heads, self.head_dim)
    v = v.view(1, s, self.heads, self.head_dim)
    if rope_freqs is not None:
        # same fused per-head RMSNorm + partial split-half rope the stock forward uses,
        # in place on the qkv buffer
        qw = mm.cast_to(self.q_norm.weight, device=device)
        kw = mm.cast_to(self.k_norm.weight, device=device)
        _ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2)
    else:
        q = self.q_norm(q)
        k = self.k_norm(k)

    n = head_chunks or 1
    if n <= 1 and isinstance(transformer_options, dict):
        # also honour a head_chunks set elsewhere in the workflow
        n = transformer_options.get("minimax_head_chunks", 1)
    n = min(n, self.heads)

    if n <= 1:
        qkv = [q, k, v]
        del q, k, v
        o = _sageattn_int8_fp8_nhd(qkv, dtype)
        return self.out_proj(o.view(s, self.heads * self.head_dim))

    # head-group slicing: quantize and attend per group so the int8/fp8 working set
    # shrinks by the group count
    out = torch.empty((s, self.heads * self.head_dim), dtype=dtype, device=device)
    out_nhd = out.view(1, s, self.heads, self.head_dim)
    hs = 0
    for i in range(n):
        he = hs + self.heads // n + (1 if i < self.heads % n else 0)
        out_nhd[:, :, hs:he] = _sageattn_int8_fp8_nhd([q[:, :, hs:he], k[:, :, hs:he], v[:, :, hs:he]], dtype)
        hs = he
    del q, k, v  # last references to the fused qkv buffer -> freed before out_proj
    return self.out_proj(out)


class _H3SageForward:
    """Descriptor that binds a fixed head_chunks value onto the patched attention forward."""

    def __init__(self, head_chunks):
        self.head_chunks = head_chunks

    def __get__(self, obj, objtype=None):
        def wrapped_forward(self_module, *args, **kwargs):
            kwargs.pop("head_chunks", None)
            return h3_sageattn_forward(self_module, *args, head_chunks=self.head_chunks, **kwargs)
        return types.MethodType(wrapped_forward, obj)


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #

class H3SageAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "head_chunks": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 56,
                        "step": 1,
                        "tooltip": "Split the attention call into this many head groups. "
                                   "Heads are independent so the result is unchanged; the "
                                   "kernel's internal transients shrink by the chunk count. "
                                   "Measured on a 56-head/128-dim block at 32k tokens: "
                                   "1 = no saving (slightly more than stock), 4 = ~7% below "
                                   "stock. 1 is the fastest, higher values trade speed for VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    OUTPUT_NODE = False

    CATEGORY = "Painter/MiniMax"
    DESCRIPTION = (
        "EXPERIMENTAL! Replaces every MiniMax H3 self-attention with int8-QK / fp8-V "
        "SageAttention called at kernel level, so the float q/k/v are freed before the "
        "kernel runs. Reduces peak VRAM and overrides the attention mode. "
        "Requires a recent sageattention build with CUDA kernels for this GPU."
    )

    def patch(self, model, head_chunks):
        if _cuda_archs is None:
            raise RuntimeError(
                "H3SageAttention: sageattention is not installed, is too old, or its CUDA "
                "arch could not be determined - the patch cannot be applied."
            )
        if _cuda_archs[0] not in SUPPORTED_ARCHS:
            raise RuntimeError(
                f"H3SageAttention: unsupported CUDA arch '{_cuda_archs[0]}'. "
                f"Supported: {sorted(SUPPORTED_ARCHS)}"
            )
        if _ck is None:
            raise RuntimeError(
                "H3SageAttention: comfy.quant_ops.ck (comfy_kitchen) is unavailable, "
                "so the fused MiniMax H3 norm+RoPE kernel cannot be used."
            )

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")

        if _MiniMaxH3Model is not None and not isinstance(diffusion_model, _MiniMaxH3Model):
            raise RuntimeError("H3SageAttention can only be applied to a MiniMax H3 model.")

        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks or not hasattr(blocks[0], "attn") or not hasattr(blocks[0].attn, "qkv_proj"):
            raise RuntimeError(
                "H3SageAttention: the model does not look like MiniMax H3 "
                "(expected diffusion_model.blocks[*].attn.qkv_proj)."
            )

        logging.info(
            f"Applying H3SageAttention to {len(blocks)} transformer blocks "
            f"(arch={_cuda_archs[0]}, sm89_plus_kernels={sageplus_sm89_available}, head_chunks={head_chunks})"
        )

        existing = getattr(model_clone, "object_patches", {})
        if any(f"diffusion_model.blocks.{i}.attn.forward" in existing for i in range(len(blocks))):
            logging.info("H3SageAttention: replacing an existing attention patch on this model.")

        for idx, block in enumerate(blocks):
            model_clone.add_object_patch(
                f"diffusion_model.blocks.{idx}.attn.forward",
                _H3SageForward(head_chunks).__get__(block.attn, block.attn.__class__),
            )

        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "H3SageAttention": H3SageAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SageAttention": "H3 SageAttention (Mem Eff)",
}
