"""Painter MiniMax FL2VA 2: fl2va + reference audio timbre, with a second conditioning
built for the upscaled canvas.

(v2 of PainterMiniMaxFL2VA. The v1 node is kept untouched and still available.)

What v2 adds on top of v1:
- Two extra INT inputs at the BOTTOM of the panel: `upscale_width` / `upscale_height`
  (both snapped to 32, i.e. step=32), e.g. 1494 / 854 -> 1504 / 864.
- A second CONDITIONING output `upscale_positive`, whose first/last-frame anchors are
  re-encoded at the upscaled canvas size.
- Two extra INT outputs `upscale_width` / `upscale_height` so the values can drive the
  `PainterMinimaxH3LatentUpscaler` node without retyping them.

So the node emits 2 conditionings + 1 latent:
    positive          -> low-res sampling (width x height, aligned to the base latent)
    upscale_positive  -> high-res refine after PainterMinimaxH3LatentUpscaler (upscale_width x upscale_height)
    latent            -> the width x height AV latent (audio stream is size-independent)

Why the two conditionings differ:
The H3 cond rows (first/last frame anchors) are encoded from image pixels at a fixed
canvas size, and `PackedLayout` derives the number of cond rows from that latent's
spatial shape, while `MiniMaxH3Model._forward` also compares the layout `signature`
against the *actual* sampling latent. Feeding a cond whose keyframe latents are
low-res into a high-res sampling pass means the anchor pixels get resampled by the DiT
-- the first/last frames come back soft. Re-encoding the frames at the upscaled canvas
gives the refine pass crisp anchors at exactly the resolution it samples at.

Both patches below are the same runtime fixes v1 needs (they are idempotent, so loading
this module next to v1 is harmless):

Patch 1 (extra_conds overwrite bug):
  cond_video_latents is shared by the keyframes / refs branches; the refs branch
  overwrites the keyframes value. An audio-only ref has no "latent" key, which
  overwrites cond_video_latents with an empty list -> first/last frame latents lost.
  -> patched to concatenate keyframes + refs.

Patch 2 (PackedLayout position misalignment bug, root cause):
  keyframes' cond-segment time positions start from text_len (first frame text_len /
  last frame text_len + video span), but the refs branch pushes cursor by refs_offset
  (audio pushes ref_audio_t, image pushes 1, video pushes its span), so the target
  video segment shifts back while the cond segment stays put -> first/last frames and
  the video are misaligned on the time axis, RoPE positions no longer match, and
  attention cannot hard-anchor the first/last frames (degrades to "identity only").
  -> patched to offset the cond segment's time position by refs_offset after __init__.

Tag numbering (reference in prompt):
    <Picture 1> = first frame (if provided)
    <Picture 2> = last frame (if provided; last frame becomes <Picture 1> when no first frame)
    <Audio 1> = reference audio (timbre)
"""

import torch
import torchaudio

import nodes
import node_helpers
import comfy.model_management
import comfy.nested_tensor
import comfy.model_base
import comfy.utils
import comfy.ldm.minimax.model as minimax_model

from comfy_api.latest import io


CANVAS_MULTIPLE = 32
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# ---------------- Patch 1: extra_conds overwrite bug ----------------
def _apply_extra_conds_patch():
    if not hasattr(comfy.model_base, "MiniMaxH3"):
        print("[PainterMiniMaxFL2VA2] WARNING: MiniMaxH3 not found, extra_conds patch skipped")
        return
    f = comfy.model_base.MiniMaxH3.extra_conds
    if getattr(f, "_minimax_kfref_patched", False):
        return
    orig = f

    def patched_extra_conds(self, **kwargs):
        out = orig(self, **kwargs)
        keyframes = kwargs.get("minimax_keyframes")
        refs = kwargs.get("minimax_refs")
        if keyframes and refs:
            payload = out["minimax_payload"].cond
            payload["cond_video_latents"] = (
                [kf["latent"] for kf in keyframes]
                + [r["latent"] for r in refs if "latent" in r]
            )
        return out

    patched_extra_conds._minimax_kfref_patched = True
    comfy.model_base.MiniMaxH3.extra_conds = patched_extra_conds


# ---------------- Patch 2: PackedLayout position misalignment bug (root cause) ----------------
def _refs_time_offset(refs):
    """Recompute how much the PackedLayout refs branch pushes the cursor (must match its internal logic)."""
    off = 0.0
    for blk in refs:
        kind = blk["kind"]
        if kind == "image":
            off += 1.0
        elif kind == "audio":
            off += float(blk["ref_audio_t"])
        elif kind in ("video", "video_audio"):
            off += max(float(blk["ref_audio_t"]), sum(minimax_model._video_t_spans(blk["latent_t"])))
    return off


def _apply_layout_patch():
    if not hasattr(minimax_model, "PackedLayout"):
        print("[PainterMiniMaxFL2VA2] WARNING: PackedLayout not found, layout patch skipped")
        return
    f = minimax_model.PackedLayout.__init__
    if getattr(f, "_minimax_kfref_layout_patched", False):
        return
    orig = f

    def patched_init(self, *args, **kwargs):
        # Current ComfyUI's PackedLayout.__init__ already offsets keyframe cond
        # positions by the refs span (the `cursor += _ref_t_span(blk)` loop runs
        # before the keyframes block), and its signature no longer accepts frame_count.
        # So we only forward the caller's arguments verbatim and never add our own
        # offset -- re-offsetting here would double-shift the first/last frames and
        # break the hard anchor.
        orig(self, *args, **kwargs)

    patched_init._minimax_kfref_layout_patched = True
    minimax_model.PackedLayout.__init__ = patched_init


_apply_extra_conds_patch()
_apply_layout_patch()


def _snap_canvas(value, fallback):
    """Round a requested canvas size to the DiT's 32px multiple (same rule as step=32)."""
    v = int(fallback if value is None else value)
    v = max(CANVAS_MULTIPLE, v)
    return max(CANVAS_MULTIPLE, round(v / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


class PainterMiniMaxFL2VA2(io.ComfyNode):
    """First/last-frame (fl2va) + reference audio timbre node, with an extra high-res conditioning."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterMiniMaxFL2VA2",
            display_name="Painter MiniMax FL2VA 2",
            category="model/conditioning/minimax",
            description=(
                "First/last frame hard anchor (fl2va) + reference audio timbre, plus a second "
                "conditioning built for the upscaled canvas. Outputs positive (width x height), "
                "upscale_positive (upscale_width x upscale_height) and the AV latent. "
                "The voice follows <Audio 1> timbre; write the lines in the prompt."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17),
                io.Image.Input("first_frame", optional=True, tooltip="First frame hard anchor (<Picture 1>)"),
                io.Image.Input("last_frame", optional=True, tooltip="Last frame hard anchor (<Picture 2>; becomes <Picture 1> when no first frame)"),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Reference audio (timbre/style, <Audio 1> and up)"),
                        prefix="ref_audio_", min=0, max=3)),
                # ---- bottom of the panel: upscale canvas ----
                io.Int.Input("upscale_width", default=1504, min=32, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="Canvas width of the upscale_positive conditioning (snapped to 32). Feed the same value to PainterMinimaxH3LatentUpscaler for the refine pass."),
                io.Int.Input("upscale_height", default=864, min=32, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="Canvas height of the upscale_positive conditioning (snapped to 32). Feed the same value to PainterMinimaxH3LatentUpscaler for the refine pass."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="upscale_positive"),
                io.Latent.Output(),
                io.Int.Output(display_name="upscale_width"),
                io.Int.Output(display_name="upscale_height"),
            ],
        )

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                first_frame=None, last_frame=None, ref_audios=None,
                upscale_width=1504, upscale_height=864) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        up_width = _snap_canvas(upscale_width, width)
        up_height = _snap_canvas(upscale_height, height)

        # first/last frame presentation targets: (canvas w, canvas h, crop, resolved frame index)
        anchors = []
        if first_frame is not None:
            anchors.append((first_frame[:1], width, height, "disabled", 0))
        if last_frame is not None:
            anchors.append((last_frame[:1], width, height, "center", frame_count - 1))

        # 1) base keyframes + ref_items: frames are resized per canvas, presentation kept separate
        keyframes = []
        ref_items = []
        for image, w, h, crop, frame_index in anchors:
            img = _resize(image, w, h, crop)
            keyframes.append({"resolved_frame_index": frame_index, "image": img})
            ref_items.append({"type": "image", "data": img})

        # 2) reference audio (timbre) -- size independent, shared by both conditionings
        ref_blocks = []
        for audio in (ref_audios or {}).values():
            if audio is None:
                continue
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        # 3) base first/last frame latents (presentation already saved into ref_items)
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))

        # 4) tokenize once; text is canvas independent, so both conditionings reuse the tokens
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if keyframes:
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)

        # 5) upscale conditioning: same text/audio, but the first/last frame anchors are encoded
        #    at the upscaled canvas so the refine pass gets crisp hard anchors.
        upscale_cond = cond
        if anchors:
            up_kfs = []
            for image, _w, _h, crop, frame_index in anchors:
                up_kfs.append({
                    "resolved_frame_index": frame_index,
                    "latent": vae.encode(_resize(image, up_width, up_height, crop)),
                })
            up_values = {"minimax_keyframes": up_kfs, "minimax_frame_count": frame_count}
            if ref_blocks:
                up_values["minimax_refs"] = ref_blocks
            upscale_cond = node_helpers.conditioning_set_values(cond, up_values)

        return io.NodeOutput(cond, upscale_cond, latent, up_width, up_height)


NODE_CLASS_MAPPINGS = {
    "PainterMiniMaxFL2VA2": PainterMiniMaxFL2VA2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterMiniMaxFL2VA2": "Painter MiniMax FL2VA 2"
}
