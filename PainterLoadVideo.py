# PainterLoadVideo
# FFmpeg based video loader for PainterNodes.
# Decodes frames through an ffmpeg rawvideo pipe, supports a start-frame
# offset, optional rate conversion, custom sizing, frame caps and model
# specific format presets. No batch manager, no VAE input, no mask output.

import os
import re
import shutil
import hashlib
import subprocess
import time
import itertools
from collections.abc import Mapping

import numpy as np
import torch

import folder_paths
from comfy.utils import ProgressBar

try:
    import psutil
except Exception:
    psutil = None


SUPPORTED_EXTENSIONS = ("webm", "mp4", "mkv", "gif", "mov", "m4v", "avi")
MAX_INT = 0xFFFFFFFFFFFFFFFF
TEXT_ENCODING = ("utf-8", "backslashreplace")

# Preset table: dim = (downscale_ratio, reserved, suggested_w, suggested_h)
# frames = (divisor, remainder[, strict]) controls the loaded frame count.
PAINTER_FORMAT_PRESETS = {
    "None": {},
    "LTXV": {"target_rate": 24, "dim": (32, 0, 768, 512), "frames": (8, 1)},
    "Hunyuan": {"target_rate": 24, "dim": (16, 0, 848, 480), "frames": (4, 1)},
    "Wan": {"target_rate": 16, "dim": (8, 0, 832, 480), "frames": (4, 1)},
    "H3": {"target_rate": 24, "dim": (32, 0, 1344, 768), "frames": (17, 5)},
}
DEFAULT_FORMAT = "H3"


def _ffmpeg_executable():
    env_path = os.environ.get("PAINTER_FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    candidates = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        candidates.append(get_ffmpeg_exe())
    except Exception:
        pass
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg is not None:
        candidates.append(system_ffmpeg)
    for fallback_name in ("ffmpeg", "ffmpeg.exe"):
        if os.path.isfile(fallback_name):
            candidates.append(os.path.abspath(fallback_name))
    if not candidates:
        return None
    return candidates[0]


def _probe_stream(ffmpeg, video_path):
    """Run a cheap copy pass so ffmpeg prints the stream summary, then
    parse resolution, fps, alpha presence and duration from stderr."""
    input_args = ["-i", video_path]
    probe_args = [ffmpeg] + input_args + ["-c", "copy", "-frames:v", "1",
                                          "-f", "null", "-"]
    try:
        probe = subprocess.run(probe_args, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("PainterLoadVideo: ffmpeg failed to probe the file:\n"
                           + exc.stderr.decode(*TEXT_ENCODING))
    summary = probe.stderr.decode(*TEXT_ENCODING)
    if "Video: vp9 " in summary:
        input_args = ["-c:v", "libvpx-vp9"] + input_args
        probe_args = [ffmpeg] + input_args + ["-c", "copy", "-frames:v", "1",
                                              "-f", "null", "-"]
        try:
            probe = subprocess.run(probe_args, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("PainterLoadVideo: ffmpeg failed to probe the vp9 stream:\n"
                               + exc.stderr.decode(*TEXT_ENCODING))
        summary = probe.stderr.decode(*TEXT_ENCODING)

    width = height = None
    fps = 1.0
    has_alpha = False
    for line in summary.split("\n"):
        size_match = re.search("^ *Stream .* Video.*, ([1-9]|\\d{2,})x(\\d+)", line)
        if size_match is None:
            continue
        width, height = int(size_match.group(1)), int(size_match.group(2))
        fps_match = re.search(", ([\\d\\.]+) fps", line)
        if fps_match:
            fps = float(fps_match.group(1))
        has_alpha = re.search("(yuva|rgba|bgra|gbra)", line) is not None
        break
    if width is None or height is None:
        raise RuntimeError("PainterLoadVideo: unable to parse video information:\n" + summary)

    duration = 0.0
    dur_match = re.search("Duration: (\\d+:\\d+:\\d+\\.\\d+),", summary)
    if dur_match:
        parts = dur_match.group(1).split(":")
        duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return width, height, fps, duration, has_alpha, input_args


def _fit_size(src_w, src_h, custom_w, custom_h, downscale_ratio):
    if downscale_ratio is None or downscale_ratio < 1:
        downscale_ratio = 8
    if custom_w == 0 and custom_h == 0:
        return src_w, src_h
    if custom_h == 0:
        src_h *= custom_w / src_w
        src_w = custom_w
    elif custom_w == 0:
        src_w *= custom_h / src_h
        src_h = custom_h
    else:
        src_w, src_h = custom_w, custom_h
    src_w = int(src_w / downscale_ratio + 0.5) * downscale_ratio
    src_h = int(src_h / downscale_ratio + 0.5) * downscale_ratio
    return src_w, src_h


def _frame_stream(video_path, force_rate, frame_load_cap, start_frame,
                  custom_width, custom_height, downscale_ratio=8):
    """Generator: first yields the stream info tuple, then decoded frames
    as float32 numpy arrays (H, W, 3) or (H, W, 4) when alpha is present."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("PainterLoadVideo: no ffmpeg executable was found.")

    src_w, src_h, src_fps, duration, has_alpha, input_args = _probe_stream(ffmpeg, video_path)

    # Convert the requested start frame into a seek time in seconds.
    start_time = 0.0
    if start_frame > 0 and src_fps > 0:
        start_time = start_frame / src_fps
    pre_seek, post_seek = [], []
    if start_time > 0:
        if start_time > 4:
            # Fast input side seek, then an accurate trim of the first 4s.
            input_args = ["-ss", str(start_time - 4)] + input_args
            post_seek = ["-ss", "4"]
        else:
            post_seek = ["-ss", str(start_time)]

    decode_args = [ffmpeg, "-v", "error", "-an"] + input_args + \
        ["-pix_fmt", "rgba64le"] + post_seek

    out_w, out_h = src_w, src_h
    filters = []
    if force_rate:
        filters.append("fps=fps=" + str(force_rate))
    if custom_width != 0 or custom_height != 0:
        out_w, out_h = _fit_size(src_w, src_h, custom_width, custom_height,
                                 downscale_ratio)
        aspect = float(out_w) / float(out_h)
        if abs(src_h * aspect - src_w) >= 1:
            # Aspect ratio changed: crop before scaling to avoid distortion.
            filters.append(
                "crop=if(gt({a}\\,a)\\,iw\\,ih*{a}):if(gt({a}\\,a)\\,iw/{a}\\,ih)".format(a=aspect))
        filters.append("scale={}:{}".format(out_w, out_h))
    if filters:
        decode_args += ["-vf", ",".join(filters)]

    frame_step = 1.0 / (force_rate or src_fps)
    expected_frames = (force_rate or src_fps) * duration
    if frame_load_cap > 0:
        decode_args += ["-frames:v", str(frame_load_cap)]
        expected_frames = min(expected_frames, frame_load_cap)

    yield (src_w, src_h, src_fps, duration, int(round(src_fps * duration)),
           frame_step, int(expected_frames), out_w, out_h, has_alpha)

    decode_args += ["-f", "rawvideo", "-"]
    pbar = ProgressBar(int(expected_frames))
    try:
        with subprocess.Popen(decode_args, stdout=subprocess.PIPE) as proc:
            bytes_per_frame = out_w * out_h * 8
            buffer = bytearray(bytes_per_frame)
            filled = 0
            previous = None
            while True:
                chunk = proc.stdout.read(bytes_per_frame - filled)
                if chunk is None:
                    time.sleep(0.1)
                    continue
                if len(chunk) == 0:
                    break
                buffer[filled:filled + len(chunk)] = chunk
                filled += len(chunk)
                if filled == bytes_per_frame:
                    if previous is not None:
                        yield previous
                        pbar.update(1)
                    previous = np.frombuffer(
                        buffer, dtype=np.dtype(np.uint16).newbyteorder("<")
                    ).reshape(out_h, out_w, 4) / (2 ** 16 - 1)
                    if not has_alpha:
                        previous = previous[:, :, :-1]
                    filled = 0
    except BrokenPipeError:
        raise RuntimeError("PainterLoadVideo: the ffmpeg decoding process crashed.")
    if previous is not None:
        yield previous


def _extract_audio(video_path, start_time=0.0, duration=0.0):
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("PainterLoadVideo: no ffmpeg executable was found.")
    args = [ffmpeg, "-i", video_path]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        result = subprocess.run(args + ["-f", "f32le", "-"],
                                capture_output=True, check=True)
        waveform = torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)
        info_match = re.search(", (\\d+) Hz, (\\w+), ",
                               result.stderr.decode(*TEXT_ENCODING))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("PainterLoadVideo: failed to extract audio:\n"
                           + exc.stderr.decode(*TEXT_ENCODING))
    if info_match:
        sample_rate = int(info_match.group(1))
        channels = {"mono": 1, "stereo": 2}.get(info_match.group(2), 2)
    else:
        sample_rate = 44100
        channels = 2
    waveform = waveform.reshape((-1, channels)).transpose(0, 1).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


class _DeferredAudio(Mapping):
    """Mapping that only runs the audio extraction when it is consumed."""
    def __init__(self, video_path, start_time, duration):
        self.path = video_path
        self.start_time = start_time
        self.duration = duration
        self._cached = None

    def _load(self):
        if self._cached is None:
            self._cached = _extract_audio(self.path, self.start_time, self.duration)
        return self._cached

    def __getitem__(self, key):
        return self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())


def _frame_budget(width, height):
    """Upper bound on how many decoded frames may be held in RAM."""
    if psutil is None:
        return MAX_INT
    try:
        free_bytes = psutil.virtual_memory().available + psutil.swap_memory().free
        free_bytes -= 2 ** 27
    except Exception:
        return MAX_INT
    return max(int(free_bytes // (width * height * 3 * 0.1)), 1)


class PainterLoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for name in os.listdir(input_dir):
            full = os.path.join(input_dir, name)
            if not os.path.isfile(full):
                continue
            parts = name.split(".")
            if len(parts) > 1 and parts[-1].lower() in SUPPORTED_EXTENSIONS:
                files.append(name)
        return {
            "required": {
                "video": (sorted(files), {"video_upload": True}),
                "force_rate": ("PAINTER_NUMBER", {"default": 0.0, "min": 0.0, "max": 120.0,
                                                  "step": 1.0, "precision": 2, "reset": 0.0}),
                "custom_width": ("PAINTER_NUMBER", {"default": 0, "min": 0, "max": 8192,
                                                    "step": 1, "precision": 0, "reset": 0}),
                "custom_height": ("PAINTER_NUMBER", {"default": 0, "min": 0, "max": 8192,
                                                     "step": 1, "precision": 0, "reset": 0}),
                "frame_load_cap": ("PAINTER_NUMBER", {"default": 0, "min": 0, "max": 1000000000,
                                                      "step": 1, "precision": 0, "reset": 0}),
                "start_frame": ("PAINTER_NUMBER", {"default": 0, "min": 0, "max": 1000000000,
                                                   "step": 1, "precision": 0, "reset": 0}),
            },
            "optional": {
                "format": (list(PAINTER_FORMAT_PRESETS.keys()),
                           {"default": DEFAULT_FORMAT}),
            },
        }

    CATEGORY = "PainterNodes"

    RETURN_TYPES = ("IMAGE", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "audio", "video_info")

    FUNCTION = "load_frames"

    def load_frames(self, video, force_rate=0.0, custom_width=0, custom_height=0,
                    frame_load_cap=0, start_frame=0, format=DEFAULT_FORMAT, **kwargs):
        video_path = folder_paths.get_annotated_filepath(video.strip().strip('"'))
        preset = PAINTER_FORMAT_PRESETS.get(format, {})
        downscale_ratio = preset.get("dim", (8,))[0] or 8

        stream = _frame_stream(video_path, float(force_rate), frame_load_cap,
                               start_frame, custom_width, custom_height,
                               downscale_ratio)
        (src_w, src_h, src_fps, src_duration, src_total, frame_step,
         _expected, out_w, out_h, has_alpha) = next(stream)

        budget = _frame_budget(src_w, src_h)
        decoded = np.fromiter(
            itertools.islice(stream, budget),
            np.dtype((np.float32, (out_h, out_w, 4 if has_alpha else 3))))
        if len(decoded) == 0:
            raise RuntimeError("PainterLoadVideo: no frames were decoded from the video.")
        images = torch.from_numpy(decoded)
        if images.size(3) == 4:
            # Alpha is not needed downstream, keep RGB only.
            images = images[:, :, :, :3]

        spec = preset.get("frames")
        if spec and len(images) % spec[0] != spec[1]:
            if len(spec) > 2 and spec[2]:
                raise RuntimeError(
                    "PainterLoadVideo: loaded {} frames, which is incompatible "
                    "with the selected format.".format(len(images)))
            divisor, remainder = spec[0], spec[1]
            keep = ((len(images) - remainder) // divisor) * divisor + remainder
            if keep <= 0:
                raise RuntimeError(
                    "PainterLoadVideo: too few frames for the selected format "
                    "({} frames required, got {}).".format(remainder, len(images)))
            images = images[:keep]

        audio_start = start_frame / src_fps if src_fps > 0 else 0.0
        audio_duration = frame_load_cap * frame_step if frame_load_cap else 0.0
        audio = _DeferredAudio(video_path, audio_start, audio_duration)

        video_info = {
            "source_fps": src_fps,
            "source_frame_count": src_total,
            "source_duration": src_duration,
            "source_width": src_w,
            "source_height": src_h,
            "loaded_fps": 1.0 / frame_step,
            "loaded_frame_count": len(images),
            "loaded_duration": len(images) * frame_step,
            "loaded_width": out_w,
            "loaded_height": out_h,
        }
        return (images, audio, video_info)

    @classmethod
    def IS_CHANGED(s, video, **kwargs):
        path = folder_paths.get_annotated_filepath(video)
        stat = os.stat(path)
        return "{}|{}|{}".format(path, stat.st_size, stat.st_mtime)

    @classmethod
    def VALIDATE_INPUTS(s, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True


def _probe_video_meta(video_path):
    """Collect fps / duration / frame count for the front-end hints."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("no ffmpeg executable was found")
    width, height, fps, duration, _alpha, _args = _probe_stream(ffmpeg, video_path)
    frames = int(round(fps * duration)) if fps > 0 and duration > 0 else 0
    return {
        "fps": fps,
        "duration": duration,
        "frames": frames,
        "width": width,
        "height": height,
    }


_META_CACHE = {}


def _cached_probe(video_path):
    stat = os.stat(video_path)
    key = (stat.st_mtime, stat.st_size)
    cached = _META_CACHE.get(video_path)
    if cached is not None and cached[0] == key:
        return cached[1]
    meta = _probe_video_meta(video_path)
    _META_CACHE[video_path] = (key, meta)
    return meta


PREVIEW_MAX_FRAMES = 240
PREVIEW_MAX_WIDTH = 960


def _build_preview(video_path, force_rate, custom_width, custom_height,
                   frame_load_cap, start_frame, fmt):
    """Render a small mp4 that reflects the current widget parameters so the
    node preview updates live. Returns the file path to serve."""
    if (force_rate == 0 and custom_width == 0 and custom_height == 0
            and frame_load_cap == 0 and start_frame == 0
            and not video_path.lower().endswith(".gif")):
        return video_path

    meta = _cached_probe(video_path)
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("no ffmpeg executable was found")

    src_fps = meta["fps"] or 24.0
    start_time = start_frame / src_fps if start_frame > 0 else 0.0
    remaining = max(0, meta["frames"] - start_frame)
    frames = frame_load_cap if frame_load_cap > 0 else remaining
    frames = max(1, min(frames, PREVIEW_MAX_FRAMES, remaining or PREVIEW_MAX_FRAMES))

    preset = PAINTER_FORMAT_PRESETS.get(fmt, {})
    downscale_ratio = preset.get("dim", (8,))[0] or 8
    if custom_width or custom_height:
        out_w, out_h = _fit_size(meta["width"], meta["height"], custom_width,
                                 custom_height, downscale_ratio)
    elif meta["width"] > PREVIEW_MAX_WIDTH:
        out_w = PREVIEW_MAX_WIDTH
        out_h = int(meta["height"] * (PREVIEW_MAX_WIDTH / meta["width"]) / 8 + 0.5) * 8
    else:
        out_w, out_h = meta["width"], meta["height"]

    cache_dir = os.path.join(folder_paths.get_temp_directory(), "painter_video_preview")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1("|".join([
        os.path.basename(video_path), str(os.stat(video_path).st_mtime),
        str(force_rate), str(custom_width), str(custom_height),
        str(frame_load_cap), str(start_frame), fmt,
    ]).encode("utf-8")).hexdigest()[:20]
    out_path = os.path.join(cache_dir, f"plv_{key}.mp4")
    if os.path.isfile(out_path):
        return out_path

    filters = []
    if force_rate:
        filters.append(f"fps={force_rate}")
    if (out_w, out_h) != (meta["width"], meta["height"]):
        aspect = float(out_w) / float(out_h)
        if abs(meta["height"] * aspect - meta["width"]) >= 1:
            # Crop to the target aspect first so the preview is never
            # stretched, matching what load_frames actually outputs.
            filters.append(
                "crop=if(gt({a}\\,a)\\,iw\\,ih*{a}):if(gt({a}\\,a)\\,iw/{a}\\,ih)".format(a=aspect))
        filters.append(f"scale={out_w}:{out_h}")

    args = [ffmpeg, "-v", "error"]
    if start_time > 0:
        args += ["-ss", f"{start_time:.4f}"]
    args += ["-i", video_path]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", "-y", out_path]
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("preview render failed:\n"
                           + exc.stderr.decode(*TEXT_ENCODING))
    return out_path


def _register_video_meta_endpoint():
    try:
        from server import PromptServer
        from aiohttp import web
        instance = getattr(PromptServer, "instance", None)
        if instance is None:
            return

        @instance.routes.get("/painternodes/video_meta")
        async def _video_meta(request):
            import asyncio
            filename = request.query.get("filename", "")
            if not filename or not folder_paths.exists_annotated_filepath(filename):
                return web.json_response({"error": "invalid video file"}, status=404)
            path = folder_paths.get_annotated_filepath(filename)
            loop = asyncio.get_event_loop()
            try:
                meta = await loop.run_in_executor(None, _cached_probe, path)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=500)
            return web.json_response(meta)

        @instance.routes.get("/painternodes/preview")
        async def _video_preview(request):
            import asyncio
            from aiohttp.web import FileResponse
            filename = request.query.get("filename", "")
            if not filename or not folder_paths.exists_annotated_filepath(filename):
                return web.json_response({"error": "invalid video file"}, status=404)
            path = folder_paths.get_annotated_filepath(filename)
            try:
                force_rate = float(request.query.get("force_rate", 0) or 0)
                custom_width = int(float(request.query.get("custom_width", 0) or 0))
                custom_height = int(float(request.query.get("custom_height", 0) or 0))
                frame_load_cap = int(float(request.query.get("frame_load_cap", 0) or 0))
                start_frame = int(float(request.query.get("start_frame", 0) or 0))
                fmt = request.query.get("format", DEFAULT_FORMAT)
            except ValueError:
                return web.json_response({"error": "bad parameters"}, status=400)
            loop = asyncio.get_event_loop()
            try:
                out = await loop.run_in_executor(
                    None, _build_preview, path, force_rate, custom_width,
                    custom_height, frame_load_cap, start_frame, fmt)
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=500)
            return FileResponse(out)
    except Exception:
        pass


_register_video_meta_endpoint()


NODE_CLASS_MAPPINGS = {
    "PainterLoadVideo": PainterLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterLoadVideo": "Painter Load Video",
}
