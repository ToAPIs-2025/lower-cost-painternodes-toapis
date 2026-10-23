# Lower-cost PainterNodes API recipes

[中文](README_zh-CN.md) · [Upstream](https://github.com/princepainter/ComfyUI-PainterNodes) · [Prices and call costs](PRICING.md)

Optional ComfyUI extension candidate with image and MP4-path nodes. Paid calls and file decoding are tested; full ComfyUI UI/torch node execution remains unverified.

## Examples

![Original image](assets/gpt-image-2-vip-e6df74ab3a.png)

gpt-image-2(VIP), low / 1K: current price **$0.0019/image**; this image settlement is pending.

![Video preview](assets/seedance-2-fast-a20e5042a9.jpg)

[Play the actual MP4](assets/seedance-2-fast-a20e5042a9.mp4)

**seedance-2-fast**: 4 seconds, 480p, 16:9, audio off. Actual charge **$0.214568**, approximately **$0.053111/s**.

The lamp video is a shared, original pricing benchmark reused across these proposals. It is an independent API output, not a recording of the upstream application. This case is text-to-video.

## Run

Set `TOAPIS_API_KEY` privately on the server. Python 3.10+, standard library only for API calls.

```sh
python unified_client.py --model gpt-image-2-vip --prompt-file prompts/image.txt --max-usd .10
# Review quotation, then explicitly submit once:
python unified_client.py --model gpt-image-2-vip --prompt-file prompts/image.txt --max-usd .10 --live --run-id reviewed-image-01
```

```sh
python unified_client.py --model seedance-2-fast --prompt-file prompts/video.txt --max-usd .75
# Add --live and a stable --run-id only after reviewing price.
```

Repeating a run with the same stable identity resumes its task. Uncertain submissions are not automatically resubmitted. Keep `.state` private. Seedance Global models are disabled.

# Model pricing and call cost

2026-10-23 · USD · ToAPIs international site (.com). Sorted by call amount; pending image quotes are labeled.

| Model | Conditions | Current API price | This call |
|---|---|---:|---:|
| gpt-image-2(VIP) | low / 1K / 1 image | $0.0019/image | Settlement pending; quote $0.0019 |
| seedance-2-fast | 480p / 4.04s / audio off | ≈ $0.053111/s in this case | $0.214568 settled |

Image price is a Key-scoped quotation; do not treat the missing settlement as zero. Video is .com `billing=settled`; $/s = this settled cost / decoded duration. Token rates are not flat per-second tariffs.

Token-rate reference for this video: $5.285714/1M total tokens; the request amount above is the observed charge.

## Provider contribution

[Integration details and acceptance checklist](integration/README.md) · [Attribution and inspected source](UPSTREAM.md).

[Register at ToAPIs](https://toapis.com/login?aff=BEWR)

## Optional ComfyUI extension candidate

Copy `integration/toapis_nodes/` into `ComfyUI/custom_nodes/toapis_nodes/`. Restart ComfyUI after setting the Key in its server environment. The image node returns IMAGE plus a cost summary; the video node returns an MP4 path plus a cost summary. `live` defaults to false. Use a stable, unique `job_id` per intended paid job. The package imports/definitions are checked, but a full ComfyUI/torch/UI run has not been performed. Existing Painter nodes are preserved.

[Model API reference](https://docs.toapis.com/docs/cn/api-reference/images/gpt-image-2-vip/generation)
