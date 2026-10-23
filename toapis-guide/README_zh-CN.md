# 更低成本的 PainterNodes API 方案

[English](README.md) · [原项目](https://github.com/princepainter/ComfyUI-PainterNodes)

基于固定源码核对的社区 API 案例与接入提案。已验证独立 API 调用与文件输出；原产品内完整流程尚未验收，未获官方背书。

## 演示

![原创案例](assets/gpt-image-2-vip-e6df74ab3a.png)

gpt-image-2(VIP) · low / 1K：**$0.0019/image**。本张报价 $0.0019，实扣回执待确认。

![视频预览](assets/seedance-2-fast-a20e5042a9.jpg)

[播放实际 MP4](assets/seedance-2-fast-a20e5042a9.mp4)

seedance-2-fast：4 秒、480p、16:9、无音频；国际站实扣 **$0.214568**，本例约 **$0.053111/s**。灯具视频是本批共享基准素材，为独立 API 生成，非原产品界面录屏。

## 复用

参照 [英文完整步骤](README.md#run)，在服务端私有环境配置 Key，先查询报价，再显式开启 `--live`；保留相同 `--run-id` 以继续已有任务。脚本禁止自动重复提交不确定的付费请求，禁止 Seedance Global。

[具体模块与验收范围](integration/README.md) · [模型价格及本次消费](PRICING.md)

# Model pricing and call cost

2026-10-23 · USD · ToAPIs international site (.com). Sorted by call amount; pending image quotes are labeled.

| Model | Conditions | Current API price | This call |
|---|---|---:|---:|
| gpt-image-2(VIP) | low / 1K / 1 image | $0.0019/image | Settlement pending; quote $0.0019 |
| seedance-2-fast | 480p / 4.04s / audio off | ≈ $0.053111/s in this case | $0.214568 settled |

Image price is a Key-scoped quotation; do not treat the missing settlement as zero. Video is .com `billing=settled`; $/s = this settled cost / decoded duration. Token rates are not flat per-second tariffs.

Token-rate reference for this video: $5.285714/1M total tokens; the request amount above is the observed charge.

[模型落地页与归因](https://go2v.toapis.com/1023_pn)

附带可选 ComfyUI 节点候选包，安装说明见英文页。完整 ComfyUI / torch 运行及界面尚未验收，原有节点保留。
