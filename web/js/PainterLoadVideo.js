import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/*
 * PainterLoadVideo front-end
 *
 * - The preview is the native one created by the "video_upload" combo.
 *   This extension only enhances it in place: autoplay, loop, muted
 *   (hover to unmute), no control bar, and its src is swapped to the
 *   parameterized backend endpoint /painternodes/preview so changing
 *   force_rate, custom size, frame cap or start_frame re-renders the
 *   clip immediately.
 * - The numeric widgets use a custom "PAINTER_NUMBER" type that draws
 *   the inline hints (source fps / frames that will be loaded) and the
 *   reset-to-default icon.
 */

const NODE_TYPE = "PainterLoadVideo";
const DEFAULT_FORMAT = "H3";
const MARGIN = 15;

const FORMAT_PRESETS = {
    "None": null,
    "LTXV": { target_rate: 24, dim: [32, 0, 768, 512], frames: [8, 1] },
    "Hunyuan": { target_rate: 24, dim: [16, 0, 848, 480], frames: [4, 1] },
    "Wan": { target_rate: 16, dim: [8, 0, 832, 480], frames: [4, 1] },
    "H3": { target_rate: 24, dim: [32, 0, 1344, 768], frames: [17, 5] },
};

const HINT_ARROW = "\u2190";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function roundFps(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "?";
    return String(Math.round(n * 100) / 100);
}

function fitText(ctx, text, maxLength) {
    text = String(text);
    if (maxLength <= 0) return ["", 0];
    const full = ctx.measureText(text).width;
    if (full <= maxLength) return [text, full];
    const cut = Math.max(0, Math.floor(maxLength / full * text.length) - 2);
    const short = text.slice(0, cut) + "\u2026";
    return [short, ctx.measureText(short).width];
}

/* ------------------------------------------------------------------ */
/* custom numeric widget with hint text and reset icon                 */
/* ------------------------------------------------------------------ */

function clampWidget(w) {
    const o = w.options || {};
    if (o.min != null && w.value < o.min) w.value = o.min;
    if (o.max != null && w.value > o.max) w.value = o.max;
}

function painterNumberDraw(ctx, node, widget_width, y, H) {
    const LG = window.LiteGraph;
    const showText = LG.vueNodesMode
        || app.canvas.ds.scale >= (app.canvas.low_quality_zoom_threshold ?? 0.5);

    ctx.strokeStyle = LG.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = LG.WIDGET_BGCOLOR;
    ctx.beginPath();
    if (showText) ctx.roundRect(MARGIN, y, widget_width - MARGIN * 2, H, [H * 0.5]);
    else ctx.rect(MARGIN, y, widget_width - MARGIN * 2, H);
    ctx.fill();
    if (!showText) return;
    if (!this.disabled) ctx.stroke();

    const showReset = this.options.reset !== undefined && this.value !== this.options.reset;
    if (showReset) {
        const cx = widget_width - MARGIN - 26;
        const cy = y + H / 2;
        ctx.save();
        ctx.strokeStyle = ctx.fillStyle = LG.WIDGET_TEXT_COLOR;
        ctx.beginPath();
        ctx.arc(cx, cy, 4, Math.PI * 1.5, Math.PI * 0.85);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(cx, cy - 1.5);
        ctx.lineTo(cx, cy - 6.5);
        ctx.lineTo(cx - 4, cy - 3.5);
        ctx.fill();
        ctx.restore();
    }

    if (!this.disabled) {
        ctx.fillStyle = LG.WIDGET_TEXT_COLOR;
        ctx.beginPath();
        ctx.moveTo(MARGIN + 16, y + 5);
        ctx.lineTo(MARGIN + 6, y + H * 0.5);
        ctx.lineTo(MARGIN + 16, y + H - 5);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(widget_width - MARGIN - 16, y + 5);
        ctx.lineTo(widget_width - MARGIN - 6, y + H * 0.5);
        ctx.lineTo(widget_width - MARGIN - 16, y + H - 5);
        ctx.fill();
    }

    let freeWidth = widget_width - (40 + MARGIN * 2 + 20);
    const valueText = this.displayValue();
    const [vText, vWidth] = fitText(ctx, valueText, freeWidth);
    freeWidth -= vWidth;

    ctx.textAlign = "left";
    ctx.fillStyle = LG.WIDGET_SECONDARY_TEXT_COLOR;
    const [name, nameWidth] = fitText(ctx, this.label || this.name, freeWidth);
    freeWidth -= nameWidth;
    ctx.fillText(name, MARGIN * 2 + 5, y + H * 0.7);

    const valueOffset = MARGIN * 2 + 20;
    ctx.textAlign = "right";
    ctx.fillStyle = LG.WIDGET_TEXT_COLOR;
    ctx.fillText(vText, widget_width - valueOffset, y + H * 0.7);

    const hint = this._plvHint?.(this.value);
    if (hint) {
        ctx.fillStyle = LG.WIDGET_OUTLINE_COLOR;
        const [hText] = fitText(ctx, hint, freeWidth);
        ctx.fillText(hText, widget_width - valueOffset - vWidth - 8, y + H * 0.7);
    }
}

function painterNumberMouse(event, [x, y], node) {
    const width = this.width || node.size[0];
    let zone = 0;
    if (x > MARGIN + 6 && x < MARGIN + 16) zone = -1;
    else if (x > width - MARGIN - 16 && x < width - MARGIN - 6) zone = 1;
    else if (x > width - MARGIN - 34 && x < width - MARGIN - 18) zone = 2;

    if (event.type === "pointermove" && event.deltaX) {
        this.value += event.deltaX * (this.options.step || 1);
        clampWidget(this);
        this.callback?.(this.value);
        node.setDirtyCanvas(true, false);
    } else if (event.type === "pointerdown") {
        if (zone === 2) {
            if (this.options.reset !== undefined) {
                this.value = this.options.reset;
                this.callback?.(this.value);
            }
        } else if (zone !== 0) {
            this.value += zone * (this.options.step || 1);
            clampWidget(this);
            this.callback?.(this.value);
        }
        node.setDirtyCanvas(true, false);
    } else if (event.type === "pointerup") {
        if ((event.click_time ?? 0) < 200 && zone === 0) {
            const self = this;
            const apply = (v) => {
                const num = parseFloat(v);
                if (Number.isFinite(num)) {
                    self.value = num;
                    clampWidget(self);
                    self.callback?.(self.value);
                    node.setDirtyCanvas(true, true);
                }
            };
            if (app.canvas.prompt) {
                app.canvas.prompt("Value", this.value, apply, event);
            } else if (window.LiteGraph.showInput) {
                window.LiteGraph.showInput("Value", String(this.value), apply);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* source metadata + hints                                             */
/* ------------------------------------------------------------------ */

async function fetchSourceMeta(node) {
    const combo = getWidget(node, "video");
    const filename = combo?.value;
    if (!filename) {
        node._plvMeta = null;
        node.setDirtyCanvas(true, false);
        return;
    }
    try {
        const res = await fetch(api.apiURL(
            `/painternodes/video_meta?filename=${encodeURIComponent(filename)}`));
        node._plvMeta = res.ok ? await res.json() : null;
    } catch (err) {
        node._plvMeta = null;
    }
    node.setDirtyCanvas(true, false);
}

function alignToFormat(count, formatName) {
    const spec = FORMAT_PRESETS[formatName]?.frames;
    if (!spec) return count;
    const [div, mod] = spec;
    if (count % div !== mod) {
        count = Math.floor((count - mod) / div) * div + mod;
    }
    return Math.max(count, 0);
}

function installHints(node) {
    const rateW = getWidget(node, "force_rate");
    const capW = getWidget(node, "frame_load_cap");
    const startW = getWidget(node, "start_frame");
    const formatW = getWidget(node, "format");

    if (rateW) {
        rateW._plvHint = (value) => {
            const meta = node._plvMeta;
            if (!meta || value) return "";
            return roundFps(meta.fps) + HINT_ARROW;
        };
    }
    if (capW) {
        capW._plvHint = (value) => {
            const meta = node._plvMeta;
            if (!meta || !meta.frames) return "";
            const start = startW?.value || 0;
            const available = Math.max(0, meta.frames - start);
            const wanted = value > 0 ? Math.min(value, available) : available;
            return alignToFormat(wanted, formatW?.value) + HINT_ARROW;
        };
    }
    if (startW) {
        startW._plvHint = (value) => {
            const meta = node._plvMeta;
            if (!meta || !meta.fps || !value) return "";
            return (Math.round(value / meta.fps * 100) / 100) + "s" + HINT_ARROW;
        };
    }
}

/* ------------------------------------------------------------------ */
/* native preview enhancement                                          */
/* ------------------------------------------------------------------ */

function collectParams(node) {
    const combo = getWidget(node, "video");
    return {
        filename: combo?.value || "",
        force_rate: getWidget(node, "force_rate")?.value ?? 0,
        custom_width: getWidget(node, "custom_width")?.value ?? 0,
        custom_height: getWidget(node, "custom_height")?.value ?? 0,
        frame_load_cap: getWidget(node, "frame_load_cap")?.value ?? 0,
        start_frame: getWidget(node, "start_frame")?.value ?? 0,
        format: getWidget(node, "format")?.value ?? DEFAULT_FORMAT,
    };
}

function previewUrl(node) {
    const params = collectParams(node);
    if (!params.filename) return "";
    return api.apiURL("/painternodes/preview?" + new URLSearchParams(
        Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)]))));
}

// Locate the native upload preview video: prefer the element we already
// claimed, otherwise scan the DOM for a /view video matching this node's
// file (the frontend may create it lazily, outside node.widgets).
function findNativeVideo(node) {
    if (node._plvVideoEl && node._plvVideoEl.isConnected) {
        return node._plvVideoEl;
    }
    node._plvVideoEl = null;
    const filename = getWidget(node, "video")?.value;
    if (!filename) return null;
    for (const v of document.querySelectorAll("video")) {
        // Never steal a video element already claimed by another node.
        if (v._plvOwner && v._plvOwner !== node) continue;
        const src = v.currentSrc || v.src || "";
        if (!src) continue;
        let url;
        try {
            url = new URL(src, window.location.origin);
        } catch (err) {
            continue;
        }
        if (!url.pathname.endsWith("/view")) continue;
        if (url.searchParams.get("filename") !== filename) continue;
        return v;
    }
    return null;
}

function setPreviewSrc(node) {
    const v = node._plvVideoEl;
    if (!v || !v.isConnected) return;
    const url = previewUrl(node);
    if (!url || node._plvPreviewUrl === url) return;
    node._plvPreviewUrl = url;
    v.src = url;
    v.load?.();
}

function enhanceNativePreview(node) {
    const v = findNativeVideo(node);
    if (!v) return false;
    node._plvVideoEl = v;
    v._plvOwner = node;

    if (!v._plvPatched) {
        v._plvPatched = true;
        v.controls = false;
        v.loop = true;
        v.muted = true;
        v.autoplay = true;
        v.playsInline = true;
        v.preload = "auto";
        v.style.objectFit = "contain";

        v.addEventListener("mouseenter", () => {
            v.muted = false;
            v.volume = 1;
        });
        v.addEventListener("mouseleave", () => {
            v.muted = true;
        });
    }

    setPreviewSrc(node);
    if (v.paused) {
        v.play().catch(() => { /* autoplay guard */ });
    }
    return true;
}

function scheduleEnhance(node, attempts = [0, 100, 300, 800, 1500, 3000, 5000, 8000]) {
    for (const delay of attempts) {
        setTimeout(() => {
            if (node.isRemoved?.()) return;
            enhanceNativePreview(node);
        }, delay);
    }
}

function schedulePreviewUpdate(node) {
    clearTimeout(node._plvTimer);
    node._plvTimer = setTimeout(() => {
        if (!enhanceNativePreview(node)) {
            scheduleEnhance(node, [0, 500]);
        }
    }, 350);
}

let enhanceObserver = null;
function startEnhanceObserver() {
    if (enhanceObserver) return;
    enhanceObserver = new MutationObserver(() => {
        const nodes = app.graph?._nodes || [];
        for (const node of nodes) {
            if (node.comfyClass === NODE_TYPE || node.type === NODE_TYPE) {
                enhanceNativePreview(node);
            }
        }
    });
    enhanceObserver.observe(document.body, { childList: true, subtree: true });
}

/* ------------------------------------------------------------------ */
/* extension registration                                              */
/* ------------------------------------------------------------------ */

app.registerExtension({
    name: "PainterNodes.PainterLoadVideo",

    setup() {
        startEnhanceObserver();
    },

    getCustomWidgets() {
        return {
            PAINTER_NUMBER(node, inputName, inputData) {
                const opts = Object.assign({}, inputData[1] || {});
                const precision = opts.precision ?? 0;
                const widget = {
                    name: inputName,
                    type: "PAINTER_NUMBER",
                    value: opts.default ?? 0,
                    draw: painterNumberDraw,
                    mouse: painterNumberMouse,
                    computeSize() {
                        return [node.size[0] - 20, 20];
                    },
                    displayValue() {
                        const v = Number(this.value);
                        return precision > 0
                            ? String(Math.round(v * 10 ** precision) / 10 ** precision)
                            : String(v | 0);
                    },
                    callback(v) {
                        const o = this.options;
                        if (o.max != null && v > o.max) v = o.max;
                        if (o.min != null && v < o.min) v = o.min;
                        if (v !== 0 && o.step) {
                            const s = o.step;
                            const sh = o.mod || 0;
                            v = Math.round((v - sh) / s) * s + sh;
                        }
                        this.value = v;
                        schedulePreviewUpdate(node);
                        node.setDirtyCanvas(true, false);
                    },
                    config: inputData,
                    options: opts,
                };
                widget.parent = node;
                if (!node.widgets) node.widgets = [];
                node.widgets.push(widget);
                return { widget };
            },
        };
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = origCreated ? origCreated.apply(this, arguments) : undefined;

            installHints(this);

            const combo = getWidget(this, "video");
            if (combo) {
                const origCallback = combo.callback;
                const self = this;
                combo.callback = function (value, ...rest) {
                    origCallback?.call(this, value, ...rest);
                    // The native widget may rebuild its media element when
                    // the file changes, so re-scan and re-claim it.
                    self._plvPreviewUrl = null;
                    self._plvVideoEl = null;
                    fetchSourceMeta(self);
                    scheduleEnhance(self, [0, 200, 800, 2000]);
                };
            }

            const formatW = getWidget(this, "format");
            if (formatW) {
                const origFormatCb = formatW.callback;
                const self = this;
                formatW.callback = function (value, ...rest) {
                    origFormatCb?.call(this, value, ...rest);
                    const preset = FORMAT_PRESETS[value];
                    const rateW = getWidget(self, "force_rate");
                    if (rateW && preset?.target_rate) {
                        rateW.options.reset = preset.target_rate;
                    }
                    if (preset?.dim) {
                        for (const name of ["custom_width", "custom_height"]) {
                            const w = getWidget(self, name);
                            if (w) w.options.step = preset.dim[0];
                        }
                    }
                    const capW = getWidget(self, "frame_load_cap");
                    if (capW) {
                        if (preset?.frames) {
                            capW.options.step = preset.frames[0];
                            capW.options.mod = preset.frames[1];
                        } else {
                            capW.options.step = 1;
                            delete capW.options.mod;
                        }
                    }
                    self.setDirtyCanvas(true, false);
                    schedulePreviewUpdate(self);
                };
            }

            fetchSourceMeta(this);
            scheduleEnhance(this);
            return result;
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = origConfigure ? origConfigure.apply(this, arguments) : undefined;
            fetchSourceMeta(this);
            scheduleEnhance(this, [50, 300, 1000, 2500]);
            return result;
        };
    },
});
