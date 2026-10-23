import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

/*
 * PainterVideoCombine3 — 前端
 *
 * 在节点第一行（images 行）上，水平居中放三个小图标按钮：
 *
 *   ⟳  同步预览：与右键菜单 "Sync preview" 完全一致 —— 工作区内所有 Painter
 *      预览视频回到起点重新播放，动图（gif/webp）刷新 src 重启动画。
 *   ⏸/▶ 同步暂停 / 继续播放：工作区内所有 Painter 预览视频一起暂停，
 *      再点一次一起继续播放。
 *   👁  显示 / 隐藏选项：切换节点面板上 frame_rate、format、
 *      filename_prefix、save_workflow_info 四个 widget 的可见性。
 *      **默认隐藏**，节点默认状态下最简洁；点一下显示，再点一下隐藏。
 *
 * 设计要点：
 *   - 按钮是**真实 DOM**（不是 canvas 绘制），放在 #graph-canvas-container
 *     里的独立覆盖层上，因此 **Nodes 2.0（Vue 节点）模式下同样存在可用**。
 *   - **水平居中于节点**：节点放大/缩小、拖宽拖窄，按钮始终居中。
 *   - 尺寸随画布缩放同步（跟着 ds.scale），视觉上永远贴着节点。
 *   - 不新增任何 widget，节点尺寸与排版保持不变。
 *   - 位置由一个全局 rAF 循环维护；两种渲染模式分别取锚点：
 *       · Nodes 2.0：节点本身就是 DOM（.lg-node[data-node-id]），
 *         images 行是第一个 .lg-slot--input。
 *       · 传统渲染：用 node.pos / ds.offset / ds.scale 换算，行位置取
 *         getConnectionPos()（回退 input.boundingRect）。
 *
 * ── widget 显隐是怎么做的（已对前端源码 + 无头浏览器实测核实） ──────────────
 * 1) 双写，两种渲染路径各读一边：
 *   · 传统 canvas：LGraphNode.getLayoutWidgets() = widgets.filter(w => !w.hidden)
 *                  LGraphNode.isWidgetVisible(w) = !(... || w.hidden || ...)
 *                  → 读 `widget.hidden`
 *   · Vue 节点：  extractWidgetDisplayOptions(w) 里 `hidden: w.options.hidden`
 *                  → 读 `widget.options.hidden`
 * 2) ⚠️ 光双写还不够 —— Nodes 2.0 下还要再 poke 一次：
 *    computeProcessedWidgets() 遍历的 node.widgets 是 extractVueNodeData() 包过的
 *    shallowReactive 快照数组，它的 getter 只在「长度变化」或「元素引用变化」时
 *    才同步真实 widgets；改 widget.options.hidden 这种深层属性**不在它的依赖里**，
 *    所以 Vue 侧不会重算，表现为"点了按钮但四项不出现"。
 *    → 补一次 pokeNodeWidgets()（临时 push 一个占位元素再立刻移除）触发重算即可。
 *    详见 pokeNodeWidgets 的注释。
 * 3) 高度贴合：litegraph 的 _arrangeWidgets 只在内容"超出"时撑高节点，不会主动收缩，
 *    且 onResize 依赖布局后的 widget.y，所以显隐后要主动 computeSize→setSize→onResize，
 *    push 一次 + 延迟再补两次。注意 computeSize() **不能传数字**（签名是 computeSize(out)，
 *    内部会执行 out[1]=…，严格模式下给 number 赋属性直接抛 TypeError，
 *    一旦抛在同一个 try 里，后面的 onResize 就永远不会执行 —— 这正是"视频溢出下边框"的成因）。
 * 4) 安全性：提交参数（graphToPrompt）只排除 `options.serialize === false`；
 *    工作流序列化/恢复也只排除 `serialize === false`。`hidden` 不在排除条件里，
 *    所以隐藏**不会**导致参数不提交、也不会丢工作流里的值。
 *
 * 其余一切（布局、widget、预览播放器、右键菜单）与原节点保持一致。
 * 原节点 PainterVideoCombine / PainterVideoCombine2 不做任何改动。
 */

// 文案 / tooltip 用 Unicode 转义，避免任何服务端编码问题。
const TIP_SYNC = "\u540c\u6b65\u9884\u89c8";        // 同步预览
const TIP_PAUSE = "\u540c\u6b65\u6682\u505c";       // 同步暂停
const TIP_RESUME = "\u7ee7\u7eed\u540c\u6b65\u64ad\u653e"; // 继续同步播放
const TIP_SHOW_OPTS = "\u663e\u793a\u9009\u9879";   // 显示选项
const TIP_HIDE_OPTS = "\u9690\u85cf\u9009\u9879";   // 隐藏选项

// 本节点类型
const THIS_TYPE = "PainterVideoCombine3";

// 同步操作覆盖的节点类型（含老节点，方便一个工作流里混用）
const SYNC_NODE_TYPES = ["PainterVideoCombine", "PainterVideoCombine2", "PainterVideoCombine3"];

// 默认隐藏的 widget：帧率、格式、文件名、保存工作流信息
const TOGGLE_WIDGETS = ["frame_rate", "format", "filename_prefix", "save_workflow_info"];

// 显隐状态持久化到 node.properties 的键名
const PROP_SHOW_OPTS = "painter_show_options";

// 按钮尺寸（CSS px，未缩放）
const BTN_W = 22;
const BTN_H = 18;
const BTN_GAP = 4;
const BTN_COUNT = 3;
const ICON_SIZE = 13;

const LAYER_CLASS = "painter-sync-btn-layer";
const STYLE_ID = "painter-sync-btn-style-3";

const ICON_SYNC = `<svg width="${ICON_SIZE}" height="${ICON_SIZE}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-3.2-6.9"/><polyline points="21 3 21 8.6 15.4 8.6"/></svg>`;

const ICON_PAUSE = `<svg width="${ICON_SIZE}" height="${ICON_SIZE}" viewBox="0 0 24 24" fill="currentColor"><rect x="6.4" y="5" width="3.7" height="14" rx="1.2"/><rect x="13.9" y="5" width="3.7" height="14" rx="1.2"/></svg>`;

const ICON_PLAY = `<svg width="${ICON_SIZE}" height="${ICON_SIZE}" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.2v13.6L19.2 12z"/></svg>`;

// 睁眼：选项当前可见（点一下会隐藏）
const ICON_EYE = `<svg width="${ICON_SIZE}" height="${ICON_SIZE}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.2 12S5.6 5.4 12 5.4 21.8 12 21.8 12 18.4 18.6 12 18.6 2.2 12 2.2 12z"/><circle cx="12" cy="12" r="2.7"/></svg>`;

// 闭眼（带斜杠）：选项当前被隐藏（点一下会显示）
const ICON_EYE_OFF = `<svg width="${ICON_SIZE}" height="${ICON_SIZE}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.2 12S5.6 5.4 12 5.4c1.7 0 3.2.4 4.5 1"/><path d="M20.4 8.6c.9 1.4 1.4 2.6 1.4 3.4 0 0-3.4 6.6-9.8 6.6-1.9 0-3.5-.5-4.9-1.3"/><path d="M9.8 9.9a3 3 0 0 0 4.2 4.2"/><line x1="3" y1="3" x2="21" y2="21"/></svg>`;

// 全局同步状态：工作区内所有预览节点共享
let syncPaused = false;

// ---------------------------------------------------------------------------
// 同步操作
// ---------------------------------------------------------------------------

function getGraph(node) {
    return node?.graph || app.canvas?.graph || app.graph;
}

function findMediaElement(node) {
    if (node.painter_media_element) return node.painter_media_element;
    if (!node.widgets) return null;
    for (const w of node.widgets) {
        const el = w.element;
        if (!el) continue;
        if (el.tagName === "VIDEO" || el.tagName === "IMG") return el;
        const m = el.querySelector?.("video, img");
        if (m) return m;
    }
    return null;
}

function collectPreviewNodes(node) {
    const all = getGraph(node)?._nodes || [];
    return all.filter((n) => SYNC_NODE_TYPES.includes(n.type));
}

function refreshAll(node) {
    for (const n of collectPreviewNodes(node)) {
        n.setDirtyCanvas?.(true, true);
    }
}

/** 与右键菜单 "Sync preview" 一致：全部回起点重播 / 动图重启动画 */
function syncPreviewAll(node) {
    for (const n of collectPreviewNodes(node)) {
        const m = n.painter_media_element || findMediaElement(n);
        if (!m) continue;
        if (m.tagName === "IMG") {
            const src = m.src;
            const sep = src && src.indexOf("?") >= 0 ? "&" : "?";
            m.src = (src || "") + sep + "_ts=" + Date.now();
        } else {
            try {
                m.pause();
                m.currentTime = 0;
                m.load();
                const p = m.play();
                if (p && typeof p.catch === "function") p.catch(() => {});
            } catch (e) {
                // best-effort; some browsers may briefly reject autoplay
            }
        }
    }
    // 手动同步预览 = 重新开始播放，状态回到「播放中」
    if (syncPaused) {
        syncPaused = false;
        applyPauseVisual();
        refreshAll(node);
    }
}

/** 全部暂停 <-> 全部继续播放 */
function syncTogglePause(node) {
    syncPaused = !syncPaused;

    for (const n of collectPreviewNodes(node)) {
        const m = n.painter_media_element || findMediaElement(n);
        if (!m || m.tagName !== "VIDEO") continue; // 动图无法暂停，跳过
        try {
            if (syncPaused) {
                m.pause();
            } else {
                const p = m.play();
                if (p && typeof p.catch === "function") p.catch(() => {});
            }
        } catch (e) {
            // best-effort
        }
    }

    applyPauseVisual();
    refreshAll(node);
    return syncPaused;
}

/** 把暂停状态反映到所有本节点的按钮图标上 */
function applyPauseVisual() {
    for (const n of collectPreviewNodes(null)) {
        if (n.type !== THIS_TYPE) continue;
        const b = n.painter_btn_pause;
        if (!b) continue;
        b.innerHTML = syncPaused ? ICON_PLAY : ICON_PAUSE;
        b.title = syncPaused ? TIP_RESUME : TIP_PAUSE;
        b.classList.toggle("painter-sync-btn--on", syncPaused);
    }
}

// ---------------------------------------------------------------------------
// widget 显隐
// ---------------------------------------------------------------------------

function isWidgetHidden(w) {
    return !!(w?.hidden || w?.options?.hidden);
}

/**
 * 设置单个 widget 的显隐。
 * 双写 widget.hidden 与 widget.options.hidden —— 前者喂传统 canvas 渲染，
 * 后者喂 Vue 节点渲染（见文件头说明）。
 */
function setWidgetHidden(node, name, hidden) {
    const w = node.widgets?.find((x) => x.name === name);
    if (!w) return false;
    w.hidden = !!hidden;
    if (!w.options) w.options = {};
    w.options.hidden = !!hidden;
    return true;
}

/** 让节点高度重新贴合内容（隐藏/显示 widget 后要收缩或长高） */function resizeNodeToFit(node) {
    const doResize = () => {
        // 每一步都单独保护：任何一步失败都不能拖垮其余步骤。
        // （踩过的坑：node.computeSize(w) 传数字会抛 TypeError —— 它的签名是
        //   computeSize(out)，内部执行 out[1]=…，严格模式下给 number 赋属性直接报错。
        //   一旦在同一个 try 里抛了，后面的 onResize 就永远不执行，节点高度永不贴合。）
        try {
            const w = Math.max(1, node.size?.[0] ?? 0);
            const need = typeof node.computeSize === "function" ? node.computeSize() : null;
            if (need && Number.isFinite(need[1]) && need[1] > 0) {
                node.setSize([Math.max(w, need[0]), need[1]]);
            }
        } catch (e) {
            // computeSize 失败不影响下面的高度贴合
        }

        // 按视频宽高比重新定高（原节点逻辑，必须单独跑）
        try {
            node.onResize?.(node.size);
        } catch (e) {
            // ignore
        }

        try {
            node.setDirtyCanvas?.(true, true);
        } catch (e) {
            // ignore
        }
    };
    // 布局（widget.y 重分配）发生在下一帧，故延迟补两次
    doResize();
    setTimeout(doResize, 60);
    setTimeout(doResize, 200);
    try { app.canvas?.setDirty?.(true, true); } catch (e) { /* ignore */ }
}

/**
 * 强制 Vue 节点重新计算 widget 列表。
 *
 * 背景（对前端 1.52.7 源码核实）：Nodes 2.0 下渲染的 widget 列表由
 * computeProcessedWidgets() 算出，它遍历的 node.widgets 是 extractVueNodeData()
 * 包装过的 shallowReactive 快照数组 —— 该 getter 只有在「长度变化」或
 * 「元素引用变化」时才把真实 widgets 同步进去。也就是说 **改 widget.options.hidden
 * 这类深层属性根本不在它的依赖里**，单纯设 hidden 在 Nodes 2.0 下看不到任何变化
 * （已用无头浏览器实测确认：live widget 的 hidden/options.hidden 都已是新值，
 *  但节点 DOM 里那几行始终不出现）。
 *
 * 做法：往 widgets 数组里临时 push 一个占位元素、再立刻移除。这一对操作会让
 * shallowReactive 数组产生一次真实变更，触发 Vue 重算；重算时读到的
 * options.hidden 已经是最新值，于是显隐立刻生效。push 与移除在同一个同步块内
 * 完成，中间状态不会暴露给其它代码（Vue 的更新调度是异步的），实测幂等。
 */
function pokeNodeWidgets(node) {
    const list = node?.widgets;
    if (!list || typeof list.push !== "function" || typeof list.splice !== "function") return false;
    const marker = { name: "__painter_poke__", type: "hidden" };
    try {
        list.push(marker);
        const i = list.indexOf(marker);
        if (i >= 0) list.splice(i, 1);
        return true;
    } catch (e) {
        try {
            const i = list.indexOf(marker);
            if (i >= 0) list.splice(i, 1);
        } catch (e2) { /* ignore */ }
        return false;
    }
}

function updateEyeButton(node) {
    const b = node.painter_btn_eye;
    if (!b) return;
    const visible = !!node.properties?.[PROP_SHOW_OPTS];
    b.innerHTML = visible ? ICON_EYE : ICON_EYE_OFF;
    b.title = visible ? TIP_HIDE_OPTS : TIP_SHOW_OPTS;
    b.classList.toggle("painter-sync-btn--active", visible);
}

/**
 * 应用显隐状态。
 * @param {*} node
 * @param {boolean} visible   true = 显示四个 widget
 * @param {boolean} persist   true = 把状态写进 node.properties（随工作流保存）
 */
function applyOptionsVisibility(node, visible, persist = true) {
    for (const name of TOGGLE_WIDGETS) setWidgetHidden(node, name, !visible);

    if (persist) {
        if (!node.properties) node.properties = {};
        node.properties[PROP_SHOW_OPTS] = !!visible;
    }

    updateEyeButton(node);
    // Nodes 2.0 必须补这一下，否则显隐在 Vue 节点上看不到变化
    pokeNodeWidgets(node);
    resizeNodeToFit(node);
}

/** 点击眼睛按钮：显隐切换 */
function toggleOptions(node) {
    const next = !node.properties?.[PROP_SHOW_OPTS];
    applyOptionsVisibility(node, next, true);
    return next;
}

/**
 * 节点创建阶段把显隐状态落到位（此时 widgets 已就绪）。
 * 新建节点 → 默认隐藏；从工作流加载 → 先按默认隐藏，
 * 随后 onConfigure 再按持久化的 properties 覆盖一次。
 */
function applyNodeVisibility(node) {
    const showOpts = !!node.properties?.[PROP_SHOW_OPTS];
    for (const name of TOGGLE_WIDGETS) setWidgetHidden(node, name, !showOpts);
    updateEyeButton(node);
    pokeNodeWidgets(node);
    resizeNodeToFit(node);
}

// ---------------------------------------------------------------------------
// 按钮（DOM 覆盖层，Nodes 2.0 与传统渲染通用）
// ---------------------------------------------------------------------------

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = `
        .painter-sync-btns { position: absolute; left: 0; top: 0; display: flex; gap: ${BTN_GAP}px;
                             transform-origin: 0 0; pointer-events: none; }
        .painter-sync-btn { width: ${BTN_W}px; height: ${BTN_H}px; box-sizing: border-box;
                            display: flex; align-items: center; justify-content: center;
                            background: #2a2a2a; border: 1px solid #5c5c5c; border-radius: 4px;
                            color: #e0e0e0; cursor: pointer; pointer-events: auto;
                            user-select: none; -webkit-user-select: none; }
        .painter-sync-btn:hover { background: #3c3c3c; border-color: #8a8a8a; }
        .painter-sync-btn:active { background: #4a4a4a; }
        .painter-sync-btn--on { background: #8c1f1f; border-color: #e35b5b; color: #ffffff; }
        .painter-sync-btn--on:hover { background: #a32323; border-color: #ff6b6b; }
        .painter-sync-btn--active { background: #1f4d8c; border-color: #5b9be3; color: #ffffff; }
        .painter-sync-btn--active:hover { background: #2360a3; border-color: #6bb0ff; }
    `;
    document.head.appendChild(s);
}

function ensureLayer() {
    const host = document.getElementById("graph-canvas-container")
        || app.canvas?.canvas?.parentElement;
    if (!host) return null;
    let layer = host.querySelector(":scope > ." + LAYER_CLASS);
    if (!layer) {
        layer = document.createElement("div");
        layer.className = LAYER_CLASS;
        layer.style.cssText = "position:absolute;left:0;top:0;width:100%;height:100%;" +
                              "pointer-events:none;z-index:5;overflow:hidden;";
        host.appendChild(layer);
    }
    return layer;
}

function makeButton(icon, title, isToggle, onClick) {
    const b = document.createElement("div");
    b.className = "painter-sync-btn" + (isToggle ? " painter-sync-btn--toggle" : "");
    b.title = title;
    b.innerHTML = icon;
    const eat = (e) => { e.stopPropagation(); };
    b.addEventListener("pointerdown", eat, true);
    b.addEventListener("mousedown", eat, true);
    b.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        try { onClick(); } catch (err) { /* ignore */ }
    });
    return b;
}

function ensureWrap(node) {
    const cur = node.painter_btn_wrap;
    if (cur && cur.isConnected && cur.parentElement?.classList?.contains(LAYER_CLASS)) return cur;

    const layer = ensureLayer();
    if (!layer) return null;

    const wrap = document.createElement("div");
    wrap.className = "painter-sync-btns";

    const b1 = makeButton(ICON_SYNC, TIP_SYNC, false, () => syncPreviewAll(node));
    const b2 = makeButton(syncPaused ? ICON_PLAY : ICON_PAUSE, syncPaused ? TIP_RESUME : TIP_PAUSE, true,
                          () => syncTogglePause(node));
    if (syncPaused) b2.classList.add("painter-sync-btn--on");

    const showOpts = !!node.properties?.[PROP_SHOW_OPTS];
    const b3 = makeButton(showOpts ? ICON_EYE : ICON_EYE_OFF, showOpts ? TIP_HIDE_OPTS : TIP_SHOW_OPTS, true,
                          () => toggleOptions(node));
    if (showOpts) b3.classList.add("painter-sync-btn--active");

    wrap.appendChild(b1);
    wrap.appendChild(b2);
    wrap.appendChild(b3);
    layer.appendChild(wrap);

    node.painter_btn_wrap = wrap;
    node.painter_btn_pause = b2;
    node.painter_btn_main = b1;
    node.painter_btn_eye = b3;
    return wrap;
}

/** images 行在节点局部坐标里的中心 y（传统渲染用） */
function getFirstRowCenterY(node) {
    // 1) 绘制后记录的 input slot 矩形中心
    const br = node.inputs?.[0]?.boundingRect;
    if (br && Number.isFinite(br[1]) && Number.isFinite(br[3]) && br[3] > 0) {
        const ly = br[1] + br[3] / 2 - node.pos[1];
        if (ly > 0 && ly < 400) return ly;
    }
    // 2) LiteGraph 官方 API（返回 slot 中心点，graph 空间）
    try {
        if (typeof node.getConnectionPos === "function") {
            const out = [0, 0];
            const p = node.getConnectionPos(true, 0, out);
            if (p && Number.isFinite(p[1])) {
                const ly = p[1] - node.pos[1];
                if (ly > 0 && ly < 400) return ly;
            }
        }
    } catch (e) {
        // fall through
    }
    // 3) 兜底：从第一个 widget 行往上推一行（slot 20 + 间距 12）
    const firstWidget = node.widgets?.find((w) => w.name !== "painter_preview");
    if (firstWidget && Number.isFinite(firstWidget.y)) {
        const ly = firstWidget.y - 32;
        if (ly > 0) return ly;
    }
    return null;
}

/**
 * 返回节点在覆盖层坐标系里的锚点：
 *   { x, y, w, rowCenterY, scale }
 * 两种渲染模式分别取不同来源。
 */
/* Per-frame DOM probing is the expensive part of this ticker: every
   getBoundingClientRect() flushes layout, and with Nodes 2.0 off the
   .lg-node querySelector always misses anyway. Cache the probe per node and
   share one layer/canvas rect across the whole frame instead of asking for
   one per node. */
const domProbe = new Map();   // node.id -> { epoch, el, slot }
let probeEpoch = 0;

function probeNodeDom(node) {
    const hit = domProbe.get(node.id);
    if (hit && hit.epoch === probeEpoch) return hit;
    const rec = { epoch: probeEpoch, el: null, slot: null };
    try {
        rec.el = document.querySelector(`.lg-node[data-node-id="${node.id}"]`);
        if (rec.el) rec.slot = rec.el.querySelector(".lg-slot--input");
    } catch (e) {
        rec.el = null;
    }
    domProbe.set(node.id, rec);
    return rec;
}

function computeAnchor(node, layer, rects) {
    const layerRect = rects ? rects.layerRect : layer.getBoundingClientRect();

    // --- Nodes 2.0：节点本身就是 DOM ---
    const probe = probeNodeDom(node);
    const el = probe.el;
    if (el) {
        const r = el.getBoundingClientRect();
        if (r.width > 1 && r.height > 1) {
            const slot = probe.slot;
            const sr = slot ? slot.getBoundingClientRect() : null;
            // slot 中心相对节点顶部约 15.3%（248px 宽时是 38px）
            const rowCenterY = sr
                ? sr.top + sr.height / 2 - layerRect.top
                : r.top - layerRect.top + r.width * 0.153;
            return {
                x: r.left - layerRect.left,
                y: r.top - layerRect.top,
                w: r.width,
                rowCenterY,
                scale: app.canvas?.ds?.scale ?? 1,
            };
        }
    }

    // --- 传统 canvas 渲染 ---
    const cvs = app.canvas?.canvas;
    if (!cvs) return null;
    const rowLy = getFirstRowCenterY(node);
    if (rowLy == null) return null;
    const ds = app.canvas.ds || { scale: 1, offset: [0, 0] };
    const cRect = (rects && rects.canvasRect) ? rects.canvasRect : cvs.getBoundingClientRect();
    const x = cRect.left - layerRect.left + (node.pos[0] + ds.offset[0]) * ds.scale;
    const y = cRect.top - layerRect.top + (node.pos[1] + ds.offset[1]) * ds.scale;
    return {
        x,
        y,
        w: node.size[0] * ds.scale,
        rowCenterY: y + rowLy * ds.scale,
        scale: ds.scale,
    };
}

function updateWrap(node, rects) {
    const wrap = ensureWrap(node);
    if (!wrap) return;

    const layer = wrap.parentElement;
    const a = layer ? computeAnchor(node, layer, rects) : null;
    if (!a) {
        if (wrap.style.display !== "none") wrap.style.display = "none";
        return;
    }

    const s = Number.isFinite(a.scale) && a.scale > 0 ? a.scale : 1;
    const totalW = BTN_W * BTN_COUNT + BTN_GAP * (BTN_COUNT - 1);
    const left = a.x + a.w / 2 - (totalW * s) / 2;   // 水平居中于节点
    const top = a.rowCenterY - (BTN_H * s) / 2;      // 垂直居中于 images 行

    const key = `${left.toFixed(1)}|${top.toFixed(1)}|${s.toFixed(3)}`;
    if (wrap.__posKey !== key) {
        wrap.__posKey = key;
        wrap.style.transform = `translate(${left}px, ${top}px) scale(${s})`;
    }
    if (wrap.style.display !== "flex") wrap.style.display = "flex";
}

let tickerStarted = false;
let idleFrames = 0;
let lastGeometryKey = "";

function startTicker() {
    if (tickerStarted) return;
    tickerStarted = true;
    const loop = () => {
        try {
            const graph = app.graph || app.canvas?.graph;
            const nodes = (graph?._nodes || []).filter((n) => n.type === THIS_TYPE);

            if (nodes.length === 0) {
                idleFrames = 0;
                lastGeometryKey = "";
            } else {
                const ds = app.canvas?.ds;
                /* Cheap numeric key: pan/zoom plus every node's geometry.
                   Reading pos/size is free; reading the DOM is not. */
                let geo = `${ds?.offset?.[0]},${ds?.offset?.[1]},${ds?.scale}`;
                for (const n of nodes) {
                    geo += `|${n.id}:${n.pos?.[0]},${n.pos?.[1]},${n.size?.[0]},${n.size?.[1]}`;
                }

                const moving = app.canvas?.isDragging || app.canvas?.dirty?.bg;
                if (geo === lastGeometryKey && !moving) {
                    /* Nothing moved. Skip the DOM pass instead of forcing layout
                       60 times a second; a few settle frames keep late DOM
                       mutations correct. */
                    idleFrames = Math.min(idleFrames + 1, 8);
                    probeEpoch++;
                    if (idleFrames >= 3) {
                        requestAnimationFrame(loop);
                        return;
                    }
                } else {
                    idleFrames = 0;
                    lastGeometryKey = geo;
                }

                const layer = document.querySelector("." + LAYER_CLASS);
                if (layer) {
                    /* One layout read per frame, shared by every node. */
                    const rects = {
                        layerRect: layer.getBoundingClientRect(),
                        canvasRect: app.canvas?.canvas?.getBoundingClientRect() ?? null,
                    };
                    for (const n of nodes) updateWrap(n, rects);
                } else {
                    for (const n of nodes) updateWrap(n, null);
                }
            }
        } catch (e) {
            // 任何绘制异常都不能影响画布
        }
        requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------

app.registerExtension({
    name: "Painter.VideoCombine3",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== THIS_TYPE) return;

        ensureStyle();
        startTicker();

        const SIDE_MARGIN = 15;
        const BOTTOM_MARGIN = 20;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.progress > 0 && this.progress < 1) {
                ctx.save();
                ctx.fillStyle = "#FFD700";
                ctx.fillRect(0, -2, this.size[0] * this.progress, 4);
                ctx.restore();
            }
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            try {
                this.painter_btn_wrap?.remove();
            } catch (e) {
                // ignore
            }
            this.painter_btn_wrap = null;
            this.painter_btn_pause = null;
            this.painter_btn_main = null;
            this.painter_btn_eye = null;
            onRemoved?.apply(this, arguments);
        };

        // 节点实例化末尾会调用 onNodeCreated，此刻 widgets 已全部建好
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            try { applyNodeVisibility(this); } catch (e) { /* ignore */ }
        };

        function getHeaderAndWidgetHeight(node) {
            let height = 24;
            if (node.widgets) {
                for (const w of node.widgets) {
                    if (w.name !== "painter_preview" && w.type !== "hidden" && !isWidgetHidden(w)) {
                        height += (w.computeSize ? w.computeSize(node.size[0])[1] : 20) + 18;
                    }
                }
            }
            return height;
        }

        function getPreviewWidget(node) {
            return node.widgets?.find(w => w.name === "painter_preview");
        }

        function getPreviewTop(node) {
            const widget = getPreviewWidget(node);
            if (widget) {
                if (typeof widget.y === "number" && widget.y > 0) return widget.y;
                if (typeof widget.last_y === "number" && widget.last_y > 0) return widget.last_y;
            }
            return getHeaderAndWidgetHeight(node);
        }

        function getFormatValue(node) {
            const w = node.widgets?.find((w) => w.name === "format");
            return w?.value || "video/h264-mp4";
        }

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            const media = findMediaElement(this);
            const isVideo = media?.tagName === "VIDEO";
            const newOptions = [];

            newOptions.push({
                content: "Save preview",
                callback: () => {
                    const params = this.properties["painter_output_cache"];
                    if (params) {
                        const url = api.apiURL(`/view?filename=${params.filename}&subfolder=${params.subfolder}&type=${params.type}`);
                        const a = document.createElement("a");
                        a.href = url; a.download = params.filename;
                        document.body.appendChild(a); a.click(); document.body.removeChild(a);
                    }
                }
            });

            newOptions.push({
                content: (isVideo && media?.paused) ? "Resume preview" : "Pause preview",
                callback: () => {
                    if (isVideo && media) media.paused ? media.play() : media.pause();
                }
            });

            newOptions.push({
                content: "Sync preview",
                callback: () => syncPreviewAll(this)
            });

            newOptions.push({
                content: this.properties?.[PROP_SHOW_OPTS] ? "Hide node options" : "Show node options",
                callback: () => toggleOptions(this)
            });

            if (options.length > 0) newOptions.push(null);
            options.unshift(...newOptions);
        };

        nodeType.prototype.onResize = function (size) {
            if (this.painter_aspect) {
                const top = getPreviewTop(this);
                const targetVideoHeight = (size[0] - SIDE_MARGIN * 2) / this.painter_aspect;
                const totalHeight = Math.ceil(top + targetVideoHeight + BOTTOM_MARGIN);

                if (Math.abs(size[1] - totalHeight) > 0.5) {
                    size[1] = totalHeight;
                }
            }

            const widget = getPreviewWidget(this);
            if (widget?.element) {
                widget.element.style.width = "100%";
                widget.element.style.left = "0px";
                const contentH = size[1] - getPreviewTop(this) - BOTTOM_MARGIN;
                widget.element.style.height = `${Math.max(0, contentH)}px`;
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            if (message?.painter_output) {
                this.properties["painter_output_cache"] = message.painter_output[0];
                updateVideoPreview(this, message.painter_output[0]);
            }
        };

        nodeType.prototype.onConfigure = function () {
            // properties 此时已从工作流恢复，按持久化状态重新落一次显隐
            applyNodeVisibility(this);

            if (this.properties?.["painter_output_cache"]) {
                updateVideoPreview(this, this.properties["painter_output_cache"]);
            }
        };

        function formatTime(seconds) {
            if (isNaN(seconds)) return "0:00";
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return `${m}:${s.toString().padStart(2, '0')}`;
        }

        function createCustomControls(video, container) {
            const controls = document.createElement("div");
            controls.className = "painter-video-controls";
            controls.style.cssText = `
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 8px 10px 6px 10px;
                background: linear-gradient(transparent, rgba(0,0,0,0.75));
                color: #fff;
                font-family: sans-serif;
                font-size: 12px;
                opacity: 0;
                transition: opacity 0.25s ease;
                display: flex;
                flex-direction: column;
                gap: 6px;
                pointer-events: none;
                z-index: 10;
                user-select: none;
            `;

            // Progress bar container
            const progressContainer = document.createElement("div");
            progressContainer.style.cssText = `
                width: 100%;
                height: 4px;
                background: rgba(255,255,255,0.3);
                border-radius: 2px;
                cursor: pointer;
                position: relative;
                pointer-events: auto;
                overflow: hidden;
            `;

            const progressBar = document.createElement("div");
            progressBar.style.cssText = `
                height: 100%;
                background: #00E5CC;
                border-radius: 2px;
                width: 0%;
                transition: width 0.1s linear;
            `;

            const progressHover = document.createElement("div");
            progressHover.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                background: rgba(0,229,204,0.35);
                width: 0%;
                pointer-events: none;
                display: none;
            `;

            progressContainer.appendChild(progressBar);
            progressContainer.appendChild(progressHover);

            // Bottom row: play button + time + duration
            const bottomRow = document.createElement("div");
            bottomRow.style.cssText = `
                display: flex;
                align-items: center;
                gap: 10px;
                pointer-events: auto;
            `;

            const playBtn = document.createElement("button");
            playBtn.innerHTML = "▶";
            playBtn.style.cssText = `
                background: transparent;
                border: none;
                color: #fff;
                cursor: pointer;
                font-size: 14px;
                width: 20px;
                height: 20px;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0;
                line-height: 1;
            `;

            const timeDisplay = document.createElement("span");
            timeDisplay.style.cssText = `
                font-size: 11px;
                color: rgba(255,255,255,0.9);
                min-width: 70px;
                white-space: nowrap;
            `;
            timeDisplay.textContent = "0:00 / 0:00";

            bottomRow.appendChild(playBtn);
            bottomRow.appendChild(timeDisplay);

            controls.appendChild(progressContainer);
            controls.appendChild(bottomRow);
            container.appendChild(controls);

            // Hover logic for container
            container.addEventListener("mouseenter", () => {
                controls.style.opacity = "1";
            });
            container.addEventListener("mouseleave", () => {
                controls.style.opacity = "0";
            });

            // Play/Pause toggle
            playBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                if (video.paused) {
                    video.play();
                } else {
                    video.pause();
                }
            });

            // Update play button icon
            video.addEventListener("play", () => {
                playBtn.innerHTML = "⏸";
            });
            video.addEventListener("pause", () => {
                playBtn.innerHTML = "▶";
            });

            // Progress update
            video.addEventListener("timeupdate", () => {
                if (video.duration) {
                    const pct = (video.currentTime / video.duration) * 100;
                    progressBar.style.width = pct + "%";
                    timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
                }
            });

            video.addEventListener("loadedmetadata", () => {
                timeDisplay.textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
            });

            // Seek on click
            let isDragging = false;

            progressContainer.addEventListener("mousedown", (e) => {
                isDragging = true;
                seek(e);
            });

            document.addEventListener("mousemove", (e) => {
                if (isDragging) {
                    const rect = progressContainer.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                    const pct = x / rect.width;
                    progressHover.style.width = (pct * 100) + "%";
                    progressHover.style.display = "block";
                }
            });

            document.addEventListener("mouseup", () => {
                if (isDragging) {
                    isDragging = false;
                    progressHover.style.display = "none";
                }
            });

            function seek(e) {
                const rect = progressContainer.getBoundingClientRect();
                const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                const pct = x / rect.width;
                if (video.duration) {
                    video.currentTime = pct * video.duration;
                }
            }

            progressContainer.addEventListener("click", (e) => {
                e.stopPropagation();
                seek(e);
            });

            progressContainer.addEventListener("mousemove", (e) => {
                const rect = progressContainer.getBoundingClientRect();
                const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
                const pct = x / rect.width;
                progressHover.style.width = (pct * 100) + "%";
                progressHover.style.display = "block";
            });

            progressContainer.addEventListener("mouseleave", () => {
                if (!isDragging) {
                    progressHover.style.display = "none";
                }
            });
        }

        function updateVideoPreview(node, data) {
            let widget = getPreviewWidget(node);

            if (!widget) {
                const element = document.createElement("div");
                element.style.display = "flex";
                element.style.justifyContent = "center";
                element.style.alignItems = "center";
                element.style.padding = "0px";
                element.style.margin = "0px";
                element.style.overflow = "hidden";
                element.style.boxSizing = "border-box";
                element.style.position = "relative";

                widget = node.addDOMWidget("painter_preview", "preview", element, {
                    serialize: false, hideOnZoom: false
                });
            }

            const url = api.apiURL(`/view?filename=${data.filename}&subfolder=${data.subfolder}&type=${data.type}`);
            const format = getFormatValue(node);
            const isAnimatedImage = typeof format === "string" && format.split("/")[0] === "image";
            widget.element.innerHTML = "";

            const triggerCtx = (e) => {
                e.preventDefault(); e.stopPropagation();
                if (app.canvas.processContextMenu) app.canvas.processContextMenu(node, e);
                else app.canvas._mousedown_callback(e);
                return false;
            };

            if (isAnimatedImage) {
                // Animated images (gif/webp) are not playable in <video>;
                // use <img> and let the browser animate natively.
                const img = document.createElement("img");
                img.src = url;
                img.style.width = "100%";
                img.style.height = "100%";
                img.style.objectFit = "cover";
                img.style.display = "block";

                img.addEventListener('contextmenu', triggerCtx, true);
                img.addEventListener('pointerdown', (e) => { if (e.button === 2) triggerCtx(e); }, true);

                img.onload = () => {
                    if (img.naturalWidth && img.naturalHeight) {
                        node.painter_aspect = img.naturalWidth / img.naturalHeight;
                        const applyResize = () => {
                            node.onResize(node.size);
                            node.setDirtyCanvas(true, true);
                        };
                        applyResize();
                        setTimeout(applyResize, 60);
                        setTimeout(applyResize, 250);
                    }
                };

                widget.element.appendChild(img);
                node.painter_media_element = img;
                node.painter_media_kind = "img";
            } else {
                // Video: keep the existing custom-controls behavior unchanged.
                const video = document.createElement("video");
                video.src = url;
                video.controls = false;
                video.loop = true;
                video.autoplay = true;
                video.muted = true;
                video.preload = "metadata";

                video.style.width = "100%";
                video.style.height = "100%";
                video.style.objectFit = "cover";
                video.style.display = "block";

                video.addEventListener('contextmenu', triggerCtx, true);
                video.addEventListener('pointerdown', (e) => { if (e.button === 2) triggerCtx(e); }, true);

                video.addEventListener('mouseenter', () => { video.muted = false; });
                video.addEventListener('mouseleave', () => { video.muted = true; });

                video.onloadedmetadata = () => {
                    if (video.videoWidth && video.videoHeight) {
                        node.painter_aspect = video.videoWidth / video.videoHeight;
                        const applyResize = () => {
                            node.onResize(node.size);
                            node.setDirtyCanvas(true, true);
                        };
                        applyResize();
                        setTimeout(applyResize, 60);
                        setTimeout(applyResize, 250);
                    }
                };

                widget.element.appendChild(video);

                // Create custom controls overlay
                createCustomControls(video, widget.element);

                node.painter_media_element = video;
                node.painter_media_kind = "video";
            }
        }
    },

    /*
     * 节点创建后（widgets 已就绪）立刻应用默认显隐；onNodeCreated 里已做一次，
     * 这里作为双保险，覆盖注册顺序/时机差异。
     */
    nodeCreated(node) {
        if (node.comfyClass !== THIS_TYPE && node.type !== THIS_TYPE) return;
        try { applyNodeVisibility(node); } catch (e) { /* ignore */ }
    }
});
