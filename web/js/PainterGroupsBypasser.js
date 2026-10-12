import { app } from "../../../scripts/app.js";

const NODE_TYPE = "PainterGroupsBypasser";
const NODE_TITLE = "Painter Groups Bypasser";
const BYPASS_MODE = 4;

const PROP_SORT = "painterSort";
const PROP_SORT_ALPHA = "painterSortAlphabet";
const PROP_COLORS = "painterMatchColors";
const PROP_TITLE = "painterMatchTitle";
const PROP_LIMIT = "painterToggleLimit";
const PROP_SUBGRAPHS = "painterIncludeSubgraphs";

const SORT_OPTIONS = ["position", "alphanumeric", "custom alphabet"];
const LIMIT_OPTIONS = ["default", "max one", "always one"];
const POLL_MS = 600;
const FAST_MS = 60;

/** Every PainterGroupsBypasser instance currently on the canvas. */
const liveNodes = new Set();
let pollTimer = null;
let fastTimer = null;

/* ------------------------------------------------------------------ */
/* graph helpers                                                       */
/* ------------------------------------------------------------------ */

function eachNodeDeep(nodes, visit) {
    const stack = Array.from(nodes);
    while (stack.length > 0) {
        const node = stack.pop();
        visit(node);
        if (node.isSubgraphNode?.() === true && node.subgraph) {
            for (const child of node.subgraph.nodes) stack.push(child);
        }
    }
}

function groupMembers(group) {
    if (typeof group.recomputeInsideNodes === "function") {
        group.recomputeInsideNodes();
    }
    const pool = group._children ? Array.from(group._children) : Array.from(group.nodes || []);
    return pool.filter((item) => item instanceof LGraphNode);
}

function groupIsLive(group) {
    return groupMembers(group).some((node) => node.mode === LiteGraph.ALWAYS);
}

function setGroupEnabled(group, enabled) {
    const mode = enabled ? LiteGraph.ALWAYS : BYPASS_MODE;
    eachNodeDeep(groupMembers(group), (node) => {
        node.mode = mode;
    });
    group.graph?.setDirtyCanvas(true, false);
}

function expandHex(value) {
    let hex = String(value ?? "").replace("#", "").trim().toLowerCase();
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    return hex.length === 6 ? `#${hex}` : "";
}

function wantedColors(raw) {
    const out = [];
    for (const chunk of String(raw ?? "").split(",")) {
        const name = chunk.trim().toLowerCase();
        if (!name) continue;
        const preset = LGraphCanvas.node_colors?.[name];
        const hex = expandHex(preset ? preset.groupcolor : name);
        if (hex) out.push(hex);
    }
    return out;
}

function collectGroups(includeSubgraphs) {
    const graph = app.canvas?.getCurrentGraph?.() ?? app.graph;
    const list = Array.from(graph?._groups ?? graph?.groups ?? []);
    if (includeSubgraphs && graph?.subgraphs) {
        for (const sub of graph.subgraphs.values()) {
            list.push(...(sub.groups ?? []));
        }
    }
    return list;
}

function alphabetRank(title, alphabet) {
    const lower = String(title ?? "").toLowerCase();
    for (let i = 0; i < alphabet.length; i++) {
        if (alphabet[i] && lower.startsWith(alphabet[i])) return i;
    }
    return -1;
}

function parseAlphabet(raw) {
    const text = String(raw ?? "").replace(/\n/g, "").trim().toLowerCase();
    if (!text) return [];
    if (text.includes(",")) {
        return text.split(",").map((entry) => entry.trim()).filter((entry) => entry.length > 0);
    }
    return text.split("");
}

function orderGroups(groups, mode, alphabet) {
    const copy = [...groups];
    if (mode === "alphanumeric") {
        copy.sort((a, b) => String(a.title ?? "").localeCompare(String(b.title ?? "")));
        return copy;
    }
    if (mode === "custom alphabet" && alphabet?.length) {
        copy.sort((a, b) => {
            const rankA = alphabetRank(a.title, alphabet);
            const rankB = alphabetRank(b.title, alphabet);
            if (rankA >= 0 && rankB >= 0 && rankA !== rankB) return rankA - rankB;
            if (rankA >= 0 && rankB < 0) return -1;
            if (rankB >= 0 && rankA < 0) return 1;
            return String(a.title ?? "").localeCompare(String(b.title ?? ""));
        });
        return copy;
    }
    copy.sort((a, b) => {
        const ay = Math.floor(a._pos[1] / 30);
        const by = Math.floor(b._pos[1] / 30);
        if (ay !== by) return ay - by;
        return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
    });
    return copy;
}

function groupKey(group) {
    return group.id != null ? `id:${group.id}` : `title:${group.title}`;
}

function focusOnGroup(group) {
    const canvas = app.canvas;
    if (!canvas || !group) return;
    canvas.centerOnNode?.(group);
    const size = group._size ?? [1, 1];
    if (typeof canvas.setZoom === "function") {
        const current = canvas.ds?.scale ?? 1;
        const zoomX = canvas.canvas.width / Math.max(size[0], 1) - 0.02;
        const zoomY = canvas.canvas.height / Math.max(size[1], 1) - 0.02;
        canvas.setZoom(Math.min(current, zoomX, zoomY), [
            canvas.canvas.width / 2,
            canvas.canvas.height / 2,
        ]);
    }
    canvas.setDirty(true, true);
}

/* ------------------------------------------------------------------ */
/* refresh scheduling                                                  */
/* ------------------------------------------------------------------ */

function stopPolling() {
    if (pollTimer != null) {
        clearInterval(pollTimer);
        pollTimer = null;
    }
}

function startPolling() {
    if (pollTimer == null && liveNodes.size > 0) {
        pollTimer = setInterval(() => {
            if (app.canvas?.isDragging) return;
            for (const node of liveNodes) {
                try {
                    node.refresh();
                } catch (error) {
                    console.error("[PainterNodes] group refresh failed", error);
                }
            }
        }, POLL_MS);
    }
}

function scheduleFast() {
    if (fastTimer != null) return;
    fastTimer = setTimeout(() => {
        fastTimer = null;
        for (const node of liveNodes) {
            try {
                node.refresh();
            } catch (error) {
                console.error("[PainterNodes] group refresh failed", error);
            }
        }
    }, FAST_MS);
}

/* ------------------------------------------------------------------ */
/* the node                                                            */
/* ------------------------------------------------------------------ */

class PainterGroupsBypasser extends LGraphNode {
    static title = NODE_TITLE;
    static category = "Painter/Groups";

    constructor(title = NODE_TITLE) {
        super(title);
        this.comfyClass = NODE_TYPE;
        this.isVirtualNode = true;
        this.serialize_widgets = false;
        this.properties[PROP_SORT] = "position";
        this.properties[PROP_SORT_ALPHA] = "";
        this.properties[PROP_COLORS] = "";
        this.properties[PROP_TITLE] = "";
        this.properties[PROP_LIMIT] = "default";
        this.properties[PROP_SUBGRAPHS] = true;
    }

    onNodeCreated() {
        this.addProperty(PROP_SORT, "position", { values: SORT_OPTIONS });
        this.addProperty(PROP_SORT_ALPHA, "", { type: "string" });
        this.addProperty(PROP_COLORS, "", { type: "string" });
        this.addProperty(PROP_TITLE, "", { type: "string" });
        this.addProperty(PROP_LIMIT, "default", { values: LIMIT_OPTIONS });
        this.addProperty(PROP_SUBGRAPHS, true, { type: "boolean" });
        liveNodes.add(this);
        startPolling();
        this.refresh();
    }

    onAdded() {
        liveNodes.add(this);
        startPolling();
        scheduleFast();
    }

    onRemoved() {
        liveNodes.delete(this);
        if (liveNodes.size === 0) stopPolling();
    }

    toggleWidgets() {
        return (this.widgets ?? []).filter((w) => w.painterKey != null);
    }

    limitMode() {
        return this.properties?.[PROP_LIMIT] || "default";
    }

    /** Rebuilds one toggle per matching group, keeping existing widgets in place. */
    refresh() {
        if (app.canvas?.isDragging) return;

        let sort = this.properties?.[PROP_SORT] || "position";
        let alphabet = [];
        if (sort === "custom alphabet") {
            alphabet = parseAlphabet(this.properties?.[PROP_SORT_ALPHA]);
            if (!alphabet.length) sort = "alphanumeric";
        }
        const groups = orderGroups(collectGroups(this.properties?.[PROP_SUBGRAPHS] !== false), sort, alphabet);
        const colors = wantedColors(this.properties?.[PROP_COLORS]);
        const rawTitle = String(this.properties?.[PROP_TITLE] ?? "").trim();

        let pattern = null;
        if (rawTitle) {
            try {
                pattern = new RegExp(rawTitle, "i");
            } catch (error) {
                pattern = null;
            }
        }

        const widgetCountBefore = this.toggleWidgets().length;
        let index = 0;

        for (const group of groups) {
            if (colors.length > 0) {
                const groupColor = expandHex(group.color);
                if (!groupColor || !colors.includes(groupColor)) continue;
            }
            if (pattern && !pattern.test(group.title ?? "")) continue;

            const key = groupKey(group);
            let widget = this.toggleWidgets().find((w) => w.painterKey === key);
            if (!widget) {
                let ref = null;
                ref = this.addWidget("toggle", "", false, () => this.handleToggle(ref), {
                    on: "yes",
                    off: "no",
                });
                ref.painterKey = key;
                widget = ref;
            }

            widget.painterGroup = group;
            widget.label = group.title?.trim() ? group.title : `Group ${index + 1}`;

            const live = groupIsLive(group);
            if (widget.value !== live) widget.value = live;

            const position = this.widgets.indexOf(widget);
            if (position !== index) {
                this.widgets.splice(position, 1);
                this.widgets.splice(index, 0, widget);
            }
            index++;
        }

        while ((this.widgets ?? [])[index]) {
            this.removeWidget(index);
        }

        if (this.toggleWidgets().length !== widgetCountBefore) {
            this.setSize(this.computeSize());
        }
    }

    handleToggle(widget) {
        const limit = this.limitMode();
        const wanted = widget.value;

        if (!wanted && limit === "always one") {
            const onCount = this.toggleWidgets().filter((w) => w.value).length;
            if (onCount <= 1) {
                widget.value = true;
                return;
            }
        }

        if (wanted && limit !== "default") {
            for (const other of this.toggleWidgets()) {
                if (other !== widget && other.value) {
                    other.value = false;
                    setGroupEnabled(other.painterGroup, false);
                }
            }
        }

        setGroupEnabled(widget.painterGroup, wanted);
    }

    setEvery(enable) {
        const limit = this.limitMode();
        this.toggleWidgets().forEach((widget, i) => {
            let wanted = enable;
            if (enable && limit !== "default" && i > 0) wanted = false;
            if (!enable && limit === "always one" && i === 0) wanted = true;
            widget.value = wanted;
            setGroupEnabled(widget.painterGroup, wanted);
        });
    }

    flipEvery() {
        const limit = this.limitMode();
        let oneIsOn = false;
        for (const widget of this.toggleWidgets()) {
            let wanted = !widget.value;
            if (limit !== "default" && oneIsOn) wanted = false;
            oneIsOn = oneIsOn || wanted;
            widget.value = wanted;
            setGroupEnabled(widget.painterGroup, wanted);
        }
        if (!oneIsOn && limit === "always one") {
            const last = this.toggleWidgets().pop();
            if (last) {
                last.value = true;
                setGroupEnabled(last.painterGroup, true);
            }
        }
    }

    getExtraMenuOptions(canvas, options) {
        const menu = options ?? [];
        menu.push({ content: "Bypass all groups", callback: () => this.setEvery(false) });
        menu.push({ content: "Enable all groups", callback: () => this.setEvery(true) });
        menu.push({ content: "Toggle all groups", callback: () => this.flipEvery() });

        const rows = this.toggleWidgets();
        if (rows.length > 0) {
            menu.push(null);
            for (const row of rows) {
                const group = row.painterGroup;
                menu.push({
                    content: `Focus group: ${row.label}`,
                    callback: () => focusOnGroup(group),
                });
            }
        }
        return menu;
    }
}

/* ------------------------------------------------------------------ */
/* registration                                                        */
/* ------------------------------------------------------------------ */

app.registerExtension({
    name: "PainterNodes.PainterGroupsBypasser",
    registerCustomNodes() {
        LiteGraph.registerNodeType(NODE_TYPE, PainterGroupsBypasser);
    },
    afterConfigure() {
        scheduleFast();
    },
    nodeCreated(node) {
        if (node?.comfyClass === NODE_TYPE || node?.type === NODE_TYPE) {
            liveNodes.add(node);
            startPolling();
        }
    },
});
