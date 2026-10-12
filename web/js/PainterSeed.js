import { app } from "../../../scripts/app.js";

const NODE_NAME = "PainterSeed";
const SEED_MAX = 1125899906842624;
const BUTTON_LABEL = "\u{1F3B2} Painter Random Seed";

const SEED_WIDGET_NAMES = ["seed", "seed_num"];
const CONTROL_WIDGET_NAMES = ["control_after_generate", "control_before_generate"];

function pickWidget(node, names) {
    if (!node.widgets) return null;
    for (const name of names) {
        const found = node.widgets.find((item) => item.name === name);
        if (found) return found;
    }
    return null;
}

app.registerExtension({
    name: "PainterNodes.PainterSeed",
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) return;

        const seedWidget = pickWidget(node, SEED_WIDGET_NAMES);
        if (!seedWidget) return;
        const controlWidget = pickWidget(node, CONTROL_WIDGET_NAMES);

        const rollButton = node.addWidget(
            "button",
            BUTTON_LABEL,
            null,
            () => {
                if (controlWidget && controlWidget.value !== "fixed") {
                    controlWidget.value = "fixed";
                }
                seedWidget.value = Math.floor(Math.random() * SEED_MAX);
                app.queuePrompt(0, 1);
            },
            { serialize: false }
        );

        const attached = [rollButton];
        if (controlWidget) attached.push(controlWidget);
        seedWidget.linkedWidgets = attached;
    },
});
