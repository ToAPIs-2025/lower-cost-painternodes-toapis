"""PainterSeed: outputs a single INT seed value for the whole workflow.

Pure passthrough node. The companion front-end script
(web/js/PainterSeed.js) adds a one-click randomize-and-run button.
"""

# Seed upper bound, kept inside the safe integer range used by ComfyUI samplers.
PAINTER_SEED_MAX = 1125899906842624


class PainterSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": PAINTER_SEED_MAX,
                        "control_after_generate": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    OUTPUT_NODE = False

    CATEGORY = "Painter/Seed"
    DESCRIPTION = "Passes a single seed value to every sampler connected to it."

    def get_seed(self, seed):
        return (seed,)


NODE_CLASS_MAPPINGS = {
    "PainterSeed": PainterSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterSeed": "Painter Seed",
}
