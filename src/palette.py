"""Named chart colour presets, modeled on Excel's built-in chart colour
themes. Purely cosmetic: STRENGTH_COLOR in charts.py (strong/moderate/weak)
is a separate, fixed semantic mapping and does not read from here, so a
palette choice can never change what strong vs. weak looks like.

Each preset gives a `series` list (cycled across per-column/per-trace lines),
a `latest` accent colour (the marker picking out the latest month or the
highest-strength point), and a `comparison` grey (dashed baselines, control
groups, gridlines)."""

PALETTES = {
    "Office": {
        "series": ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"],
        "latest": "#C00000",
        "comparison": "#A5A5A5",
    },
    "Colorful": {
        "series": ["#5B9BD5", "#ED7D31", "#A5A5A5", "#FFC000", "#4472C4", "#70AD47"],
        "latest": "#FF0000",
        "comparison": "#808080",
    },
    "Monochrome Blue": {
        "series": ["#1F4E79", "#2E75B6", "#5B9BD5", "#9DC3E6", "#BDD7EE", "#DEEBF7"],
        "latest": "#1F4E79",
        "comparison": "#9DC3E6",
    },
    "Grayscale": {
        "series": ["#262626", "#525252", "#7F7F7F", "#A6A6A6", "#BFBFBF", "#D9D9D9"],
        "latest": "#000000",
        "comparison": "#A6A6A6",
    },
}

DEFAULT_PALETTE = "Office"


def get_palette(name):
    """A named preset's colours, falling back to the default for an unknown
    or missing name (e.g. before the sidebar widget has run once)."""
    return PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
