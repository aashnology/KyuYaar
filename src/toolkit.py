"""
The tools the orchestrator may call, their JSON schemas, and a dispatcher.

The model only ever supplies a tool name and a small set of validated
arguments. It never touches the dataframes, and every Evidence object that
comes back is stamped with the tool and arguments that produced it, which is
what lets the UI show where a number came from.
"""

from dataclasses import asdict

from channels import marketing_channel_analysis
from decomposition import aov_volume_decomposition
from evidence import Evidence, baseline_trend, segment_breakdown
from effects import marketing_effect, price_effect

# Only dimensions with a matching cause-testing tool are exposed. Small
# segments (customer tier) need a significance test on the association itself
# before they can be ranked reliably. Channel is covered by
# marketing_channel_analysis, which tests it against comparison regions rather
# than ranking it.
DIMENSIONS = ["region", "category"]
METRICS = ["revenue", "quantity"]

SOURCE_FILE = {
    "baseline_trend": "src/evidence.py",
    "segment_breakdown": "src/evidence.py",
    "aov_volume_decomposition": "src/decomposition.py",
    "marketing_effect": "src/effects.py",
    "marketing_channel_analysis": "src/channels.py",
    "price_effect": "src/effects.py",
}

TOOL_SPECS = [
    {
        "name": "baseline_trend",
        "description": (
            "Compare the latest month's total of a metric against the average of all "
            "earlier months. Establishes THAT the metric moved and by how much, not why. "
            "Call this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"metric": {"type": "string", "enum": METRICS}},
            "required": [],
        },
    },
    {
        "name": "segment_breakdown",
        "description": (
            "Rank the values of one dimension (for example each region or each product "
            "category) by how disproportionately they contributed to the latest month's "
            "change. Shows WHERE the change is concentrated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": DIMENSIONS},
                "metric": {"type": "string", "enum": METRICS},
            },
            "required": ["dimension"],
        },
    },
    {
        "name": "aov_volume_decomposition",
        "description": (
            "Split the latest month's revenue change into a part from the number of orders and a "
            "part from average order value, and test each against how much it normally moves "
            "month to month. Answers whether the business is getting fewer orders or smaller "
            "orders. Call it with no dimension for the overall picture, or with region or "
            "category for each segment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"dimension": {"type": "string", "enum": DIMENSIONS}},
            "required": [],
        },
    },
    {
        "name": "marketing_effect",
        "description": (
            "For every region, test whether a change in marketing spend lines up with a "
            "change in order volume, compared against regions whose spend did not change. "
            "Returns statistical evidence per region, including regions where marketing "
            "is NOT supported as an explanation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "marketing_channel_analysis",
        "description": (
            "For every paid channel in every region, test whether a change in that channel's "
            "spend lines up with a change in the orders recorded under it, compared against "
            "regions whose spend did not change. For a channel whose spend moved it also checks "
            "whether the order loss is concentrated in that channel or shared by the region's "
            "other channels (region-wide), and whether cost per order changed. Run it after "
            "marketing_effect; it shows WHICH channel, and whether channel data can support that."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "price_effect",
        "description": (
            "For every product category, test whether a like-for-like price change lines "
            "up with a change in order volume, compared against categories whose prices "
            "did not change. Returns statistical evidence per category, including "
            "categories where pricing is NOT supported as an explanation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


class ToolError(ValueError):
    """Raised for an unknown tool or an invalid argument."""


class Toolkit:
    def __init__(self, orders, marketing):
        self.orders = orders
        self.marketing = marketing

    @staticmethod
    def specs():
        return TOOL_SPECS

    def run(self, name, args=None):
        args = dict(args or {})
        if name == "baseline_trend":
            metric = args.get("metric", "revenue")
            self._check("metric", metric, METRICS)
            result = [baseline_trend(self.orders, metric_col=metric)]
            clean = {"metric": metric}
        elif name == "segment_breakdown":
            dimension = args.get("dimension")
            metric = args.get("metric", "revenue")
            self._check("dimension", dimension, DIMENSIONS)
            self._check("metric", metric, METRICS)
            result = segment_breakdown(self.orders, dimension_col=dimension, metric_col=metric)
            clean = {"dimension": dimension, "metric": metric}
        elif name == "aov_volume_decomposition":
            dimension = args.get("dimension")
            if dimension is not None:
                self._check("dimension", dimension, DIMENSIONS)
            result = aov_volume_decomposition(self.orders, dimension_col=dimension)
            clean = {} if dimension is None else {"dimension": dimension}
        elif name == "marketing_effect":
            result = marketing_effect(self.orders, self.marketing)
            clean = {}
        elif name == "marketing_channel_analysis":
            result = marketing_channel_analysis(self.orders, self.marketing)
            clean = {}
        elif name == "price_effect":
            result = price_effect(self.orders)
            clean = {}
        else:
            raise ToolError(f"unknown tool '{name}'")

        for ev in result:
            ev.details["provenance"] = {
                "tool": name,
                "args": clean,
                "source": SOURCE_FILE[name],
            }
        return result

    @staticmethod
    def _check(field, value, allowed):
        if value not in allowed:
            raise ToolError(f"{field} must be one of {allowed}, got {value!r}")


def evidence_to_payload(ev: Evidence) -> dict:
    """JSON-safe view of an Evidence object, as handed back to the model."""
    d = asdict(ev)
    d["details"] = {k: v for k, v in d["details"].items() if k != "provenance"}
    return d
