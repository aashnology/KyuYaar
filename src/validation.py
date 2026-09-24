"""
Bring-your-own-data: read, validate and clean the four tables.

The investigation tools assume a clean frame, so a person's own export has to be
checked before it reaches them. The rules here follow one principle: never
change the data silently and never crash on it. Every problem is either an
error that says what to fix, or a warning that says exactly what was done
(rows dropped, text read as numbers, a partial month) and how many rows it
touched. A few rows that cannot be used are dropped and counted; a large share
is treated as a sign the wrong file was uploaded.

Free-text labels (region, category, channel, segment) are treated as untrusted:
over-length values are unusable rows, and instruction-like values reject the file.

Nothing here estimates or fills in a value.
"""

import difflib
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_loader import enrich

TABLES = ("orders", "products", "customers", "marketing")

# column -> kind. "required" columns are used by the tools; "optional" ones are
# read when present.
SCHEMA = {
    "orders": {
        "required": {"order_date": "date", "customer_id": "id", "product_id": "id",
                     "quantity": "number", "revenue": "number", "channel": "text"},
        "optional": {"order_id": "id"},
    },
    "products": {
        "required": {"product_id": "id", "category": "text", "cost": "number"},
        "optional": {"price": "number"},
    },
    "customers": {
        "required": {"customer_id": "id", "region": "text"},
        "optional": {"segment": "text", "signup_date": "date"},
    },
    "marketing": {
        "required": {"date": "date", "channel": "text", "region": "text", "spend": "number"},
        "optional": {"impressions": "number"},
    },
}

# Share of a table's rows that may be dropped as unusable before the file is
# rejected outright.
MAX_DROP_SHARE = 0.05
MIN_MONTHS = 3            # fewer months cannot be compared at all
COMFORTABLE_MONTHS = 8    # earlier month-on-month changes needed to judge "normal" is 6
MAX_ORDERS = 500_000
_EXAMPLES = 3

# Region, category, channel and segment names are free text from the upload and
# end up inside the evidence the model reads. Real names are short, so a
# longer value is treated as an unusable row. A value that reads like an
# instruction to an AI system is not a name at all: the file is rejected. The
# list is deliberately short and literal; it catches obvious phrasing, and the
# prompt boundary plus the numeric guardrail cover the rest.
MAX_TEXT_LENGTH = 60
_INJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(rules?|instructions?|prompts?|guidelines?|guardrails?|constraints?|above|previous|prior)\b",
    r"\b(system|developer)\s+(prompt|message|instructions?)\b",
    r"\bnew\s+instructions?\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\s+(a|an|if)\b",
    r"\bpretend\s+(to\s+be|you)\b",
    r"\b(reveal|print|repeat|show)\b.{0,30}\b(prompt|instructions?)\b",
    r"\bjailbreak\b",
    r"^\s*(system|assistant|user|human)\s*:",
    r"</?\s*(system|assistant|user|tool|instructions?)\b",
    r"\[/?inst\]|<\|[^|]{0,30}\|>",
)]


class DataValidationError(ValueError):
    """Raised by load_validated when the upload has blocking problems."""

    def __init__(self, result):
        self.result = result
        super().__init__("\n".join(result.errors))


@dataclass
class ValidationResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    partial_last_month: bool = False
    data: tuple | None = None          # (orders, marketing), enriched, only when ok

    @property
    def ok(self):
        return not self.errors


# ---------------------------------------------------------------- reading ---

def _to_bytes(source):
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError(f"cannot read {type(source).__name__}")


def read_table(name, source):
    """Parse one CSV. Returns (DataFrame | None, errors, notes); errors mean the
    file could not be read at all."""
    try:
        raw = _to_bytes(source)
    except Exception as exc:  # unreadable path, closed file
        return None, [f"{name}.csv: could not be opened ({exc})"], []
    if not raw.strip():
        return None, [f"{name}.csv: the file is empty"], []

    frame, last_exc, notes = None, None, []
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=encoding)
            if frame.shape[1] == 1:   # semicolon- or tab-separated exports
                frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str, encoding=encoding)
            if encoding != "utf-8-sig":
                notes.append(f"{name}.csv: not valid UTF-8, read as Latin-1; check accented text looks right")
            break
        except Exception as exc:
            last_exc = exc
    if frame is None:
        return None, [f"{name}.csv: could not be read as a CSV ({last_exc})"], []
    return frame, [], notes


# ---------------------------------------------------------------- cleaning ---

def _normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [re.sub(r"[\s\-]+", "_", str(c).strip().lower()) for c in frame.columns]
    return frame


def _examples(values):
    vals = [str(v) for v in pd.Series(values).dropna().unique()[:_EXAMPLES]]
    return ", ".join(repr(v) for v in vals) if vals else "blank values"


def _reads_like_instruction(text):
    folded = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).lower()
    return any(p.search(folded) for p in _INJECTION_PATTERNS)


def _screen_labels(name, col, s, result):
    """Record an error when any value in a text column reads like an instruction."""
    flagged = sorted(v for v in s.dropna().unique() if _reads_like_instruction(v))
    if not flagged:
        return
    n_rows = int(s.isin(flagged).sum())
    shown = ", ".join(repr(v if len(v) <= 40 else v[:40] + "...") for v in flagged[:_EXAMPLES])
    result.errors.append(
        f"{name}.csv: '{col}' has {n_rows} row(s) with text that reads like an instruction to an AI "
        f"system (for example {shown}). Region, category, channel and segment values must be plain "
        f"names, so the file was not loaded. Rename or remove those values and upload it again."
    )


def _check_columns(name, frame, result):
    """Missing required columns, with a suggestion when a close name exists."""
    spec = SCHEMA[name]
    have = list(frame.columns)
    missing = [c for c in spec["required"] if c not in have]
    if not missing:
        return True
    # Only suggest columns that are not already meaningful to this table.
    known = set(spec["required"]) | set(spec["optional"])
    candidates = [c for c in have if c not in known]
    parts = []
    for col in missing:
        close = difflib.get_close_matches(col, candidates, n=1, cutoff=0.5)
        parts.append(f"'{col}'" + (f" (found '{close[0]}'; rename it to '{col}'?)" if close else ""))
    result.errors.append(
        f"{name}.csv: missing required column(s): {', '.join(parts)}. "
        f"Columns found: {', '.join(have) if have else 'none'}."
    )
    return False


def _coerce(name, frame, result):
    """Convert each column to its type and return a boolean mask of rows that
    are usable, with a reason recorded for every row that is not."""
    spec = {**SCHEMA[name]["required"], **SCHEMA[name]["optional"]}
    bad = pd.Series(False, index=frame.index)
    reasons = {}

    def flag(mask, why):
        nonlocal bad
        mask = mask.fillna(False).astype(bool) & ~bad
        if mask.any():
            reasons[why] = reasons.get(why, 0) + int(mask.sum())
            bad = bad | mask

    for col, kind in spec.items():
        if col not in frame.columns:
            continue
        required = col in SCHEMA[name]["required"]
        s = frame[col].astype("string").str.strip()
        blank = s.isna() | (s == "")

        if kind in ("id", "text"):
            frame[col] = s.astype(object)
            if required:
                flag(blank, f"blank {col}")
            if kind == "text":
                _screen_labels(name, col, s, result)
                flag(s.str.len() > MAX_TEXT_LENGTH, f"{col} is longer than {MAX_TEXT_LENGTH} characters")
        elif kind == "date":
            parsed = pd.to_datetime(s, errors="coerce")
            if required:
                flag(blank, f"blank {col}")
                unreadable = ~blank & parsed.isna()
                flag(unreadable, f"{col} is not a date (for example {_examples(s[unreadable.fillna(False)])})")
            frame[col] = parsed
        else:  # number
            cleaned = s.str.replace(r"[,\s$€£₹]", "", regex=True)
            cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (12.50) -> -12.50
            plain = pd.to_numeric(s, errors="coerce")
            num = pd.to_numeric(cleaned, errors="coerce")
            rescued = int((plain.isna() & num.notna() & ~blank).sum())
            if rescued:
                result.warnings.append(
                    f"{name}.csv: read {rescued} value(s) in '{col}' with currency symbols or thousands "
                    f"separators as plain numbers."
                )
            if required:
                flag(blank, f"blank {col}")
                unreadable = ~blank & num.isna()
                flag(unreadable, f"{col} is not a number (for example {_examples(s[unreadable.fillna(False)])})")
            frame[col] = num.astype(float)

    return bad, reasons


def _unify_case(name, frame, result):
    """Spellings that differ only by capitalisation or padding would be tested
    as separate regions or channels; use the most common spelling and say so."""
    for col, kind in SCHEMA[name]["required"].items():
        if kind != "text":
            continue
        counts = frame[col].value_counts()
        by_key = {}
        for value, n in counts.items():
            by_key.setdefault(str(value).lower(), []).append((n, value))
        merged = {}
        for variants in by_key.values():
            if len(variants) > 1:
                canonical = max(variants)[1]
                merged.update({v: canonical for _, v in variants if v != canonical})
        if merged:
            frame[col] = frame[col].replace(merged)
            shown = ", ".join(f"'{a}' -> '{b}'" for a, b in list(merged.items())[:_EXAMPLES])
            result.warnings.append(
                f"{name}.csv: '{col}' had spellings that differ only by capitalisation ({shown}); "
                f"treated as the same value."
            )
    return frame


def _drop(name, frame, mask, reasons, result, extra_reasons=None):
    """Drop flagged rows if they are few; otherwise reject the table."""
    reasons = dict(reasons)
    for why, m in (extra_reasons or {}).items():
        m = m & ~mask
        if m.any():
            reasons[why] = reasons.get(why, 0) + int(m.sum())
            mask = mask | m
    n_bad, total = int(mask.sum()), len(frame)
    if not n_bad:
        return frame
    detail = "; ".join(f"{why}: {count}" for why, count in reasons.items())
    if n_bad / total > MAX_DROP_SHARE:
        result.errors.append(
            f"{name}.csv: {n_bad} of {total} rows ({n_bad / total:.0%}) cannot be used ({detail}). "
            f"That is more than {MAX_DROP_SHARE:.0%} of the file, so it looks like the wrong file or "
            f"column mapping; fix the source rather than dropping that many rows."
        )
        return frame
    result.warnings.append(
        f"{name}.csv: excluded {n_bad} of {total} rows that cannot be used ({detail})."
    )
    return frame.loc[~mask].reset_index(drop=True)


def _clean_table(name, raw, result):
    frame = _normalize_columns(raw)
    if frame.empty:
        result.errors.append(f"{name}.csv: the file has a header but no rows")
        return None
    if not _check_columns(name, frame, result):
        return None

    bad, reasons = _coerce(name, frame, result)
    extra = {}
    if name == "orders":
        extra["quantity is not above zero"] = frame["quantity"] <= 0
        extra["negative revenue (refunds are not supported)"] = frame["revenue"] < 0
    elif name == "products":
        extra["negative cost"] = frame["cost"] < 0
    elif name == "marketing":
        extra["negative spend"] = frame["spend"] < 0
    # the date filter above already recorded unreadable dates as reasons
    frame = _drop(name, frame, bad, reasons, result, extra)
    if result.errors and any(e.startswith(f"{name}.csv") for e in result.errors):
        return None

    frame = _unify_case(name, frame, result)

    if name == "orders" and "order_id" in frame.columns:
        dup = frame["order_id"].duplicated()
        if dup.any():
            share = dup.mean()
            if share > MAX_DROP_SHARE:
                result.errors.append(
                    f"orders.csv: {int(dup.sum())} of {len(frame)} rows ({share:.0%}) repeat an earlier "
                    f"order_id; check that each order appears once."
                )
                return None
            result.warnings.append(
                f"orders.csv: {int(dup.sum())} repeated order_id row(s) were excluded (first one kept)."
            )
            frame = frame.loc[~dup].reset_index(drop=True)

    if name in ("products", "customers"):
        key = "product_id" if name == "products" else "customer_id"
        dup = frame[key].duplicated(keep=False)
        if dup.any():
            result.errors.append(
                f"{name}.csv: {key} must be unique, but {int(frame.loc[dup, key].nunique())} value(s) "
                f"appear more than once (for example {_examples(frame.loc[dup, key])}). "
                f"Repeated keys would multiply orders when the tables are joined."
            )
            return None
    if name == "customers" and "segment" not in frame.columns:
        frame["segment"] = "Unknown"
    if name == "customers":
        frame["segment"] = frame["segment"].fillna("Unknown")
    return frame


# ------------------------------------------------------------ cross-table ---

def _link_orders(orders, products, customers, result):
    """Drop the few orders whose product or customer is unknown; reject if many."""
    unknown_p = ~orders["product_id"].isin(products["product_id"])
    unknown_c = ~orders["customer_id"].isin(customers["customer_id"])
    reasons = {}
    if unknown_p.any():
        reasons[f"product_id not in products.csv (for example {_examples(orders.loc[unknown_p, 'product_id'])})"] = int(unknown_p.sum())
    if unknown_c.any():
        reasons[f"customer_id not in customers.csv (for example {_examples(orders.loc[unknown_c, 'customer_id'])})"] = int(unknown_c.sum())
    if not reasons:
        return orders
    mask = unknown_p | unknown_c
    n_bad = int(mask.sum())
    detail = "; ".join(f"{why}: {count}" for why, count in reasons.items())
    if n_bad / len(orders) > MAX_DROP_SHARE:
        result.errors.append(
            f"orders.csv: {n_bad} of {len(orders)} orders ({n_bad / len(orders):.0%}) do not match a "
            f"product or customer ({detail}). Check that the ids in orders.csv are the same ids used "
            f"in products.csv and customers.csv."
        )
        return orders
    result.warnings.append(f"orders.csv: excluded {n_bad} orders that do not match a product or customer ({detail}).")
    return orders.loc[~mask].reset_index(drop=True)


def _month_end_reached(last_ts):
    return last_ts.normalize() >= (last_ts + pd.offsets.MonthEnd(0)).normalize()


def _sufficiency(orders, marketing, customers, products, result):
    period = orders["order_date"].dt.to_period("M")
    months = sorted(period.unique())
    n_months = len(months)
    first, last = orders["order_date"].min(), orders["order_date"].max()
    full_range = pd.period_range(months[0], months[-1], freq="M")
    empty = [str(p) for p in full_range if p not in set(months)]

    result.summary.update({
        "orders": len(orders), "products": len(products), "customers": len(customers),
        "marketing_rows": len(marketing), "months": n_months,
        "first_order": str(first.date()), "last_order": str(last.date()),
    })

    if n_months < MIN_MONTHS:
        result.errors.append(
            f"orders.csv covers {n_months} calendar month(s) ({first.date()} to {last.date()}). "
            f"KyuYaar compares the latest month with its own history, so it needs at least "
            f"{MIN_MONTHS}, and about {COMFORTABLE_MONTHS} or more to judge what is normal."
        )
    elif n_months < COMFORTABLE_MONTHS:
        result.warnings.append(
            f"Only {n_months} months of orders. The orders-versus-order-value check needs about "
            f"{COMFORTABLE_MONTHS} months to tell a real move from ordinary variation, so it will "
            f"grade those findings weak; the cause tests still run."
        )
    if empty:
        result.warnings.append(
            f"No orders at all in {', '.join(empty[:4])}{' and more' if len(empty) > 4 else ''}. "
            f"A gap in the middle of the history can look like a collapse; check the export is complete."
        )

    if not _month_end_reached(last):
        result.partial_last_month = True
        result.warnings.append(
            f"The latest month ({months[-1]}) ends on {last.date()}, before the month ends. Every "
            f"figure for a partial month will look like a drop. Tick the box to leave it out and "
            f"investigate the last complete month instead."
        )

    # Comparison groups: the cause tests compare a region or category with the others.
    regions = customers["region"].nunique()
    cats = products["category"].nunique()
    if regions < 2 or cats < 2:
        result.errors.append(
            f"The cause tests compare each region and category with the others, so they need at "
            f"least 2 of each; found {regions} region(s) and {cats} categor{'y' if cats == 1 else 'ies'}."
        )
    elif regions < 3 or cats < 3:
        result.warnings.append(
            f"Only {regions} region(s) and {cats} categor{'y' if cats == 1 else 'ies'}: with so few "
            f"comparison groups, a change that hits half of them cannot be separated from a general one."
        )

    # Marketing must cover the two months the tests compare.
    m_last = marketing["date"].max()
    if m_last.to_period("M") < months[-1]:
        result.errors.append(
            f"marketing.csv ends {m_last.date()} but orders run to {last.date()}. The marketing tests "
            f"compare the latest two months of spend, so marketing.csv must cover them."
        )
    m_regions = set(marketing["region"])
    o_regions = set(customers["region"])
    if not (m_regions & o_regions):
        result.errors.append(
            f"No region in marketing.csv ({_examples(sorted(m_regions))}) matches a region in "
            f"customers.csv ({_examples(sorted(o_regions))}); spell them identically."
        )
    else:
        missing = sorted(o_regions - m_regions)
        extra = sorted(m_regions - o_regions)
        if missing:
            result.warnings.append(f"marketing.csv has no rows for region(s): {', '.join(missing)}; marketing cannot be tested there.")
        if extra:
            result.warnings.append(f"marketing.csv lists region(s) with no customers: {', '.join(extra)}.")
    if (marketing["spend"] > 0).sum() == 0:
        result.warnings.append("Every spend value in marketing.csv is zero, so no marketing change can be detected.")


def _smoke_test(orders, marketing, result):
    """Run the standard investigation plan once. A data shape the tools cannot
    handle becomes an error the person can read instead of a traceback."""
    from orchestrator import STANDARD_PLAN
    from toolkit import Toolkit

    tk = Toolkit(orders, marketing)
    for tool, args in STANDARD_PLAN:
        try:
            tk.run(tool, args)
        except Exception as exc:
            result.errors.append(
                f"The investigation tools could not run on this data ({tool}: {type(exc).__name__}: {exc}). "
                f"This usually means too little history or a column with unexpected values."
            )
            return


# ------------------------------------------------------------------ entry ---

def validate_tables(raw, drop_partial_last_month=False, smoke_test=True):
    """Validate and clean a dict of raw DataFrames keyed orders/products/customers/marketing.
    `result.data` holds the enriched (orders, marketing) pair when result.ok."""
    result = ValidationResult()
    absent = [t for t in TABLES if t not in raw or raw[t] is None]
    if absent:
        result.errors.append(f"Missing file(s): {', '.join(t + '.csv' for t in absent)}. All four tables are required.")
        return result

    clean = {}
    for name in TABLES:
        clean[name] = _clean_table(name, raw[name], result)
    if any(v is None for v in clean.values()):
        return result

    orders = _link_orders(clean["orders"], clean["products"], clean["customers"], result)
    if result.errors:
        return result
    if len(orders) > MAX_ORDERS:
        result.errors.append(
            f"orders.csv has {len(orders):,} usable rows; this demo is limited to {MAX_ORDERS:,}. "
            f"Filter to the last 12 to 24 months."
        )
        return result

    _sufficiency(orders, clean["marketing"], clean["customers"], clean["products"], result)
    if drop_partial_last_month and result.partial_last_month:
        cutoff = orders["order_date"].max().to_period("M").start_time
        before = len(orders)
        orders = orders[orders["order_date"] < cutoff].reset_index(drop=True)
        marketing = clean["marketing"]
        clean["marketing"] = marketing[marketing["date"] < cutoff].reset_index(drop=True)
        result.warnings = [w for w in result.warnings if "before the month ends" not in w]
        result.warnings.append(
            f"Left out the partial latest month ({before - len(orders)} orders); the investigation "
            f"uses the last complete month."
        )
        result.summary["last_order"] = str(orders["order_date"].max().date())
        result.summary["months"] -= 1
        result.summary["orders"] = len(orders)
        result.partial_last_month = False
        if result.summary["months"] < MIN_MONTHS:
            result.errors.append(f"Only {result.summary['months']} complete month(s) remain after leaving out the partial one.")
    if result.errors:
        return result

    data = enrich(orders, clean["products"], clean["customers"], clean["marketing"])
    if smoke_test:
        _smoke_test(*data, result)
    if result.ok:
        result.data = data
    return result


def load_upload(sources, drop_partial_last_month=False, smoke_test=True):
    """Read and validate four uploaded CSVs. `sources` maps table name to a
    path, bytes or file-like object. Never raises for bad data; read
    `result.ok`, `result.errors` and `result.warnings`."""
    raw, read_errors, notes = {}, [], []
    for name in TABLES:
        if sources.get(name) is None:
            raw[name] = None
            continue
        frame, errors, table_notes = read_table(name, sources[name])
        raw[name] = frame
        read_errors.extend(errors)
        notes.extend(table_notes)

    if read_errors:
        result = ValidationResult(errors=read_errors, warnings=notes)
        absent = [t for t in TABLES if sources.get(t) is None]
        if absent:
            result.errors.append(f"Missing file(s): {', '.join(t + '.csv' for t in absent)}. All four tables are required.")
        return result

    result = validate_tables(raw, drop_partial_last_month=drop_partial_last_month, smoke_test=smoke_test)
    result.warnings = notes + result.warnings
    return result


def load_validated(sources, drop_partial_last_month=False):
    """Like load_upload but raises DataValidationError; returns (orders, marketing)."""
    result = load_upload(sources, drop_partial_last_month=drop_partial_last_month)
    if not result.ok:
        raise DataValidationError(result)
    return result.data
