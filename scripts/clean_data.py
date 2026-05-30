"""Clean raw Squarespace order export into normalized uniform order data.

Filters to uniform items only (tops, bottoms, socks), normalizes sizes,
parses colors and numbers from product names, and assigns uniform sets.

Input:  data/orders.csv (raw Squarespace export)
Output: data/cleaned_orders.csv
"""

import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "orders.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned_orders.csv"

# ---------------------------------------------------------------------------
# Product category classification
# ---------------------------------------------------------------------------
TOPS_KEYWORDS = [
    "game jersey", "game tee", "away jersey", "home jersey",
    "3rd kit jersey", "3rd jersey", "kit tee",
    "temporary jersey", "temporary home jersey", "temporary away jersey",
    "temporary 3rd jersey", "temporary uniform",
]
BOTTOMS_KEYWORDS = [
    "game short", "away short", "home short",
    "3rd kit short", "3rd short",
    "temporary short", "temporary home short", "temporary away short",
    "temporary adidas short", "temporary home game adidas short",
    "temporary away game adidas short",
]
SOCKS_KEYWORDS = [
    "game sock", "away sock", "home sock",
    "copa zone", "3rd kit sock", "3rd sock",
    "temporary sock", "temporary home sock", "temporary away sock",
    "temporary navi sock", "temporary navy sock",
    "temporary white sock", "temporary red sock", "temporary black sock",
]

# Exclude practice gear and non-uniform items
EXCLUDE_KEYWORDS = [
    "practice", "training", "hoodie", "hoody", "jacket", "polo",
    "backpack", "ball", "camp", "shoe", "pump", "hat", "beanie",
    "crew", "pant", "water bottle", "hydroflask", "sackpack", "bib",
    "1/4 zip", "quarter zip", "bracelet", "hair bow", "metro",
    "goal keeper", "goalkeeper", "gk ",
]


def classify_product(name: str) -> str | None:
    """Classify a product name into top/bottom/socks or None.

    Only official game uniform items are included — practice gear is excluded.
    """
    lower = name.lower()

    # Exclude non-uniform items first
    for kw in EXCLUDE_KEYWORDS:
        if kw in lower:
            return None

    # Check more specific categories first to avoid misclassification
    for kw in SOCKS_KEYWORDS:
        if kw in lower:
            return "socks"
    for kw in BOTTOMS_KEYWORDS:
        if kw in lower:
            return "bottom"
    for kw in TOPS_KEYWORDS:
        if kw in lower:
            return "top"

    return None


# ---------------------------------------------------------------------------
# Uniform set classification
# ---------------------------------------------------------------------------
def classify_uniform_set(name: str, order_month: int) -> str:
    """Determine uniform set from product name or order month fallback."""
    lower = name.lower()

    if "fall" in lower:
        return "fall"
    if "winter" in lower:
        return "winter"
    if "spring" in lower:
        return "spring"
    if lower.startswith("cfa"):
        return "cfa"

    # Fallback by order month
    if order_month in (7, 8, 9, 10):
        return "fall"
    if order_month in (11, 12, 1, 2):
        return "winter"
    if order_month in (3, 4, 5, 6):
        return "spring"

    return "unknown"


# ---------------------------------------------------------------------------
# Gender/age parsing
# ---------------------------------------------------------------------------
def parse_gender_age(name: str) -> str:
    """Parse gender/age group from product name."""
    lower = name.lower()

    women_patterns = ["women", "womens", "women's", "girls"]
    for pattern in women_patterns:
        if pattern in lower:
            return "womens"

    return "mens_youth"


# ---------------------------------------------------------------------------
# Size normalization
# ---------------------------------------------------------------------------
VALID_SIZES = {
    "YXS", "YS", "YM", "YL", "YXL",
    "AS", "AM", "AL", "AXL", "AXXL",
    "WXS", "WS", "WM", "WL", "WXL",
}

# Map common variant patterns to standard sizes
SIZE_MAP = {
    "youth xs": "YXS", "youth xsmall": "YXS", "youth x-small": "YXS",
    "youth s": "YS", "youth small": "YS",
    "youth m": "YM", "youth medium": "YM", "youth med": "YM",
    "youth l": "YL", "youth large": "YL",
    "youth xl": "YXL", "youth x-large": "YXL",
    "adult s": "AS", "adult small": "AS", "men s": "AS", "men's s": "AS",
    "adult m": "AM", "adult medium": "AM", "adult med": "AM",
    "men m": "AM", "men's m": "AM",
    "adult l": "AL", "adult large": "AL", "men l": "AL", "men's l": "AL",
    "adult xl": "AXL", "adult x-large": "AXL",
    "adult xxl": "AXXL", "adult xx-large": "AXXL",
    "women xs": "WXS", "women's xs": "WXS", "woman xs": "WXS",
    "women s": "WS", "women's s": "WS", "woman s": "WS",
    "women m": "WM", "women's m": "WM", "woman m": "WM",
    "women l": "WL", "women's l": "WL", "woman l": "WL",
    "women xl": "WXL", "women's xl": "WXL", "woman xl": "WXL",
    "women xsmall": "WXS", "women small": "WS", "women medium": "WM",
    "women large": "WL", "women xlarge": "WXL",
    "woman xsmall": "WXS", "woman small": "WS", "woman medium": "WM",
    "woman large": "WL", "woman xlarge": "WXL",
}

# Direct code matches (case-insensitive)
DIRECT_SIZE_CODES = {
    "yxs": "YXS", "ys": "YS", "ym": "YM", "yl": "YL", "yxl": "YXL",
    "as": "AS", "am": "AM", "al": "AL", "axl": "AXL", "axs": "AS",
    "axxl": "AXXL", "2xl": "AXXL",
    "wxs": "WXS", "ws": "WS", "wm": "WM", "wl": "WL", "wxl": "WXL",
}


def normalize_size(variant: str) -> str:
    """Extract and normalize size from the variant string."""
    if pd.isna(variant):
        return "UNKNOWN"

    original = str(variant).strip()
    lower = original.lower()

    # Try direct match first (the variant IS just a size code)
    if lower in DIRECT_SIZE_CODES:
        return DIRECT_SIZE_CODES[lower]

    # Try matching size code at start or end after splitting on "/"
    parts = re.split(r"[/,]", lower)
    for part in parts:
        part = part.strip()
        if part in DIRECT_SIZE_CODES:
            return DIRECT_SIZE_CODES[part]

    # Try the longer text patterns
    for pattern, size in SIZE_MAP.items():
        if pattern in lower:
            return size

    # Try regex for embedded size codes like "Black/YM" or "Entrada 22/AL"
    size_match = re.search(
        r"\b(yxs|ys|ym|yl|yxl|axxl|axl|as|am|al|wxs|ws|wm|wl|wxl)\b",
        lower,
    )
    if size_match:
        return DIRECT_SIZE_CODES[size_match.group(1)]

    # Try "Large/Black", "Medium/Navy", "Small/White", "X-Large/Black" patterns
    desc_match = re.search(
        r"\b(x-?small|small|medium|large|x-?large|xlarge)\b", lower,
    )
    if desc_match:
        desc_map = {
            "xsmall": "AS", "x-small": "AS", "small": "AS",
            "medium": "AM", "large": "AL",
            "xlarge": "AXL", "x-large": "AXL",
        }
        return desc_map.get(desc_match.group(1), "UNKNOWN")

    # Try standalone S/M/L/XL patterns (less specific)
    standalone = re.search(r"\b(xxl|xl|xs|s|m|l)\b", lower)
    if standalone:
        size_letter = standalone.group(1).upper()
        size_prefix_map = {
            "XS": "AS", "S": "AS", "M": "AM", "L": "AL",
            "XL": "AXL", "XXL": "AXXL",
        }
        return size_prefix_map.get(size_letter, "UNKNOWN")

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Color parsing
# ---------------------------------------------------------------------------
COLORS = [
    "navy", "white", "black", "red", "royal", "yellow", "green",
    "sky blue", "scuba blue", "orange", "pink", "purple", "grey", "gray",
    "neon green", "neon orange", "neon yellow",
]


def parse_color(name: str) -> str:
    """Extract jersey color from product name."""
    lower = name.lower()

    # Check multi-word colors first
    for color in sorted(COLORS, key=len, reverse=True):
        if color in lower:
            return color.title()

    return "Unknown"


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------
def parse_number(variant: str) -> str | None:
    """Extract jersey number info from variant string."""
    if pd.isna(variant):
        return None

    lower = str(variant).lower()

    if "no number" in lower:
        return None
    if "with name & number" in lower or "with name and number" in lower:
        return "custom"
    if "with name, no number" in lower:
        return None

    number_match = re.search(r"add number.*?for.*?\$", lower)
    if number_match:
        return "custom"

    # Check for actual number in variant
    num_match = re.search(r"#\s*(\d+)", str(variant))
    if num_match:
        return num_match.group(1)

    return None


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------
def clean_data() -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned DataFrame."""
    print(f"Reading {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Parse dates
    df["parsed_date"] = pd.to_datetime(df["Created at"], utc=True, errors="coerce")
    df["order_month"] = df["parsed_date"].dt.month
    df["order_year"] = df["parsed_date"].dt.year

    # Filter to paid orders only
    df = df[df["Financial Status"] == "PAID"].copy()
    print(f"  After paid filter: {len(df):,}")

    # Drop rows with no product name
    df = df[df["Lineitem name"].notna()].copy()

    # Classify products
    df["product_category"] = df["Lineitem name"].apply(classify_product)
    df = df[df["product_category"].notna()].copy()
    print(f"  After uniform filter: {len(df):,}")

    # Build cleaned columns
    df["order_id"] = df["Order ID"].astype(int)
    df["order_date"] = df["parsed_date"].dt.date
    df["season"] = df["order_month"].apply(
        lambda m: "fall" if m in (7, 8, 9, 10)
        else "winter" if m in (11, 12, 1, 2)
        else "spring"
    )
    df["year"] = df["order_year"].astype(int)
    df["uniform_set"] = df.apply(
        lambda row: classify_uniform_set(row["Lineitem name"], row["order_month"]),
        axis=1,
    )
    # Player names excluded from output for privacy
    df["gender_age"] = df["Lineitem name"].apply(parse_gender_age)
    df["size"] = df["Lineitem variant"].apply(normalize_size)
    df["color"] = df.apply(
        lambda row: parse_color(row["Lineitem name"])
        if row["product_category"] == "top"
        else None,
        axis=1,
    )
    df["number"] = df.apply(
        lambda row: parse_number(row["Lineitem variant"])
        if row["product_category"] == "top"
        else None,
        axis=1,
    )
    df["quantity"] = pd.to_numeric(df["Lineitem quantity"], errors="coerce").fillna(1).astype(int)
    df["unit_price"] = pd.to_numeric(df["Lineitem price"], errors="coerce")

    # Select final columns
    output_cols = [
        "order_id", "order_date", "season", "year", "uniform_set",
        "product_category", "gender_age", "size",
        "color", "number", "quantity", "unit_price",
    ]
    result = df[output_cols].copy()

    # Sort by date
    result = result.sort_values(["order_date", "order_id"]).reset_index(drop=True)

    return result


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the cleaned data."""
    print(f"\n{'='*60}")
    print(f"CLEANED DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(df):,}")
    print(f"Unique orders: {df['order_id'].nunique():,}")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"\nBy product category:")
    print(df["product_category"].value_counts().to_string())
    print(f"\nBy uniform set:")
    print(df["uniform_set"].value_counts().to_string())
    print(f"\nBy gender/age:")
    print(df["gender_age"].value_counts().to_string())
    print(f"\nBy size (top 15):")
    print(df["size"].value_counts().head(15).to_string())
    print(f"\nUNKNOWN sizes: {(df['size'] == 'UNKNOWN').sum():,} "
          f"({(df['size'] == 'UNKNOWN').mean()*100:.1f}%)")
    print(f"\nBy color (tops only):")
    tops = df[df["product_category"] == "top"]
    print(tops["color"].value_counts().head(10).to_string())


if __name__ == "__main__":
    cleaned = clean_data()
    print_summary(cleaned)

    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved to {CLEAN_PATH}")
