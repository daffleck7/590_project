"""
data_module.py
==============
Pure Python — zero LLM involvement.
Daniel's responsibility (he already has clean_data.py, this is the next layer).

Input:  path to cleaned_orders.csv  +  ProblemConfig
Output: DataBundle  (features DataFrame, target Series, metadata)

The DataBundle is what the Prediction Module trains on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# We import lazily so this module can be imported even if problem_config
# is not on sys.path yet — the caller wires the path.
try:
    from src.models.problem_config import ProblemConfig, DataRequirements
except ImportError:
    try:
        from problem_config import ProblemConfig, DataRequirements
    except ImportError:
        ProblemConfig = None  # type: ignore
        DataRequirements = None  # type: ignore


# ---------------------------------------------------------------------------
# DataBundle — the output contract
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """
    Everything downstream modules need. Nothing more.

    features_train / features_test  : DataFrames of engineered features
    target_train  / target_test     : Series of demand quantity
    demand_pivot                    : (year, season, size, category) → quantity
                                       Used directly by the Optimizer as demand samples
    metadata                        : sizes, categories, seasons present, train/test years
    input_hash                      : SHA-256 of the raw CSV — for trace logger
    """
    features_train: pd.DataFrame
    target_train: pd.Series
    features_test: pd.DataFrame
    target_test: pd.Series
    demand_pivot: pd.DataFrame          # aggregated demand per group
    metadata: dict = field(default_factory=dict)
    input_hash: str = ""

    def summary(self) -> str:
        return (
            f"DataBundle | "
            f"train={len(self.features_train)} rows | "
            f"test={len(self.features_test)} rows | "
            f"demand_groups={len(self.demand_pivot)} | "
            f"hash={self.input_hash[:8]}..."
        )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DataModule:
    """
    Loads cleaned_orders.csv, engineers features, chronological split,
    validates against ProblemConfig.data_requirements.

    Usage:
        dm = DataModule(config)
        bundle = dm.load("data/cleaned_orders.csv")
    """

    def __init__(self, config: "ProblemConfig"):
        self.config = config
        self.req: "DataRequirements" = config.data_requirements

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, csv_path: str | Path) -> DataBundle:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        raw = pd.read_csv(csv_path)
        input_hash = self._hash_file(csv_path)

        self._validate_columns(raw)
        df = self._clean(raw)
        df = self._engineer_features(df)

        demand_pivot = self._aggregate_demand(df)

        train_df, test_df = self._split(df)

        features_train, target_train = self._make_features_target(train_df)
        features_test,  target_test  = self._make_features_target(test_df)

        metadata = {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_years": sorted(train_df["year"].unique().tolist()),
            "test_years": sorted(test_df["year"].unique().tolist()),
            "sizes": sorted(df["size"].unique().tolist()),
            "categories": sorted(df["product_category"].unique().tolist()),
            "seasons": sorted(df["season"].unique().tolist()),
            "uniform_sets": sorted(df["uniform_set"].unique().tolist()) if "uniform_set" in df.columns else [],
            "demand_groups": len(demand_pivot),
        }

        return DataBundle(
            features_train=features_train,
            target_train=target_train,
            features_test=features_test,
            target_test=target_test,
            demand_pivot=demand_pivot,
            metadata=metadata,
            input_hash=input_hash,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = set(self.req.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Parse dates
        df["order_date"] = pd.to_datetime(df["order_date"])

        # Normalise string columns (only those present)
        for col in ["season", "product_category", "size", "uniform_set", "gender_age"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.lower()

        # Drop rows where key fields are null (should be zero from clean_data.py but be safe)
        before = len(df)
        df = df.dropna(subset=["size", "product_category", "season", "year", "quantity"])
        if len(df) < before:
            print(f"[DataModule] Dropped {before - len(df)} rows with null key fields")

        # Drop rows with zero or negative quantity (returns / errors)
        df = df[df["quantity"] > 0]

        # colour and number columns have lots of NaN — fine, we don't use them
        df = df.drop(columns=["color", "number"], errors="ignore")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- Size group encoding ---
        # Youth = YXS/YS/YM/YL/YXL
        # Adult men = AS/AM/AL/AXL
        # Women = WXS/WS/WM/WL/WXL
        size_group_map = {}
        for s in df["size"].unique():
            if s.startswith("y"):
                size_group_map[s] = "youth"
            elif s.startswith("w"):
                size_group_map[s] = "women"
            else:
                size_group_map[s] = "adult_men"
        df["size_group"] = df["size"].map(size_group_map)

        # --- Size numeric rank (for ordinal encoding) ---
        size_order = {
            "yxs": 1, "ys": 2, "ym": 3, "yl": 4, "yxl": 5,
            "wxs": 6, "ws": 7, "wm": 8, "wl": 9, "wxl": 10,
            "as": 11, "am": 12, "al": 13, "axl": 14,
        }
        df["size_rank"] = df["size"].map(size_order).fillna(7)   # unknown → mid rank

        # --- Season numeric (for gradient boosting) ---
        season_map = {"fall": 1, "winter": 2, "spring": 3, "cfa": 4}
        df["season_num"] = df["season"].map(season_map).fillna(0)

        # --- Jersey lifecycle year ---
        # Condivo 22 → years 2022–2023 (Year 1 = 2022, Year 2 = 2023)
        # Tiro 25    → years 2024–2026 (Year 1 = 2024, Year 2 = 2025/2026)
        # Simplified: even years in lifecycle = Year 1, next = Year 2
        def lifecycle_year(row):
            y = row["year"]
            if y in [2022, 2024]:
                return 1
            elif y in [2023, 2025, 2026]:
                return 2
            return 1  # 2020/2021 = older model, treat as Year 1
        df["lifecycle_year"] = df.apply(lifecycle_year, axis=1)

        # --- Year relative to dataset start (trend feature) ---
        df["year_idx"] = df["year"] - df["year"].min()

        # --- Month extracted ---
        df["order_month"] = df["order_date"].dt.month

        return df

    def _aggregate_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to (year, season, uniform_set, product_category, size) → total quantity.
        This is the demand table the Optimizer samples from.
        """
        keys = self.req.groupby_keys + ["lifecycle_year", "size_group", "size_rank", "season_num"]
        # Only keep keys that exist in df
        keys = [k for k in keys if k in df.columns]

        demand = (
            df.groupby(keys, observed=True)["quantity"]
            .sum()
            .reset_index()
            .rename(columns={"quantity": "demand"})
        )
        return demand

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split — never random for time series."""
        train = df[df["year"].isin(self.req.train_years)].copy()
        test  = df[df["year"].isin(self.req.test_years)].copy()
        return train, test

    def _make_features_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build feature matrix X and target y from a split.

        Features used for prediction:
          - size_rank          (ordinal size encoding)
          - season_num         (1=fall, 2=winter, 3=spring, 4=cfa)
          - lifecycle_year     (1 or 2)
          - year_idx           (0, 1, 2 … — trend)
          - order_month        (seasonality within year)
          - is_youth           (1/0)
          - is_women           (1/0)
          - is_top / is_bottom / is_socks
        """
        df = df.copy()

        # One-hot style binary flags (better than get_dummies for explicit control)
        df["is_youth"]   = (df["size_group"] == "youth").astype(int)
        df["is_women"]   = (df["size_group"] == "women").astype(int)
        df["is_top"]     = (df["product_category"] == "top").astype(int)
        df["is_bottom"]  = (df["product_category"] == "bottom").astype(int)
        df["is_socks"]   = (df["product_category"] == "socks").astype(int)

        feature_cols = [
            "size_rank", "season_num", "lifecycle_year",
            "year_idx", "order_month",
            "is_youth", "is_women",
            "is_top", "is_bottom", "is_socks",
        ]
        X = df[feature_cols].reset_index(drop=True)
        y = df["quantity"].reset_index(drop=True)
        return X, y

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from problem_config import build_cfa_default_config

    config = build_cfa_default_config()
    dm = DataModule(config)

    csv_path = Path(__file__).parent.parent.parent / "data" / "cleaned_orders.csv"
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}. Copy cleaned_orders.csv to data/ folder.")
        sys.exit(1)

    bundle = dm.load(csv_path)
    print(bundle.summary())
    print("\nTrain features shape:", bundle.features_train.shape)
    print("Test features shape:", bundle.features_test.shape)
    print("\nDemand pivot sample:")
    print(bundle.demand_pivot.head(10).to_string())
    print("\nMetadata:")
    for k, v in bundle.metadata.items():
        print(f"  {k}: {v}")
