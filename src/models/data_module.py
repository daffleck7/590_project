"""
data_module.py
==============
Generic data loader for the optimization pipeline.

Input:  path to cleaned CSV + ProblemConfig
Output: DataBundle (features, target, demand pivot, metadata)

The DataBundle is what the Prediction Module trains on.
Feature engineering is done automatically based on column types —
no domain-specific logic hardcoded here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Everything downstream modules need.

    features_train / features_test  : DataFrames of auto-engineered features
    target_train  / target_test     : Series of demand quantity
    demand_pivot                    : aggregated demand per group
    metadata                        : column info, train/test years, etc.
    input_hash                      : SHA-256 of the raw CSV for tracing
    """

    features_train: pd.DataFrame
    target_train: pd.Series
    features_test: pd.DataFrame
    target_test: pd.Series
    demand_pivot: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    input_hash: str = ""

    def summary(self) -> str:
        """One-line summary of the bundle."""
        return (
            f"DataBundle | "
            f"train={len(self.features_train)} rows | "
            f"test={len(self.features_test)} rows | "
            f"demand_groups={len(self.demand_pivot)} | "
            f"features={len(self.features_train.columns)} | "
            f"hash={self.input_hash[:8]}..."
        )


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class DataModule:
    """Generic data loader: load CSV, validate, aggregate, split, auto-feature.

    Usage:
        dm = DataModule(config)
        bundle = dm.load("data/cleaned.csv")
    """

    def __init__(self, config: "ProblemConfig"):
        self.config = config
        self.req: "DataRequirements" = config.data_requirements

    def load(self, csv_path: str | Path) -> DataBundle:
        """Load CSV and produce a DataBundle."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        raw = pd.read_csv(csv_path, low_memory=False)
        input_hash = self._hash_file(csv_path)

        self._validate_columns(raw)
        df = self._clean(raw)

        demand_pivot = self._aggregate_demand(df)
        demand_pivot = self._auto_engineer_features(demand_pivot)

        train_df, test_df = self._split(demand_pivot)
        features_train, target_train = self._make_features_target(train_df)
        features_test, target_test = self._make_features_target(test_df)

        # Collect metadata from the data itself
        metadata = {
            "total_rows": len(df),
            "demand_groups": len(demand_pivot),
            "train_groups": len(train_df),
            "test_groups": len(test_df),
            "feature_columns": list(features_train.columns),
            "n_features": len(features_train.columns),
            "train_years": sorted(train_df["year"].unique().tolist()) if "year" in train_df.columns else [],
            "test_years": sorted(test_df["year"].unique().tolist()) if "year" in test_df.columns else [],
        }

        # Add unique values for categorical groupby keys
        for key in self.req.groupby_keys:
            if key in df.columns and key != "year":
                metadata[f"{key}_values"] = sorted(df[key].dropna().unique().tolist())

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
        """Check that required columns exist."""
        if not self.req.required_columns:
            return
        missing = set(self.req.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: parse dates, normalize strings, drop nulls."""
        df = df.copy()

        # Parse date column if present
        date_col = self.req.date_column
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Normalize string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip().str.lower()

        # Drop rows where target is null or non-positive
        target = self.req.target_column
        if target in df.columns:
            before = len(df)
            df = df.dropna(subset=[target])
            df = df[df[target] > 0]
            dropped = before - len(df)
            if dropped > 0:
                print(f"[DataModule] Dropped {dropped} rows with null/non-positive {target}")

        return df

    def _aggregate_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate raw rows to demand per group using groupby_keys."""
        keys = [k for k in self.req.groupby_keys if k in df.columns]
        target = self.req.target_column

        if not keys:
            demand = df.copy()
            demand = demand.rename(columns={target: "demand"})
            return demand

        demand = (
            df.groupby(keys, observed=True)[target]
            .sum()
            .reset_index()
            .rename(columns={target: "demand"})
        )

        return demand

    def _auto_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-generate features from the demand pivot.

        - Numeric columns pass through as-is
        - Categorical/string columns get label-encoded as ordinal integers
        - Year gets a relative index (trend feature)
        - Date components extracted if date column present
        """
        df = df.copy()

        # Year relative index (trend)
        if "year" in df.columns:
            df["year_idx"] = df["year"] - df["year"].min()

        # Label-encode categorical columns
        for col in df.select_dtypes(include=["object"]).columns:
            unique_vals = sorted(df[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            df[f"{col}_encoded"] = df[col].map(mapping).fillna(-1).astype(int)

        return df

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split by train/test years."""
        if "year" not in df.columns:
            # No year column — use 80/20 split
            split_idx = int(len(df) * 0.8)
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

        train = df[df["year"].isin(self.req.train_years)].copy()
        test = df[df["year"].isin(self.req.test_years)].copy()
        return train, test

    def _make_features_target(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix X and target y.

        Uses encoded categorical columns and engineered numeric columns.
        Excludes raw groupby keys, IDs, the target, and datetime columns.
        """
        target_col = "demand"
        exclude = {target_col, "year"}  # year is captured by year_idx

        # Exclude raw string/object columns
        for col in df.select_dtypes(include=["object"]).columns:
            exclude.add(col)

        # Exclude datetime columns
        for col in df.select_dtypes(include=["datetime64"]).columns:
            exclude.add(col)

        # Exclude raw groupby keys that aren't encoded
        for col in self.req.groupby_keys:
            exclude.add(col)

        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

        x_df = df[feature_cols].fillna(0).reset_index(drop=True)
        y_series = df[target_col].reset_index(drop=True)
        return x_df, y_series

    @staticmethod
    def _hash_file(path: Path) -> str:
        """SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
