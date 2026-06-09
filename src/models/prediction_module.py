"""
prediction_module.py
====================
Generic prediction module for inventory optimization.

Input:  DataBundle (from data_module.py) + ProblemConfig
Output: PredictionResult

Three models compared on newsvendor cost:
  1. XGBoost    -- gradient boosting
  2. LightGBM   -- Microsoft's gradient boosting
  3. Ridge      -- linear baseline

Each model also has PTO-Adjusted variant (critical ratio shift).
Winner (lowest NV cost) passed to optimizer.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)


@dataclass
class PredictionParams:
    """Tunable hyperparameters for the prediction models."""

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    lgbm_n_estimators: int = 300
    lgbm_max_depth: int = 4
    lgbm_learning_rate: float = 0.05
    lgbm_subsample: float = 0.8
    ridge_alpha: float = 1.0
    pto_strength: float = 0.8
    train_years: list[int] = field(default_factory=lambda: [2020, 2021, 2022, 2023])
    test_years: list[int] = field(default_factory=lambda: [2024])


@dataclass
class PredictionResult:
    """Output of the prediction module."""

    predicted_demand: np.ndarray
    demand_df: pd.DataFrame
    sigma: np.ndarray
    P10: np.ndarray
    P50: np.ndarray
    P90: np.ndarray
    best_model_name: str
    all_costs: dict
    baseline_cost: float
    model_xgb: object
    model_lgbm: object
    feature_importance: pd.Series
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary."""
        best_cost = self.all_costs[self.best_model_name]
        saving = (1 - best_cost / self.baseline_cost) * 100 if self.baseline_cost > 0 else 0
        return (
            f"PredictionResult | winner={self.best_model_name} | "
            f"NV cost=${best_cost:,.0f} | {saving:+.1f}% vs baseline | "
            f"groups={len(self.predicted_demand)}"
        )


class PredictionModule:
    """Generic prediction module — auto-detects features from the DataBundle."""

    def __init__(self, config, params: PredictionParams | None = None):
        self.config = config
        self.params = params or PredictionParams()

    def run(self, bundle) -> PredictionResult:
        """Run all models and return the best by newsvendor cost."""
        train_years = self.params.train_years
        test_years = self.params.test_years

        pivot = bundle.demand_pivot

        if "year" in pivot.columns:
            dtr = pivot[pivot["year"].isin(train_years)].copy().reset_index(drop=True)
            dte = pivot[pivot["year"].isin(test_years)].copy().reset_index(drop=True)
        else:
            split_idx = int(len(pivot) * 0.8)
            dtr = pivot.iloc[:split_idx].copy().reset_index(drop=True)
            dte = pivot.iloc[split_idx:].copy().reset_index(drop=True)

        Xtr, ytr = self._build_features(dtr)
        Xte, yte = self._build_features(dte)

        baseline_preds, baseline_cost = self._baseline(dtr, dte)

        model_xgb, pred_xgb, cost_xgb = self._model_xgboost(Xtr, ytr, Xte, dte)
        model_lgbm, pred_lgbm, cost_lgbm = self._model_lightgbm(Xtr, ytr, Xte, dte)
        _, pred_ridge, cost_ridge = self._model_ridge(Xtr, ytr, Xte, dte)

        pred_xgb_adj, cost_xgb_adj = self._pto_adjust(pred_xgb, dte)
        pred_lgbm_adj, cost_lgbm_adj = self._pto_adjust(pred_lgbm, dte)
        pred_ridge_adj, cost_ridge_adj = self._pto_adjust(pred_ridge, dte)

        sigma_arr = self._compute_sigma(dtr, model_xgb, Xtr)

        all_costs = {
            "Baseline": baseline_cost,
            "XGBoost": cost_xgb,
            "XGBoost+PTO": cost_xgb_adj,
            "LightGBM": cost_lgbm,
            "LightGBM+PTO": cost_lgbm_adj,
            "Ridge": cost_ridge,
            "Ridge+PTO": cost_ridge_adj,
        }

        model_preds = {
            "XGBoost": pred_xgb,
            "XGBoost+PTO": pred_xgb_adj,
            "LightGBM": pred_lgbm,
            "LightGBM+PTO": pred_lgbm_adj,
            "Ridge": pred_ridge,
            "Ridge+PTO": pred_ridge_adj,
        }

        best_name = min(model_preds, key=lambda x: all_costs[x])
        best_preds = np.maximum(model_preds[best_name], self._get_moq())

        print(f"[PredictionModule] Train: {len(dtr)} groups | Test: {len(dte)} groups")
        print(f"[PredictionModule] Features: {list(Xtr.columns)}")
        print(f"[PredictionModule] Results:")
        for name, cost in sorted(all_costs.items(), key=lambda x: x[1]):
            chg = (cost / baseline_cost - 1) * 100 if baseline_cost > 0 else 0
            winner = " <-- WINNER" if name == best_name else ""
            print(f"  {name:<20}: ${cost:>9,.0f}  ({chg:+.1f}%){winner}")

        # Build sigma array — one per test group
        sigma_val = np.full(len(dte), sigma_arr) if np.isscalar(sigma_arr) else sigma_arr
        P10 = np.maximum(best_preds + sigma_val * norm.ppf(0.10), 0)
        P50 = np.maximum(best_preds + sigma_val * norm.ppf(0.50), 0)
        P90 = np.maximum(best_preds + sigma_val * norm.ppf(0.90), 0)

        return PredictionResult(
            predicted_demand=best_preds,
            demand_df=dte,
            sigma=sigma_val,
            P10=P10, P50=P50, P90=P90,
            best_model_name=best_name,
            all_costs=all_costs,
            baseline_cost=baseline_cost,
            model_xgb=model_xgb,
            model_lgbm=model_lgbm,
            feature_importance=pd.Series(
                model_xgb.feature_importances_, index=Xtr.columns
            ).sort_values(ascending=False),
            metadata={
                "train_years": train_years,
                "test_years": test_years,
                "n_train": len(dtr),
                "n_test": len(dte),
                "feature_columns": list(Xtr.columns),
            },
        )

    def _build_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Auto-detect feature columns from the demand pivot.

        Uses all numeric columns except 'demand' as features.
        Excludes raw string/object/datetime columns.
        """
        exclude = {"demand"}
        feature_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]
        return df[feature_cols].reset_index(drop=True), df["demand"].reset_index(drop=True)

    def _baseline(self, dtr: pd.DataFrame, dte: pd.DataFrame):
        """Baseline: per-group average demand from training data.

        Groups by all non-numeric, non-year columns present in both sets.
        """
        # Find categorical grouping columns (exclude year, demand, numeric features)
        group_cols = [
            c for c in dtr.columns
            if not pd.api.types.is_numeric_dtype(dtr[c]) and c != "demand"
        ]
        if not group_cols:
            # No categorical columns — use global mean
            bq = np.full(len(dte), dtr["demand"].mean())
            return bq, self._total_nv_cost(bq, dte)

        avg = (
            dtr.groupby(group_cols)["demand"]
            .mean()
            .reset_index()
            .rename(columns={"demand": "bq"})
        )
        dte_b = dte.merge(avg, on=group_cols, how="left")
        dte_b["bq"] = dte_b["bq"].fillna(dtr["demand"].mean())
        return dte_b["bq"].values, self._total_nv_cost(dte_b["bq"].values, dte_b.reset_index(drop=True))

    def _model_xgboost(self, Xtr, ytr, Xte, dte):
        """Train XGBoost regressor."""
        p = self.params
        m = XGBRegressor(
            n_estimators=p.xgb_n_estimators, max_depth=p.xgb_max_depth,
            learning_rate=p.xgb_learning_rate, subsample=p.xgb_subsample,
            colsample_bytree=p.xgb_colsample_bytree, random_state=42, verbosity=0,
        )
        m.fit(Xtr, ytr)
        pred = np.maximum(m.predict(Xte), 0)
        return m, pred, self._total_nv_cost(pred, dte)

    def _model_lightgbm(self, Xtr, ytr, Xte, dte):
        """Train LightGBM regressor."""
        p = self.params
        m = LGBMRegressor(
            n_estimators=p.lgbm_n_estimators, max_depth=p.lgbm_max_depth,
            learning_rate=p.lgbm_learning_rate, subsample=p.lgbm_subsample,
            random_state=42, verbose=-1,
        )
        m.fit(Xtr, ytr)
        pred = np.maximum(m.predict(Xte), 0)
        return m, pred, self._total_nv_cost(pred, dte)

    def _model_ridge(self, Xtr, ytr, Xte, dte):
        """Train Ridge regression (linear baseline)."""
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        m = Ridge(alpha=self.params.ridge_alpha)
        m.fit(Xtr_s, ytr)
        pred = np.maximum(m.predict(Xte_s), 0)
        return m, pred, self._total_nv_cost(pred, dte)

    def _pto_adjust(self, pred, dte):
        """Apply predict-then-optimize adjustment using critical ratio."""
        strength = self.params.pto_strength

        def scale_factor(row_idx):
            cat = dte.loc[row_idx, "product_category"] if "product_category" in dte.columns else None
            period = int(dte.loc[row_idx, "lifecycle_year"]) if "lifecycle_year" in dte.columns else None
            try:
                co = self._get_cost(cat, "overage_cost", period)
                cu = self._get_cost(cat, "underage_cost", period)
                return 1.0 + (cu / (cu + co) - 0.5) * strength
            except (KeyError, TypeError):
                return 1.0

        adjusted = np.array([
            pred[i] * scale_factor(i) for i in range(len(dte))
        ])
        adjusted = np.maximum(adjusted, 0)
        return adjusted, self._total_nv_cost(adjusted, dte)

    def _compute_sigma(self, dtr, model, Xtr):
        """Compute residual standard deviation for uncertainty bounds."""
        Xtr_features, _ = self._build_features(dtr)
        preds = model.predict(Xtr_features)
        residuals = dtr["demand"].values - preds
        sigma = np.std(residuals)
        if sigma == 0:
            sigma = 10.0
        return sigma

    def _nv_cost(self, q, demand, co, cu):
        """Single-item newsvendor cost."""
        return co * max(q - demand, 0) + cu * max(demand - q, 0)

    def _total_nv_cost(self, preds, df):
        """Total newsvendor cost across all items."""
        total = 0.0
        for i in range(len(df)):
            cat = df.loc[i, "product_category"] if "product_category" in df.columns else None
            period = int(df.loc[i, "lifecycle_year"]) if "lifecycle_year" in df.columns else None
            try:
                co = self._get_cost(cat, "overage_cost", period)
                cu = self._get_cost(cat, "underage_cost", period)
            except (KeyError, TypeError):
                co, cu = 1.0, 1.0  # fallback to symmetric cost
            total += self._nv_cost(float(preds[i]), float(df.loc[i, "demand"]), co, cu)
        return total

    def _get_cost(self, cat, cost_name, period=None):
        """Look up cost from config.cost_structure."""
        for item in self.config.cost_structure:
            if item.name == cost_name and item.product_category == cat and item.period == period:
                return item.value
        for item in self.config.cost_structure:
            if item.name == cost_name and item.product_category == cat and item.period is None:
                return item.value
        for item in self.config.cost_structure:
            if item.name == cost_name and item.period is None and item.product_category is None:
                return item.value
        raise KeyError(f"{cost_name} not found for category={cat}, period={period}")

    def _get_moq(self):
        """Get minimum order quantity from config constraints."""
        for c in self.config.constraints:
            if "minimum" in c.name.lower() or "moq" in c.name.lower():
                for key in ["moq_per_size", "moq", "min_qty"]:
                    if key in c.parameters:
                        return int(c.parameters[key])
        return 0


def add_prediction(bundle, config, params: PredictionParams | None = None) -> PredictionResult:
    """Run prediction with optional tunable parameters.

    Args:
        bundle: DataBundle from DataModule.
        config: ProblemConfig for cost lookups.
        params: Optional hyperparameters. Uses defaults if None.

    Returns:
        PredictionResult with model comparison and best predictions.
    """
    return PredictionModule(config, params).run(bundle)
