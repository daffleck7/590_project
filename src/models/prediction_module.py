"""
prediction_module.py
====================
Pure Python — no LLM involvement.

Input:  DataBundle (from data_module.py) + ProblemConfig
Output: PredictionResult

Three models compared on newsvendor cost:
  1. XGBoost    — gradient boosting, 300 trees
  2. LightGBM   — Microsoft's gradient boosting, faster
  3. Ridge      — linear baseline for comparison

Each model also has PTO-Adjusted variant (critical ratio shift).
Winner (lowest NV cost) passed to OptimizerModule.

Train: 2020-2023 | Test: 2024 (complete year)
Output includes: mean + sigma + P10/P50/P90 scenarios for SAA optimizer.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)


@dataclass
class PredictionResult:
    predicted_demand:   np.ndarray
    demand_df:          pd.DataFrame
    sigma:              np.ndarray
    P10:                np.ndarray
    P50:                np.ndarray
    P90:                np.ndarray
    best_model_name:    str
    all_costs:          dict
    baseline_cost:      float
    model_xgb:          object
    model_lgbm:         object
    feature_importance: pd.Series
    metadata:           dict = field(default_factory=dict)

    def summary(self) -> str:
        best_cost = self.all_costs[self.best_model_name]
        saving    = (1 - best_cost / self.baseline_cost) * 100
        return (
            f"PredictionResult | winner={self.best_model_name} | "
            f"NV cost=${best_cost:,.0f} | {saving:+.1f}% vs baseline | "
            f"groups={len(self.predicted_demand)}"
        )


class PredictionModule:

    def __init__(self, config):
        self.config = config

    def run(self, bundle) -> PredictionResult:
        train_years = [2020, 2021, 2022, 2023]
        test_years  = [2024]

        dtr = bundle.demand_pivot[bundle.demand_pivot["year"].isin(train_years)].copy().reset_index(drop=True)
        dte = bundle.demand_pivot[bundle.demand_pivot["year"].isin(test_years)].copy().reset_index(drop=True)

        Xtr, ytr = self._build_features(dtr)
        Xte, yte = self._build_features(dte)

        baseline_preds, baseline_cost = self._baseline(dtr, dte)

        model_xgb,      pred_xgb,      cost_xgb      = self._model_xgboost(Xtr, ytr, Xte, dte)
        model_lgbm,     pred_lgbm,     cost_lgbm     = self._model_lightgbm(Xtr, ytr, Xte, dte)
        model_histgb,   pred_histgb,   cost_histgb   = self._model_histgb(Xtr, ytr, Xte, dte)
        _,              pred_ridge,    cost_ridge    = self._model_ridge(Xtr, ytr, Xte, dte)

        pred_xgb_adj,    cost_xgb_adj    = self._pto_adjust(pred_xgb,    dte)
        pred_lgbm_adj,   cost_lgbm_adj   = self._pto_adjust(pred_lgbm,   dte)
        pred_histgb_adj, cost_histgb_adj = self._pto_adjust(pred_histgb, dte)
        pred_ridge_adj,  cost_ridge_adj  = self._pto_adjust(pred_ridge,  dte)

        sigma_cat = self._compute_sigma(dtr, model_xgb, Xtr)

        all_costs = {
            "Baseline":            baseline_cost,
            "XGBoost":             cost_xgb,
            "XGBoost+PTO":         cost_xgb_adj,
            "LightGBM":            cost_lgbm,
            "LightGBM+PTO":        cost_lgbm_adj,
            "HistGB":              cost_histgb,
            "HistGB+PTO":          cost_histgb_adj,
            "Ridge":               cost_ridge,
            "Ridge+PTO":           cost_ridge_adj,
        }

        model_preds = {
            "XGBoost":             pred_xgb,
            "XGBoost+PTO":         pred_xgb_adj,
            "LightGBM":            pred_lgbm,
            "LightGBM+PTO":        pred_lgbm_adj,
            "HistGB":              pred_histgb,
            "HistGB+PTO":          pred_histgb_adj,
            "Ridge":               pred_ridge,
            "Ridge+PTO":           pred_ridge_adj,
        }

        best_name  = min(model_preds, key=lambda x: all_costs[x])
        best_preds = np.maximum(model_preds[best_name], self._get_moq())

        print(f"[PredictionModule] Train: {len(dtr)} groups | Test: {len(dte)} groups")
        print(f"[PredictionModule] Results:")
        for name, cost in sorted(all_costs.items(), key=lambda x: x[1]):
            chg    = (cost / baseline_cost - 1) * 100
            winner = " ← WINNER" if name == best_name else ""
            print(f"  {name:<20}: ${cost:>9,.0f}  ({chg:+.1f}%){winner}")

        sigma_arr = np.array([sigma_cat.get(dte.loc[i,"product_category"], 10.0) for i in range(len(dte))])
        P10 = np.maximum(best_preds + sigma_arr * norm.ppf(0.10), 0)
        P50 = np.maximum(best_preds + sigma_arr * norm.ppf(0.50), 0)
        P90 = np.maximum(best_preds + sigma_arr * norm.ppf(0.90), 0)

        return PredictionResult(
            predicted_demand   = best_preds,
            demand_df          = dte,
            sigma              = sigma_arr,
            P10                = P10, P50 = P50, P90 = P90,
            best_model_name    = best_name,
            all_costs          = all_costs,
            baseline_cost      = baseline_cost,
            model_xgb          = model_xgb,
            model_lgbm         = model_lgbm,
            feature_importance = pd.Series(
                model_xgb.feature_importances_, index=Xtr.columns
            ).sort_values(ascending=False),
            metadata = {
                "train_years": train_years,
                "test_years":  test_years,
                "n_train":     len(dtr),
                "n_test":      len(dte),
                "sigma_by_cat": sigma_cat,
            }
        )

    def _build_features(self, df):
        df = df.copy(); df["year_idx"] = df["year"] - 2020
        df["is_youth"]  = (df["size_group"] == "youth").astype(int)
        df["is_women"]  = (df["size_group"] == "women").astype(int)
        df["is_top"]    = (df["product_category"] == "top").astype(int)
        df["is_bottom"] = (df["product_category"] == "bottom").astype(int)
        df["is_socks"]  = (df["product_category"] == "socks").astype(int)
        cols = ["size_rank","season_num","lifecycle_year","year_idx",
                "is_youth","is_women","is_top","is_bottom","is_socks"]
        return df[cols].reset_index(drop=True), df["demand"].reset_index(drop=True)

    def _baseline(self, dtr, dte):
        avg  = dtr.groupby(["product_category","size"])["demand"].mean().reset_index().rename(columns={"demand":"bq"})
        dte_b = dte.merge(avg, on=["product_category","size"], how="left")
        dte_b["bq"] = dte_b["bq"].fillna(dtr["demand"].mean())
        return dte_b["bq"].values, self._total_nv_cost(dte_b["bq"].values, dte_b.reset_index(drop=True))

    def _model_xgboost(self, Xtr, ytr, Xte, dte):
        m = XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
        m.fit(Xtr, ytr); p = np.maximum(m.predict(Xte), 0)
        return m, p, self._total_nv_cost(p, dte)

    def _model_lightgbm(self, Xtr, ytr, Xte, dte):
        m = LGBMRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42,verbose=-1)
        m.fit(Xtr, ytr); p = np.maximum(m.predict(Xte), 0)
        return m, p, self._total_nv_cost(p, dte)

    def _model_histgb(self, Xtr, ytr, Xte, dte):
        m = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
            l2_regularization=0.05,
        )
        m.fit(Xtr, ytr)
        p = np.maximum(m.predict(Xte), 0)
        return m, p, self._total_nv_cost(p, dte)

    def _model_ridge(self, Xtr, ytr, Xte, dte):
        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        m = Ridge(alpha=1.0); m.fit(Xtr_s, ytr); p = np.maximum(m.predict(Xte_s), 0)
        return m, p, self._total_nv_cost(p, dte)

    def _pto_adjust(self, pred, dte):
        def sf(cat, lyr):
            co = self._get_cost(cat,"overage_cost",lyr); cu = self._get_cost(cat,"underage_cost",lyr)
            return 1.0 + (cu/(cu+co) - 0.5) * 0.8
        p = np.array([pred[i]*sf(dte.loc[i,"product_category"],int(dte.loc[i,"lifecycle_year"])) for i in range(len(dte))])
        return np.maximum(p,0), self._total_nv_cost(np.maximum(p,0), dte)

    def _compute_sigma(self, dtr, model, Xtr):
        dtr_r = dtr.copy(); dtr_r["pred"] = model.predict(Xtr); dtr_r["resid"] = dtr_r["demand"] - dtr_r["pred"]
        return dtr_r.groupby("product_category")["resid"].std().to_dict()

    def _nv_cost(self, q, D, co, cu): return co*max(q-D,0)+cu*max(D-q,0)

    def _total_nv_cost(self, preds, df):
        total = 0.0
        for i in range(len(df)):
            co = self._get_cost(df.loc[i,"product_category"],"overage_cost",int(df.loc[i,"lifecycle_year"]))
            cu = self._get_cost(df.loc[i,"product_category"],"underage_cost",int(df.loc[i,"lifecycle_year"]))
            total += self._nv_cost(float(preds[i]),float(df.loc[i,"demand"]),co,cu)
        return total

    def _get_cost(self, cat, cost_name, lyr=None):
        for i in self.config.cost_structure:
            if i.name==cost_name and i.product_category==cat and i.lifecycle_year==lyr: return i.value
        for i in self.config.cost_structure:
            if i.name==cost_name and i.product_category==cat and i.lifecycle_year is None: return i.value
        for i in self.config.cost_structure:
            if i.name==cost_name: return i.value
        raise KeyError(cost_name)

    def _get_moq(self):
        for c in self.config.constraints:
            if c.name=="minimum_order_quantity": return int(c.parameters.get("moq_per_size",6))
        return 6


def add_prediction(bundle, config) -> PredictionResult:
    """
    Single function call for orchestrator.
    
    Usage:
        from prediction_module import add_prediction
        result = add_prediction(bundle, config)
        print(result.summary())
        # result.predicted_demand, result.P10, result.P90 → Step 4
    """
    return PredictionModule(config).run(bundle)
