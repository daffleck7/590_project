"""OptimizationResult Pydantic model — output contract from the optimizer module."""

from pydantic import BaseModel


class OptimizationResult(BaseModel):
    """Output produced by the optimizer (Step 4 — Fabian).

    Keys in order_quantities and shadow_prices use the SKU format:
    "{product_category}_{gender_age}_{size}"  e.g. "top_mens_youth_ym"
    """

    order_quantities: dict[str, float]
    objective_value: float
    total_spend: float
    shadow_prices: dict[str, float]
    solver_status: str
    selected_model: str
