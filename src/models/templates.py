"""
templates.py
============
Domain-specific default configs. Each template pre-fills a ProblemConfig
for a particular business problem so it can be used for testing, demos,
or as a fallback.
"""

from src.models.problem_config import (
    ProblemConfig,
    Objective,
    Constraint,
    CostItem,
    DataRequirements,
    UncertainParameter,
)


def build_cfa_default_config() -> ProblemConfig:
    """CFA youth soccer uniform ordering — newsvendor optimization.

    Returns a fully populated ProblemConfig for the CFA problem.
    Used as:
      1. Fallback if LLM output fails validation
      2. Ground truth for testing Intake Module quality
      3. Starting point for stakeholder modifications
      4. Dev shortcut for skip-intake testing
    """

    sizes = [
        "YXS", "YS", "YM", "YL", "YXL",   # Youth
        "AS", "AM", "AL", "AXL",            # Adult men
        "WXS", "WS", "WM", "WL", "WXL",    # Women
    ]
    categories = ["top", "bottom", "socks"]

    constraints = [
        Constraint(
            name="seasonal_budget",
            description="Total procurement spend per season cannot exceed league budget",
            type="hard",
            parameters={"budget_usd": 40000, "note": "Estimated — update with real budget"},
        ),
        Constraint(
            name="minimum_order_quantity",
            description="Supplier requires minimum 6 units per size per order",
            type="hard",
            parameters={"moq_per_size": 6},
        ),
        Constraint(
            name="non_negativity",
            description="Cannot order negative units",
            type="hard",
            parameters={},
        ),
        Constraint(
            name="carryover_inventory",
            description="Year 1 unsold stock carries to Year 2 at zero holding cost",
            type="hard",
            parameters={"holding_cost_per_unit": 0.0},
        ),
    ]

    cost_structure = [
        # --- Tops (jerseys) ---
        CostItem(name="unit_cost",     value=25.0, product_category="top",
                 description="Wholesale price per jersey"),
        CostItem(name="overage_cost",  value=5.0,  product_category="top", period=1,
                 description="Year 1: storage / opportunity cost of unsold jersey"),
        CostItem(name="overage_cost",  value=15.0, product_category="top", period=2,
                 description="Year 2: sold at loss — salvage discount + fees"),
        CostItem(name="underage_cost", value=12.0, product_category="top", period=1,
                 description="Year 1: rush order premium + extra shipping"),
        CostItem(name="underage_cost", value=60.0, product_category="top", period=2,
                 description="Year 2: discontinued model — secondhand premium or lost sale"),
        # --- Bottoms (shorts) ---
        CostItem(name="unit_cost",     value=16.0, product_category="bottom",
                 description="Wholesale shorts price"),
        CostItem(name="overage_cost",  value=3.0,  product_category="bottom", period=1,
                 description="Low holding cost — shorts not size-critical"),
        CostItem(name="overage_cost",  value=8.0,  product_category="bottom", period=2,
                 description="Salvage loss on discontinued shorts"),
        CostItem(name="underage_cost", value=8.0,  product_category="bottom", period=1,
                 description="Rush order premium for shorts"),
        CostItem(name="underage_cost", value=25.0, product_category="bottom", period=2,
                 description="Discontinued shorts — harder to source"),
        # --- Socks ---
        CostItem(name="unit_cost",     value=10.0, product_category="socks",
                 description="Socks wholesale"),
        CostItem(name="overage_cost",  value=1.0,  product_category="socks",
                 description="Socks easy to carry — minimal overage cost"),
        CostItem(name="underage_cost", value=5.0,  product_category="socks",
                 description="Generic socks — easier to source at premium"),
    ]

    uncertain_parameters = [
        UncertainParameter(
            name=f"demand_{size}_{cat}",
            description=f"Number of {size} {cat} units ordered in the upcoming season",
            data_column="quantity",
            unit="units",
        )
        for cat in categories
        for size in sizes
    ]

    return ProblemConfig(
        problem_description=(
            "CFA youth soccer league in Orange County processes 2,000-3,000 uniform "
            "orders per season across jerseys, shorts, and socks in 14 size codes. "
            "Orders must be placed before demand is realized. The current Adidas jersey "
            "model (Tiro 25) is in its final lifecycle year — unsold inventory cannot be "
            "reordered next season. Minimize total cost across overage, underage, and "
            "procurement subject to a seasonal budget and supplier minimum order quantities."
        ),
        problem_title="CFA Uniform Ordering Optimization",
        objective=Objective(
            direction="minimize",
            description=(
                "Minimize total expected newsvendor cost across all sizes and product "
                "categories — weighted by period-specific cost structure."
            ),
            metric_name="total_newsvendor_cost",
        ),
        constraints=constraints,
        uncertain_parameters=uncertain_parameters,
        cost_structure=cost_structure,
        solver_hint="newsvendor",
        item_categories=categories,
        item_sizes=sizes,
        time_periods=["fall", "winter", "spring"],
        custom_fields={"lifecycle_year": 2, "supplier": "Adidas"},
        data_requirements=DataRequirements(
            required_columns=[
                "order_id", "order_date", "season", "year",
                "uniform_set", "product_category", "gender_age",
                "size", "quantity", "unit_price",
            ],
            target_column="quantity",
            date_column="order_date",
            train_years=[2020, 2021, 2022, 2023, 2024],
            test_years=[2025, 2026],
            groupby_keys=["year", "season", "uniform_set", "product_category", "size"],
        ),
    )
