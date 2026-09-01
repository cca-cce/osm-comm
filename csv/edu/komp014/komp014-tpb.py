#!/usr/bin/env python3
"""Generate a synthetic child-adapted Theory of Planned Behavior dataset.

The output contains three categorical demographic variables and three integer
Likert items (1 = strongly disagree, 5 = strongly agree) for each TPB
construct: Attitude (ATT), Subjective Norms (SN), Perceived Behavioral Control
(PBC), and Behavioral Intention (BI).

This dataset is synthetic and must not be represented as observed survey data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


N_ROWS = 1_000
START_SEED = 14_014
MAX_ATTEMPTS = 1_000
OUTPUT_PATH = Path(__file__).with_suffix(".tsv")

LIKERT_THRESHOLDS = np.array([-1.05, -0.35, 0.35, 1.05])
CONSTRUCT_ITEMS = {
    "ATT": [
        "att_fun_exciting",
        "att_valuable_use_of_time",
        "att_feel_good_about_self",
    ],
    "SN": [
        "sn_parent_guardian_approval",
        "sn_friend_approval",
        "sn_teacher_encouragement",
    ],
    "PBC": [
        "pbc_confident_can_learn",
        "pbc_enough_spare_time",
        "pbc_instrument_access",
    ],
    "BI": [
        "bi_plan_next_6_months",
        "bi_try_best_to_learn_soon",
        "bi_want_to_start_school_year",
    ],
}


def to_likert(values: np.ndarray) -> np.ndarray:
    """Discretize a continuous response into the five Likert categories."""
    return np.digitize(values, LIKERT_THRESHOLDS) + 1


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Calculate Cronbach's alpha for a multi-item scale."""
    values = items.to_numpy(dtype=float)
    n_items = values.shape[1]
    item_variance = values.var(axis=0, ddof=1).sum()
    total_variance = values.sum(axis=1).var(ddof=1)
    return float(n_items / (n_items - 1) * (1 - item_variance / total_variance))


def generate_tpb_items(seed: int, n_rows: int) -> pd.DataFrame:
    """Generate correlated latent TPB constructs and their observed items."""
    rng = np.random.default_rng(seed)

    # Predictors are distinct but modestly correlated, as expected in TPB data.
    predictor_correlation = np.full((3, 3), 0.15)
    np.fill_diagonal(predictor_correlation, 1.0)
    attitude, subjective_norms, perceived_control = rng.multivariate_normal(
        mean=np.zeros(3), cov=predictor_correlation, size=n_rows
    ).T

    # Weak positive structural paths leave most variation in intention unexplained.
    intention = (
        0.14 * attitude
        + 0.12 * subjective_norms
        + 0.13 * perceived_control
        + rng.normal(
            0,
            np.sqrt(1 - (0.14**2 + 0.12**2 + 0.13**2)),
            size=n_rows,
        )
    )

    latent_constructs = {
        "ATT": attitude,
        "SN": subjective_norms,
        "PBC": perceived_control,
        "BI": intention,
    }
    observed = {}
    for construct, latent_values in latent_constructs.items():
        for column in CONSTRUCT_ITEMS[construct]:
            continuous_item = 0.80 * latent_values + rng.normal(0, 0.60, n_rows)
            observed[column] = to_likert(continuous_item)

    return pd.DataFrame(observed)


def generate_demographics(seed: int, n_rows: int) -> pd.DataFrame:
    """Generate plausible categorical demographics for children aged 10–13."""
    rng = np.random.default_rng(seed + 1_000_000)
    grades = rng.choice(["Grade 4", "Grade 5", "Grade 6"], n_rows, p=[0.32, 0.34, 0.34])

    age_by_grade = {
        "Grade 4": (["10 years", "11 years", "12 years"], [0.72, 0.25, 0.03]),
        "Grade 5": (
            ["10 years", "11 years", "12 years", "13 years"],
            [0.08, 0.70, 0.20, 0.02],
        ),
        "Grade 6": (["11 years", "12 years", "13 years"], [0.08, 0.67, 0.25]),
    }
    ages = np.empty(n_rows, dtype=object)
    for grade, (choices, probabilities) in age_by_grade.items():
        selected = grades == grade
        ages[selected] = rng.choice(choices, selected.sum(), p=probabilities)

    gender = rng.choice(
        ["Girl", "Boy", "Non-binary", "Prefer not to say"],
        n_rows,
        p=[0.48, 0.48, 0.025, 0.015],
    )
    return pd.DataFrame(
        {"age_group": ages, "grade_level": grades, "gender": gender}
    )


def scale_indices(items: pd.DataFrame) -> pd.DataFrame:
    """Average the three items for each construct without adding output columns."""
    return pd.DataFrame(
        {
            construct: items[columns].mean(axis=1)
            for construct, columns in CONSTRUCT_ITEMS.items()
        }
    )


def diagnostics(
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], float]:
    """Return regression, reliability, VIF, and model fit diagnostics."""
    indices = scale_indices(items)
    standardized = (indices - indices.mean()) / indices.std(ddof=1)
    design = sm.add_constant(standardized[["ATT", "SN", "PBC"]])
    model = sm.OLS(
        standardized["BI"],
        design,
    ).fit()
    regression = pd.DataFrame(
        {
            "standardized_beta": model.params[["ATT", "SN", "PBC"]],
            "p_value": model.pvalues[["ATT", "SN", "PBC"]],
        }
    )
    alphas = {
        construct: cronbach_alpha(items[columns])
        for construct, columns in CONSTRUCT_ITEMS.items()
    }
    vifs = {
        predictor: float(variance_inflation_factor(design.to_numpy(), position))
        for position, predictor in enumerate(design.columns)
        if predictor != "const"
    }
    return regression, alphas, vifs, float(model.rsquared)


def acceptable(
    items: pd.DataFrame,
    regression: pd.DataFrame,
    alphas: dict[str, float],
    vifs: dict[str, float],
) -> bool:
    """Require reliable scales and weak, positive, significant TPB effects."""
    effects = regression["standardized_beta"]
    return bool(
        effects.between(0.05, 0.20, inclusive="both").all()
        and (regression["p_value"] < 0.05).all()
        and min(alphas.values()) >= 0.70
        and max(vifs.values()) < 5.0
        and not items.isna().any().any()
        and all(set(items[column].unique()) == {1, 2, 3, 4, 5} for column in items)
    )


def main() -> None:
    for seed in range(START_SEED, START_SEED + MAX_ATTEMPTS):
        items = generate_tpb_items(seed, N_ROWS)
        regression, alphas, vifs, r_squared = diagnostics(items)
        if acceptable(items, regression, alphas, vifs):
            demographics = generate_demographics(seed, N_ROWS)
            dataset = pd.concat([demographics, items], axis=1)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(OUTPUT_PATH, sep="\t", index=False)

            print(f"Created {OUTPUT_PATH} with {len(dataset):,} rows using seed {seed}.")
            print("\nStandardized regression predicting the BI index:")
            print(regression.to_string(float_format=lambda value: f"{value:.6f}"))
            print(f"\nModel R-squared: {r_squared:.6f}")
            print("Scale reliability (Cronbach's alpha):")
            for construct, alpha in alphas.items():
                print(f"  {construct}: {alpha:.6f}")
            print("Variance inflation factors (VIF):")
            for predictor, vif in vifs.items():
                print(f"  {predictor}: {vif:.6f}")
            return

    raise RuntimeError(
        f"No acceptable dataset found in {MAX_ATTEMPTS} deterministic attempts."
    )


if __name__ == "__main__":
    main()
