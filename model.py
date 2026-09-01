"""Reusable ML experiments for the Young People Survey.

An experiment only uses ``input_columns``. If the game asks five questions,
evaluation uses those same five answers, never the rest of a survey row.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class TargetResult:
    """Held-out performance for one possible prediction target."""

    target: str
    model_mae: float
    baseline_mae: float
    improvement: float
    improvement_percent: float
    test_samples: int


def load_survey(path: str = "data/responses.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _validate_inputs(df: pd.DataFrame, input_columns: Sequence[str]) -> list[str]:
    columns = list(input_columns)
    if not columns:
        raise ValueError("At least one input question is required.")
    if len(columns) != len(set(columns)):
        raise ValueError("Input questions must not contain duplicates.")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown input question(s): {missing}")
    non_numeric = [
        column for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(
            "This first version supports numeric input questions only. "
            f"Non-numeric question(s): {non_numeric}"
        )
    return columns


def make_regression_pipeline(random_state: int = 42) -> Pipeline:
    """Build preprocessing and model as one leakage-safe pipeline."""

    # Pipeline.fit learns imputation medians from training participants only.
    # Random forests do not require feature scaling.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]
    )


def discover_input_questions(
    df: pd.DataFrame,
    *,
    question_count: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[str]:
    """Choose broadly informative 1-5 questions without looking at test people.

    We greedily choose questions that give the best correlation coverage across
    the other 1-5 survey responses. After one question is selected, later choices
    are rewarded for explaining responses the existing questions do not already
    cover. This is a simple, understandable first question-selection strategy.

    Correlation only selects the inputs. Random Forest performance on untouched
    test participants still decides whether those inputs truly predict a target.
    """

    if question_count < 1:
        raise ValueError("question_count must be at least 1.")

    train_indices, _ = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state
    )
    discovery_rows = df.iloc[train_indices]

    # The game currently accepts 1-5 ratings, so age, height, weight, and other
    # differently-scaled numeric columns are not eligible input questions.
    candidates = [
        column
        for column in discovery_rows.select_dtypes(include="number").columns
        if discovery_rows[column].dropna().between(1, 5).all()
    ]
    if question_count > len(candidates):
        raise ValueError("Not enough 1-5 numeric questions in the dataset.")

    # Spearman correlation works with ordered ratings and does not assume that
    # the relationship must be perfectly linear.
    correlations = discovery_rows[candidates].corr(method="spearman").abs()
    diagonal = np.eye(len(correlations), dtype=bool)
    correlations = correlations.mask(diagonal, 0.0).fillna(0.0)

    selected: list[str] = []
    current_coverage = pd.Series(0.0, index=candidates)
    for _ in range(question_count):
        best_question = max(
            (question for question in candidates if question not in selected),
            key=lambda question: np.maximum(
                current_coverage, correlations[question]
            ).mean(),
        )
        selected.append(best_question)
        current_coverage = np.maximum(
            current_coverage, correlations[best_question]
        )

    return selected


def evaluate_numeric_targets(
    df: pd.DataFrame,
    input_columns: Sequence[str],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[TargetResult]:
    """Rank unanswered numeric targets by held-out gain over baseline.

    One participant split is reused for every target. Missing target answers are
    omitted for that target only. The baseline predicts the training median,
    which is the appropriate constant prediction when measuring MAE.
    """

    inputs = _validate_inputs(df, input_columns)
    numeric_targets = [
        column for column in df.select_dtypes(include="number").columns
        if column not in inputs
    ]
    train_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state
    )
    results: list[TargetResult] = []

    for target in numeric_targets:
        train_rows = df.iloc[train_indices]
        test_rows = df.iloc[test_indices]
        # A missing target cannot teach the model or be used to score it.
        train_rows = train_rows[train_rows[target].notna()]
        test_rows = test_rows[test_rows[target].notna()]
        if train_rows.empty or test_rows.empty:
            continue

        model = make_regression_pipeline(random_state)
        model.fit(train_rows[inputs], train_rows[target])
        model_mae = mean_absolute_error(
            test_rows[target], model.predict(test_rows[inputs])
        )

        baseline_value = train_rows[target].median()
        baseline_mae = mean_absolute_error(
            test_rows[target], np.full(len(test_rows), baseline_value)
        )
        improvement = baseline_mae - model_mae
        improvement_percent = (
            100.0 * improvement / baseline_mae if baseline_mae > 0 else 0.0
        )
        results.append(TargetResult(
            target=target,
            model_mae=model_mae,
            baseline_mae=baseline_mae,
            improvement=improvement,
            improvement_percent=improvement_percent,
            test_samples=len(test_rows),
        ))

    # Percentage gain makes targets with different scales comparable. For
    # example, a one-kilogram gain is not equivalent to a one-point rating gain.
    return sorted(
        results, key=lambda result: result.improvement_percent, reverse=True
    )


def predict_targets(
    df: pd.DataFrame,
    user_answers: Mapping[str, float],
    targets: Sequence[str],
    *,
    random_state: int = 42,
) -> dict[str, float]:
    """Refit selected targets on all participants and predict the user.

    Target selection happens separately on held-out data. Only after that honest
    test do we use all available rows to fit the final prediction model.
    """

    inputs = _validate_inputs(df, list(user_answers))
    user_row = pd.DataFrame([[user_answers[c] for c in inputs]], columns=inputs)
    predictions: dict[str, float] = {}

    for target in targets:
        if target in inputs or target not in df.columns:
            raise ValueError(f"Invalid prediction target: {target}")
        if not pd.api.types.is_numeric_dtype(df[target]):
            raise ValueError(f"Target is not numeric: {target}")
        training_rows = df[df[target].notna()]
        model = make_regression_pipeline(random_state)
        model.fit(training_rows[inputs], training_rows[target])
        predictions[target] = float(model.predict(user_row)[0])

    return predictions


def print_ranking(results: Sequence[TargetResult], limit: int = 10) -> None:
    """Print a compact experiment table."""

    print(f"{'Target':30} {'Model MAE':>10} {'Baseline':>10} {'Gain':>9} {'Gain %':>8}")
    print("-" * 72)
    for result in results[:limit]:
        print(
            f"{result.target[:30]:30} {result.model_mae:10.3f} "
            f"{result.baseline_mae:10.3f} {result.improvement:9.3f} "
            f"{result.improvement_percent:7.1f}%"
        )


if __name__ == "__main__":
    survey = load_survey()
    discovered_inputs = discover_input_questions(survey)
    ranking = evaluate_numeric_targets(survey, discovered_inputs)
    print("\nFive questions selected from training data:")
    print(", ".join(discovered_inputs))
    print_ranking(ranking)
