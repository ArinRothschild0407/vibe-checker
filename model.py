"""Reusable ML experiments for the Young People Survey.

An experiment only uses ``input_columns``. If the game asks five questions,
evaluation uses those same five answers, never the rest of a survey row.
"""

from dataclasses import dataclass
from pathlib import Path
from random import SystemRandom
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TargetResult:
    """Held-out performance for one possible prediction target."""

    target: str
    model_mae: float
    baseline_mae: float
    improvement: float
    improvement_percent: float
    test_samples: int


@dataclass(frozen=True)
class PredictionExperiment:
    """A five-question/target relationship tested on untouched participants."""

    target: str
    input_questions: tuple[str, ...]
    validation_improvement_percent: float
    test_model_mae: float
    test_baseline_mae: float
    test_improvement_percent: float
    test_samples: int
    model_name: str = "Random Forest"


@dataclass(frozen=True)
class QuestionGroupSearch:
    """A random question group that passed pre-game confirmation."""

    input_questions: tuple[str, ...]
    experiments: tuple[PredictionExperiment, ...]
    attempts: int
    confirmed_before_final_test: int


def load_survey(path: str = "data/responses.csv") -> pd.DataFrame:
    """Load responses and replace every shortened name with original wording."""

    response_path = Path(path)
    survey = pd.read_csv(response_path)
    column_guide = pd.read_csv(response_path.with_name("columns.csv"))
    short_to_original = dict(zip(column_guide["short"], column_guide["original"]))

    missing_metadata = set(survey.columns) - set(short_to_original)
    if missing_metadata:
        raise ValueError(
            "columns.csv has no original wording for: "
            f"{sorted(missing_metadata)}"
        )
    return survey.rename(columns=short_to_original)


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


def make_regression_pipeline(
    random_state: int = 42,
    *,
    n_estimators: int = 150,
) -> Pipeline:
    """Build preprocessing and model as one leakage-safe pipeline."""

    # Pipeline.fit learns imputation medians from training participants only.
    # Random forests do not require feature scaling.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=3,
                random_state=random_state,
                # This dataset is small. One worker avoids noisy joblib warnings
                # seen with this project's Python/scikit-learn combination.
                n_jobs=1,
            )),
        ]
    )


def make_candidate_models(
    random_state: int = 42,
    *,
    fast_screening: bool = False,
) -> dict[str, Pipeline]:
    """Build leakage-safe candidate models for the model tournament."""

    tree_count = 50 if fast_screening else 150
    return {
        "Random Forest": make_regression_pipeline(
            random_state, n_estimators=tree_count
        ),
        "Extra Trees": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=tree_count,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=1,
            )),
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                n_estimators=60 if fast_screening else 120,
                loss="huber",
                min_samples_leaf=3,
                random_state=random_state,
            )),
        ]),
        "KNN": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=25, weights="distance")),
        ]),
        "Ridge": Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
    }


def make_model_by_name(
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    """Rebuild the selected algorithm for confirmation or prediction."""

    models = make_candidate_models(random_state, fast_screening=False)
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]


def _mae_comparison(
    model: Pipeline,
    train_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    inputs: Sequence[str],
    target: str,
) -> tuple[float, float, float]:
    """Fit one experiment and compare it with a training-median baseline."""

    train_rows = train_rows[train_rows[target].notna()]
    evaluation_rows = evaluation_rows[evaluation_rows[target].notna()]
    model.fit(train_rows[list(inputs)], train_rows[target])
    model_mae = mean_absolute_error(
        evaluation_rows[target], model.predict(evaluation_rows[list(inputs)])
    )
    baseline_value = train_rows[target].median()
    baseline_mae = mean_absolute_error(
        evaluation_rows[target], np.full(len(evaluation_rows), baseline_value)
    )
    improvement_percent = (
        100.0 * (baseline_mae - model_mae) / baseline_mae
        if baseline_mae > 0 else 0.0
    )
    return model_mae, baseline_mae, improvement_percent


def discover_best_prediction_experiment(
    df: pd.DataFrame,
    *,
    question_count: int = 5,
    random_state: int = 42,
) -> PredictionExperiment:
    """Discover which five questions best predict an unanswered response.

    Training selects target-specific inputs, validation ranks all targets, and
    the single best discovery is then confirmed on untouched test participants.
    """

    all_indices = np.arange(len(df))
    train_validation_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=random_state
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices, test_size=0.25, random_state=random_state
    )
    train_rows = df.iloc[train_indices]
    validation_rows = df.iloc[validation_indices]
    test_rows = df.iloc[test_indices]

    numeric_targets = list(df.select_dtypes(include="number").columns)
    # Inputs must be questions a player can answer on a 1-5 scale.
    candidate_inputs = [
        column
        for column in train_rows.select_dtypes(include="number").columns
        if train_rows[column].dropna().between(1, 5).all()
    ]
    discoveries: list[tuple[float, str, tuple[str, ...]]] = []

    for target in numeric_targets:
        possible_inputs = [question for question in candidate_inputs if question != target]
        # Correlation screening sees training participants only.
        correlations = train_rows[possible_inputs].corrwith(
            train_rows[target], method="spearman"
        ).abs().fillna(0.0)
        inputs = tuple(correlations.nlargest(question_count).index)
        if len(inputs) < question_count:
            continue
        _, _, validation_gain = _mae_comparison(
            make_regression_pipeline(random_state),
            train_rows,
            validation_rows,
            inputs,
            target,
        )
        discoveries.append((validation_gain, target, inputs))

    if not discoveries:
        raise ValueError("No numeric prediction experiments could be evaluated.")

    validation_gain, target, inputs = max(discoveries, key=lambda item: item[0])
    # The final test group influenced neither question nor target selection.
    development_rows = df.iloc[train_validation_indices]
    test_model_mae, test_baseline_mae, test_gain = _mae_comparison(
        make_regression_pipeline(random_state),
        development_rows,
        test_rows,
        inputs,
        target,
    )
    return PredictionExperiment(
        target=target,
        input_questions=inputs,
        validation_improvement_percent=validation_gain,
        test_model_mae=test_model_mae,
        test_baseline_mae=test_baseline_mae,
        test_improvement_percent=test_gain,
        test_samples=int(test_rows[target].notna().sum()),
    )


def choose_random_questions(
    df: pd.DataFrame,
    *,
    question_count: int = 5,
) -> list[str]:
    """Choose random questions whose recorded answers use a 1-5 scale."""

    candidates = [
        column
        for column in df.select_dtypes(include="number").columns
        if df[column].dropna().between(1, 5).all()
    ]
    if question_count > len(candidates):
        raise ValueError("Not enough 1-5 questions in the dataset.")
    # SystemRandom intentionally gives the game a fresh group on each run.
    return SystemRandom().sample(candidates, question_count)


def evaluate_question_group(
    df: pd.DataFrame,
    input_questions: Sequence[str],
    *,
    target_count: int = 5,
    random_state: int = 42,
) -> list[PredictionExperiment]:
    """Find and finally confirm what one exact question group can predict.

    Models are ranked on validation people. Only the top targets are then scored
    on untouched test people, so the game never evaluates using hidden answers.
    """

    inputs = tuple(_validate_inputs(df, input_questions))
    all_indices = np.arange(len(df))
    train_validation_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=random_state
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices, test_size=0.25, random_state=random_state
    )
    train_rows = df.iloc[train_indices]
    validation_rows = df.iloc[validation_indices]
    test_rows = df.iloc[test_indices]

    validation_results: list[tuple[float, str]] = []
    targets = [
        column for column in df.select_dtypes(include="number").columns
        if column not in inputs
    ]
    for target in targets:
        _, _, validation_gain = _mae_comparison(
            make_regression_pipeline(random_state),
            train_rows,
            validation_rows,
            inputs,
            target,
        )
        validation_results.append((validation_gain, target))

    # Validation chooses at most five targets before final test is examined.
    finalists = sorted(validation_results, reverse=True)[:target_count]
    development_rows = df.iloc[train_validation_indices]
    experiments: list[PredictionExperiment] = []
    for validation_gain, target in finalists:
        test_model_mae, test_baseline_mae, test_gain = _mae_comparison(
            make_regression_pipeline(random_state),
            development_rows,
            test_rows,
            inputs,
            target,
        )
        experiments.append(PredictionExperiment(
            target=target,
            input_questions=inputs,
            validation_improvement_percent=validation_gain,
            test_model_mae=test_model_mae,
            test_baseline_mae=test_baseline_mae,
            test_improvement_percent=test_gain,
            test_samples=int(test_rows[target].notna().sum()),
        ))
    return experiments


def find_predictive_random_group(
    df: pd.DataFrame,
    *,
    question_count: int = 10,
    target_count: int = 5,
    minimum_confirmed_targets: int = 3,
    minimum_improvement_percent: float = 5.0,
    max_attempts: int = 10,
    random_state: int = 42,
) -> QuestionGroupSearch | None:
    """Search random groups before the player is asked any questions.

    Data roles are kept separate: training fits models, screening ranks targets,
    confirmation decides whether a random group is worth asking, and final test
    estimates the selected group's performance on untouched participants.
    """

    all_indices = np.arange(len(df))
    development_indices, final_test_indices = train_test_split(
        all_indices, test_size=0.2, random_state=random_state
    )
    train_indices, selection_indices = train_test_split(
        development_indices, test_size=0.375, random_state=random_state
    )
    screening_indices, confirmation_indices = train_test_split(
        selection_indices, test_size=0.5, random_state=random_state
    )
    train_rows = df.iloc[train_indices]
    screening_rows = df.iloc[screening_indices]
    confirmation_rows = df.iloc[confirmation_indices]
    final_test_rows = df.iloc[final_test_indices]

    for attempt in range(1, max_attempts + 1):
        inputs = tuple(choose_random_questions(df, question_count=question_count))
        targets = [
            column for column in df.select_dtypes(include="number").columns
            if column not in inputs
        ]
        screening_results: list[tuple[float, str, str]] = []
        for target in targets:
            model_results: list[tuple[float, str]] = []
            for model_name, model in make_candidate_models(
                random_state, fast_screening=True
            ).items():
                _, _, gain = _mae_comparison(
                    model, train_rows, screening_rows, inputs, target
                )
                model_results.append((gain, model_name))
            best_gain, best_model_name = max(model_results)
            screening_results.append((best_gain, target, best_model_name))

        # Only these targets move forward; confirmation did not rank them.
        finalists = sorted(
            screening_results, key=lambda item: item[0], reverse=True
        )[:target_count]
        training_and_screening = df.iloc[
            np.concatenate([train_indices, screening_indices])
        ]
        confirmed: list[tuple[float, float, str, str]] = []
        for screening_gain, target, model_name in finalists:
            _, _, confirmation_gain = _mae_comparison(
                make_model_by_name(model_name, random_state),
                training_and_screening,
                confirmation_rows,
                inputs,
                target,
            )
            if (
                screening_gain >= minimum_improvement_percent
                and confirmation_gain >= minimum_improvement_percent
            ):
                confirmed.append(
                    (screening_gain, confirmation_gain, target, model_name)
                )

        if len(confirmed) < minimum_confirmed_targets:
            continue

        # The group qualified without looking at final-test answers.
        final_training_rows = df.iloc[development_indices]
        experiments: list[PredictionExperiment] = []
        for screening_gain, _, target, model_name in confirmed:
            model_mae, baseline_mae, final_gain = _mae_comparison(
                make_model_by_name(model_name, random_state),
                final_training_rows,
                final_test_rows,
                inputs,
                target,
            )
            experiments.append(PredictionExperiment(
                target=target,
                input_questions=inputs,
                validation_improvement_percent=screening_gain,
                test_model_mae=model_mae,
                test_baseline_mae=baseline_mae,
                test_improvement_percent=final_gain,
                test_samples=int(final_test_rows[target].notna().sum()),
                model_name=model_name,
            ))
        return QuestionGroupSearch(
            input_questions=inputs,
            experiments=tuple(experiments),
            attempts=attempt,
            confirmed_before_final_test=len(confirmed),
        )

    return None


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


def predict_experiments(
    df: pd.DataFrame,
    user_answers: Mapping[str, float],
    experiments: Sequence[PredictionExperiment],
    *,
    random_state: int = 42,
) -> dict[str, float]:
    """Predict with the same algorithm that won each target's tournament."""

    inputs = _validate_inputs(df, list(user_answers))
    user_row = pd.DataFrame([[user_answers[c] for c in inputs]], columns=inputs)
    predictions: dict[str, float] = {}
    for experiment in experiments:
        training_rows = df[df[experiment.target].notna()]
        model = make_model_by_name(experiment.model_name, random_state)
        model.fit(training_rows[inputs], training_rows[experiment.target])
        predictions[experiment.target] = float(model.predict(user_row)[0])
    return predictions


def print_ranking(results: Sequence[TargetResult], limit: int = 10) -> None:
    """Print results without shortening the original survey questions."""

    for rank, result in enumerate(results[:limit], start=1):
        print(f"{rank}. {result.target}")
        print(
            f"   Model MAE: {result.model_mae:.3f} | "
            f"Baseline MAE: {result.baseline_mae:.3f} | "
            f"Improvement: {result.improvement_percent:.1f}%"
        )


if __name__ == "__main__":
    survey = load_survey()
    search = find_predictive_random_group(
        survey, question_count=10, minimum_improvement_percent=10.0
    )
    if search is None:
        raise SystemExit("No qualifying random group found.")
    print(f"\nQualifying group found after {search.attempts} attempt(s):")
    for question in search.input_questions:
        print(f"- {question}")
    print("\nConfirmed targets checked on final test:")
    for experiment in search.experiments:
        print(f"- {experiment.target}")
        print(
            f"  Validation gain: {experiment.validation_improvement_percent:.1f}% | "
            f"Final gain: {experiment.test_improvement_percent:.1f}%"
        )
