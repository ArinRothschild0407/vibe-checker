"""Terminal version of the survey prediction experiment.

This file handles questions and printed results. The reusable data splitting,
model testing, and prediction logic stays in model.py.
"""

from model import (
    find_predictive_random_group,
    load_survey,
    predict_experiments,
    question_category,
)


MIN_IMPROVEMENT_PERCENT = 5.0
QUALIFYING_IMPROVEMENT_PERCENT = 10.0
QUESTION_COUNT = 10
TARGET_COUNT = 5
MIN_CONFIRMED_TARGETS = 3
MAX_GROUP_ATTEMPTS = 10

# These cutoffs keep the program from presenting every model output as a real
# discovery. A group must find at least three targets with a 10% gain during
# selection. A target is finally shown only if it still beats the baseline by 5%
# on the final test participants. Finding nothing is a valid result.


def question_scale(question: str, survey_columns: list[str]) -> str:
    """Return the original questionnaire scale for any numeric 1-5 question."""

    position = survey_columns.index(question)
    positions = {name: index for index, name in enumerate(survey_columns)}
    if question == "I prefer.":
        return "1 = slow paced music, 5 = fast paced music"
    if positions["Dance, Disco, Funk"] <= position <= positions["Opera"]:
        return "1 = don't enjoy at all, 5 = enjoy very much"
    if positions["Horror movies"] <= position <= positions["Action movies"]:
        return "1 = don't enjoy at all, 5 = enjoy very much"
    if positions["History"] <= position <= positions["Pets"]:
        return "1 = not interested, 5 = very interested"
    if positions["Flying"] <= position <= positions["Public speaking"]:
        return "1 = not afraid at all, 5 = very afraid"
    if positions["I save all the money I can."] <= position <= positions[
        "I will hapilly pay more money for good, quality or healthy food."
    ]:
        return "1 = never, 5 = always"
    return "1 = strongly disagree, 5 = strongly agree"


def ask_rating(question: str, survey_columns: list[str]) -> int:
    """Ask until the player supplies an integer from 1 through 5."""

    while True:
        try:
            scale = question_scale(question, survey_columns)
            answer = int(input(f"{question}\n  {scale}: "))
            if 1 <= answer <= 5:
                return answer
        except ValueError:
            pass
        print("Please enter a whole number from 1 to 5.")


def main() -> None:
    survey = load_survey()
    survey_columns = list(survey.columns)
    print("\nSearching for a useful data-guided 10-question group...")
    # "Random" does not mean all questions have an equal chance. The ML code
    # first makes a pool with useful cross-category relationships, then samples
    # a varied group and tests it before asking the player anything.
    search = find_predictive_random_group(
        survey,
        question_count=QUESTION_COUNT,
        target_count=TARGET_COUNT,
        minimum_confirmed_targets=MIN_CONFIRMED_TARGETS,
        minimum_improvement_percent=QUALIFYING_IMPROVEMENT_PERCENT,
        max_attempts=MAX_GROUP_ATTEMPTS,
    )
    if search is None:
        print(
            f"No random group predicted at least {MIN_CONFIRMED_TARGETS} "
            f"targets after {MAX_GROUP_ATTEMPTS} attempts."
        )
        print("Try running the experiment again; no unsupported guesses were made.")
        return
    game_questions = list(search.input_questions)

    print("\n===============================")
    print("  SURVEY PREDICTION EXPERIMENT")
    print("===============================")
    print(f"\nFound a qualifying guided group after {search.attempts} attempt(s).")
    print(
        "These 10 questions predicted at least three targets during both "
        "screening and confirmation:"
    )
    for question in game_questions:
        print(f"- [{question_category(survey, question)}] {question}")

    print("\nAnswer the 10 questions using the scale shown.\n")
    user_answers = {
        question: ask_rating(question, survey_columns)
        for question in game_questions
    }

    # Final-test results were computed only after this group passed screening
    # and confirmation; they determine which predictions are safe to display.
    experiments = search.experiments
    confirmed = [
        experiment for experiment in experiments
        if experiment.validation_improvement_percent > 0
        and experiment.test_improvement_percent >= MIN_IMPROVEMENT_PERCENT
    ]

    if not confirmed:
        print(
            f"\nThese {QUESTION_COUNT} answers did not reliably predict "
            "another response."
        )
        print("That negative result is part of the experiment; no guess will be made.")
        return

    predictions = predict_experiments(survey, user_answers, confirmed)
    print(f"\nReliable predictions found: {len(confirmed)} of {TARGET_COUNT}")
    for experiment in confirmed:
        target_values = survey[experiment.target].dropna()
        if target_values.between(1, 5).all():
            suffix = " out of 5"
        elif experiment.target == "Age":
            suffix = " years"
        else:
            suffix = ""
        print(f"\n- {experiment.target}: {predictions[experiment.target]:.2f}{suffix}")
        print(f"  Target category: {question_category(survey, experiment.target)}")
        print(f"  Winning model: {experiment.model_name}")
        print(
            f"  Final MAE {experiment.test_model_mae:.3f} vs "
            f"baseline {experiment.test_baseline_mae:.3f} "
            f"({experiment.test_improvement_percent:.1f}% better)"
        )
        # These correlations explain which answers were most associated with the
        # target. They are clues, not proof of causation and not a replacement
        # for the held-out MAE comparison above.
        correlations = survey[list(experiment.input_questions)].corrwith(
            survey[experiment.target], method="spearman"
        ).dropna()
        strongest = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        ).head(3)
        print("  Strongest cross-category associations:")
        for question, correlation in strongest.items():
            direction = "positive" if correlation >= 0 else "negative"
            print(
                f"  - {question_category(survey, question)}: {question} "
                f"({direction}, correlation {correlation:+.2f})"
            )
    print("\nThese are statistical estimates, not facts about you.")


if __name__ == "__main__":
    main()
