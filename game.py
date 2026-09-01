"""Interactive game: random questions followed by evidence-based predictions."""

from model import (
    choose_random_questions,
    evaluate_question_group,
    load_survey,
    predict_targets,
)


MIN_IMPROVEMENT_PERCENT = 5.0
QUESTION_COUNT = 5
TARGET_COUNT = 5


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
    game_questions = choose_random_questions(
        survey, question_count=QUESTION_COUNT
    )

    print("\n===============================")
    print("  SURVEY PREDICTION EXPERIMENT")
    print("===============================")
    print("\nThis run randomly selected these five questions:")
    for question in game_questions:
        print(f"- {question}")

    print("\nAnswer the five questions using the scale shown.\n")
    user_answers = {
        question: ask_rating(question, survey_columns)
        for question in game_questions
    }

    print("\nTesting what this exact random group predicts...")
    experiments = evaluate_question_group(
        survey, game_questions, target_count=TARGET_COUNT
    )
    confirmed = [
        experiment for experiment in experiments
        if experiment.validation_improvement_percent > 0
        and experiment.test_improvement_percent >= MIN_IMPROVEMENT_PERCENT
    ]

    if not confirmed:
        print("\nThese five answers did not reliably predict another response.")
        print("That negative result is part of the experiment; no guess will be made.")
        return

    targets = [experiment.target for experiment in confirmed]
    predictions = predict_targets(survey, user_answers, targets)
    print(f"\nReliable predictions found: {len(confirmed)} of {TARGET_COUNT}")
    for experiment in confirmed:
        target_values = survey[experiment.target].dropna()
        suffix = " out of 5" if target_values.between(1, 5).all() else ""
        print(f"\n- {experiment.target}: {predictions[experiment.target]:.2f}{suffix}")
        print(
            f"  Final MAE {experiment.test_model_mae:.3f} vs "
            f"baseline {experiment.test_baseline_mae:.3f} "
            f"({experiment.test_improvement_percent:.1f}% better)"
        )
    print("\nThese are statistical estimates, not facts about you.")


if __name__ == "__main__":
    main()
