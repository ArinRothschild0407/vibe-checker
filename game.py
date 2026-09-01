"""Small interaction layer for the survey prediction experiment."""

from model import (
    discover_input_questions,
    evaluate_numeric_targets,
    load_survey,
    predict_targets,
    print_ranking,
)


# The CSV uses shortened column names. Keep those names for the model, while
# showing the original survey wording to the player.
QUESTION_PROMPTS = {
    "Countryside, outdoors": "Outdoor activities",
    "Life struggles": "I cry when I feel down or things don't go the right way.",
    "Classical music": "Classical",
    "Energy levels": "I am always full of life and energy.",
    "Spending on looks": "I spend a lot of money on my appearance.",
    "Romantic": "Romantic movies",
}

QUESTION_SCALES = {
    "Life struggles": "1 = strongly disagree, 5 = strongly agree",
    "Classical music": "1 = don't enjoy at all, 5 = enjoy very much",
    "Energy levels": "1 = strongly disagree, 5 = strongly agree",
    "Spending on looks": "1 = never, 5 = always",
    "Romantic": "1 = don't enjoy at all, 5 = enjoy very much",
}

# Testing many targets can produce small chance wins. This modest relative
# threshold excludes weak and negative results; repeated validation comes later.
MIN_IMPROVEMENT_PERCENT = 5.0
MAX_PREDICTIONS = 5


def ask_rating(question: str) -> int:
    """Ask until the player supplies an integer from 1 through 5."""

    while True:
        try:
            prompt = QUESTION_PROMPTS.get(question, question)
            scale = QUESTION_SCALES.get(question, "1 = low, 5 = high")
            answer = int(input(f"{prompt}\n  {scale}: "))
            if 1 <= answer <= 5:
                return answer
        except ValueError:
            pass
        print("Please enter a whole number from 1 to 5.")


def main() -> None:
    survey = load_survey()
    game_questions = discover_input_questions(survey, question_count=5)
    print("\n===============================")
    print("  SURVEY PREDICTION EXPERIMENT")
    print("===============================")
    print("\nThe training data selected these five broadly informative questions:")
    for question in game_questions:
        print(f"- {QUESTION_PROMPTS.get(question, question)}")
    print("\nAnswer each question from 1 to 5 using the scale shown.")
    print("The model will test what these five answers genuinely predict.\n")

    user_answers = {question: ask_rating(question) for question in game_questions}
    print("\nEvaluating unanswered numeric questions on held-out participants...")
    results = evaluate_numeric_targets(survey, game_questions)
    print("\nHighest-ranked held-out results:")
    print_ranking(results)

    selected = [
        result for result in results
        if result.improvement_percent >= MIN_IMPROVEMENT_PERCENT
    ][:MAX_PREDICTIONS]
    if not selected:
        print("\nNone cleared the minimum held-out improvement threshold.")
        print("For these five inputs, the honest conclusion is: no useful predictions.")
        return

    predictions = predict_targets(
        survey, user_answers, [result.target for result in selected]
    )
    print(
        f"\nPredictions for you (only baseline improvement "
        f">= {MIN_IMPROVEMENT_PERCENT:.1f}%):"
    )
    for result in selected:
        print(
            f"- {result.target}: {predictions[result.target]:.2f} "
            f"(held-out MAE {result.model_mae:.3f}, "
            f"baseline {result.baseline_mae:.3f})"
        )
    print("\nThese are statistical estimates, not facts about you.")


if __name__ == "__main__":
    main()
