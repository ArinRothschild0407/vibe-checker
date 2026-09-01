"""Small interaction layer for the survey prediction experiment."""

from model import evaluate_numeric_targets, load_survey, predict_targets, print_ranking


# This starting group is developer-chosen, not claimed to be optimal.
GAME_QUESTIONS = [
    "Rock",
    "Horror",
    "Reading",
    "Countryside, outdoors",
    "Spending on gadgets",
]

# Testing many targets can produce small chance wins. This modest relative
# threshold excludes weak and negative results; repeated validation comes later.
MIN_IMPROVEMENT_PERCENT = 5.0
MAX_PREDICTIONS = 5


def ask_rating(question: str) -> int:
    """Ask until the player supplies an integer from 1 through 5."""

    while True:
        try:
            answer = int(input(f"{question} (1-5): "))
            if 1 <= answer <= 5:
                return answer
        except ValueError:
            pass
        print("Please enter a whole number from 1 to 5.")


def main() -> None:
    survey = load_survey()
    print("\n===============================")
    print("  SURVEY PREDICTION EXPERIMENT")
    print("===============================")
    print("\nAnswer each question from 1 to 5.")
    print("The model will test what these five answers genuinely predict.\n")

    user_answers = {question: ask_rating(question) for question in GAME_QUESTIONS}
    print("\nEvaluating unanswered numeric questions on held-out participants...")
    results = evaluate_numeric_targets(survey, GAME_QUESTIONS)
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
