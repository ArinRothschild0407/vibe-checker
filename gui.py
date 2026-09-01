"""Native desktop interface for the survey prediction experiment."""

from threading import Thread
import tkinter as tk
from tkinter import messagebox, ttk

from model import (
    find_predictive_random_group,
    load_survey,
    predict_experiments,
    question_category,
)


QUESTION_COUNT = 10
TARGET_COUNT = 5
MIN_CONFIRMED_TARGETS = 3
QUALIFYING_IMPROVEMENT_PERCENT = 10.0
FINAL_IMPROVEMENT_PERCENT = 5.0
MAX_GROUP_ATTEMPTS = 10

BACKGROUND = "#f4f1ea"
CARD = "#ffffff"
INK = "#172126"
MUTED = "#617078"
ACCENT = "#27756a"
ACCENT_DARK = "#17564f"
SOFT_ACCENT = "#dcece8"
WARNING = "#a65b24"


def question_scale(question: str, survey_columns: list[str]) -> str:
    """Return the scale used by the original questionnaire."""

    position = survey_columns.index(question)
    positions = {name: index for index, name in enumerate(survey_columns)}
    if question == "I prefer.":
        return "1 = slow paced music    •    5 = fast paced music"
    if positions["Dance, Disco, Funk"] <= position <= positions["Opera"]:
        return "1 = don't enjoy at all    •    5 = enjoy very much"
    if positions["Horror movies"] <= position <= positions["Action movies"]:
        return "1 = don't enjoy at all    •    5 = enjoy very much"
    if positions["History"] <= position <= positions["Pets"]:
        return "1 = not interested    •    5 = very interested"
    if positions["Flying"] <= position <= positions["Public speaking"]:
        return "1 = not afraid at all    •    5 = very afraid"
    return "1 = strongly disagree    •    5 = strongly agree"


class SurveyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Survey Signals")
        self.root.geometry("980x780")
        self.root.minsize(760, 620)
        self.root.configure(bg=BACKGROUND)

        self.survey = load_survey()
        self.survey_columns = list(self.survey.columns)
        self.search = None
        self.answer_variables: dict[str, tk.IntVar] = {}
        self.active_canvas: tk.Canvas | None = None

        self._configure_styles()
        self._build_shell()
        self.root.bind_all("<MouseWheel>", self._scroll_wheel)
        self.root.bind_all("<Button-4>", self._scroll_linux)
        self.root.bind_all("<Button-5>", self._scroll_linux)
        self.root.after(100, self._bring_to_front)
        self.start_experiment()

    def _bring_to_front(self) -> None:
        """Make a newly launched GUI visible instead of hiding behind the terminal."""

        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()
        self.root.after(800, lambda: self.root.attributes("-topmost", False))

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            borderwidth=0,
            padding=(20, 12),
            font=("Helvetica", 12, "bold"),
        )
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
        style.configure(
            "Secondary.TButton",
            background=SOFT_ACCENT,
            foreground=ACCENT_DARK,
            borderwidth=0,
            padding=(18, 10),
            font=("Helvetica", 11, "bold"),
        )
        style.configure("Survey.TRadiobutton", background=CARD, font=("Helvetica", 11))

    def _build_shell(self) -> None:
        header = tk.Frame(self.root, bg=INK, padx=34, pady=22)
        header.pack(fill="x")
        tk.Label(
            header,
            text="SURVEY SIGNALS",
            bg=INK,
            fg="#9bd4ca",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="What can ten answers really predict?",
            bg=INK,
            fg="white",
            font=("Helvetica", 24, "bold"),
        ).pack(anchor="w", pady=(4, 0))

        self.content = tk.Frame(self.root, bg=BACKGROUND)
        self.content.pack(fill="both", expand=True)

    def _clear_content(self) -> None:
        self.active_canvas = None
        for child in self.content.winfo_children():
            child.destroy()

    def _scroll_wheel(self, event) -> str | None:
        if self.active_canvas is not None and event.delta:
            direction = -1 if event.delta > 0 else 1
            self.active_canvas.yview_scroll(direction * 3, "units")
            return "break"
        return None

    def _scroll_linux(self, event) -> str | None:
        if self.active_canvas is not None:
            self.active_canvas.yview_scroll(-3 if event.num == 4 else 3, "units")
            return "break"
        return None

    def start_experiment(self) -> None:
        self._clear_content()
        self.search = None
        self.answer_variables = {}

        panel = tk.Frame(self.content, bg=BACKGROUND, padx=60, pady=80)
        panel.pack(fill="both", expand=True)
        tk.Label(
            panel,
            text="Finding an informative question group…",
            bg=BACKGROUND,
            fg=INK,
            font=("Helvetica", 22, "bold"),
        ).pack()
        tk.Label(
            panel,
            text=(
                "Testing data-guided groups across music, movies, interests, "
                "fears, personality, and spending."
            ),
            bg=BACKGROUND,
            fg=MUTED,
            font=("Helvetica", 12),
            wraplength=620,
            justify="center",
        ).pack(pady=(14, 26))
        progress = ttk.Progressbar(panel, mode="indeterminate", length=360)
        progress.pack()
        progress.start(12)
        self.status_label = tk.Label(
            panel,
            text="Comparing five ML algorithms on held-out participants…",
            bg=BACKGROUND,
            fg=ACCENT,
            font=("Helvetica", 11),
        )
        self.status_label.pack(pady=16)

        Thread(target=self._search_worker, daemon=True).start()

    def _search_worker(self) -> None:
        try:
            result = find_predictive_random_group(
                self.survey,
                question_count=QUESTION_COUNT,
                target_count=TARGET_COUNT,
                minimum_confirmed_targets=MIN_CONFIRMED_TARGETS,
                minimum_improvement_percent=QUALIFYING_IMPROVEMENT_PERCENT,
                max_attempts=MAX_GROUP_ATTEMPTS,
            )
            self.root.after(0, lambda: self._search_finished(result))
        except Exception as error:  # Keep worker errors visible in the GUI.
            self.root.after(0, lambda: self._show_error(str(error)))

    def _search_finished(self, result) -> None:
        if result is None:
            self._clear_content()
            panel = tk.Frame(self.content, bg=BACKGROUND, padx=60, pady=80)
            panel.pack(fill="both", expand=True)
            tk.Label(
                panel,
                text="No qualifying group found",
                bg=BACKGROUND,
                fg=INK,
                font=("Helvetica", 22, "bold"),
            ).pack()
            tk.Label(
                panel,
                text=(
                    "No group predicted at least three targets strongly enough. "
                    "That is a valid negative result—try another search."
                ),
                bg=BACKGROUND,
                fg=MUTED,
                font=("Helvetica", 12),
                wraplength=580,
                justify="center",
            ).pack(pady=18)
            ttk.Button(
                panel,
                text="Search again",
                style="Accent.TButton",
                command=self.start_experiment,
            ).pack(pady=12)
            return

        self.search = result
        self._show_questions()

    def _scrolling_page(self) -> tuple[tk.Canvas, tk.Frame]:
        canvas = tk.Canvas(self.content, bg=BACKGROUND, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content, orient="vertical", command=canvas.yview)
        body = tk.Frame(canvas, bg=BACKGROUND)
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(window, width=event.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        self.active_canvas = canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        return canvas, body

    def _show_questions(self) -> None:
        self._clear_content()
        canvas, body = self._scrolling_page()
        intro = tk.Frame(body, bg=BACKGROUND, padx=48, pady=30)
        intro.pack(fill="x")
        tk.Label(
            intro,
            text="A useful, cross-category group was found",
            bg=BACKGROUND,
            fg=INK,
            font=("Helvetica", 20, "bold"),
        ).pack(anchor="w")
        tk.Label(
            intro,
            text=(
                f"Qualified after {self.search.attempts} search attempt(s). "
                "Answer every question to reveal only predictions that survived final testing."
            ),
            bg=BACKGROUND,
            fg=MUTED,
            font=("Helvetica", 11),
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        for number, question in enumerate(self.search.input_questions, start=1):
            card = tk.Frame(body, bg=CARD, padx=24, pady=18, highlightthickness=1)
            card.configure(highlightbackground="#d9dedc")
            card.pack(fill="x", padx=48, pady=7)
            category = question_category(self.survey, question)
            tk.Label(
                card,
                text=f"{number:02d}  ·  {category.upper()}",
                bg=CARD,
                fg=ACCENT,
                font=("Helvetica", 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                card,
                text=question,
                bg=CARD,
                fg=INK,
                font=("Helvetica", 13, "bold"),
                wraplength=800,
                justify="left",
            ).pack(anchor="w", pady=(5, 4))
            tk.Label(
                card,
                text=question_scale(question, self.survey_columns),
                bg=CARD,
                fg=MUTED,
                font=("Helvetica", 10),
            ).pack(anchor="w", pady=(0, 10))

            variable = tk.IntVar(value=0)
            self.answer_variables[question] = variable
            choices = tk.Frame(card, bg=CARD)
            choices.pack(anchor="w")
            for value in range(1, 6):
                ttk.Radiobutton(
                    choices,
                    text=str(value),
                    value=value,
                    variable=variable,
                    style="Survey.TRadiobutton",
                ).pack(side="left", padx=(0, 22))

        action = tk.Frame(body, bg=BACKGROUND, padx=48, pady=26)
        action.pack(fill="x")
        ttk.Button(
            action,
            text="Analyze my answers",
            style="Accent.TButton",
            command=lambda: self._begin_prediction(canvas),
        ).pack(side="right")

    def _begin_prediction(self, canvas: tk.Canvas) -> None:
        unanswered = [
            question for question, variable in self.answer_variables.items()
            if variable.get() == 0
        ]
        if unanswered:
            messagebox.showinfo(
                "Complete all questions",
                f"Please answer all {QUESTION_COUNT} questions before continuing.",
            )
            return
        answers = {
            question: variable.get()
            for question, variable in self.answer_variables.items()
        }
        canvas.yview_moveto(0)
        self._clear_content()
        panel = tk.Frame(self.content, bg=BACKGROUND, padx=60, pady=90)
        panel.pack(fill="both", expand=True)
        tk.Label(
            panel,
            text="Building your predictions…",
            bg=BACKGROUND,
            fg=INK,
            font=("Helvetica", 22, "bold"),
        ).pack()
        progress = ttk.Progressbar(panel, mode="indeterminate", length=340)
        progress.pack(pady=26)
        progress.start(12)
        Thread(
            target=self._prediction_worker,
            args=(answers,),
            daemon=True,
        ).start()

    def _prediction_worker(self, answers: dict[str, int]) -> None:
        try:
            confirmed = [
                experiment for experiment in self.search.experiments
                if experiment.validation_improvement_percent > 0
                and experiment.test_improvement_percent >= FINAL_IMPROVEMENT_PERCENT
            ]
            predictions = predict_experiments(self.survey, answers, confirmed)
            self.root.after(
                0,
                lambda: self._show_results(confirmed, predictions),
            )
        except Exception as error:
            self.root.after(0, lambda: self._show_error(str(error)))

    def _show_results(self, experiments, predictions: dict[str, float]) -> None:
        self._clear_content()
        _canvas, body = self._scrolling_page()
        intro = tk.Frame(body, bg=BACKGROUND, padx=48, pady=30)
        intro.pack(fill="x")
        tk.Label(
            intro,
            text=f"{len(experiments)} reliable prediction(s)",
            bg=BACKGROUND,
            fg=INK,
            font=("Helvetica", 22, "bold"),
        ).pack(anchor="w")
        tk.Label(
            intro,
            text="Associations are evidence from this dataset—not facts or causes.",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Helvetica", 11),
        ).pack(anchor="w", pady=(7, 0))

        if not experiments:
            tk.Label(
                body,
                text="The pre-screened relationships did not survive final testing.",
                bg=BACKGROUND,
                fg=WARNING,
                font=("Helvetica", 14, "bold"),
            ).pack(padx=48, pady=30)

        for experiment in experiments:
            card = tk.Frame(body, bg=CARD, padx=26, pady=22, highlightthickness=1)
            card.configure(highlightbackground="#d9dedc")
            card.pack(fill="x", padx=48, pady=9)
            category = question_category(self.survey, experiment.target)
            tk.Label(
                card,
                text=category.upper(),
                bg=CARD,
                fg=ACCENT,
                font=("Helvetica", 9, "bold"),
            ).pack(anchor="w")
            tk.Label(
                card,
                text=experiment.target,
                bg=CARD,
                fg=INK,
                font=("Helvetica", 15, "bold"),
                wraplength=800,
                justify="left",
            ).pack(anchor="w", pady=(5, 3))

            target_values = self.survey[experiment.target].dropna()
            if target_values.between(1, 5).all():
                value_text = f"{predictions[experiment.target]:.2f} / 5"
            elif experiment.target == "Age":
                value_text = f"{predictions[experiment.target]:.1f} years"
            else:
                value_text = f"{predictions[experiment.target]:.2f}"
            tk.Label(
                card,
                text=value_text,
                bg=CARD,
                fg=ACCENT_DARK,
                font=("Helvetica", 28, "bold"),
            ).pack(anchor="w", pady=(3, 10))
            tk.Label(
                card,
                text=(
                    f"{experiment.model_name}  •  MAE {experiment.test_model_mae:.3f} "
                    f"vs baseline {experiment.test_baseline_mae:.3f}  •  "
                    f"{experiment.test_improvement_percent:.1f}% better"
                ),
                bg=CARD,
                fg=MUTED,
                font=("Helvetica", 10),
                wraplength=800,
                justify="left",
            ).pack(anchor="w")

            correlations = self.survey[list(experiment.input_questions)].corrwith(
                self.survey[experiment.target], method="spearman"
            ).dropna()
            strongest = correlations.reindex(
                correlations.abs().sort_values(ascending=False).index
            ).head(3)
            tk.Label(
                card,
                text="Strongest cross-category associations",
                bg=CARD,
                fg=INK,
                font=("Helvetica", 10, "bold"),
            ).pack(anchor="w", pady=(15, 5))
            for question, correlation in strongest.items():
                direction = "positive" if correlation >= 0 else "negative"
                tk.Label(
                    card,
                    text=(
                        f"• {question_category(self.survey, question)}: {question} "
                        f"({direction} {correlation:+.2f})"
                    ),
                    bg=CARD,
                    fg=MUTED,
                    font=("Helvetica", 10),
                    wraplength=800,
                    justify="left",
                ).pack(anchor="w", pady=2)

        action = tk.Frame(body, bg=BACKGROUND, padx=48, pady=28)
        action.pack(fill="x")
        ttk.Button(
            action,
            text="Run another experiment",
            style="Accent.TButton",
            command=self.start_experiment,
        ).pack(side="right")

    def _show_error(self, message: str) -> None:
        messagebox.showerror("Survey Signals", message)
        self.start_experiment()


def main() -> None:
    root = tk.Tk()
    SurveyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
