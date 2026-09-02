"""Native desktop interface for the survey prediction experiment.

The GUI does not contain the model-selection logic. It collects answers, calls
the reusable functions in model.py, and explains results in playful language.
"""

from threading import Thread
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from model import (
    find_predictive_random_group,
    load_survey,
    predict_experiments,
    question_category,
)


QUESTION_COUNT = 20
TARGET_COUNT = 5
MIN_CONFIRMED_TARGETS = 3
QUALIFYING_IMPROVEMENT_PERCENT = 10.0
FINAL_IMPROVEMENT_PERCENT = 5.0
MAX_GROUP_ATTEMPTS = 10

# I started with five and then ten questions. Twenty is the current compromise:
# more answers give the models more information, but also make the game longer
# and still do not guarantee that every target can be predicted.

# Regression estimates tend to cluster near 3 because they average many possible
# responses.  The game can make a bolder call by stretching that estimate away
# from the neutral midpoint.  We still display the untouched ML estimate so this
# presentation choice is not confused with extra model accuracy.
GUESS_BOLDNESS = 1.8

# Preserve macOS trackpad momentum while making each gesture feel responsive.
TRACKPAD_SCROLL_SPEED = 3

BACKGROUND = "#f7e5bd"
CARD = "#fff8e7"
INK = "#43281f"
MUTED = "#785b4b"
ACCENT = "#e85d3f"
ACCENT_DARK = "#a93c2b"
SOFT_ACCENT = "#f2bd3b"
WARNING = "#b34f32"
MUSTARD = "#e9a72f"
AVOCADO = "#71853a"
TEAL = "#2c7a73"
PINK = "#cf6681"
BROWN = "#5b3428"

CATEGORY_COLORS = {
    "Music": PINK,
    "Movies": ACCENT,
    "Interests": TEAL,
    "Fears": MUSTARD,
    "Personality and lifestyle": AVOCADO,
    "Spending": "#9b5f9e",
    "Demographics": BROWN,
}


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


def prediction_label(question: str, score: float, survey_columns: list[str]) -> str:
    """Translate a regression score into the nearest original response category."""

    rating = max(1, min(5, int(score + 0.5)))
    position = survey_columns.index(question)
    positions = {name: index for index, name in enumerate(survey_columns)}

    music_genre = positions["Dance, Disco, Funk"] <= position <= positions["Opera"]
    movie_genre = positions["Horror movies"] <= position <= positions["Action movies"]
    if music_genre or movie_genre:
        return {
            1: "Would probably hate it",
            2: "Would probably dislike it",
            3: "Feels pretty neutral",
            4: "Would probably like it",
            5: "Would probably love it",
        }[rating]

    if positions["History"] <= position <= positions["Pets"]:
        return {
            1: "Not interested at all",
            2: "Probably not very interested",
            3: "Somewhere in the middle",
            4: "Probably interested",
            5: "Very interested",
        }[rating]

    if positions["Flying"] <= position <= positions["Public speaking"]:
        return {
            1: "Probably not afraid at all",
            2: "Only a little uneasy",
            3: "Somewhat uneasy",
            4: "Probably afraid",
            5: "Probably very afraid",
        }[rating]

    if question == "I prefer.":
        return {
            1: "Strongly prefers slow music",
            2: "Leans toward slow music",
            3: "Likes both about equally",
            4: "Leans toward fast music",
            5: "Strongly prefers fast music",
        }[rating]

    return {
        1: "Would strongly disagree",
        2: "Would probably disagree",
        3: "Feels somewhere in the middle",
        4: "Would probably agree",
        5: "Would strongly agree",
    }[rating]


def make_bolder_guess(score: float, midpoint: float = 3.0) -> float:
    """Stretch a 1–5 estimate away from neutral, without leaving the scale."""

    bold_score = midpoint + GUESS_BOLDNESS * (score - midpoint)
    return max(1.0, min(5.0, bold_score))


class SurveyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Vibe Check")
        self.root.geometry("980x780")
        self.root.minsize(760, 620)
        self.root.configure(bg=BACKGROUND)

        self.survey = load_survey()
        self.survey_columns = list(self.survey.columns)
        self.search = None
        self.answer_variables: dict[str, tk.IntVar] = {}
        self.answer_buttons: dict[str, list[tk.Button]] = {}
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
            font=("Avenir Next", 12, "bold"),
        )
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
        style.configure(
            "Secondary.TButton",
            background=MUSTARD,
            foreground=BROWN,
            borderwidth=0,
            padding=(18, 10),
            font=("Avenir Next", 11, "bold"),
        )
        style.configure(
            "Survey.TRadiobutton",
            background=CARD,
            foreground=INK,
            font=("Avenir Next", 12, "bold"),
        )

    def _build_shell(self) -> None:
        header = tk.Frame(self.root, bg=BROWN, padx=34, pady=14)
        header.pack(fill="x")
        badge_row = tk.Frame(header, bg=BROWN)
        badge_row.pack(fill="x")
        tk.Label(
            badge_row,
            text="  VIBE CHECK '75  ",
            bg=MUSTARD,
            fg=BROWN,
            font=("Phosphate", 14),
            padx=8,
            pady=4,
        ).pack(side="left")
        for color in (ACCENT, PINK, TEAL, AVOCADO):
            tk.Label(
                badge_row,
                text="●",
                bg=BROWN,
                fg=color,
                font=("Helvetica", 16, "bold"),
            ).pack(side="right", padx=3)
        tk.Label(
            header,
            text=f"{QUESTION_COUNT} answers. A few groovy guesses.",
            bg=BROWN,
            fg="#fff4d6",
            font=("SignPainter", 27),
        ).pack(anchor="w", pady=(10, 0))

        stripe = tk.Canvas(
            self.root,
            height=12,
            bg=BROWN,
            highlightthickness=0,
        )
        stripe.pack(fill="x")
        stripe_colors = (ACCENT, MUSTARD, AVOCADO, TEAL, PINK, "#8a6b9b")
        for index, color in enumerate(stripe_colors):
            stripe.create_rectangle(
                index * 170,
                0,
                (index + 1) * 170,
                12,
                fill=color,
                outline=color,
            )

        self.content = tk.Frame(self.root, bg=BACKGROUND)
        self.content.pack(fill="both", expand=True)

    def _retro_pattern(self, parent: tk.Widget) -> tk.Canvas:
        """Draw a wide 1970s poster pattern inspired by the supplied reference."""

        art = tk.Canvas(
            parent,
            width=880,
            height=330,
            bg="#f9e7cd",
            highlightthickness=2,
            highlightbackground=BROWN,
        )
        palette = (ACCENT, MUSTARD, AVOCADO, TEAL, PINK, "#8a6b9b")

        # Left: parallel flowing ribbons that enter and leave the frame.
        wave = (-80, 82, 60, 18, 155, 96, 280, 36, 385, 104)
        for offset, color in reversed(list(enumerate(palette[:4]))):
            shifted = tuple(
                value + (offset * 18 if index % 2 else 0)
                for index, value in enumerate(wave)
            )
            art.create_line(
                shifted,
                smooth=True,
                splinesteps=32,
                width=27,
                fill=BROWN,
            )
            art.create_line(
                shifted,
                smooth=True,
                splinesteps=32,
                width=23,
                fill=color,
            )

        # Right: a low sunburst gives the composition a poster-panel rhythm.
        centre_x, centre_y = 742, 330
        ray_points = (430, 330, 505, 35, 585, 330, 635, 0, 690, 330, 755, 0,
                      785, 330, 875, 25, 880, 330)
        rays = [ray_points[index:index + 6] for index in range(0, 24, 6)]
        for index, points in enumerate(rays):
            art.create_polygon(
                centre_x,
                centre_y,
                *points,
                fill=palette[(index + 1) % len(palette)],
                outline=BROWN,
                width=2,
            )

        # Bottom-left: oversized crop of a lava-loop shape.
        art.create_oval(-125, 178, 330, 520, fill=AVOCADO, outline=BROWN, width=3)
        art.create_oval(-66, 214, 278, 472, fill=MUSTARD, outline=BROWN, width=3)
        art.create_oval(-12, 248, 222, 430, fill="#f9e7cd", outline=BROWN, width=3)

        # A horizontal title strip ties the different patterns together.
        art.create_rectangle(0, 130, 880, 224, fill="#ca5e2b", outline=BROWN, width=3)
        art.create_text(
            440,
            164,
            text="TUNING THE VIBE MACHINE",
            fill="#fff1d3",
            font=("Phosphate", 25),
        )
        art.create_text(
            440,
            199,
            text=f"finding {QUESTION_COUNT} questions with something interesting to say",
            fill="#fff1d3",
            font=("SignPainter", 18),
        )
        return art

    def _clear_content(self) -> None:
        self.active_canvas = None
        for child in self.content.winfo_children():
            child.destroy()

    def _scroll_wheel(self, event) -> str | None:
        if self.active_canvas is not None and event.delta:
            if sys.platform == "darwin":
                # macOS trackpads already send small, momentum-aware deltas.
                # Keeping their magnitude makes scrolling follow the gesture
                # instead of converting every event into the same large jump.
                amount = -event.delta * TRACKPAD_SCROLL_SPEED
            else:
                # A traditional Windows mouse wheel normally reports 120 per
                # notch, so normalize it to a comfortable pixel movement.
                amount = round(-event.delta / 120 * 36)
            self.active_canvas.yview_scroll(amount, "units")
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
        self.answer_buttons = {}

        panel = tk.Frame(self.content, bg=BACKGROUND, padx=22, pady=22)
        panel.pack(fill="both", expand=True)
        self._retro_pattern(panel).pack(fill="x", pady=(0, 16))
        tk.Label(
            panel,
            text=(
                "Mixing music, movies, interests, fears, personality, and spending…"
            ),
            bg=BACKGROUND,
            fg=MUTED,
            font=("Avenir Next", 12),
            wraplength=620,
            justify="center",
        ).pack(pady=(14, 26))
        progress = ttk.Progressbar(panel, mode="indeterminate", length=360)
        progress.pack()
        progress.start(12)
        self.status_label = tk.Label(
            panel,
            text="Looking for a question mix with genuinely good vibes…",
            bg=BACKGROUND,
            fg=ACCENT,
            font=("Avenir Next", 11, "bold"),
        )
        self.status_label.pack(pady=16)

        # This search trains many small models and can take a few seconds. A
        # worker thread keeps the window and progress animation responsive;
        # root.after() sends the finished result back to Tkinter's main thread.
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
                text="The stars didn't align this time",
                bg=BACKGROUND,
                fg=INK,
                font=("Avenir Next", 22, "bold"),
            ).pack()
            tk.Label(
                panel,
                text=(
                    "That question mix didn't have enough signal for three fair "
                    "guesses. Let's shuffle the deck and try again."
                ),
                bg=BACKGROUND,
                fg=MUTED,
                font=("Avenir Next", 12),
                wraplength=580,
                justify="center",
            ).pack(pady=18)
            ttk.Button(
                panel,
                text="Shuffle again",
                style="Accent.TButton",
                command=self.start_experiment,
            ).pack(pady=12)
            return

        self.search = result
        self._show_questions()

    def _scrolling_page(self) -> tuple[tk.Canvas, tk.Frame]:
        canvas = tk.Canvas(
            self.content,
            bg=BACKGROUND,
            highlightthickness=0,
            # One scroll unit equals one pixel, allowing trackpad deltas to be
            # applied gradually rather than jumping by widget-sized units.
            yscrollincrement=1,
        )
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
            text=f"Your {QUESTION_COUNT}-question mix is ready",
            bg=BACKGROUND,
            fg=INK,
            font=("Phosphate", 24),
        ).pack(anchor="w")
        tk.Label(
            intro,
            text=(
                f"The deck clicked after {self.search.attempts} shuffle(s). "
                "Pick a number for every question, then let the vibe reader do its thing."
            ),
            bg=BACKGROUND,
            fg=MUTED,
            font=("Avenir Next", 11),
            wraplength=780,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))
        self.answer_progress_label = tk.Label(
            intro,
            text=f"0 / {QUESTION_COUNT} dials tuned",
            bg=BACKGROUND,
            fg=ACCENT,
            font=("Phosphate", 13),
        )
        self.answer_progress_label.pack(anchor="w", pady=(12, 0))

        cards_grid = tk.Frame(body, bg=BACKGROUND, padx=40)
        cards_grid.pack(fill="x")
        cards_grid.grid_columnconfigure(0, weight=1, uniform="questions")
        cards_grid.grid_columnconfigure(1, weight=1, uniform="questions")

        for number, question in enumerate(self.search.input_questions, start=1):
            category = question_category(self.survey, question)
            category_color = CATEGORY_COLORS.get(category, ACCENT)
            card = tk.Frame(
                cards_grid,
                bg=CARD,
                padx=20,
                pady=17,
                highlightthickness=3,
                highlightbackground=category_color,
            )
            row, column = divmod(number - 1, 2)
            card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
            tk.Label(
                card,
                text=f"  {number:02d}  ·  {category.upper()}  ",
                bg=category_color,
                fg="white",
                font=("Avenir Next", 9, "bold"),
                padx=4,
                pady=3,
            ).pack(anchor="w")
            tk.Label(
                card,
                text=question,
                bg=CARD,
                fg=INK,
                font=("Avenir Next", 13, "bold"),
                wraplength=360,
                justify="left",
            ).pack(anchor="w", pady=(5, 4))
            tk.Label(
                card,
                text=question_scale(question, self.survey_columns),
                bg=CARD,
                fg=MUTED,
                font=("Avenir Next", 10),
            ).pack(anchor="w", pady=(0, 10))

            variable = tk.IntVar(value=0)
            self.answer_variables[question] = variable
            self.answer_buttons[question] = []
            choices = tk.Frame(card, bg=CARD)
            choices.pack(anchor="w")
            for value in range(1, 6):
                button = tk.Button(
                    choices,
                    text=str(value),
                    width=3,
                    bg="#f3dfb7",
                    fg=INK,
                    activebackground=MUSTARD,
                    activeforeground=INK,
                    font=("Futura", 11, "bold"),
                    relief="flat",
                    bd=0,
                    padx=3,
                    pady=5,
                    cursor="hand2",
                    command=lambda q=question, v=value, c=category_color: (
                        self._select_answer(q, v, c)
                    ),
                )
                button.pack(side="left", padx=(0, 7))
                self.answer_buttons[question].append(button)

        action = tk.Frame(body, bg=BACKGROUND, padx=48, pady=26)
        action.pack(fill="x")
        ttk.Button(
            action,
            text="Read my vibe",
            style="Accent.TButton",
            command=lambda: self._begin_prediction(canvas),
        ).pack(side="right")

    def _select_answer(self, question: str, value: int, color: str) -> None:
        """Record an answer and make the chosen number visually unmistakable."""

        self.answer_variables[question].set(value)
        for number, button in enumerate(self.answer_buttons[question], start=1):
            selected = number == value
            button.configure(
                bg=color if selected else "#f3dfb7",
                fg="white" if selected else INK,
                relief="sunken" if selected else "flat",
                bd=2 if selected else 0,
            )
        answered = sum(variable.get() != 0 for variable in self.answer_variables.values())
        self.answer_progress_label.configure(
            text=f"{answered} / {QUESTION_COUNT} dials tuned"
        )

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
            text="Reading the room…",
            bg=BACKGROUND,
            fg=INK,
            font=("Avenir Next", 23, "bold"),
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
            # Passing the earlier screening is not enough. The final untouched
            # participants get the last word about which guesses may be shown.
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
            text=f"Your vibe report has {len(experiments)} good guess(es)",
            bg=BACKGROUND,
            fg=INK,
            font=("Phosphate", 24),
        ).pack(anchor="w")
        tk.Label(
            intro,
            text="Fun patterns from the survey—not destiny, not a personality verdict.",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Avenir Next", 11),
        ).pack(anchor="w", pady=(7, 0))

        if not experiments:
            tk.Label(
                body,
                text="Plot twist: the promising patterns didn't hold up at the finish line.",
                bg=BACKGROUND,
                fg=WARNING,
                font=("Avenir Next", 14, "bold"),
            ).pack(padx=48, pady=30)

        result_grid = tk.Frame(body, bg=BACKGROUND, padx=40)
        result_grid.pack(fill="x")
        result_grid.grid_columnconfigure(0, weight=1, uniform="results")
        result_grid.grid_columnconfigure(1, weight=1, uniform="results")
        result_colors = ("#fff3c9", "#e4efe4", "#f8ddd4", "#dcebea", "#f2dce7")

        for index, experiment in enumerate(experiments):
            result_bg = result_colors[index % len(result_colors)]
            category = question_category(self.survey, experiment.target)
            category_color = CATEGORY_COLORS.get(category, ACCENT)
            card = tk.Frame(
                result_grid,
                bg=result_bg,
                padx=22,
                pady=20,
                highlightthickness=4,
                highlightbackground=category_color,
            )
            row, column = divmod(index, 2)
            card.grid(row=row, column=column, sticky="nsew", padx=8, pady=8)
            tk.Label(
                card,
                text=f"  {category.upper()}  ",
                bg=category_color,
                fg="white",
                font=("Avenir Next", 9, "bold"),
                padx=4,
                pady=3,
            ).pack(anchor="w")
            tk.Label(
                card,
                text=experiment.target,
                bg=result_bg,
                fg=INK,
                font=("Avenir Next", 15, "bold"),
                wraplength=360,
                justify="left",
            ).pack(anchor="w", pady=(5, 3))

            target_values = self.survey[experiment.target].dropna()
            if target_values.between(1, 5).all():
                raw_score = predictions[experiment.target]
                bold_score = make_bolder_guess(raw_score)
                value_text = prediction_label(
                    experiment.target,
                    bold_score,
                    self.survey_columns,
                )
                # The headline gets the more decisive game score, but both
                # values stay visible so presentation is not confused with an
                # improvement in the actual ML model.
                score_detail = (
                    f"bold game guess {bold_score:.2f} / 5  ·  "
                    f"raw model score {raw_score:.2f} / 5"
                )
            elif experiment.target == "Age":
                value_text = f"{predictions[experiment.target]:.1f} years"
                score_detail = "numeric estimate"
            else:
                value_text = f"{predictions[experiment.target]:.2f}"
                score_detail = "numeric estimate"
            tk.Label(
                card,
                text=value_text,
                bg=result_bg,
                fg=category_color,
                font=("Phosphate", 24),
                wraplength=360,
                justify="left",
            ).pack(anchor="w", pady=(3, 10))
            tk.Label(
                card,
                text=(
                    f"BEHIND THE CURTAIN  ·  {score_detail}  ·  {experiment.model_name}  ·  "
                    f"average miss {experiment.test_model_mae:.2f}  ·  "
                    f"{experiment.test_improvement_percent:.1f}% better than a basic guess"
                ),
                bg=result_bg,
                fg=MUTED,
                font=("Avenir Next", 9, "bold"),
                wraplength=360,
                justify="left",
            ).pack(anchor="w")

            # Correlation makes the result easier to interpret. The held-out MAE
            # above—not correlation—is what decided this target was predictable.
            correlations = self.survey[list(experiment.input_questions)].corrwith(
                self.survey[experiment.target], method="spearman"
            ).dropna()
            strongest = correlations.reindex(
                correlations.abs().sort_values(ascending=False).index
            ).head(3)
            tk.Label(
                card,
                text="What connected the dots",
                bg=result_bg,
                fg=INK,
                font=("Avenir Next", 11, "bold"),
            ).pack(anchor="w", pady=(15, 5))
            for question, correlation in strongest.items():
                direction = "positive" if correlation >= 0 else "negative"
                tk.Label(
                    card,
                    text=(
                        f"• {question_category(self.survey, question)}: {question} "
                        f"({direction} {correlation:+.2f})"
                    ),
                    bg=result_bg,
                    fg=MUTED,
                    font=("Avenir Next", 10),
                    wraplength=360,
                    justify="left",
                ).pack(anchor="w", pady=2)

        action = tk.Frame(body, bg=BACKGROUND, padx=48, pady=28)
        action.pack(fill="x")
        ttk.Button(
            action,
            text="Shuffle a new vibe check",
            style="Accent.TButton",
            command=self.start_experiment,
        ).pack(side="right")

    def _show_error(self, message: str) -> None:
        messagebox.showerror("Vibe Check '75", message)
        self.start_experiment()


def main() -> None:
    root = tk.Tk()
    SurveyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
