# Vibe Check: Survey Response Predictor

Note: ChatGPT and Codex were used for portions of the project. More specific
notes are available in separate files.


This project investigates whether a small group of answers from the Kaggle
**Young People Survey** (https://www.kaggle.com/datasets/miroslavsabo/young-people-survey) can predict other unanswered survey responses. It is an
ML experiment that I have chosen to make interactive through a colorful game.
It does not claim to tell the user something definitive about their personality.

The program searches for a useful group of survey questions, asks the user to
answer them, and only displays targets that performed better than a constant
baseline on held-out survey participants.

## Requirements

- Python 3.10 or newer
- `pip`
- Tkinter for the desktop GUI (included with Python from python.org on macOS
  and Windows; see troubleshooting below for Homebrew and Linux)

The required dataset files are already included:

- `data/responses.csv` — survey responses
- `data/columns.csv` — original full survey question wording

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/ArinRothschild0407/vibe-checker.git
cd vibe-checker
```

Create and activate a virtual environment:

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Each time you open a new terminal to work on the project, return to the cloned
directory and reactivate the environment:

```bash
cd vibe-checker
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For later PowerShell sessions, reactivate the environment after entering the
project directory:

```powershell
cd vibe-checker
.venv\Scripts\Activate.ps1
```

## Tkinter troubleshooting

Test the Python interpreter from the activated virtual environment:

```bash
python -m tkinter
```

A small Tk window should open. If Python reports `No module named '_tkinter'`
or `No module named 'tkinter'`, install Tkinter for the same Python version used
to create the virtual environment.

### macOS with Homebrew

Homebrew installs Tkinter separately from Python. For the current Homebrew
Python 3.14 packages:

```bash
brew install python@3.14 python-tk@3.14
```

If you use another Homebrew Python version, install the matching formula (for
example, `python-tk@3.13` for `python@3.13`). Then recreate and reactivate the
virtual environment so it uses that interpreter:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m tkinter
```

### Linux

Install the operating-system package before creating the virtual environment.
For Ubuntu or Debian:

```bash
sudo apt update
sudo apt install python3-tk
```

For Fedora:

```bash
sudo dnf install python3-tkinter
```

For Arch Linux:

```bash
sudo pacman -S tk
```

If the environment was created before Tkinter was installed, recreate it using
the macOS/Linux installation commands above. Tkinter is part of the Python
installation and intentionally is not listed in `requirements.txt`.

## Running the project

Launch the graphical version:

```bash
python gui.py
```

The initial search may take several seconds while models are trained and
evaluated. The GUI asks 20 survey questions and then shows up to five supported
predictions.

There is also a terminal version that asks 10 questions, which I used for
testing:

```bash
python game.py
```

There is also a file I used during setup. It prints basic information about the
dataset, column types, ranges, and missing values:

```bash
python explore.py
```

## What the machine-learning code does

The reusable ML code is in `model.py`; the interaction layers are in `gui.py`
and `game.py`.

1. It searches for an informative, cross-category group of input questions.
2. It tests unanswered numeric survey questions as possible targets.
3. Missing-value preprocessing is fitted only on training participants to avoid
   leakage.
4. A fast Ridge model screens possible targets.
5. Promising targets enter a tournament containing Random Forest, Extra Trees,
   Gradient Boosting, K-nearest neighbors, and Ridge regression.
6. Each model is compared with a constant prediction equal to the **training-set
   median**. Performance is measured with mean absolute error (MAE).
7. Separate screening, confirmation, and final-test participants are used. A
   prediction is shown only if it continues to improve over the baseline on
   held-out data.

The program is allowed to find no reliable predictions. This is an intentional
part of the experiment rather than an error.

Height and weight are excluded as prediction targets. Input and target
questions from the same category are also kept separate, and highly similar
question pairs are rejected to reduce trivial predictions.

## Understanding the results

- **Raw model score:** the regression model's actual estimate on the original
  survey scale.
- **Average miss / MAE:** the average absolute distance between predictions and
  real held-out answers. Lower is better.
- **Better than a basic guess:** improvement over always predicting the
  training-set median.
- **Correlation:** a descriptive association in the survey data. It does not
  establish causation.

For a more playful result, the GUI stretches the raw 1–5 estimate away from the
neutral midpoint and labels it with phrases such as “would probably love it.”
The card displays both the **bold game guess** and the untouched **raw model
score**. This display adjustment does not improve or alter the evaluated ML
accuracy.

## Project files

| File | Purpose |
| --- | --- |
| `model.py` | Data loading, question selection, model training, evaluation, and prediction |
| `gui.py` | Tkinter desktop interface |
| `game.py` | Terminal-based interaction |
| `explore.py` | Basic dataset inspection |
| `data/responses.csv` | Young People Survey response data |
| `data/columns.csv` | Original survey question names and wording |

## Reproducibility note

Model train/test splits use fixed random states, while question-group selection
intentionally includes randomness so different runs can investigate different
sets of questions. Therefore, the questions and resulting predictions may vary
between runs.
