# Vibe Check: Survey Response Predictor

Note: ChatGPT and CodeX were used for protions of the prject, more specific notes in seperate files


This project investigates whether a small group of answers from the Kaggle
**Young People Survey** (https://www.kaggle.com/datasets/miroslavsabo/young-people-survey) can predict other unanswered survey responses. It is an
ML experiment which I have chosen to make interactive through a colorful game and not claim that tries to tell the user something about their personality.

The program searches for a useful group of survey questions, asks the user to
answer them, and only displays targets that performed better than a constant
baseline on held-out survey participants.

## Requirements

- Python 3.10 or newer
- `pip`
- Tkinter for the desktop GUI (included with standard Python installations on
  macOS and Windows)

The required dataset files are already included:

- `data/responses.csv` — survey responses
- `data/columns.csv` — original full survey question wording

## Installation

Clone the repository and enter its directory:

```bash
git clone <repository-url>
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

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On some Linux systems, Tkinter must be installed separately before launching
the GUI. For Ubuntu/Debian:

```bash
sudo apt install python3-tk
```

## Running the project

Launch the graphical version:

```bash
python gui.py
```

The initial search may take several seconds while models are trained and
evaluated. The GUI asks 20 survey questions and then shows up to five supported
predictions.

There is also a terminal version that asks 10 questions whihc I used for testing:

```bash
python game.py
```

There is alsoa file I used for set up which can be used to print basic information about the dataset, column types, ranges, and missing
values:

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
