# sweet.tv TV programme recommender (RecSys final exam, GSB HSE)

For each of the 1,260 users of the Kaggle community competition
[sweet.tv — TV program recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender) the code
predicts the five programmes the user will most often watch for at least 80% of an airing in weeks 31 to 42 of 2020.

Final submission: `submissions/ranker_v3.csv`, a LightGBM re-ranker over popularity, a habit-times-schedule
heuristic, ALS, personalised PageRank, association rules, content-based filtering and viewing-habit features.
Returned Kaggle score: public 0.45244, private 0.44757. Hold-out MAP@5 (weeks 21 to 30): 0.3595.
The report is `report/starchenko_exam_report.pdf`.

## Data

The competition data is not in the repository. Accept the competition rules on Kaggle, then

```
kaggle competitions download -c sweettv-tv-program-recommender -p data/raw
```

and unzip the archive into `data/raw/` (five csv files). The first notebook parses them once into `data/cache/*.parquet`.

## Environment

Windows 11, Python 3.12.14, packages pinned in `requirements.txt` (NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1,
scikit-learn 1.9.1, LightGBM 4.7.0, pyarrow 26.0.0). No GPU and no torch. The re-ranker notebooks need about 6 GB of RAM.

```
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt
```

## Reproducing the results

Every notebook is written as a percent-format script `*.py` next to it. `tools/py2nb.py` turns the script into the
notebook and `tools/run_nb.py` executes it in place (the kernel runs on the same interpreter). From the repository root:

```
python tools/py2nb.py modelling/baselines.py modelling/baselines.ipynb
python tools/run_nb.py modelling/baselines.ipynb
```

Run in this order; times are on an AMD Zen 4 laptop with 16 threads.

| Step | Notebook | Writes | Time |
|---|---|---|---|
| 1 | `eda/eda.ipynb` | parquet cache, `eda/eda_summary.json`, `eda/figures/` | 72 s |
| 2 | `modelling/baselines.ipynb` | popularity and habit x schedule submissions, `baselines_results.json` | 60 s |
| 3 | `modelling/cf_models.ipynb` | ALS, PPR, rules, content settings (`cf_results.json`) and submissions | 215 s |
| 4 | `modelling/ranker.ipynb` | re-ranker round 1 and its submissions | 497 s |
| 5 | `modelling/ranker_v2.ipynb` | round 2, `submissions/ranker_v2.csv` | 463 s |
| 6 | `modelling/ranker_v3.ipynb` | round 3, the final `submissions/ranker_v3.csv` | 1,488 s |
| 7 | `modelling/analysis.ipynb` | hold-out comparison, bootstrap, list properties, example users | 35 s |
| 8 | `tests/check_core.py` | hand-made checks of MAP@5, the join, the target rule, ALS and the submission format | seconds |
| 9 | `report/make_figures.py`, `report/make_tables.py`, then `pdflatex report.tex` in `report/` | report figure, tables and PDF | seconds |

Round 1 (`ranker.ipynb`) was executed with the first ALS setting (32 factors, alpha 3, reg 100), before the ALS grid in
`cf_models.ipynb` was extended; rerunning it now reads the current `cf_results.json` and gives slightly different
round-1 numbers. Rounds 2 and 3 used the current setting. Seeds are fixed (0 for ALS, sampling and the bootstrap, 0 to 2
for LightGBM), and round 3 reproduced round 2 to four decimals.

## Layout

```
src/sweettv/   data loading, view-schedule join, target rule, time snapshots, baselines,
               ALS / PPR / association rules / content, candidates and features, LightGBM wrapper, top-5 lists
eda/           exploratory analysis
modelling/     methods, re-ranker rounds, analysis and their saved results
submissions/   every submitted file
tests/         checks
report/        LaTeX source, generated tables, figure script, Kaggle score export, PDF
tools/         notebook builder and runner, Kaggle kernel packer used for an early EDA run
```

## Main choices

- A view matches every slot of the same channel whose airing overlaps it. An airing counts as watched when the user's
  views cover at least 0.8 of it, the organisers' rule from the competition data page.
- `tv_show_id = 0` rows are never items, but all airtime feeds the viewing-habit features.
- Hold-out: history weeks 11 to 20, target weeks 21 to 30. The ranker learns from snapshots whose labels end by week 20.
- Candidate pool: labelled programmes that air in the target window. Repeats are allowed for every method, empty places
  are filled from one popularity ranking, which is also the answer for users without history.
