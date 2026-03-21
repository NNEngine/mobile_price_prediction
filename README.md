# <div align = "center">mobile_price_prediction</div>

A mobile price prediction MLOps Project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

```
Since you are working with DVC, avoid reading data directly in feature scripts.
So,we are building load_data.py in src/data
```

```
run: `python -m src.features.build_features`` from root dir
```

```
ruff (linting + formatting)
    - Fastest to set up, replaces both flake8 and black
    - Add to pyproject.toml:

run in terminal

- ruff check src/data
- ruff check src/data --fix  (fixable issue only not errors)
```

**Key design decisions:**

- **`metric_map` with lambdas** — each metric is only computed if requested in `params.yaml`, so you can turn metrics on/off without touching code
- **`try/except` per metric** — one failing metric (e.g. `roc_auc` on binary) won't crash the entire evaluation
- **Timestamped JSON reports** — multiple runs don't overwrite each other, maps cleanly to MLflow later
- **`confusion_matrix` always included** — too useful to make optional, stored as a plain list for JSON compatibility
- **Takes `comparison` DataFrame** — plugs directly into `predict_model()` output with no transformation needed

The full pipeline flow is now:
```
load_data → reg_vs_clf → select_model → train_model → predict_model → evaluate_model
```
