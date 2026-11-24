# Diabetes Prediction

**FOR TEAMMATES:** Read sections 1 and 2 to get set up for your part of the project

## 1. Setup

```bash
conda create -n diabetes python=3.11
conda activate diabetes
pip install -r requirements.txt
```

## Data

Download the dataset from Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

(Kaggle may ask you to sign in.) Click "Download" → "Download dataset as zip", open the zip, and move `diabetes_binary_health_indicators_BRFSS2015.csv` into `data/raw/`.

**Everyone must use this same file.** It is the default in `src/data.py`, so no code changes are needed. If you use a different file, update `DATA_FILE` in `src/data.py`, but tell the team first so we all stay consistent.

The dataset is also listed on the UCI repository (https://archive.ics.uci.edu/dataset/891), which links to the Kaggle copy. Please follow the acknowledgment policy on the Kaggle page when citing the data.

Run everything from the project root.

## 2. Getting the train / validation / test splits

All preprocessing lives in `src/data.py`. You only need one function:

```python
from src.data import get_splits

d = get_splits()
```

`get_splits()` loads the CSV, drops exact duplicate rows, makes a stratified 70/15/15 split, and standardizes the non-binary columns using statistics from the training set only. It returns a dictionary:

| Key | Contents |
|---|---|
| `X_train`, `y_train` | training features and labels |
| `X_val`, `y_val` | validation data (tuning, early stopping, threshold selection) |
| `X_test`, `y_test` | test data (final evaluation only) |
| `feature_names` | list of column names |
| `scaler` | the fitted `Standardizer`, for transforming new data |

The split is deterministic. The same file, `seed` (default 42), and NumPy version give identical splits for everyone. Please don't change the seed or `drop_duplicates` setting without telling the team.

Quick sanity check (prints split shapes and positive rates, writes no files):

```bash
python -m src.data
```

## 3. Training and evaluating the MLP

```python
from src.data import get_splits
from src.mlp import MLP, evaluate, best_f1_threshold, gradient_check

d = get_splits()
n_features = d["X_train"].shape[1]

# Sanity-check backprop on a small batch before training (expect error ~1e-7 or smaller)
model = MLP([n_features, 64, 32, 1], l2=1e-4, seed=0)
print(gradient_check(model, d["X_train"][:30], d["y_train"][:30]))

# Train (early stopping on validation loss)
history = model.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                    epochs=50, lr=1e-3, optimizer="adam")

# Tune the threshold on VALIDATION, then evaluate once on TEST
thr, _ = best_f1_threshold(d["y_val"], model.predict_proba(d["X_val"]))
print(evaluate(d["y_test"], model.predict_proba(d["X_test"]), thr))
```

Notes:
- `layer_sizes` is configurable, e.g. `[n_features, 128, 64, 32, 1]`. The last size must be 1.
- Use `optimizer="adam"` with `lr` around 1e-3, or `optimizer="sgd"` (momentum) with `lr` around 1e-1.
- The positive class is a minority, so accuracy is misleading. Compare models on AUC, F1, precision, and recall. `pos_weight` in `fit` up-weights the positive class.
- Never tune anything on the test set.

## 4. Repo structure

```
diabetes-mlp/
  data/
    raw/           original CSV (git-ignored)
    processed/     optional cached splits (git-ignored)
  src/
    data.py        loading, stratified split, standardization
    mlp.py         MLP, optimizers, loss, gradient check, metrics
  notebooks/       EDA and experiments (import from src/)
  results/
    figures/       plots for the report and poster
  requirements.txt
  README.md
```

- **`src/data.py`**: `load_raw`, `stratified_split`, `Standardizer`, `get_splits`.
- **`src/mlp.py`**: the `MLP` class, plus hand-written `confusion_matrix`, `precision_recall_f1`, `roc_auc`, `roc_curve`, `best_f1_threshold`, `evaluate`, and `gradient_check`.
- **Notebooks** should import from `src/` rather than redefine code. Add these two lines at the top so module edits reload automatically:
  ```python
  %load_ext autoreload
  %autoreload 2
  ```

## 5. Conventions
- Keep model code in `src/`, and keep notebooks for experiments and plots.
- Don't commit data files.
- Don't use scikit-learn or other prebuilt modeling packages in the MLP code, including helpers like `train_test_split` or `roc_auc_score`.
