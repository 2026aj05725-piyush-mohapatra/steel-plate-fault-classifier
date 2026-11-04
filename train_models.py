# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Steel Plate Fault Classification — Model Training
# **BITS Pilani · Machine Learning · Assignment 2**
#
# This notebook (and its twin script `train_models.py`) trains five classifiers on the
# **UCI Steel Plates Faults** dataset, evaluates each on a held-out test split with six
# metrics, and saves the fitted pipelines to `model/` for the Streamlit app.
#
# | Step | What happens |
# |---|---|
# | 1 | Load the raw UCI file `data/Faults.NNA` and turn the 7 one-hot fault columns into one label |
# | 2 | Quick EDA: class balance, skew, correlation |
# | 3 | Stratified 80/20 split → `test_data.csv` |
# | 4 | 5-fold stratified grid search for each model (scoring = macro-F1) |
# | 5 | Test-set metrics: Accuracy, AUC, Precision, Recall, F1, MCC |
# | 6 | Confusion matrices + classification report |
# | 7 | Save pipelines (`.joblib`) + `model_card.json`, then reload from disk to double-check |
#
# > If a package is missing on the lab image, run `%pip install -r requirements.txt` in a new cell first.

# %%
import getpass
import json
import platform
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fault_toolkit import (
    FAULT_CLASSES, FEATURE_COLUMNS, METRIC_ORDER, MODEL_DIR_NAME, MODEL_REGISTRY,
    TARGET_COLUMN, load_uci_faults, score_predictions,
)

IN_NOTEBOOK = "__file__" not in globals()
ROOT = Path.cwd() if IN_NOTEBOOK else Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / MODEL_DIR_NAME
FIG_DIR = ROOT / "reports" / "figures"
for folder in (DATA_DIR, MODEL_DIR, FIG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

SPLIT_SEED = 1729          # Hardy–Ramanujan number; fixes split, CV folds and model seeds
TEST_FRACTION = 0.20

# Chart colours (validated categorical palette, light mode) + ink colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})


def finish(fig, filename):
    """Save a figure under reports/figures/ and show it (notebook) or close it (script)."""
    fig.savefig(FIG_DIR / filename)
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


print(f"Run started  : {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Host / user  : {socket.gethostname()} / {getpass.getuser()}")
print(f"Python       : {platform.python_version()} ({platform.system()})")
print(f"scikit-learn : {sklearn.__version__} | numpy {np.__version__} | pandas {pd.__version__}")
print(f"Project root : {ROOT}")

# %% [markdown]
# ## 1. Load the UCI file
# `Faults.NNA` has no header: 27 numeric features followed by 7 binary columns
# (one per fault type). Exactly one of the 7 is 1 in every row, so they collapse
# cleanly into a single multi-class target `Fault_Type`.

# %%
plates = load_uci_faults(DATA_DIR / "Faults.NNA")
plates.to_csv(DATA_DIR / "steel_plates_faults.csv", index=False)

print("Shape:", plates.shape, "| missing values:", int(plates.isna().sum().sum()),
      "| duplicate rows:", int(plates.duplicated().sum()))
plates.head()

# %% [markdown]
# ## 2. Exploratory look

# %%
class_counts = plates[TARGET_COLUMN].value_counts()
print((class_counts / len(plates) * 100).round(1).astype(str).add(" %").to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.6))
bars = ax.barh(class_counts.index[::-1], class_counts.values[::-1], color=SERIES[0], height=0.6)
ax.bar_label(bars, padding=3, color=MUTED, fontsize=9)
ax.set_title(f"Class distribution — {len(plates):,} plates, {len(class_counts)} fault types", loc="left")
ax.set_xlabel("Number of plates")
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
finish(fig, "class_distribution.png")

# %%
skew = plates[FEATURE_COLUMNS].skew().sort_values(key=np.abs, ascending=False)
print("Most skewed features (|skew|):")
print(skew.head(8).round(2).to_string())

corr = plates[FEATURE_COLUMNS].corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print(f"\nFeature pairs with |r| > 0.9: {int((upper.abs() > 0.9).sum())}")

fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(corr, cmap=sns.diverging_palette(250, 15, s=75, l=50, as_cmap=True),
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.3, linecolor="white",
            cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
ax.set_title("Feature correlation — several near-duplicate geometry features", loc="left")
finish(fig, "feature_correlation.png")

# %% [markdown]
# ## 3. Stratified train / test split
# 20 % of the plates (stratified by fault type) are held out. They are written to
# `test_data.csv` with all 27 features **and** the `Fault_Type` label — this is the
# file to upload into the Streamlit app.

# %%
X = plates[FEATURE_COLUMNS]
y = plates[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_FRACTION, stratify=y, random_state=SPLIT_SEED)

test_frame = X_test.copy()
test_frame[TARGET_COLUMN] = y_test
test_frame.to_csv(ROOT / "test_data.csv", index=False)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
pd.DataFrame({"train": y_train.value_counts(), "test": y_test.value_counts()})

# %% [markdown]
# ## 4. Model definitions and hyper-parameter search
# Every model is a scikit-learn `Pipeline`, so the scaler is saved **inside** the model
# file and the app never needs a separate preprocessing artifact.
#
# * Scale-sensitive models (LR, KNN, NB, SVM) let the grid search choose between
#   `StandardScaler` and a Yeo-Johnson `PowerTransformer` (many features are heavily skewed).
# * The Decision Tree is scale-invariant, so it gets no scaler.
# * Grid search uses 5-fold **stratified** CV on the training split only, scored by
#   **macro-F1** so the rare classes (Dirtiness, Stains) count as much as the common ones.

# %%
SCALERS = [StandardScaler(), PowerTransformer(method="yeo-johnson")]

search_plan = {
    "Logistic Regression": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000))]),
        {"scale": SCALERS, "clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
         "clf__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        Pipeline([("clf", DecisionTreeClassifier(random_state=SPLIT_SEED))]),
        {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [None, 6, 8, 10, 12, 16],
         "clf__min_samples_leaf": [1, 2, 4, 8], "clf__class_weight": [None, "balanced"]},
    ),
    "K-Nearest Neighbors": (
        Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"scale": SCALERS, "clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
         "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
    ),
    "Naive Bayes (Gaussian)": (
        Pipeline([("scale", StandardScaler()), ("clf", GaussianNB())]),
        {"scale": SCALERS, "clf__var_smoothing": np.logspace(-11, -1, 11)},
    ),
    "Support Vector Machine": (
        Pipeline([("scale", StandardScaler()),
                  ("clf", SVC(kernel="rbf", probability=True, random_state=SPLIT_SEED))]),
        {"scale": SCALERS, "clf__C": [1, 3, 10, 30, 100],
         "clf__gamma": ["scale", 0.01, 0.03, 0.1], "clf__class_weight": [None, "balanced"]},
    ),
}
assert list(search_plan) == list(MODEL_REGISTRY), "model names must match fault_toolkit.MODEL_REGISTRY"

# %%
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
fitted, tuning_log = {}, {}

for name, (pipeline, grid) in search_plan.items():
    started = time.perf_counter()
    search = GridSearchCV(pipeline, grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    seconds = time.perf_counter() - started

    fitted[name] = search.best_estimator_
    readable = {k.replace("clf__", ""): (type(v).__name__ if k == "scale"
                                         else v.item() if isinstance(v, np.generic) else v)
                for k, v in search.best_params_.items()}
    tuning_log[name] = {
        "best_params": readable,
        "cv_macro_f1": round(float(search.best_score_), 4),
        "candidates_tried": len(search.cv_results_["params"]),
        "search_seconds": round(seconds, 1),
    }
    print(f"{name:<24} CV macro-F1 = {search.best_score_:.4f}  "
          f"({len(search.cv_results_['params'])} candidates, {seconds:.0f}s)  best = {readable}")

# %% [markdown]
# ## 5. Test-set evaluation (six metrics)
# * **AUC** is one-vs-rest ROC-AUC, macro-averaged over the 7 classes.
# * **Precision / Recall / F1** are macro-averaged (each fault type weighted equally).
# * **MCC** is the multi-class Matthews correlation coefficient.

# %%
results = {}
for name, model in fitted.items():
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    results[name] = score_predictions(y_test, pred, proba, model.classes_, average="macro")

    # sanity check: our AUC helper equals scikit-learn's multi-class OvR AUC
    sk_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=model.classes_)
    assert abs(sk_auc - results[name]["AUC"]) < 1e-12

metrics_table = pd.DataFrame(results).T[METRIC_ORDER]
metrics_table.index.name = "ML Model Name"
metrics_table.round(4).to_csv(MODEL_DIR / "test_metrics.csv")

best_model_name = metrics_table["F1"].idxmax()
print(f"Best model on the test split (macro-F1): {best_model_name}\n")
metrics_table.round(4)

# %%
fig, ax = plt.subplots(figsize=(10, 4.2))
width = 0.16
positions = np.arange(len(METRIC_ORDER))
for i, (name, row) in enumerate(metrics_table.iterrows()):
    ax.bar(positions + (i - 2) * width, row[METRIC_ORDER].values, width * 0.88,
           color=SERIES[i], label=name)
ax.set_xticks(positions, METRIC_ORDER)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Score on test split")
ax.set_title(f"Model comparison on the {len(X_test)}-plate test split", loc="left")
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9)
finish(fig, "model_comparison.png")

# %% [markdown]
# ## 6. Where do the models go wrong?

# %%
label_order = sorted(FAULT_CLASSES)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for ax, (name, model) in zip(axes, fitted.items()):
    cm = confusion_matrix(y_test, model.predict(X_test), labels=label_order)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                linewidths=0.5, linecolor="white", xticklabels=label_order,
                yticklabels=label_order, ax=ax, annot_kws={"fontsize": 8})
    ax.set_title(f"{name}\nacc {results[name]['Accuracy']:.3f} · MCC {results[name]['MCC']:.3f}",
                 fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
axes[-1].axis("off")
axes[-1].text(0.05, 0.55, "How to read these matrices\n\n• rows = actual fault type\n"
              "• columns = predicted fault type\n• diagonal = correct predictions\n"
              "• darker cell = more plates", fontsize=11, color=MUTED, va="center")
fig.tight_layout()
finish(fig, "confusion_matrices.png")

print(f"Classification report — {best_model_name}\n")
print(classification_report(y_test, fitted[best_model_name].predict(X_test), digits=3))

# %% [markdown]
# ## 7. Save the models and a model card, then reload them from disk
# The reload step scores `test_data.csv` exactly the way the Streamlit app does, which
# proves the app will reproduce the table above.

# %%
for name, model in fitted.items():
    joblib.dump(model, MODEL_DIR / MODEL_REGISTRY[name], compress=3)

model_card = {
    "dataset": "UCI Steel Plates Faults (Buscema, Terzi & Tastle, 2010) — DOI 10.24432/C5J88N",
    "trained_at": datetime.now().isoformat(timespec="seconds"),
    "split": {"test_fraction": TEST_FRACTION, "random_state": SPLIT_SEED, "stratified": True,
              "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
    "cv": "StratifiedKFold(n_splits=5, shuffle=True), scoring=f1_macro",
    "features": FEATURE_COLUMNS,
    "target": TARGET_COLUMN,
    "classes": label_order,
    "metric_averaging": "macro",
    "models": {name: {"file": MODEL_REGISTRY[name], **tuning_log[name],
                      "test_metrics": {k: round(float(v), 4) for k, v in results[name].items()}}
               for name in fitted},
    "best_model_by_test_f1": best_model_name,
    "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "joblib": joblib.__version__},
}
(MODEL_DIR / "model_card.json").write_text(json.dumps(model_card, indent=2))

# reload check
reloaded_frame = pd.read_csv(ROOT / "test_data.csv")
for name, filename in MODEL_REGISTRY.items():
    model = joblib.load(MODEL_DIR / filename)
    again = score_predictions(reloaded_frame[TARGET_COLUMN],
                              model.predict(reloaded_frame[FEATURE_COLUMNS]),
                              model.predict_proba(reloaded_frame[FEATURE_COLUMNS]),
                              model.classes_)
    assert all(abs(again[m] - results[name][m]) < 1e-12 for m in METRIC_ORDER), name
    size_kb = (MODEL_DIR / filename).stat().st_size / 1024
    print(f"✔ {filename:<30} {size_kb:7.1f} KB  reloaded and re-scored identically")

# %% [markdown]
# ## 8. Keep README and requirements.txt in step with this run
# * The metrics table in `README.md` (between the `METRICS` markers) is regenerated.
# * The `scikit-learn` pin in `requirements.txt` is set to the version that produced the
#   `.joblib` files — a mismatch is the most common reason a deployed app fails to load models.

# %%
def markdown_table(table: pd.DataFrame) -> str:
    header = "| ML Model Name | " + " | ".join(METRIC_ORDER) + " |"
    divider = "|---|" + "|".join([":---:"] * len(METRIC_ORDER)) + "|"
    rows = []
    for name, row in table.iterrows():
        cells = []
        for metric in METRIC_ORDER:
            text = f"{row[metric]:.4f}"
            cells.append(f"**{text}**" if row[metric] == table[metric].max() else text)
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def refresh_readme(readme: Path, table_md: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    pattern = r"(<!-- METRICS:START -->).*?(<!-- METRICS:END -->)"
    if not re.search(pattern, text, flags=re.S):
        return False
    readme.write_text(re.sub(pattern, lambda m: f"{m.group(1)}\n{table_md}\n{m.group(2)}", text,
                             flags=re.S), encoding="utf-8")
    return True


def pin_requirement(req_file: Path, package: str, version: str) -> bool:
    if not req_file.exists():
        return False
    lines = req_file.read_text(encoding="utf-8").splitlines()
    changed = [f"{package}=={version}" if re.match(rf"^{re.escape(package)}\s*([=<>!~].*)?$", ln.strip())
               else ln for ln in lines]
    req_file.write_text("\n".join(changed) + "\n", encoding="utf-8")
    return changed != lines


table_md = markdown_table(metrics_table)
(ROOT / "reports" / "metrics_table.md").write_text(table_md + "\n", encoding="utf-8")
print(table_md)
print("\nREADME table refreshed:", refresh_readme(ROOT / "README.md", table_md))
print("requirements.txt scikit-learn pin changed:",
      pin_requirement(ROOT / "requirements.txt", "scikit-learn", sklearn.__version__))
print(f"\nRun finished : {datetime.now():%Y-%m-%d %H:%M:%S}")
