"""
Steel Plate Fault Classifier — Streamlit front end
BITS Pilani · Machine Learning · Assignment 2

Loads the five pre-trained scikit-learn pipelines saved in model/ and scores an
uploaded (or the bundled) test CSV. Nothing is retrained in this app.
Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from fault_toolkit import (
    FAULT_CLASSES, FAULT_NOTES, FEATURE_COLUMNS, METRIC_ORDER, MODEL_DIR_NAME,
    MODEL_REGISTRY, TARGET_COLUMN, one_hot_to_label, score_predictions,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / MODEL_DIR_NAME
SAMPLE_CSV = APP_DIR / "test_data.csv"
FULL_CSV = APP_DIR / "data" / "steel_plates_faults.csv"
UCI_URL = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"

# validated categorical palette (one fixed colour per model) + chart ink
MODEL_COLOURS = dict(zip(MODEL_REGISTRY, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED, GRID, SURFACE = "#1c2430", "#52514e", "#e1e0d9", "#fcfcfb"

METRIC_HELP = {
    "Accuracy": "Share of plates whose fault type was predicted correctly.",
    "AUC": "One-vs-rest ROC-AUC averaged over the fault types present: how well the "
           "predicted probabilities rank each class above the others.",
    "Precision": "Of the plates predicted as a fault type, how many really were that type.",
    "Recall": "Of the plates that really were a fault type, how many the model found.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient: a balanced score from -1 to 1 that stays "
           "honest when classes are imbalanced.",
}

st.set_page_config(page_title="Steel Plate Fault Classifier", page_icon="🔩",
                   layout="wide", initial_sidebar_state="auto")

st.markdown(
    """
    <style>
      .hero {background: linear-gradient(118deg, #17202c 0%, #26344a 55%, #3a4d66 100%);
             border-left: 6px solid #d9622b; border-radius: 14px;
             padding: 1.3rem 1.6rem 1.1rem; margin-bottom: 1.1rem;}
      .hero h1 {color: #ffffff; font-size: 1.85rem; margin: 0 0 .35rem 0; padding: 0;}
      .hero p  {color: #c9d2de; margin: 0; font-size: .97rem; max-width: 60rem;}
      .hero .chips span {display: inline-block; margin: .75rem .4rem 0 0; padding: .14rem .65rem;
             border-radius: 999px; font-size: .78rem; color: #e8edf3;
             background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2);}
      .step {font-weight: 700; color: #d9622b; letter-spacing: .02em; margin: .9rem 0 .2rem 0;}
      div[data-testid="stMetricValue"] {font-size: 1.75rem;}
      section[data-testid="stSidebar"][aria-expanded="true"] {min-width: 320px; max-width: 320px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the five saved models…")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_REGISTRY.items()}


@st.cache_data
def load_model_card() -> dict:
    path = MODEL_DIR / "model_card.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data
def read_table(raw: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.shape[1] == 1:  # probably ';' or tab separated
        frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    return frame


def _row_keys(frame: pd.DataFrame) -> list[tuple]:
    return list(map(tuple, frame[FEATURE_COLUMNS].astype(float).round(6).to_numpy()))


@st.cache_data
def training_row_keys() -> set:
    """Feature fingerprints of the rows the models were trained on
    (full dataset minus the held-out test split)."""
    if not (FULL_CSV.exists() and SAMPLE_CSV.exists()):
        return set()
    return set(_row_keys(pd.read_csv(FULL_CSV))) - set(_row_keys(pd.read_csv(SAMPLE_CSV)))


@st.cache_data(show_spinner="Scoring the plates with all five models…")
def score_all_models(features: pd.DataFrame) -> dict:
    out = {}
    for name, model in load_models().items():
        out[name] = {"pred": model.predict(features),
                     "proba": model.predict_proba(features),
                     "classes": list(model.classes_)}
    return out


def prepare_upload(frame: pd.DataFrame) -> dict:
    """Validate an uploaded frame and split it into features / labels."""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = one_hot_to_label(frame)
    by_lower = {c.lower(): c for c in frame.columns}
    renames = {by_lower[c.lower()]: c for c in FEATURE_COLUMNS + [TARGET_COLUMN]
               if c not in frame.columns and c.lower() in by_lower}
    frame = frame.rename(columns=renames)

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        return {"error": missing}

    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    notes, keep = [], features.notna().all(axis=1)
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} row(s) had blank or non-numeric feature values and were skipped.")

    labels = None
    if TARGET_COLUMN in frame.columns:
        labels = frame[TARGET_COLUMN].astype(str).str.strip()
        unknown = ~labels.isin(FAULT_CLASSES)
        if (unknown & keep).any():
            bad = sorted(labels[unknown & keep].unique())[:5]
            notes.append(f"{int((unknown & keep).sum())} row(s) had an unknown {TARGET_COLUMN} "
                         f"({', '.join(bad)}) and were skipped.")
        keep &= ~unknown
        labels = labels[keep].reset_index(drop=True)

    extra = [c for c in frame.columns if c not in FEATURE_COLUMNS + [TARGET_COLUMN]]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {', '.join(extra[:6])}"
                     + ("…" if len(extra) > 6 else ""))
    return {"X": features[keep].reset_index(drop=True), "y": labels, "notes": notes,
            "row_ids": np.flatnonzero(keep.to_numpy()) + 1}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def confusion_figure(y_true, y_pred, labels, normalise: bool):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor=SURFACE)
    sns.heatmap(shown, annot=True, fmt=".2f" if normalise else "d", cmap="Blues",
                cbar=False, square=True, linewidths=0.6, linecolor="white",
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"fontsize": 9},
                vmin=0, vmax=1 if normalise else None)
    ax.set_xlabel("Predicted fault type", color=INK)
    ax.set_ylabel("Actual fault type", color=INK)
    ax.tick_params(axis="x", rotation=40, labelsize=9, colors=MUTED)
    ax.tick_params(axis="y", rotation=0, labelsize=9, colors=MUTED)
    fig.tight_layout()
    return fig, cm


def comparison_figure(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 3.9), facecolor=SURFACE)
    style_axes(ax)
    positions = np.arange(len(METRIC_ORDER))
    width = 0.8 / len(table)
    for i, (name, row) in enumerate(table.iterrows()):
        ax.bar(positions + (i - (len(table) - 1) / 2) * width, row[METRIC_ORDER].values,
               width * 0.88, color=MODEL_COLOURS[name], label=name)
    ax.set_xticks(positions, METRIC_ORDER)
    low = float(np.nanmin(table[METRIC_ORDER].to_numpy()))
    ax.set_ylim(max(0.0, np.floor(low * 10) / 10 - 0.1), 1.0)
    ax.set_ylabel("Score", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(table), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.tight_layout()
    return fig


def fmt(value: float) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.4f}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
card = load_model_card()
best_on_test = card.get("best_model_by_test_f1", "Support Vector Machine")

with st.sidebar:
    st.markdown("## 🔩 Fault Inspector")
    st.caption("Steel plate surface-fault classifier · BITS ML Assignment 2")

    st.markdown('<div class="step">① TEST DATA</div>', unsafe_allow_html=True)
    upload = st.file_uploader(
        "Upload a test CSV", type=["csv"],
        help=f"Needs the 27 feature columns and a `{TARGET_COLUMN}` column "
             "(or the 7 original one-hot fault columns). Upload held-out test data only.")
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test_data.csv", data=SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch",
                           icon=":material/download:")

    st.markdown('<div class="step">② MODEL</div>', unsafe_allow_html=True)
    model_names = list(MODEL_REGISTRY)
    model_name = st.selectbox(
        "Classifier", model_names,
        index=model_names.index(best_on_test) if best_on_test in model_names else 0,
        format_func=lambda n: f"{n}  ★" if n == best_on_test else n,
        help="★ marks the model with the best macro-F1 on the held-out test split.")

    st.markdown('<div class="step">③ AVERAGING</div>', unsafe_allow_html=True)
    averaging = st.radio(
        "Precision · Recall · F1 · AUC", ["macro", "weighted"], horizontal=True,
        help="macro = every fault type counts equally (used in the README). "
             "weighted = each fault type counts by its number of plates.")

    st.divider()
    st.caption(f"Models are pre-trained pipelines loaded from `{MODEL_DIR_NAME}/`; "
               "uploading data never retrains them.  \n"
               f"Data: [UCI Steel Plates Faults]({UCI_URL}) · CC BY 4.0")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <h1>Steel Plate Fault Classifier</h1>
      <p>Seven kinds of surface fault, 27 geometric and luminosity measurements per plate,
         and five pre-trained classifiers. Pick a model and score held-out plates to see
         how well it tells the faults apart.</p>
      <div class="chips"><span>UCI dataset #198</span><span>1,941 plates</span>
        <span>27 features</span><span>7 classes</span><span>5 models</span>
        <span>macro / weighted metrics</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    load_models()
except Exception as exc:  # most often a scikit-learn version mismatch
    trained_with = card.get("library_versions", {}).get("scikit-learn", "?")
    st.error(f"Could not load the saved models ({type(exc).__name__}: {exc}). They were saved "
             f"with scikit-learn {trained_with}; this server runs {sklearn.__version__}. "
             "Pin the same version in requirements.txt.")
    st.stop()

trained_with = card.get("library_versions", {}).get("scikit-learn")
if trained_with and trained_with != sklearn.__version__:
    st.warning(f"Models were saved with scikit-learn {trained_with} but this server runs "
               f"{sklearn.__version__}. Results may differ slightly.")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
if upload is not None:
    source_label, raw_bytes = f"uploaded file **{upload.name}**", upload.getvalue()
elif SAMPLE_CSV.exists():
    source_label, raw_bytes = "bundled **test_data.csv** (held-out 20 % split)", SAMPLE_CSV.read_bytes()
else:
    st.info("Upload a test CSV in the sidebar to begin.")
    st.stop()

try:
    uploaded_frame = read_table(raw_bytes)
except Exception as exc:
    st.error(f"That file could not be read as CSV: {exc}")
    st.stop()

prepared = prepare_upload(uploaded_frame)
if "error" in prepared:
    st.error("The file is missing required feature column(s): " + ", ".join(prepared["error"]))
    with st.expander("Expected columns"):
        st.code(", ".join(FEATURE_COLUMNS + [TARGET_COLUMN]))
    st.stop()

X, y = prepared["X"], prepared["y"]
if X.empty:
    st.error("No usable rows were left after validation.")
    st.stop()

has_labels = y is not None
info = (f"Scoring {source_label} · **{len(X):,}** plates · "
        + (f"**{y.nunique()}** fault types present" if has_labels else "no labels"))
if upload is None:
    info += " — upload your own file in the sidebar to replace it."
st.info(info, icon=":material/dataset:")
for note in prepared["notes"]:
    st.caption("⚠ " + note)

train_keys = training_row_keys()
seen_in_training = sum(k in train_keys for k in _row_keys(X))
if seen_in_training:
    st.warning(f"{seen_in_training} of {len(X)} rows were part of the training split, so the "
               "metrics on this file are optimistic. Upload held-out test data only.")

with st.expander("Preview the data"):
    left, right = st.columns([3, 1])
    left.dataframe(X.join(y) if has_labels else X, height=260, width="stretch")
    if has_labels:
        right.dataframe(y.value_counts().rename_axis("Fault type").rename("Plates"), width="stretch")

if not has_labels:
    st.warning(f"No `{TARGET_COLUMN}` column found, so metrics cannot be computed. "
               "Showing predictions only.")

scored = score_all_models(X)
current = scored[model_name]

all_metrics = None
if has_labels:
    all_metrics = pd.DataFrame(
        {name: score_predictions(y, s["pred"], s["proba"], s["classes"], average=averaging)
         for name, s in scored.items()}).T[METRIC_ORDER]
    all_metrics.index.name = "Model"

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_metrics, tab_cm, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Evaluation metrics", "🧩 Confusion matrix & report", "🏁 Compare all models",
     "🔍 Predictions", "ℹ️ About"])

with tab_metrics:
    st.subheader(f"{model_name}")
    if has_labels:
        mine, average_row = all_metrics.loc[model_name], all_metrics.mean()
        cols = st.columns(6)
        for col, metric in zip(cols, METRIC_ORDER):
            diff = mine[metric] - average_row[metric]
            col.metric(metric, fmt(mine[metric]),
                       delta=None if np.isnan(diff) else f"{diff:+.3f} vs avg",
                       help=METRIC_HELP[metric], border=True)
        st.caption(f"Precision, recall, F1 and AUC use **{averaging}** averaging over the fault "
                   "types. The delta compares this model with the mean of all five models.")
    else:
        st.info("Add a `Fault_Type` column to see metrics.")

    model_info = card.get("models", {}).get(model_name, {})
    steps = " → ".join(type(step).__name__ for _, step in load_models()[model_name].steps)
    with st.container(border=True):
        a, b, c = st.columns([2, 3, 1.2])
        a.markdown(f"**Pipeline**  \n`{steps}`")
        params = model_info.get("best_params", {})
        b.markdown("**Tuned hyper-parameters**  \n" +
                   (", ".join(f"`{k}={v}`" for k, v in params.items()) or "—"))
        c.markdown(f"**CV macro-F1**  \n{model_info.get('cv_macro_f1', '—')}")

with tab_cm:
    if has_labels:
        labels = sorted(set(y) | set(current["pred"]))
        normalise = st.toggle("Row-normalise (show recall per actual class)", value=False)
        left, right = st.columns([1.05, 1])
        with left:
            fig, cm = confusion_figure(y, current["pred"], labels, normalise)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with right:
            st.markdown("**Classification report**")
            report = pd.DataFrame(classification_report(
                y, current["pred"], output_dict=True, zero_division=0)).T
            report = report.drop(index="accuracy", errors="ignore")
            report["support"] = report["support"].astype(int)
            report.index.name = "Class"
            st.dataframe(
                report, width="stretch",
                column_config={**{k: st.column_config.ProgressColumn(k, min_value=0.0, max_value=1.0,
                                                                      format="%.3f", width="small")
                                  for k in ("precision", "recall", "f1-score")},
                               "support": st.column_config.NumberColumn("support", width="small")})
            off_diag = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
                        for j in range(len(labels)) if i != j and cm[i, j] > 0]
            if off_diag:
                st.markdown("**Most frequent mix-ups**")
                for count, actual, predicted in sorted(off_diag, reverse=True)[:3]:
                    st.markdown(f"- `{actual}` predicted as `{predicted}` — **{count}** plates")
    else:
        st.info("A confusion matrix needs the true `Fault_Type` labels.")

with tab_compare:
    if has_labels:
        winner = all_metrics["F1"].idxmax()
        st.success(f"Best on this data: **{winner}** with F1 {all_metrics.loc[winner, 'F1']:.4f} "
                   f"and MCC {all_metrics.loc[winner, 'MCC']:.4f} ({averaging} averaging).",
                   icon=":material/trophy:")
        shown = all_metrics.copy()
        shown.insert(0, "F1 rank", shown["F1"].rank(ascending=False, method="min").astype(int))
        st.dataframe(
            shown, width="stretch",
            column_config={"F1 rank": st.column_config.NumberColumn("F1 rank", width="small"),
                           **{m: st.column_config.ProgressColumn(m, min_value=0.0, max_value=1.0,
                                                                 format="%.4f", width="small")
                              for m in METRIC_ORDER}})
        fig = comparison_figure(all_metrics)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.download_button("Download this comparison (CSV)", all_metrics.round(4).to_csv().encode(),
                           file_name=f"model_comparison_{averaging}.csv", mime="text/csv",
                           icon=":material/table:")
    else:
        st.info("Model comparison needs the true `Fault_Type` labels.")

with tab_pred:
    proba = current["proba"]
    order = np.argsort(proba, axis=1)
    classes = np.array(current["classes"])
    table = pd.DataFrame({
        "Row": prepared["row_ids"],
        "Predicted": current["pred"],
        "Confidence": proba.max(axis=1),
        "Runner-up": classes[order[:, -2]],
    })
    if has_labels:
        table.insert(1, "Actual", y.to_numpy())
        table["Correct"] = np.where(table["Actual"] == table["Predicted"], "✅", "❌")
        n_wrong = int((table["Correct"] == "❌").sum())
        only_wrong = st.toggle(f"Show only the {n_wrong} misclassified plates", value=False,
                               key="only_wrong")
        if only_wrong:
            table = table[table["Correct"] == "❌"]
    st.dataframe(table, hide_index=True, width="stretch", height=420,
                 column_config={"Confidence": st.column_config.ProgressColumn(
                     "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})
    export = X.copy()
    if has_labels:
        export[TARGET_COLUMN] = y.to_numpy()
    export["Predicted"], export["Confidence"] = current["pred"], proba.max(axis=1).round(4)
    st.download_button("Download predictions (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_REGISTRY[model_name].split('.')[0]}.csv",
                       mime="text/csv", icon=":material/download:")

with tab_about:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            f"""
**Dataset.** [Steel Plates Faults]({UCI_URL}) from the UCI Machine Learning Repository
(Buscema, Terzi & Tastle, 2010 · DOI 10.24432/C5J88N · CC BY 4.0). Each of the 1,941
rows is one fault region on a steel plate, described by 27 measurements: its bounding
box and position, area and perimeters, luminosity statistics, steel type and plate
thickness, and a set of derived shape indices. The task is to name the fault type.

**Training.** 80 / 20 stratified split (seed 1729). Each model is a scikit-learn
pipeline tuned by 5-fold stratified grid search on the training split only, scored by
macro-F1. The 389 held-out rows are `test_data.csv`.
            """)
        counts = y.value_counts() if has_labels else pd.Series(dtype=int)
        st.dataframe(pd.DataFrame({
            "Fault type": FAULT_CLASSES,
            "What it is": [FAULT_NOTES[c] for c in FAULT_CLASSES],
            "Plates in this file": [int(counts.get(c, 0)) for c in FAULT_CLASSES],
        }), hide_index=True, width="stretch")
    with right:
        versions = card.get("library_versions", {})
        st.markdown("**Run details**")
        st.dataframe(pd.DataFrame({
            "Item": ["Models trained at", "Train / test rows", "scikit-learn (training)",
                     "scikit-learn (this server)", "Python (training)"],
            "Value": [card.get("trained_at", "—"),
                      f"{card.get('split', {}).get('train_rows', '—')} / {card.get('split', {}).get('test_rows', '—')}",
                      versions.get("scikit-learn", "—"), sklearn.__version__,
                      versions.get("python", "—")],
        }), hide_index=True, width="stretch")
        st.markdown("**Test-split scores recorded at training time (macro)**")
        recorded = pd.DataFrame({n: m.get("test_metrics", {}) for n, m in card.get("models", {}).items()}).T
        if not recorded.empty:
            st.dataframe(recorded[METRIC_ORDER], width="stretch",
                         column_config={m: st.column_config.NumberColumn(m, format="%.3f", width="small")
                                        for m in METRIC_ORDER})
