#!/usr/bin/env python3
"""Text classification baseline: TF-IDF or Doc2Vec features and a linear classifier.

A reusable version of temp/pre-classification.py that works on any table with a
text column and a label column. Import it (as the notebooks do) or run it from
the command line for a quick feasibility check.

- stratified_holdout() sets a test set aside with the same class proportions as the data.
- balanced_sample() draws a training set with the classes as equal as the data allows.
- make_pipeline() builds TF-IDF + logistic regression; any scikit-learn classifier
  can be swapped in.
- make_doc2vec_pipeline() builds the same with Doc2Vec document vectors instead of
  TF-IDF (needs gensim).
- evaluate() trains a pipeline and reports precision, recall, F1 and a confusion matrix.
- top_features() lists the words a fitted TF-IDF model weighs most for each class.
- plot_confusion_matrix(), plot_model_comparison() and plot_macro_f1() draw the
  results with matplotlib.

Usage:
    python classification.py data/processed/narratives_preprocessed.csv --text lemmas \\
        --stop-words none --test-size 0.1 --train-size 20000
    python classification.py data/processed/narratives_preprocessed.csv --text lemmas \\
        --features doc2vec --test-size 0.1 --train-size 20000
    python classification.py temp/asrs_selected_features.pkl --text Combined_Narrative --top-k 4
"""

import argparse
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature settings of the original feasibility baseline.
DEFAULT_TFIDF = {"max_features": 3000, "ngram_range": (1, 2)}


def top_classes(labels: pd.Series, k: int) -> list[str]:
    """The k most frequent labels, most frequent first."""
    return labels.value_counts().nlargest(k).index.tolist()


def stratified_holdout(df: pd.DataFrame, label: str, test_size: float = 0.2,
                       random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into (rest, test), keeping the class proportions in both."""
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label])


def balanced_sample(df: pd.DataFrame, label: str, n: int, random_state: int = 42) -> pd.DataFrame:
    """Draw `n` rows with the classes as equal as the data allows.

    Every class gets an equal share. A class with fewer rows than its share
    contributes all of them, and the shortfall is split evenly among the others.
    """
    counts = df[label].value_counts().sort_values()
    if n > counts.sum():
        raise ValueError(f"Asked for {n:,} rows but only {counts.sum():,} are available")
    quotas, remaining = {}, n
    for i, (cls, available) in enumerate(counts.items()):
        quotas[cls] = min(available, remaining // (len(counts) - i))
        remaining -= quotas[cls]
    parts = [df[df[label] == cls].sample(quota, random_state=random_state) for cls, quota in quotas.items()]
    return pd.concat(parts).sample(frac=1, random_state=random_state)


def default_classifier() -> LogisticRegression:
    """Logistic regression with balanced class weights, as in the original baseline."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def make_pipeline(classifier=None, stop_words: str | None = "english", **tfidf_params) -> Pipeline:
    """TF-IDF features followed by `classifier`.

    The defaults match the original baseline: up to 3,000 unigrams and bigrams,
    scikit-learn's English stop words removed, and logistic regression with
    balanced class weights. Pass stop_words=None for text that is already
    preprocessed; scikit-learn's list also removes words like "not", "no",
    "up" and "down". Extra keyword arguments go to TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(stop_words=stop_words, **{**DEFAULT_TFIDF, **tfidf_params})
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier or default_classifier())])


def simple_tokens(text: str) -> list[str]:
    """Lowercase words and numbers; punctuation is dropped."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Doc2VecVectorizer(BaseEstimator, TransformerMixin):
    """Turns texts into Doc2Vec document vectors; a drop-in alternative to TfidfVectorizer.

    fit() trains a gensim Doc2Vec model on the texts. transform() infers a
    vector for each text, for training texts too, so that training and test
    vectors are produced the same way. The default is PV-DBOW (dm=0), which is
    fast and usually works best for classifying documents.
    """

    def __init__(self, vector_size: int = 100, dm: int = 0, window: int = 5, min_count: int = 2,
                 epochs: int = 20, workers: int = 4, seed: int = 42, tokenizer=None):
        self.vector_size = vector_size
        self.dm = dm
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.seed = seed
        self.tokenizer = tokenizer

    def _tokens(self, text: str) -> list[str]:
        return (self.tokenizer or simple_tokens)(text)

    def fit(self, texts, y=None):
        # Imported here so the TF-IDF part of this module works without gensim installed.
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        documents = [TaggedDocument(self._tokens(text), [i]) for i, text in enumerate(texts)]
        self.model_ = Doc2Vec(documents, vector_size=self.vector_size, dm=self.dm, window=self.window,
                              min_count=self.min_count, epochs=self.epochs, workers=self.workers,
                              seed=self.seed)
        return self

    def transform(self, texts) -> np.ndarray:
        return np.vstack([self.model_.infer_vector(self._tokens(text), epochs=self.epochs)
                          for text in texts])


def make_doc2vec_pipeline(classifier=None, **doc2vec_params) -> Pipeline:
    """Doc2Vec document vectors, standardized, followed by `classifier`.

    Keyword arguments go to Doc2VecVectorizer (vector_size, dm, epochs, ...).
    The classifier defaults to the same logistic regression as the TF-IDF baseline.
    """
    return Pipeline([("doc2vec", Doc2VecVectorizer(**doc2vec_params)),
                     ("scale", StandardScaler()),
                     ("classifier", classifier or default_classifier())])


@dataclass
class Evaluation:
    """Predictions on a test set, with the usual metrics."""

    labels: list[str]
    y_true: pd.Series
    y_pred: pd.Series

    @property
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    @property
    def macro_f1(self) -> float:
        return f1_score(self.y_true, self.y_pred, labels=self.labels, average="macro", zero_division=0)

    def per_class(self) -> pd.DataFrame:
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.labels, zero_division=0)
        return pd.DataFrame({"precision": precision, "recall": recall, "f1": f1, "support": support},
                            index=self.labels)

    def confusion(self) -> pd.DataFrame:
        matrix = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        return pd.DataFrame(matrix, index=[f"True: {c}" for c in self.labels],
                            columns=[f"Pred: {c}" for c in self.labels])

    def report(self) -> str:
        """scikit-learn's classification report: precision, recall and F1 per class."""
        return classification_report(self.y_true, self.y_pred, labels=self.labels, digits=3,
                                     zero_division=0)

    def __str__(self) -> str:
        return (f"{self.report()}\nConfusion matrix (rows: true class, columns: predicted):\n"
                f"{self.confusion().to_string()}")


def evaluate(train: pd.DataFrame, test: pd.DataFrame, text: str, label: str,
             pipeline: Pipeline | None = None,
             labels: list[str] | None = None) -> tuple[Pipeline, Evaluation]:
    """Fit `pipeline` on `train` and predict `test`.

    Returns the fitted pipeline and its Evaluation. Classes are reported in
    `labels` order, by default most frequent in the test set first.
    """
    pipeline = pipeline if pipeline is not None else make_pipeline()
    pipeline.fit(train[text], train[label])
    predictions = pd.Series(pipeline.predict(test[text]), index=test.index, name="predicted")
    labels = labels or test[label].value_counts().index.tolist()
    return pipeline, Evaluation(labels, test[label], predictions)


def top_features(pipeline: Pipeline, n: int = 15) -> pd.DataFrame:
    """The n highest-weighted features for each class of a fitted TF-IDF pipeline."""
    if "tfidf" not in pipeline.named_steps:
        raise ValueError("top_features needs a TF-IDF pipeline: Doc2Vec dimensions aren't words")
    vocabulary = pipeline.named_steps["tfidf"].get_feature_names_out()
    model = pipeline.named_steps["classifier"]
    return pd.DataFrame({cls: vocabulary[weights.argsort()[::-1][:n]]
                         for cls, weights in zip(model.classes_, model.coef_)},
                        index=range(1, n + 1))


# --- Charts ---------------------------------------------------------------------
# Light theme from the reference palette. Series take these colors in this fixed
# order (never cycled); the heatmap uses one hue, light (low) to dark (high).
SURFACE = "#fcfcfb"
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def _luminance(color) -> float:
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink_on(fill) -> str:
    """White or near-black text, whichever contrasts more with `fill`."""
    fill_lum = _luminance(fill)
    on_white = 1.05 / (fill_lum + 0.05)
    on_dark = (fill_lum + 0.05) / (_luminance(TEXT_PRIMARY) + 0.05)
    return "#ffffff" if on_white >= on_dark else TEXT_PRIMARY


def _style(ax, grid_axis: str | None = "x") -> None:
    """Recessive chrome: no box, no tick marks, hairline grid behind the data."""
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=TEXT_MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def plot_confusion_matrix(evaluation: Evaluation, ax=None, title: str | None = None):
    """Confusion matrix as a heatmap; returns the axes.

    Each row shows how the reports of one true class were classified, as shares
    of that class, so a rare class reads as clearly as a common one. The report
    count is printed under each share.
    """
    counts = evaluation.confusion().to_numpy()
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    n = len(evaluation.labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.3 * n + 2.4, 1.1 * n + 1.6))
    ax.figure.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis=None)

    # Cell edges in the surface color leave a 2px gap between cells.
    ax.pcolormesh(shares, cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)  # first class on top
    for i in range(n):
        for j in range(n):
            ink = _ink_on(SEQUENTIAL(shares[i, j]))
            ax.text(j + 0.5, i + 0.42, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=11, color=ink)
            ax.text(j + 0.5, i + 0.70, f"{counts[i, j]:,}", ha="center", va="center", fontsize=8, color=ink)

    centers = np.arange(n) + 0.5
    ax.set_xticks(centers, [textwrap.fill(label, 12) for label in evaluation.labels])
    ax.set_yticks(centers, [textwrap.fill(label, 14) for label in evaluation.labels])
    ax.tick_params(labelcolor=TEXT_SECONDARY, labelsize=9)
    ax.set_xlabel("Predicted class", color=TEXT_MUTED, fontsize=9, labelpad=8)
    ax.set_ylabel("True class", color=TEXT_MUTED, fontsize=9, labelpad=8)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, pad=24)
    ax.text(0, 1.02, "Share of each true class (row), with report counts", transform=ax.transAxes,
            fontsize=9, color=TEXT_SECONDARY, va="bottom")
    return ax


def plot_model_comparison(evaluations: dict[str, Evaluation],
                          metrics: tuple[str, ...] = ("precision", "recall", "f1"),
                          title: str | None = None):
    """Per-class scores of two or three models; returns the figure.

    One panel per metric, one row per class plus the macro average, and a dot
    per model, joined by a line so the gap between the models stands out.
    """
    names = list(evaluations)
    if not 2 <= len(names) <= len(SERIES_COLORS):
        raise ValueError(f"Compare two to {len(SERIES_COLORS)} models, got {len(names)}")
    labels = evaluations[names[0]].labels
    rows = labels + ["Macro average"]
    values = {}
    for name, evaluation in evaluations.items():
        table = evaluation.per_class().reindex(labels)
        values[name] = {m: list(table[m]) + [table[m].mean()] for m in metrics}
    everything = [v for name in names for m in metrics for v in values[name][m]]
    low, high = max(0.0, min(everything) - 0.05), min(1.0, max(everything) + 0.05)

    fig, axes = plt.subplots(1, len(metrics), sharey=True, sharex=True,
                             figsize=(2.9 * len(metrics) + 1.9, 0.5 * len(rows) + 1.8))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    for ax, metric in zip(axes, metrics):
        _style(ax, grid_axis="x")
        for k in y:
            xs = [values[name][metric][k] for name in names]
            ax.plot([min(xs), max(xs)], [k, k], color=BASELINE, linewidth=2, solid_capstyle="round", zorder=1)
        for name, color in zip(names, SERIES_COLORS):
            ax.scatter(values[name][metric], y, s=70, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.axhline(len(labels) - 0.5, color=GRID, linewidth=0.8)  # sets the average apart
        ax.set_title("F1" if metric == "f1" else metric.capitalize(), loc="left", fontsize=11,
                     color=TEXT_PRIMARY)
        ax.set_xlim(low, high)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    axes[0].set_yticks(y, rows)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].tick_params(axis="y", labelcolor=TEXT_SECONDARY, labelsize=10)

    legend = [Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=color,
                     markeredgecolor=SURFACE, markeredgewidth=2, label=name)
              for name, color in zip(names, SERIES_COLORS)]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(1, 1), ncol=len(names), frameon=False,
               fontsize=10, labelcolor=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()  # also leaves room for the title
    return fig


def plot_macro_f1(evaluations: dict[str, Evaluation], groups: dict[str, str] | None = None,
                  title: str | None = None):
    """Macro F1 of several models as dots on one axis, colored by group; returns the figure.

    `groups` maps each model name to a group (e.g. its feature type); groups
    take the series colors in order of first appearance. The best score is labeled.
    """
    names = list(evaluations)
    groups = groups or {name: name for name in names}
    group_names = list(dict.fromkeys(groups[name] for name in names))
    if len(group_names) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} groups, got {len(group_names)}")
    color = dict(zip(group_names, SERIES_COLORS))
    scores = np.array([evaluations[name].macro_f1 for name in names])

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.7))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid_axis="x")
    y = np.arange(len(names))
    ax.scatter(scores, y, s=80, color=[color[groups[name]] for name in names], edgecolors=SURFACE,
               linewidths=2, zorder=3)
    best = int(scores.argmax())
    ax.annotate(f"{scores[best]:.3f}", xy=(scores[best], best), xytext=(9, 0), textcoords="offset points",
                va="center", fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(y, names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, labelsize=10)
    ax.set_xlim(max(0.0, scores.min() - 0.04), min(1.0, scores.max() + 0.04))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("Macro F1 on the test set", color=TEXT_MUTED, fontsize=9, labelpad=8)

    legend = [Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=color[g],
                     markeredgecolor=SURFACE, markeredgewidth=2, label=g) for g in group_names]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(1, 1), ncol=len(group_names),
               frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, color=TEXT_PRIMARY)
    fig.tight_layout()  # also leaves room for the title
    return fig


def load_table(path: Path, header_row: int = 0) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, header=header_row, dtype=str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="CSV or pickle file")
    parser.add_argument("--text", required=True, help="column with the text to classify")
    parser.add_argument("--label", default="Primary Problem", help="column with the class labels")
    parser.add_argument("--features", choices=["tfidf", "doc2vec"], default="tfidf")
    parser.add_argument("--top-k", type=int, help="keep only the k most frequent labels")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of each class held out")
    parser.add_argument("--train-size", type=int,
                        help="draw a class-balanced training set of this many rows from the rest")
    parser.add_argument("--stop-words", default="english",
                        help='TF-IDF only: "english" (default) or "none" for already preprocessed text')
    parser.add_argument("--max-features", type=int, default=DEFAULT_TFIDF["max_features"],
                        help="TF-IDF only")
    parser.add_argument("--vector-size", type=int, default=100, help="Doc2Vec only")
    parser.add_argument("--header-row", type=int, default=0,
                        help="row holding the column names (1 for raw ASRS exports)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_table(args.input, args.header_row)
    missing = {args.text, args.label} - set(df.columns)
    if missing:
        sys.exit(f"Column(s) not found in {args.input}: {', '.join(sorted(missing))}")
    df = df.dropna(subset=[args.text, args.label])
    if args.top_k:
        df = df[df[args.label].isin(top_classes(df[args.label], args.top_k))]

    rest, test = stratified_holdout(df, args.label, args.test_size, args.seed)
    train = balanced_sample(rest, args.label, args.train_size, args.seed) if args.train_size else rest
    print(pd.DataFrame({"train": train[args.label].value_counts(),
                        "test": test[args.label].value_counts()}).to_string(), end="\n\n")

    if args.features == "doc2vec":
        pipeline = make_doc2vec_pipeline(vector_size=args.vector_size, seed=args.seed)
    else:
        stop_words = None if args.stop_words.lower() == "none" else args.stop_words
        pipeline = make_pipeline(stop_words=stop_words, max_features=args.max_features)
    _, result = evaluate(train, test, args.text, args.label, pipeline)
    print(f"Accuracy {result.accuracy:.3f}, macro F1 {result.macro_f1:.3f}\n")
    print(result)


if __name__ == "__main__":
    main()
