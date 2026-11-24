"""
Comprehensive Evaluation Metrics and High-Resolution Plotting Engine.
Optimized for multi-class intent classification and confidence calibration.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for scripts and headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# ----------------------------------------------------------------------
# Warm, Editorial Color Palette Constants
# ----------------------------------------------------------------------
WARM_CANVAS = "#F5F3EE"
WARM_CARD = "#FBFAF7"
WARM_TEXT_PRIMARY = "#2B2823"
WARM_TEXT_SECONDARY = "#6B655B"
WARM_BORDER = "#E4DFD5"
WARM_CLAY = "#C96442"
WARM_CLAY_DARK = "#8A3A20"
WARM_CLAY_TINT = "#F6E7E0"
WARM_SAGE = "#5E7D5B"
WARM_OCHRE = "#B7862B"
WARM_BRICK = "#A4453A"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["text.color"] = WARM_TEXT_PRIMARY
plt.rcParams["axes.labelcolor"] = WARM_TEXT_PRIMARY
plt.rcParams["xtick.color"] = WARM_TEXT_SECONDARY
plt.rcParams["ytick.color"] = WARM_TEXT_SECONDARY


def _apply_warm_theme(fig, axes):
    """Applies warm, editorial aesthetic styling to figure and all axes."""
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    fig.patch.set_facecolor(WARM_CARD)
    for ax in axes:
        ax.set_facecolor(WARM_CARD)
        for spine in ax.spines.values():
            spine.set_color(WARM_BORDER)
            spine.set_linewidth(1.0)
        ax.tick_params(colors=WARM_TEXT_SECONDARY, labelsize=9)
        ax.xaxis.label.set_color(WARM_TEXT_PRIMARY)
        ax.yaxis.label.set_color(WARM_TEXT_PRIMARY)
        ax.title.set_color(WARM_TEXT_PRIMARY)
        ax.grid(True, linestyle=":", color=WARM_BORDER, alpha=0.9, linewidth=0.8)


def calculate_top_k_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int = 3) -> float:
    """Computes Top-K accuracy given true label indices and probability matrix."""
    if y_probs is None or y_probs.ndim != 2:
        return 0.0
    num_classes = y_probs.shape[1]
    effective_k = min(k, num_classes)
    top_k_preds = np.argsort(y_probs, axis=1)[:, -effective_k:]
    hits = [y in preds for y, preds in zip(y_true, top_k_preds)]
    return float(np.mean(hits))


def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """
    Computes Expected Calibration Error (ECE) and bin-wise reliability statistics.
    ECE measures the difference between expected accuracy and confidence.
    """
    if y_probs is None:
        return 0.0, []

    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_confidence = float(np.mean(confidences[mask]))
            avg_accuracy = float(np.mean(accuracies[mask]))
            weight = bin_size / len(confidences)
            ece += weight * abs(avg_accuracy - avg_confidence)

            bin_stats.append({
                "bin_range": (bin_lower, bin_upper),
                "count": int(bin_size),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
            })
        else:
            bin_stats.append({
                "bin_range": (bin_lower, bin_upper),
                "count": 0,
                "avg_confidence": (bin_lower + bin_upper) / 2.0,
                "avg_accuracy": 0.0,
            })

    return float(ece), bin_stats


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray = None,
    class_names: list[str] = None,
) -> dict:
    """
    Computes an extensive set of multi-class classification and calibration metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    num_classes = len(class_names) if class_names else len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Top-K Accuracies
    top3_acc = calculate_top_k_accuracy(y_true, y_probs, k=3) if y_probs is not None else acc
    top5_acc = calculate_top_k_accuracy(y_true, y_probs, k=5) if y_probs is not None else acc

    # Calibration & Loss Metrics
    ece_score, bin_stats = (0.0, [])
    brier_score = 0.0
    cross_entropy = 0.0

    if y_probs is not None:
        ece_score, bin_stats = calculate_ece(y_true, y_probs, n_bins=10)
        try:
            # One-hot true labels for multi-class Brier score
            one_hot_true = np.zeros_like(y_probs)
            for idx, label in enumerate(y_true):
                if label < y_probs.shape[1]:
                    one_hot_true[idx, label] = 1.0
            brier_score = float(np.mean(np.sum((y_probs - one_hot_true) ** 2, axis=1)))
            cross_entropy = float(log_loss(y_true, y_probs, labels=list(range(num_classes))))
        except Exception:
            brier_score = 0.0
            cross_entropy = 0.0

    # Per-class F1 distribution
    _, _, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    class_performance = []
    for idx, name in enumerate(class_names):
        class_performance.append({
            "class_id": idx,
            "name": name,
            "f1": float(f1_per_class[idx]),
            "support": int(support_per_class[idx]),
        })

    class_performance_sorted = sorted(class_performance, key=lambda x: x["f1"])

    # Confusion matrix and most common confusion pairs
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    
    confusion_pairs = []
    for actual in range(num_classes):
        for predicted in range(num_classes):
            count = int(cm_no_diag[actual, predicted])
            if count > 0:
                confusion_pairs.append({
                    "actual": class_names[actual],
                    "predicted": class_names[predicted],
                    "count": count,
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    metrics = {
        "accuracy": acc,
        "top3_accuracy": float(top3_acc),
        "top5_accuracy": float(top5_acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "matthews_corrcoef": mcc,
        "expected_calibration_error": ece_score,
        "brier_score": brier_score,
        "cross_entropy": cross_entropy,
        "class_performance": class_performance,
        "hardest_classes": class_performance_sorted[:10],
        "best_classes": class_performance_sorted[-10:][::-1],
        "top_confusions": confusion_pairs[:10],
        "calibration_bins": bin_stats,
        "confusion_matrix": cm,
    }
    return metrics


# ----------------------------------------------------------------------
# High-Resolution Visualization Functions
# ----------------------------------------------------------------------

def plot_training_curves(trainer_history: list[dict], output_path: Path | str) -> str:
    """Plots training and validation loss, along with Macro-F1 across epochs."""
    epochs, train_losses, eval_losses, eval_f1s = [], [], [], []

    for entry in trainer_history:
        if "epoch" in entry:
            ep = entry["epoch"]
            if "loss" in entry:
                epochs.append(ep)
                train_losses.append(entry["loss"])
            if "eval_loss" in entry:
                eval_losses.append((ep, entry["eval_loss"]))
            if "eval_macro_f1" in entry:
                eval_f1s.append((ep, entry["eval_macro_f1"]))

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    _apply_warm_theme(fig, ax1)

    # Plot losses with warm editorial colors
    if eval_losses:
        ep_eval, ev_losses = zip(*eval_losses)
        ax1.plot(ep_eval, ev_losses, marker="o", markersize=6, color=WARM_CLAY, linewidth=2.2, label="Validation Loss")
    if epochs and train_losses:
        ax1.plot(epochs, train_losses, linestyle="--", color=WARM_TEXT_SECONDARY, alpha=0.85, linewidth=1.8, label="Training Loss")
    
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cross Entropy Loss", color=WARM_CLAY, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=WARM_CLAY)
    ax1.grid(True, linestyle=":", color=WARM_BORDER, alpha=0.85)

    # Plot Macro F1 on secondary axis
    if eval_f1s:
        ax2 = ax1.twinx()
        ep_f1, f1_vals = zip(*eval_f1s)
        ax2.plot(ep_f1, f1_vals, marker="s", markersize=6, color=WARM_SAGE, linewidth=2.5, label="Validation Macro-F1")
        ax2.set_ylabel("Macro-F1 Score", color=WARM_SAGE, fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=WARM_SAGE)
        ax2.set_ylim([0.0, 1.02])
        for spine in ax2.spines.values():
            spine.set_color(WARM_BORDER)

    # Unified clean legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if eval_f1s:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", facecolor=WARM_CARD, edgecolor=WARM_BORDER)
    else:
        ax1.legend(loc="upper right", facecolor=WARM_CARD, edgecolor=WARM_BORDER)

    plt.title("Model Training & Validation Trajectory", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WARM_CARD)
    plt.close(fig)
    return str(output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path | str, top_k: int = 20) -> str:
    """
    Plots a filtered confusion matrix focused on the top most frequent / confused classes
    using a warm clay gradient to match the editorial design system.
    """
    num_classes = len(class_names)
    if num_classes > top_k:
        row_sums = cm.sum(axis=1)
        top_indices = np.argsort(row_sums)[-top_k:]
        sub_cm = cm[np.ix_(top_indices, top_indices)]
        sub_names = [class_names[i] for i in top_indices]
        title_suffix = f"(Top {top_k} Most Frequent Classes out of {num_classes})"
    else:
        sub_cm = cm
        sub_names = class_names
        title_suffix = f"(All {num_classes} Classes)"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    _apply_warm_theme(fig, ax)

    row_sums = sub_cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = sub_cm.astype(float) / row_sums

    warm_clay_cmap = LinearSegmentedColormap.from_list(
        "warm_clay",
        [WARM_CARD, WARM_CLAY_TINT, "#E8A892", WARM_CLAY, WARM_CLAY_DARK]
    )

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=warm_clay_cmap,
        xticklabels=sub_names,
        yticklabels=sub_names,
        cbar_kws={"label": "Normalized Recall"},
        linewidths=0.5,
        linecolor=WARM_BORDER,
        ax=ax,
    )
    plt.title(f"Normalized Confusion Matrix Heatmap\n{title_suffix}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Intent", fontsize=10, fontweight="bold")
    plt.ylabel("Actual Intent", fontsize=10, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WARM_CARD)
    plt.close(fig)
    return str(output_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    ece: float,
    output_path: Path | str,
    n_bins: int = 10,
) -> str:
    """
    Generates a Reliability Diagram (empirical accuracy vs predicted confidence)
    along with confidence distribution histogram using the warm editorial palette.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            bin_accs.append(np.mean(accuracies[mask]))
            bin_confs.append(np.mean(confidences[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_centers[i])
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), dpi=300, sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    _apply_warm_theme(fig, [ax1, ax2])

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], linestyle="--", color=WARM_TEXT_SECONDARY, linewidth=1.5, label="Perfect Calibration")
    ax1.bar(
        bin_centers,
        bin_accs,
        width=1.0 / n_bins,
        alpha=0.75,
        color=WARM_CLAY,
        edgecolor=WARM_BORDER,
        linewidth=1.0,
        label=f"Outputs (ECE = {ece:.3f})",
    )
    ax1.scatter(bin_confs, bin_accs, color=WARM_CLAY_DARK, s=40, zorder=5)
    ax1.set_ylabel("Empirical Accuracy", fontsize=10, fontweight="bold")
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper left", facecolor=WARM_CARD, edgecolor=WARM_BORDER)
    ax1.set_title("Reliability Diagram & Expected Calibration Error (ECE)", fontsize=12, fontweight="bold")

    # Confidence distribution histogram
    ax2.hist(confidences, bins=bins, color=WARM_OCHRE, edgecolor=WARM_BORDER, alpha=0.8)
    ax2.set_xlabel("Confidence Bin", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax2.set_xlim([0.0, 1.0])

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WARM_CARD)
    plt.close(fig)
    return str(output_path)


def plot_class_performance(class_performance: list[dict], output_path: Path | str, top_n: int = 10) -> str:
    """Plots a comparative horizontal bar chart of hardest vs best performing classes by F1."""
    sorted_classes = sorted(class_performance, key=lambda x: x["f1"])
    hardest = sorted_classes[:top_n]
    best = sorted_classes[-top_n:][::-1]

    names = [c["name"] for c in hardest] + ["---"] + [c["name"] for c in best]
    scores = [c["f1"] for c in hardest] + [0.0] + [c["f1"] for c in best]
    colors = [WARM_BRICK] * top_n + [WARM_CARD] + [WARM_SAGE] * top_n

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    _apply_warm_theme(fig, ax)

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=colors, edgecolor=WARM_BORDER, linewidth=0.8, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("F1-Score", fontsize=10, fontweight="bold")
    ax.set_xlim([0.0, 1.05])
    ax.set_title(f"Class-Wise F1 Disparity: Top {top_n} Hardest vs Top {top_n} Best", fontsize=12, fontweight="bold")
    ax.axvline(0.9, color=WARM_CLAY, linestyle=":", linewidth=1.5, alpha=0.8, label="90% Benchmark")
    ax.legend(loc="lower right", facecolor=WARM_CARD, edgecolor=WARM_BORDER)

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WARM_CARD)
    plt.close(fig)
    return str(output_path)


def plot_metrics_summary(metrics: dict, output_path: Path | str) -> str:
    """Generates an executive metrics dashboard card displaying all key metrics in warm editorial tones."""
    labels = ["Top-1 Acc", "Top-3 Acc", "Top-5 Acc", "Macro-F1", "Weighted-F1", "MCC"]
    values = [
        metrics.get("accuracy", 0.0),
        metrics.get("top3_accuracy", 0.0),
        metrics.get("top5_accuracy", 0.0),
        metrics.get("macro_f1", 0.0),
        metrics.get("weighted_f1", 0.0),
        metrics.get("matthews_corrcoef", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    _apply_warm_theme(fig, ax)

    palette = [WARM_CLAY, "#D47C5E", WARM_OCHRE, WARM_SAGE, "#7A9A77", WARM_TEXT_SECONDARY]
    bars = ax.bar(labels, values, color=palette, edgecolor=WARM_BORDER, width=0.55, linewidth=1.0, alpha=0.92)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=WARM_TEXT_PRIMARY,
        )

    ax.set_ylim([0.0, 1.15])
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Executive Performance Metrics Summary\n"
        f"ECE: {metrics.get('expected_calibration_error', 0.0):.3f} | Brier Score: {metrics.get('brier_score', 0.0):.3f}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WARM_CARD)
    plt.close(fig)
    return str(output_path)
    plt.close(fig)
    return str(output_path)
