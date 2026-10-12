#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr


FACTORS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

MODEL_DISPLAY_NAMES = {
    "google/gemma-3-1b-it": "Gemma3-1B",
    "google/gemma-3-4b-it": "Gemma3-4B",
    "google/gemma-3-12b-it": "Gemma3-12B",
    "google/gemma-3-27b-it": "Gemma3-27B",
}

RANK_COLORS = {
    1: "#0173B2",
    2: "#029E73",
    3: "#CC78A8",
    4: "#1A1A1A",
    5: "#D55E00",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_sarr_from_005(result: dict) -> float:
    """
    计算 alpha 从 0.05 到 0.40 的 SARR：

    SARR = 1 / (0.40 - 0.05) * integral R(alpha) d_alpha

    使用梯形积分，不使用 alpha=0 的 baseline。
    """
    by_factor = result.get("by_factor", {})

    x_values = []
    y_values = []

    for factor in FACTORS:
        factor_key = f"{factor:.2f}"

        if factor_key not in by_factor:
            raise ValueError(
                f"Missing refusal statistics for factor={factor_key}."
            )

        refusal_rate = by_factor[factor_key].get("refusal_rate")

        if refusal_rate is None:
            raise ValueError(
                f"refusal_rate is None for factor={factor_key}."
            )

        x_values.append(factor)
        y_values.append(float(refusal_rate))

    area = np.trapz(y_values, x_values)
    interval_length = FACTORS[-1] - FACTORS[0]

    return float(area / interval_length)


def load_ranked_feature_info(
    ranked_features_path: Path,
    model_name: str,
) -> dict[int, dict]:
    ranked_data = load_json(ranked_features_path)

    records = ranked_data.get(model_name)

    if not isinstance(records, list):
        raise ValueError(
            f"Cannot find model {model_name!r} "
            f"in {ranked_features_path}"
        )

    return {
        int(record["feature_idx"]): {
            "rank": int(record["rank"]),
            "cos_sim": float(record["cos_sim"]),
        }
        for record in records
    }


def collect_sarr_records(
    statistics_path: Path,
    ranked_features_path: Path,
    model_name: str,
) -> list[dict]:
    statistics = load_json(statistics_path)

    stored_model_name = statistics.get("model_name")

    if stored_model_name != model_name:
        raise ValueError(
            f"model_name mismatch:\n"
            f"  --model_name: {model_name}\n"
            f"  statistics JSON: {stored_model_name}"
        )

    cosine_info = load_ranked_feature_info(
        ranked_features_path=ranked_features_path,
        model_name=model_name,
    )

    results = statistics.get("results")

    if not isinstance(results, dict):
        raise ValueError(
            f"Cannot find results in {statistics_path}"
        )

    records = []

    for file_name, result in results.items():
        if "error" in result:
            continue

        file_info = result.get("file_info", {})

        if file_info.get("method") != "ranked_sae":
            continue

        feature_idx = file_info.get("feature_idx")

        if feature_idx is None:
            continue

        feature_idx = int(feature_idx)

        if feature_idx not in cosine_info:
            print(
                f"Skip {file_name}: feature={feature_idx} "
                "not found in ranked_features.json"
            )
            continue

        try:
            sarr = compute_sarr_from_005(result)
        except ValueError as error:
            print(f"Skip {file_name}: {error}")
            continue

        ranked_record = cosine_info[feature_idx]

        records.append(
            {
                "model": model_name,
                "dataset": file_info.get("dataset", "unknown"),
                "use_safe": bool(file_info.get("use_safe")),
                "feature_idx": feature_idx,
                "rank": ranked_record["rank"],
                "cos_sim": ranked_record["cos_sim"],
                "sarr": sarr,
                "source_file": file_name,
            }
        )

    return records


def macro_average_sarr(records: list[dict]) -> list[dict]:
    """
    对同一 SAE feature 在各数据集上的 SARR 进行宏平均。
    Safe 与 Unsafe 分开计算。
    """
    grouped = defaultdict(list)

    for record in records:
        group_key = (
            record["model"],
            record["use_safe"],
            record["feature_idx"],
        )
        grouped[group_key].append(record)

    macro_rows = []

    for (
        model,
        use_safe,
        feature_idx,
    ), feature_records in grouped.items():
        first_record = feature_records[0]

        datasets = sorted(
            {record["dataset"] for record in feature_records}
        )

        macro_sarr = float(
            np.mean(
                [record["sarr"] for record in feature_records]
            )
        )

        macro_rows.append(
            {
                "model": model,
                "use_safe": use_safe,
                "feature_idx": feature_idx,
                "rank": first_record["rank"],
                "cos_sim": first_record["cos_sim"],
                "macro_sarr": macro_sarr,
                "datasets": datasets,
            }
        )

    return macro_rows


def compute_correlations(
    rows: list[dict],
) -> tuple[float, float]:
    x_values = [row["cos_sim"] for row in rows]
    y_values = [row["macro_sarr"] for row in rows]

    if len(x_values) < 2:
        return float("nan"), float("nan")

    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return float("nan"), float("nan")

    spearman_rho = float(
        spearmanr(x_values, y_values).statistic
    )
    kendall_tau = float(
        kendalltau(x_values, y_values).statistic
    )

    return spearman_rho, kendall_tau


def format_statistic(value: float) -> str:
    if math.isnan(value):
        return "N/A"

    return f"{value:.2f}"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_one_safety_group(
    macro_rows: list[dict],
    model_name: str,
    use_safe: bool,
    output_pdf_path: Path,
) -> None:
    """分别绘制并保存 Safe 或 Unsafe 的单张 PDF。"""
    setup_plot_style()

    model_display_name = MODEL_DISPLAY_NAMES.get(
        model_name,
        model_name,
    )
    safety_name = "Safe" if use_safe else "Unsafe"

    rows = sorted(
        [
            row
            for row in macro_rows
            if row["use_safe"] == use_safe
        ],
        key=lambda row: row["rank"],
    )

    if not rows:
        print(f"No valid {safety_name.lower()} data to plot.")
        return

    x_values = [row["cos_sim"] for row in rows]
    y_values = [row["macro_sarr"] for row in rows]

    spearman_rho, kendall_tau = compute_correlations(rows)

    fig, ax = plt.subplots(figsize=(6.7, 5.2))

    for row in rows:
        rank = row["rank"]

        ax.scatter(
            row["cos_sim"],
            row["macro_sarr"],
            s=78,
            color=RANK_COLORS.get(rank, "#666666"),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=f"Rank {rank}",
        )

    ax.set_title(
        f"{model_display_name}\n"
        f"Spearman $\\rho$ = "
        f"{format_statistic(spearman_rho)}; "
        f"Kendall $\\tau$ = "
        f"{format_statistic(kendall_tau)}",
        fontweight="bold",
        pad=14,
    )

    ax.set_xlabel("Cosine similarity to the DiM vector")
    ax.set_ylabel(
        f"Macro-average SARR ({safety_name.lower()})"
    )

    ax.grid(
        axis="y",
        color="#d9d9d9",
        linewidth=0.8,
        alpha=0.9,
        zorder=0,
    )

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    x_padding = max((x_max - x_min) * 0.15, 0.015)
    y_padding = max((y_max - y_min) * 0.15, 0.02)

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(
        max(-0.02, y_min - y_padding),
        min(1.02, y_max + y_padding),
    )

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # 为顶部图例和模型标题预留空间。
    fig.subplots_adjust(
        top=0.76,
        bottom=0.14,
        left=0.16,
        right=0.98,
    )

    fig.savefig(
        output_pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_summary(macro_rows: list[dict]) -> None:
    print("\n===== Macro-average SARR: alpha in [0.05, 0.40] =====")

    for use_safe in [True, False]:
        safety_name = "safe" if use_safe else "unsafe"

        rows = sorted(
            [
                row
                for row in macro_rows
                if row["use_safe"] == use_safe
            ],
            key=lambda row: row["rank"],
        )

        if not rows:
            print(f"\n{safety_name}: no valid rows")
            continue

        spearman_rho, kendall_tau = compute_correlations(rows)

        print(
            f"\n{safety_name}: "
            f"Spearman rho={format_statistic(spearman_rho)}, "
            f"Kendall tau={format_statistic(kendall_tau)}"
        )

        for row in rows:
            print(
                f"  rank={row['rank']}, "
                f"feature={row['feature_idx']}, "
                f"cos_sim={row['cos_sim']:.6f}, "
                f"macro_sarr={row['macro_sarr']:.6f}, "
                f"datasets={','.join(row['datasets'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cosine similarity versus macro-average SARR "
            "from alpha=0.05 to alpha=0.40."
        )
    )

    parser.add_argument(
        "--model_name",
        required=True,
        choices=tuple(MODEL_DISPLAY_NAMES.keys()),
        help="Model to analyze.",
    )

    parser.add_argument(
        "--statistics_json",
        type=Path,
        required=True,
        help="Ranked refusal statistics JSON for this model.",
    )

    parser.add_argument(
        "--ranked_features_json",
        type=Path,
        required=True,
        help="ranked_features.json for this model.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("sarr_correlation_results"),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_sarr_records(
        statistics_path=args.statistics_json,
        ranked_features_path=args.ranked_features_json,
        model_name=args.model_name,
    )

    if not records:
        raise ValueError(
            "No valid SARR records were loaded. "
            "Check that the statistics JSON contains ranked_sae "
            "results and all factors from 0.05 to 0.40."
        )

    macro_rows = macro_average_sarr(records)

    if not macro_rows:
        raise ValueError("No macro-average SARR records were created.")

    print_summary(macro_rows)

    model_tag = args.model_name.replace("/", "_")

    safe_pdf_path = (
        args.output_dir
        / f"{model_tag}_cosine_macro_sarr_safe_005_to_040.pdf"
    )

    unsafe_pdf_path = (
        args.output_dir
        / f"{model_tag}_cosine_macro_sarr_unsafe_005_to_040.pdf"
    )

    plot_one_safety_group(
        macro_rows=macro_rows,
        model_name=args.model_name,
        use_safe=True,
        output_pdf_path=safe_pdf_path,
    )

    plot_one_safety_group(
        macro_rows=macro_rows,
        model_name=args.model_name,
        use_safe=False,
        output_pdf_path=unsafe_pdf_path,
    )

    print(f"\nSaved Safe PDF: {safe_pdf_path}")
    print(f"Saved Unsafe PDF: {unsafe_pdf_path}")


if __name__ == "__main__":
    main()