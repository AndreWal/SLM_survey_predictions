import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import polars as pl
    return Path, pl, plt


@app.cell
def _(plt):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.labelcolor": "#374151",
            "text.color": "#111827",
            "axes.edgecolor": "#E5E7EB",
            "grid.color": "#E5E7EB",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return ()


@app.cell
def _(Path):
    results_dir = Path("data/results")
    results_path = results_dir / "lfmdat.parquet"
    llama_path = results_dir / "llama32dat.parquet"
    qwen_path = results_dir / "qwen3dat.parquet"
    output_dir = Path("data/figures_tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    return llama_path, output_dir, qwen_path, results_path


@app.cell
def _(llama_path, pl, qwen_path, results_path):
    df = pl.read_parquet(results_path)
    if llama_path.exists():
        llama_df = pl.read_parquet(llama_path).select(["LLAMA32"])
        df = df.with_columns(llama_df["LLAMA32"])
    if qwen_path.exists():
        qwen_df = pl.read_parquet(qwen_path).select(["QWEN3"])
        df = df.with_columns(qwen_df["QWEN3"])
    return (df,)


@app.cell
def _(df, output_dir, plt, pl):
    vote_counts = (
        df.with_columns(pl.col("vote").str.to_lowercase().alias("vote_lower"))
        .group_by("vote_lower")
        .len()
        .sort("vote_lower")
    )
    vote_map = {"yes": 0, "no": 1}
    vote_counts = vote_counts.with_columns(
        pl.col("vote_lower").replace_strict(vote_map).alias("order")
    ).sort("order")
    labels = vote_counts["vote_lower"].to_list()
    values = vote_counts["len"].to_list()

    _total = sum(values)
    shares = [value / _total if _total else 0 for value in values]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    colors = ["#2a9d8f", "#e76f51"]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_title("Vote Distribution (Ground Truth)")
    ax.set_xlabel("Vote")
    ax.set_ylabel("Count")
    ax.bar_label(
        bars,
        labels=[f"{value:,}\n({share:.1%})" for value, share in zip(values, shares)],
        padding=4,
        fontsize=9,
        color="#111827",
    )
    ax.set_ylim(0, max(values) * 1.18 if values else 1)
    fig.tight_layout()
    fig.savefig(output_dir / "vote_distribution.png", dpi=200)
    plt.close(fig)
    return labels, values


@app.cell
def _(df, pl):
    mapped = df.with_columns(
        pl.col("vote")
        .str.to_lowercase()
        .replace_strict({"yes": 1, "no": 0})
        .alias("true")
    )
    model_specs = [
        {"column": "LFM", "label": "LFM"},
        {"column": "LLAMA32", "label": "Llama-3.2"},
        {"column": "QWEN3", "label": "Qwen3"},
    ]
    model_specs = [
        spec for spec in model_specs if spec["column"] in mapped.columns
    ]
    return mapped, model_specs


@app.cell
def _(mapped, model_specs, output_dir, pl):
    def counts_for(model_col):
        data = (
            mapped.with_columns(
                pl.col(model_col)
                .str.to_lowercase()
                .replace_strict({"yes": 1, "no": 0})
                .alias("pred")
            )
            .filter(pl.col("true").is_not_null() & pl.col("pred").is_not_null())
        )

        tp = data.filter((pl.col("true") == 1) & (pl.col("pred") == 1)).height
        tn = data.filter((pl.col("true") == 0) & (pl.col("pred") == 0)).height
        fp = data.filter((pl.col("true") == 0) & (pl.col("pred") == 1)).height
        fn = data.filter((pl.col("true") == 1) & (pl.col("pred") == 0)).height
        total = tp + tn + fp + fn
        yes = data.filter(pl.col("true") == 1).height
        no = data.filter(pl.col("true") == 0).height
        yes_share = yes / total if total else float("nan")
        return tp, tn, fp, fn, total, yes, no, yes_share

    def metrics_from_counts(tp, tn, fp, fn, total):
        accuracy = (tp + tn) / total if total else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else float("nan")
        )
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        balanced_acc = (recall + specificity) / 2
        return accuracy, precision, recall, f1, specificity, balanced_acc

    metrics_rows = []
    per_metric_rows = []

    for spec in model_specs:
        tp, tn, fp, fn, total, yes, no, yes_share = counts_for(
            spec["column"]
        )
        accuracy, precision, recall, f1, specificity, balanced_acc = (
            metrics_from_counts(tp, tn, fp, fn, total)
        )
        metrics_rows.append(
            {
                "model": spec["label"],
                "n": total,
                "yes": yes,
                "no": no,
                "yes_share": yes_share,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "balanced_accuracy": balanced_acc,
            }
        )
        per_metric_rows.extend(
            [
                {"model": spec["label"], "metric": "accuracy", "value": accuracy},
                {"model": spec["label"], "metric": "precision", "value": precision},
                {"model": spec["label"], "metric": "recall", "value": recall},
                {"model": spec["label"], "metric": "f1", "value": f1},
                {
                    "model": spec["label"],
                    "metric": "specificity",
                    "value": specificity,
                },
                {
                    "model": spec["label"],
                    "metric": "balanced_accuracy",
                    "value": balanced_acc,
                },
            ]
        )

    yes = mapped.filter(pl.col("true") == 1).height
    no = mapped.filter(pl.col("true") == 0).height
    total_overall = yes + no
    yes_share = yes / total_overall if total_overall else float("nan")

    majority_pred_yes = yes >= no
    base_tp = yes if majority_pred_yes else 0
    base_tn = 0 if majority_pred_yes else no
    base_fp = no if majority_pred_yes else 0
    base_fn = 0 if majority_pred_yes else yes

    base_accuracy, base_precision, base_recall, base_f1, base_specificity, base_balanced = (
        metrics_from_counts(base_tp, base_tn, base_fp, base_fn, total_overall)
    )

    metrics_rows.append(
        {
            "model": "Majority",
            "n": total_overall,
            "yes": yes,
            "no": no,
            "yes_share": yes_share,
            "accuracy": base_accuracy,
            "precision": base_precision,
            "recall": base_recall,
            "f1": base_f1,
            "specificity": base_specificity,
            "balanced_accuracy": base_balanced,
        }
    )
    per_metric_rows.extend(
        [
            {"model": "Majority", "metric": "accuracy", "value": base_accuracy},
            {"model": "Majority", "metric": "precision", "value": base_precision},
            {"model": "Majority", "metric": "recall", "value": base_recall},
            {"model": "Majority", "metric": "f1", "value": base_f1},
            {
                "model": "Majority",
                "metric": "specificity",
                "value": base_specificity,
            },
            {
                "model": "Majority",
                "metric": "balanced_accuracy",
                "value": base_balanced,
            },
        ]
    )

    overall_metrics = pl.DataFrame(metrics_rows)
    overall_metrics.write_csv(output_dir / "metrics_overall.csv")

    model_metrics_df = pl.DataFrame(per_metric_rows)
    model_metrics_df.write_csv(output_dir / "model_metrics.csv")

    lfm_metrics = [
        row for row in per_metric_rows if row["model"] == "LFM"
    ]
    if lfm_metrics:
        pl.DataFrame(lfm_metrics).write_csv(output_dir / "lfm_metrics.csv")

    return (overall_metrics,)


@app.cell
def _(mapped, model_specs, output_dir, pl):
    label_map = {
        "v_677": "677 Umweltverantwortung",
        "v_678": "678 Eigenmietwert",
        "v_679": "679 E-ID",
    }

    def grouped_metrics_for(spec):
        data = (
            mapped.with_columns(
                pl.col(spec["column"])
                .str.to_lowercase()
                .replace_strict({"yes": 1, "no": 0})
                .alias("pred")
            )
            .filter(pl.col("true").is_not_null() & pl.col("pred").is_not_null())
        )
        return (
            data.group_by("vote_type")
            .agg(
                pl.len().alias("n"),
                (pl.col("true") == 1).sum().alias("yes"),
                (pl.col("true") == 0).sum().alias("no"),
                ((pl.col("true") == 1) & (pl.col("pred") == 1)).sum().alias(
                    "tp"
                ),
                ((pl.col("true") == 0) & (pl.col("pred") == 0)).sum().alias(
                    "tn"
                ),
                ((pl.col("true") == 0) & (pl.col("pred") == 1)).sum().alias(
                    "fp"
                ),
                ((pl.col("true") == 1) & (pl.col("pred") == 0)).sum().alias(
                    "fn"
                ),
            )
            .with_columns(
                (
                    pl.col("tp")
                    + pl.col("tn")
                    + pl.col("fp")
                    + pl.col("fn")
                ).alias("total")
            )
            .with_columns(
                (pl.col("yes") / pl.col("n")).alias("yes_share"),
                ((pl.col("tp") + pl.col("tn")) / pl.col("total")).alias(
                    "accuracy"
                ),
                (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).alias(
                    "precision"
                ),
                (pl.col("tp") / (pl.col("tp") + pl.col("fn"))).alias(
                    "recall"
                ),
                (
                    2
                    * (pl.col("tp") / (pl.col("tp") + pl.col("fp")))
                    * (pl.col("tp") / (pl.col("tp") + pl.col("fn")))
                    / (
                        (pl.col("tp") / (pl.col("tp") + pl.col("fp")))
                        + (pl.col("tp") / (pl.col("tp") + pl.col("fn")))
                    )
                ).alias("f1"),
                (pl.col("tn") / (pl.col("tn") + pl.col("fp"))).alias(
                    "specificity"
                ),
            )
            .with_columns(
                (
                    (pl.col("recall") + pl.col("specificity")) / 2
                ).alias("balanced_accuracy")
            )
            .with_columns(
                pl.col("vote_type")
                .map_elements(lambda v: label_map.get(v, v))
                .alias("vote_label")
            )
            .with_columns(pl.lit(spec["label"]).alias("model"))
            .select(
                [
                    "model",
                    "vote_type",
                    "vote_label",
                    "n",
                    "yes",
                    "no",
                    "yes_share",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                    "balanced_accuracy",
                ]
            )
            .sort("vote_type")
        )

    truth_counts = (
        mapped.group_by("vote_type")
        .agg(
            pl.len().alias("n"),
            (pl.col("true") == 1).sum().alias("yes"),
            (pl.col("true") == 0).sum().alias("no"),
        )
        .with_columns((pl.col("yes") / pl.col("n")).alias("yes_share"))
        .with_columns(
            pl.when(pl.col("yes") >= pl.col("no"))
            .then(pl.lit("yes"))
            .otherwise(pl.lit("no"))
            .alias("majority_class")
        )
        .with_columns(
            pl.when(pl.col("majority_class") == "yes")
            .then(pl.col("yes"))
            .otherwise(pl.lit(0))
            .alias("tp"),
            pl.when(pl.col("majority_class") == "no")
            .then(pl.col("no"))
            .otherwise(pl.lit(0))
            .alias("tn"),
            pl.when(pl.col("majority_class") == "yes")
            .then(pl.col("no"))
            .otherwise(pl.lit(0))
            .alias("fp"),
            pl.when(pl.col("majority_class") == "no")
            .then(pl.col("yes"))
            .otherwise(pl.lit(0))
            .alias("fn"),
        )
        .with_columns(
            (pl.col("tp") + pl.col("tn") + pl.col("fp") + pl.col("fn")).alias(
                "total"
            )
        )
        .with_columns(
            ((pl.col("tp") + pl.col("tn")) / pl.col("total")).alias("accuracy"),
            (pl.col("tp") / (pl.col("tp") + pl.col("fp"))).alias("precision"),
            (pl.col("tp") / (pl.col("tp") + pl.col("fn"))).alias("recall"),
            (
                2
                * (pl.col("tp") / (pl.col("tp") + pl.col("fp")))
                * (pl.col("tp") / (pl.col("tp") + pl.col("fn")))
                / (
                    (pl.col("tp") / (pl.col("tp") + pl.col("fp")))
                    + (pl.col("tp") / (pl.col("tp") + pl.col("fn")))
                )
            ).alias("f1"),
            (pl.col("tn") / (pl.col("tn") + pl.col("fp"))).alias(
                "specificity"
            ),
        )
        .with_columns(
            (
                (pl.col("recall") + pl.col("specificity")) / 2
            ).alias("balanced_accuracy")
        )
        .with_columns(
            pl.col("vote_type")
            .map_elements(lambda v: label_map.get(v, v))
            .alias("vote_label"),
            pl.lit("Majority").alias("model"),
        )
        .select(
            [
                "model",
                "vote_type",
                "vote_label",
                "n",
                "yes",
                "no",
                "yes_share",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "balanced_accuracy",
            ]
        )
        .sort("vote_type")
    )

    model_metric_tables = [grouped_metrics_for(spec) for spec in model_specs]
    grouped = pl.concat([*model_metric_tables, truth_counts]).sort(
        ["vote_type", "model"]
    )

    grouped.write_csv(output_dir / "metrics_by_vote_type.csv")
    return (grouped,)


if __name__ == "__main__":
    app.run()
