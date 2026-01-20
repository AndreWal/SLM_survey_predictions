# 🗳️ LLM Survey Predictions

This project examines whether small language models (SLMs) can
simulate human decisions such as political behavior where prior research
has largely depended on closed, proprietary LLMs and has delivered mixed
results, especially for populations outside the United States. That reliance on commericial LLMs
raises concerns about privacy, reproducibility, and access. SLMs can mitigate these issues because they run locally, but their
performance remains uncertain. To address this gap, the repo evaluates several
SLMs by predicting Swiss voters' choices in three direct democratic votes in 2025 using vote
proposal summaries, party recommendations, and commonly used voter attributes
for explaining vote behavior.

## 🔎 At a glance
- 📦 Local, reproducible runs with Docker + GGUF models.
- 🧪 Per-vote evaluation against real survey ground truth.
- 📈 Ready-to-share figures and CSV metrics.

## ✨ What this repo does
- Prepares cleaned survey data with respondent profiles and issue-specific vote
  summaries.
- Prompts local LLMs to predict binary votes (yes/no) for each respondent.
- Stores model outputs alongside ground truth for analysis and visualization.

## 🧭 Repository layout
- `src/`: model download, prompting, and evaluation notebooks/scripts.
- `data/`: processed datasets, results, and summaries (raw survey data excluded).
- `models/`: local GGUF model files (ignored by git).
- `docker/`: container setup for reproducible runs.
- `start.sh`: convenience script to run the Docker image.

## 🧩 Prompt blueprint
Each example combines (1) the vote summary, (2) party recommendations, and
(3) a respondent profile. The model must answer with exactly `yes` or `no`.

```text
System: You are simulating the Swiss voter described by the user.
        Output exactly one of: yes, no.

User:
<vote summary text>
The major parties issued the following voting recommendations:
<party position text>

You are an <age> old <gender> with <education> education that earns <income>
per month. You are <political interest> in politics and you identify with
<party>.

How would you vote? Answer exactly: yes or no.
```

## 📊 Results snapshot (current)
The `data/results` folder contains datasets with the ground-truth vote in the
`vote` column and model predictions in columns named after each model
(e.g., `LFM`). LFM, Llama-3.2, and Qwen3 results are available.

**Models evaluated**

| Model | Params | Abbrev | Status |
| --- | --- | --- | --- |
| LiquidAI LFM2.5-1.2B-Instruct | 1.2B | lfm | complete |
| Llama-3.2-3B-Instruct | 3B | llama32 | complete |
| Qwen3-4B-Instruct-2507 | 4B | qwen3 | complete |

## 🔍 Insights
- 🧠 Summary: SLMs remain highly sensitive to prompt design and do not reliably outperform a simple majority-class baseline.
- 🧭 Qwen3 edges out the other SLMs overall, driven by a strong result on the Umweltverantwortung vote.
- 📌 LFM performs slightly better than the baseline, but analysis by vote shows this is driven by a single vote.
- ⚖️ Llama shows strong performance for one vote as well but does not consistently outperform the baseline across multiple votes.
- 📉 Majority baseline stays competitive or ahead for Eigenmietwert and E-ID.
- 🧰 Fine-tuning might help, but SLMs still have a long way to go before they can reliably employed to simulate human voting behavior.
- 🔄 Future work could explore which features predict miss-classification once SLMs consistently outperform the majority-class baseline.

## 📈 Figures
![Vote distribution](data/figures_tables/vote_distribution.png)

## 🧮 Metrics (overall)
Computed on the 3,883 examples. Majority is a baseline that
always predicts the most common class.

| Model | Accuracy | Precision | Recall | F1 | Specificity | Balanced acc. |
| --- | --- | --- | --- | --- | --- | --- |
| LFM | 0.604 | 0.599 | 0.609 | 0.604 | 0.599 | 0.604 |
| Llama-3.2 | 0.593 | 0.679 | 0.341 | 0.454 | 0.842 | 0.591 |
| Qwen3 | **0.609** | 0.636 | 0.495 | 0.557 | 0.722 | **0.608** |
| Majority | 0.504 | n/a | 0.000 | n/a | 1.000 | 0.500 |

Metrics are saved to `data/figures_tables/metrics_overall.csv` and the
per-metric export to `data/figures_tables/model_metrics.csv`.

## 🧾 Metrics by vote_type
Computed on the 3,883 examples and saved to
`data/figures_tables/metrics_by_vote_type.csv`. Majority is a baseline that
always predicts the most common class.

| Model | Vote | n | Yes | No | Yes share | Accuracy | F1 | Balanced acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LFM | 677 Umweltverantwortung | 1,514 | 535 | 979 | 0.353 | 0.668 | 0.458 | 0.607 |
| Llama-3.2 | 677 Umweltverantwortung | 1,514 | 535 | 979 | 0.353 | 0.715 | 0.410 | 0.617 |
| Qwen3 | 677 Umweltverantwortung | 1,514 | 535 | 979 | 0.353 | **0.740** | 0.623 | **0.710** |
| Majority | 677 Umweltverantwortung | 1,514 | 535 | 979 | 0.353 | 0.647 | n/a | 0.500 |
| LFM | 678 Eigenmietwert | 1,180 | 715 | 465 | 0.606 | 0.564 | 0.680 | 0.509 |
| Llama-3.2 | 678 Eigenmietwert | 1,180 | 715 | 465 | 0.606 | 0.566 | 0.638 | 0.549 |
| Qwen3 | 678 Eigenmietwert | 1,180 | 715 | 465 | 0.606 | 0.579 | 0.654 | **0.558** |
| Majority | 678 Eigenmietwert | 1,180 | 715 | 465 | 0.606 | **0.606** | 0.755 | 0.500 |
| LFM | 679 E-ID | 1,189 | 676 | 513 | 0.569 | 0.562 | 0.613 | **0.554** |
| Llama-3.2 | 679 E-ID | 1,189 | 676 | 513 | 0.569 | 0.464 | 0.147 | 0.525 |
| Qwen3 | 679 E-ID | 1,189 | 676 | 513 | 0.569 | 0.473 | 0.335 | 0.511 |
| Majority | 679 E-ID | 1,189 | 676 | 513 | 0.569 | **0.569** | 0.725 | 0.500 |

## ⚡ Quickstart
1. Download models (uses Hugging Face Hub):
   `python3 src/download_models.py`
2. Run model inference:
   `python3 src/run_models.py`
3. Inspect outputs in `data/results/`.
4. Generate figures and metrics tables:
   `python3 src/analysis_results.py`

## 📝 Notes
- Results files are named `<abbrev>dat.parquet` and include all original
  respondent features plus model predictions in a new column.
- Use `SKIP_MODELS=llama32,qwen3` to skip specific models.
