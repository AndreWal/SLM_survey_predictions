# LLM Survey Predictions

This project builds and evaluates small language model (SLM) predictions for survey-style political preference questions. It prepares survey datasets, runs model inference, and aggregates results into summaries suitable for analysis and visualization.

Core components
- `src/`: data preparation, model download/inference, and evaluation utilities
- `data/`: processed datasets, results, and vote summaries (raw survey data excluded)
- `models/`: local model artifacts (ignored by git)
- `docker/`: container setup for reproducible runs
