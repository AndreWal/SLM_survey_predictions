import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")

MODELS = [
    {
        "repo_id": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        "filename": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        "abbr": "lfm",
    },
    {
        "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q6_K.gguf",
        "abbr": "llama32",
    },
    {
        "repo_id": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "filename": "Qwen3-4B-Instruct-2507-Q6_K.gguf",
        "abbr": "qwen3",
    },
]


@app.cell
def _():
    import marimo as mo
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    return Path, hf_hub_download


@app.cell
def _(Path):
    TARGET_DIR = Path("models")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS, TARGET_DIR


@app.cell
def _(MODELS, TARGET_DIR, hf_hub_download):
    for model in MODELS:
        target_path = TARGET_DIR / model["filename"]
        if target_path.exists():
            print(f"Skipping existing file: {target_path}")
            continue
        model_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=TARGET_DIR,
        )
        print(f"Model downloaded to: {model_path}")
    return


if __name__ == "__main__":
    app.run()
