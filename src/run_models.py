import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from llama_cpp import Llama, LlamaGrammar
    import os
    from pathlib import Path
    import polars as pl
    from download_models import MODELS
    return Llama, LlamaGrammar, MODELS, Path, os, pl


@app.cell
def _(pl):
    # Survey
    dat = pl.read_parquet("data/survey_data/dat_clean.parquet")
    return (dat,)


@app.cell
def _(pl):
    # Vote summaries
    summa = pl.read_json("data/vote_summaries/summaries.json")
    return (summa,)


@app.cell
def _(pl):
    # Party positions
    part = pl.read_json("data/party_positions/party_positions.json")
    return (part,)


@app.cell
def _(LlamaGrammar, os):
    n_ctx = int(os.getenv("LLAMA_N_CTX", "2048"))
    n_threads = int(os.getenv("LLAMA_N_THREADS", "12"))

    grammar = LlamaGrammar.from_string(
        'root ::= "yes" | "no"'
    )
    return grammar, n_ctx, n_threads


@app.cell
def _(
    Llama,
    MODELS,
    Path,
    dat,
    grammar,
    n_ctx,
    n_threads,
    os,
    part,
    pl,
    summa,
):
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    skip_models = {
        s.strip().lower()
        for s in os.getenv("SKIP_MODELS", "").split(",")
        if s.strip()
    }

    model_outputs = {}
    for model in MODELS:
        abbr = model.get("abbr") or Path(model["filename"]).stem.lower()
        if abbr in skip_models:
            print(f"Skipping model via SKIP_MODELS: {abbr}")
            continue
        column_name = abbr.upper()
        out_path = results_dir / f"{abbr}dat.parquet"
        if out_path.exists():
            print(f"Skipping existing results: {out_path}")
            continue

        model_path = model_dir / model["filename"]
        if not model_path.exists():
            print(f"Missing model file: {model_path}")
            continue

        try:
            llm = Llama(
                model_path=str(model_path),
                chat_format="chatml",
                n_ctx=n_ctx,
                n_threads=n_threads,
            )
        except ValueError as exc:
            print(f"Failed to load model {model_path}: {exc}")
            continue

        resp = []
        for i in range(dat.shape[0]):
            vote = summa[dat[i, 7]].item()
            party = part[dat[i, 7]].item()

            personal = (
                f"You are an {dat[i,1]} old {dat[i,2]} with {dat[i,4]} "
                f"education that earns {dat[i,5]} per month. You are {dat[i,3]} "
                f"in politics and you identify with {dat[i,0]}.\n"
            )

            comb = (
                vote
                + "\n"
                + "The major parties issued the following voting recommendations:"
                + party
                + "\n"
                + personal
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are simulating the Swiss voter described by the user. "
                        "Output exactly one of: yes, no."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        comb
                        + "\nHow would you vote? Answer exactly: yes or no."
                    ),
                },
            ]

            llm.reset()
            llm._ctx.kv_cache_clear()

            out = llm.create_chat_completion(
                messages=messages,
                max_tokens=3,
                temperature=0.0,
                stop=["<|im_end|>", "\n"],
                grammar=grammar,
            )

            resp.append(out["choices"][0]["message"]["content"].strip())
            print(f"{model['filename']}: {i + 1} of {dat.shape[0]} done.")

        cleaned = [v.rstrip(".") for v in resp]
        model_df = dat.with_columns(pl.Series(column_name, cleaned))
        model_df.write_parquet(out_path)
        model_outputs[abbr] = model_df
        print(f"Saved results to: {out_path}")
    return


if __name__ == "__main__":
    app.run()
