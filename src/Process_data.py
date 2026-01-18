import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    return Path, pl


@app.cell
def _(Path):
    path = Path("data/survey_data")
    return (path,)


@app.cell
def _(pl):
    # Party identification
    PAR_DICT = {
        1: "SVP",
        2: "SP",
        3: "FDP",
        4: "Mitte",
        5: "GPS",
        6: "EVP",
        7: "Lega",
        8: "CSP",
        9: "GLP",
        11: "PdA",
        12: "EDU",
        13: "Piraten",
        14: "MCG",
        15: "with a smaller party",
        16: "no party",
        17: "multiple parties",
    }

    def party_code_expr(col: str, default: str = "no response") -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Int64, strict=False)
            .replace_strict(PAR_DICT, default=default, return_dtype=pl.Utf8)
        )

    # Gender

    gender_expr = (
        pl.col("S11").cast(pl.Int64, strict=False)
        .pipe(lambda s:  
            pl.when(s == 1).then(pl.lit("man"))
              .when(s == 2).then(pl.lit("woman"))
              .when(s == 3).then(pl.lit("non-binary"))
              .otherwise(pl.lit("no response"))
        )
        .alias("gender")
    )

    # Political interest

    polint_expr = (
        pl.col("POLINT").cast(pl.Int64, strict=False)
        .pipe(lambda s:  
            pl.when(s == 1).then(pl.lit("very interested"))
              .when(s == 2).then(pl.lit("somewhat interested"))
              .when(s == 3).then(pl.lit("not very interested"))
              .when(s == 4).then(pl.lit("not at all interested"))
              .otherwise(pl.lit("no response"))
        )
        .alias("polinterest")
    )

    # Education (EDUC)

    def education_level_expr(col: str) -> pl.Expr:
        edu = pl.col(col).cast(pl.Int64, strict=False)

        return (
            pl.when(edu.is_in([10, 21]))
              .then(pl.lit("primary or less"))

            .when(edu.is_in([22, 31, 32, 33, 40]))
              .then(pl.lit("secondary"))

            .when(edu.is_in([51, 52, 60]))
              .then(pl.lit("tertiary"))

            .otherwise(pl.lit("no response"))  # 97, 98, invalid
        )

    # Income (INCOME)

    INCOME_DICT = {
        1:  "weniger als 2'000 CHF",
        2:  "2'001-3'000 CHF",
        3:  "3'001-4'000 CHF",
        4:  "4'001-5'000 CHF",
        5:  "5'001-6'000 CHF",
        6:  "6'001-7'000 CHF",
        7:  "7'001-8'000 CHF",
        8:  "8'001-9'000 CHF",
        9:  "9'001-10'000 CHF",
        10: "10'001-11'000 CHF",
        11: "11'001-12'000 CHF",
        12: "12'001-13'000 CHF",
        13: "13'001-14'000 CHF",
        14: "14'001-15'000 CHF",
        15: "mehr als 15'000 CHF",
        98: "no response",
    }

    def income_category_expr(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Int64, strict=False)
            .replace_strict(
                INCOME_DICT,
                default=None,
                return_dtype=pl.Utf8
            )
        )
    # Votes 

    VOTE_DICT = {
        1: "yes",
        2: "no",
        3: "no vote",
        8: "no response",
    }

    def vote_decision_expr(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .cast(pl.Int64, strict=False)
            .replace_strict(
                VOTE_DICT,
                default=None,
                return_dtype=pl.Utf8
            )
        )
    return (
        education_level_expr,
        gender_expr,
        income_category_expr,
        party_code_expr,
        polint_expr,
        vote_decision_expr,
    )


@app.cell
def _(path, pl):
    # Vote 677
    v77 = pl.read_csv(path / "V677.csv", encoding="windows-1252")
    return (v77,)


@app.cell
def _(
    education_level_expr,
    gender_expr,
    income_category_expr,
    party_code_expr,
    pl,
    polint_expr,
    v77,
    vote_decision_expr,
):
    v77_rec = v77.with_columns(
        party_code_expr("PARTY15").alias("party_id"),
        (2025 - pl.col("BIRTHYEARR")).alias("age"),
        gender_expr,
        polint_expr, 
        education_level_expr("EDUC").alias("education"),
        income_category_expr("INCOME").alias("income_cat"), 
        vote_decision_expr("VOTE1").alias("vote"), 
        pl.lit("v_677").alias("vote_type")
    )
    return (v77_rec,)


@app.cell
def _(pl, v77_rec):
    numcols = ["age"]

    catcols = ["vote","party_id", "gender", "polinterest","education","income_cat"]

    v77_clean = v77_rec.select(
        ["party_id", "age", "gender", "polinterest","education","income_cat","vote","vote_type"]
    ).filter(
            pl.all_horizontal(pl.col(numcols).is_not_null()) &
            ~pl.any_horizontal(pl.col(catcols) == "no response")&
            (pl.col("vote") != "no vote")
    )
    return catcols, numcols, v77_clean


@app.cell
def _(path, pl):
    # Vote 678 and 679 combined
    v78  = pl.read_csv(path / "V679.csv", encoding="windows-1252",
        truncate_ragged_lines=True,
        ignore_errors=True)
    return (v78,)


@app.cell
def _(pl):
    gender_expr2 = (
        pl.col("s11").cast(pl.Int64, strict=False)
        .pipe(lambda s:  
            pl.when(s == 1).then(pl.lit("man"))
              .when(s == 2).then(pl.lit("woman"))
              .when(s == 3).then(pl.lit("non-binary"))
              .otherwise(pl.lit("no response"))
        )
        .alias("gender")
    )

    # Political interest

    polint_expr2 = (
        pl.col("polint").cast(pl.Int64, strict=False)
        .pipe(lambda s:  
            pl.when(s == 1).then(pl.lit("very interested"))
              .when(s == 2).then(pl.lit("somewhat interested"))
              .when(s == 3).then(pl.lit("not very interested"))
              .when(s == 4).then(pl.lit("not at all interested"))
              .otherwise(pl.lit("no response"))
        )
        .alias("polinterest")
    )
    return gender_expr2, polint_expr2


@app.cell
def _(
    education_level_expr,
    gender_expr2,
    income_category_expr,
    party_code_expr,
    pl,
    polint_expr2,
    v78,
    vote_decision_expr,
):
    v78_rec = v78.with_columns(
        party_code_expr("party15").alias("party_id"),
        (2025 - pl.col("birthyearr")).alias("age"),
        gender_expr2,
        polint_expr2, 
        education_level_expr("educ").alias("education"),
        income_category_expr("income").alias("income_cat"), 
        vote_decision_expr("vote1").alias("vote"), 
        pl.lit("v_678").alias("vote_type")
    )
    return (v78_rec,)


@app.cell
def _(catcols, numcols, pl, v78_rec):
    v78_clean = v78_rec.select(
        ["party_id", "age", "gender", "polinterest","education","income_cat","vote","vote_type"]
    ).filter(
            pl.all_horizontal(pl.col(numcols).is_not_null()) &
            ~pl.any_horizontal(pl.col(catcols) == "no response")&
            (pl.col("vote") != "no vote")
    )
    return (v78_clean,)


@app.cell
def _(path, pl):
    v79  = pl.read_csv(path / "V679.csv", encoding="windows-1252",
        truncate_ragged_lines=True,
        ignore_errors=True)
    return (v79,)


@app.cell
def _(
    education_level_expr,
    gender_expr2,
    income_category_expr,
    party_code_expr,
    pl,
    polint_expr2,
    v79,
    vote_decision_expr,
):
    v79_rec = v79.with_columns(
        party_code_expr("party15").alias("party_id"),
        (2025 - pl.col("birthyearr")).alias("age"),
        gender_expr2,
        polint_expr2, 
        education_level_expr("educ").alias("education"),
        income_category_expr("income").alias("income_cat"), 
        vote_decision_expr("vote2").alias("vote"), 
        pl.lit("v_679").alias("vote_type")
    )
    return (v79_rec,)


@app.cell
def _(catcols, numcols, pl, v79_rec):
    v79_clean = v79_rec.select(
        ["party_id", "age", "gender", "polinterest","education","income_cat","vote","vote_type"]
    ).filter(
            pl.all_horizontal(pl.col(numcols).is_not_null()) &
            ~pl.any_horizontal(pl.col(catcols) == "no response") &
            (pl.col("vote") != "no vote")
    )
    return (v79_clean,)


@app.cell
def _(pl, v77_clean, v78_clean, v79_clean):
    vall = pl.concat([v77_clean, v78_clean, v79_clean])
    vall
    return (vall,)


@app.cell
def _(path, vall):
    vall.write_parquet(path / "dat_clean.parquet")
    return


if __name__ == "__main__":
    app.run()
