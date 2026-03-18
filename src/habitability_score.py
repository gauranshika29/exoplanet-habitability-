import pandas as pd
import numpy as np

def compute_habitability(df):
    """
    Compute a habitability score (0–1) for exoplanets
    based on Earth-like features.
    """
    # Filter out extreme/gas giants
    df = df[
        (df["pl_rade"] > 0) & (df["pl_rade"] < 2.5) &   # radius in Earth radii
        (df["pl_bmasse"] > 0) & (df["pl_bmasse"] < 5) & # mass in Earth masses
        (df["pl_eqt"] > 150) & (df["pl_eqt"] < 400)     # temp in Kelvin
    ].copy()

    # Compute normalized scores
    radius_score = 1 - abs(df["pl_rade"] - 1.0) / 1.5
    temp_score   = 1 - abs(df["pl_eqt"] - 288) / 150
    flux_score   = 1 - abs(df["pl_insol"] - 1) / 1.0
    mass_score   = 1 - abs(df["pl_bmasse"] - 1) / 2

    # Clip between 0–1
    radius_score = radius_score.clip(0,1)
    temp_score   = temp_score.clip(0,1)
    flux_score   = flux_score.clip(0,1)
    mass_score   = mass_score.clip(0,1)

    # Weighted habitability score
    df["habitability_score"] = (
        0.3 * radius_score +
        0.3 * temp_score +
        0.2 * flux_score +
        0.2 * mass_score
    )

    return df

if __name__ == "__main__":
    # Load cleaned dataset
    df = pd.read_csv("data/clean_exoplanets.csv")
    print(f"Original dataset: {df.shape[0]} rows")

    # Compute habitability
    df = compute_habitability(df)
    print(f"After filtering: {df.shape[0]} rows")

    # Top 10 habitable planets
    top10 = df.sort_values("habitability_score", ascending=False).head(10)
    print("\nTop 10 potentially habitable exoplanets:")
    print(top10[["pl_name", "pl_rade", "pl_bmasse", "pl_eqt", "habitability_score"]])

    # Save scored dataset
    df.to_csv("data/habitable_scored_exoplanets.csv", index=False)
    print("\nSaved scored dataset to data/habitable_scored_exoplanets.csv")
