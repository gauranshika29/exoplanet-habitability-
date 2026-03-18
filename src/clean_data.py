import pandas as pd

def clean_exoplanet_data(path):
    df = pd.read_csv(
        path,
        comment="#",
        engine="python",
        sep=","
    )

    print("Columns found:")
    print(df.columns.tolist())

    cols = [
        "pl_name",
        "pl_rade",
        "pl_bmasse",
        "pl_orbsmax",
        "pl_orbper",
        "pl_insol",
        "pl_eqt",
        "pl_orbeccen",
        "st_teff",
        "st_rad",
        "st_mass"
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("Before cleaning:", df.shape)

    df = df.dropna(subset=[
        "pl_rade",
        "pl_bmasse",
        "pl_orbsmax",
        "pl_insol",
        "pl_eqt"
    ])

    print("After cleaning:", df.shape)
    return df


if __name__ == "__main__":
    df = clean_exoplanet_data("data/exoplanets.csv")
    df.to_csv("data/clean_exoplanets.csv", index=False)
    print("Saved cleaned data to data/clean_exoplanets.csv")
