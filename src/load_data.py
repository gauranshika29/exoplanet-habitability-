import pandas as pd

def load_exoplanet_data(path):
    df = pd.read_csv(
        path,
        sep=",",
        comment="#",
        engine="python"
    )

    print("Columns found:")
    print(df.columns.tolist())

    cols = [
        "pl_rade",
        "pl_bmasse",
        "pl_orbper",
        "pl_eqt",
        "st_teff",
        "st_rad"
    ]

    df = df[cols]
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_exoplanet_data("data/exoplanets.csv")
    print(df.head())
    print("Shape:", df.shape)
