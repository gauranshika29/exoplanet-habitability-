import pandas as pd
import plotly.express as px

# Load scored dataset
df = pd.read_csv("data/habitable_scored_exoplanets.csv")

# ---------- Scatter plot: Radius vs Temperature ----------
fig_scatter = px.scatter(
    df,
    x="pl_rade",
    y="pl_eqt",
    color="habitability_score",
    hover_data=["pl_name", "pl_bmasse", "pl_eqt"],
    color_continuous_scale="Viridis",
    labels={
        "pl_rade": "Planet Radius (Earth Radii)",
        "pl_eqt": "Equilibrium Temperature (K)",
        "habitability_score": "Habitability Score"
    },
    title="Exoplanets: Radius vs Equilibrium Temperature"
)
fig_scatter.show()

# ---------- Histogram: Habitability Score ----------
fig_hist = px.histogram(
    df,
    x="habitability_score",
    nbins=20,
    title="Distribution of Habitability Scores",
    labels={"habitability_score": "Habitability Score"},
    marginal="box",
    color_discrete_sequence=["skyblue"]
)
fig_hist.show()

# ---------- Top 10 Most Habitable Planets ----------
top10 = df.sort_values("habitability_score", ascending=False).head(10)
fig_top10 = px.bar(
    top10,
    x="habitability_score",
    y="pl_name",
    orientation="h",
    title="Top 10 Most Habitable Exoplanets",
    labels={"habitability_score": "Habitability Score", "pl_name": "Planet Name"},
    text="habitability_score",
    color="habitability_score",
    color_continuous_scale="Viridis"
)
fig_top10.update_layout(yaxis={'categoryorder':'total ascending'})
fig_top10.show()
