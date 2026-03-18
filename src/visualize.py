import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load scored dataset
df = pd.read_csv("data/habitable_scored_exoplanets.csv")

# ---------- Scatter plot: Radius vs Temperature ----------
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    df["pl_rade"], df["pl_eqt"],
    c=df["habitability_score"], cmap="viridis", s=50, alpha=0.8
)
plt.colorbar(scatter, label="Habitability Score")
plt.xlabel("Planet Radius (Earth Radii)")
plt.ylabel("Equilibrium Temperature (K)")
plt.title("Exoplanets: Radius vs Equilibrium Temperature")
plt.grid(True)
plt.savefig("plots/scatter_radius_temp.png", dpi=300)
plt.show()

# ---------- Histogram: Habitability Score ----------
plt.figure(figsize=(10,6))
sns.histplot(df["habitability_score"], bins=20, kde=True, color="skyblue")
plt.xlabel("Habitability Score")
plt.ylabel("Number of Planets")
plt.title("Distribution of Habitability Scores")
plt.grid(True)
plt.savefig("plots/histogram_habitability.png", dpi=300)
plt.show()

# ---------- Top 10 Most Habitable Planets ----------
top10 = df.sort_values("habitability_score", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(
    x="habitability_score",
    y="pl_name",
    data=top10,
    palette="viridis"
)
plt.xlabel("Habitability Score")
plt.ylabel("Planet Name")
plt.title("Top 10 Most Habitable Exoplanets")
plt.xlim(0,1)
plt.grid(axis="x")
plt.savefig("plots/top10_habitability.png", dpi=300)
plt.show()

print("Plots saved to 'plots/' folder")
