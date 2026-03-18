import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# ─────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────
df = pd.read_csv("data/habitable_scored_exoplanets.csv")

df["discoverymethod"] = df["discoverymethod"] if "discoverymethod" in df.columns else "Unknown"
df["discoverymethod"] = df["discoverymethod"].fillna("Unknown")

# Habitable label: score >= 0.8
df["habitable"] = (df["habitability_score"] >= 0.8).astype(int)

# ─────────────────────────────────────────────
# Train Random Forest with improved recall
# ─────────────────────────────────────────────
FEATURES = ["pl_rade", "pl_bmasse", "pl_eqt"]
X = df[FEATURES]
y = df["habitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

THRESHOLD = 0.20
y_proba = model.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)

print("\nModel Performance:")
print(f"  Accuracy  : {accuracy:.3f}")
print(f"  Precision : {precision:.3f}")
print(f"  Recall    : {recall:.3f}")
print(f"  Threshold : {THRESHOLD}")

# ─────────────────────────────────────────────
# Plain-English explanation helper
# ─────────────────────────────────────────────
def explain_prediction(r, m, t):
    good, bad = [], []

    if 0.8 <= r <= 1.3:
        good.append("Earth-like radius")
    elif r > 1.3:
        bad.append(f"radius too large ({r:.1f} R⊕)")
    else:
        bad.append(f"radius too small ({r:.1f} R⊕)")

    if 0.5 <= m <= 2.0:
        good.append("Earth-like mass")
    elif m > 2.0:
        bad.append(f"mass too high ({m:.1f} M⊕)")
    else:
        bad.append(f"mass too low ({m:.1f} M⊕)")

    if 200 <= t <= 310:
        good.append("temperature in habitable range")
    elif t > 310:
        bad.append(f"too hot ({t:.0f} K)")
    else:
        bad.append(f"too cold ({t:.0f} K)")

    if good and not bad:
        return "✓ " + ", ".join(good)
    elif good and bad:
        return "✓ " + ", ".join(good) + "  |  ✗ " + ", ".join(bad)
    else:
        return "✗ " + ", ".join(bad)

# ─────────────────────────────────────────────
# Shared styles
# ─────────────────────────────────────────────
CARD = {
    "flex": "1",
    "padding": "20px 16px",
    "borderRadius": "12px",
    "textAlign": "center",
    "minWidth": "120px",
}

METRIC_COLORS = {
    "accuracy":  ("#e8f5e9", "#2e7d32"),
    "precision": ("#e3f2fd", "#1565c0"),
    "recall":    ("#fff3e0", "#e65100"),
}

def stat_card(label, value, bg, fg):
    return html.Div([
        html.P(label, style={"margin": "0 0 6px", "fontSize": "13px", "opacity": ".7"}),
        html.P(value, style={"margin": 0, "fontSize": "22px", "fontWeight": "700", "color": fg}),
    ], style={**CARD, "backgroundColor": bg})

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = Dash(__name__)
app.title = "Exoplanet Habitability Dashboard"

app.layout = html.Div(style={"backgroundColor": "#eef2f7", "minHeight": "100vh", "paddingBottom": "60px"}, children=[

    # ── Header ───────────────────────────────
    html.Div(style={
        "background": "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
        "padding": "36px 20px 28px",
        "textAlign": "center",
        "marginBottom": "30px",
    }, children=[
        html.H1("🪐 Exoplanet Habitability Dashboard",
                style={"color": "white", "margin": 0, "fontSize": "28px", "fontWeight": "600"}),
        html.P("Interactive ML-powered explorer using NASA Exoplanet Archive data",
               style={"color": "rgba(255,255,255,0.65)", "margin": "8px 0 0", "fontSize": "14px"}),
    ]),

    # ── Summary stats (dynamic) ───────────────
    html.Div(id="summary-stats", style={"width": "88%", "margin": "0 auto 24px"}),

    # ── Model metrics (static) ────────────────
    html.Div(style={
        "width": "88%", "margin": "0 auto 28px",
        "backgroundColor": "white", "borderRadius": "14px",
        "padding": "20px 24px", "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"
    }, children=[
        html.H3("Model Performance", style={"margin": "0 0 16px", "fontSize": "16px", "fontWeight": "600"}),
        html.P(f"Random Forest  |  threshold {THRESHOLD}  |  class_weight='balanced'",
               style={"fontSize": "12px", "opacity": ".55", "margin": "0 0 14px"}),
        html.Div(style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}, children=[
            stat_card("Accuracy",  f"{accuracy:.3f}",  *METRIC_COLORS["accuracy"]),
            stat_card("Precision", f"{precision:.3f}", *METRIC_COLORS["precision"]),
            stat_card("Recall",    f"{recall:.3f}",    *METRIC_COLORS["recall"]),
        ])
    ]),

    # ── Filters ───────────────────────────────
    html.Div(style={
        "width": "88%", "margin": "0 auto 28px",
        "backgroundColor": "white", "borderRadius": "14px",
        "padding": "20px 24px", "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"
    }, children=[
        html.H3("Filters", style={"margin": "0 0 16px", "fontSize": "16px", "fontWeight": "600"}),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px 40px"}, children=[
            html.Div([
                html.Label("Habitability Score", style={"fontSize": "13px", "fontWeight": "500"}),
                dcc.RangeSlider(id="score-slider",
                    min=round(df["habitability_score"].min(), 2),
                    max=round(df["habitability_score"].max(), 2),
                    step=0.01,
                    value=[round(df["habitability_score"].min(), 2), round(df["habitability_score"].max(), 2)],
                    marks={0.5: "0.5", 0.7: "0.7", 0.9: "0.9"},
                    tooltip={"placement": "bottom", "always_visible": False})
            ]),
            html.Div([
                html.Label("Planet Radius (R⊕)", style={"fontSize": "13px", "fontWeight": "500"}),
                dcc.RangeSlider(id="radius-slider",
                    min=round(df["pl_rade"].min(), 1),
                    max=round(df["pl_rade"].max(), 1),
                    step=0.1,
                    value=[round(df["pl_rade"].min(), 1), round(df["pl_rade"].max(), 1)],
                    marks={1: "1", 1.5: "1.5", 2: "2"},
                    tooltip={"placement": "bottom", "always_visible": False})
            ]),
            html.Div([
                html.Label("Equilibrium Temperature (K)", style={"fontSize": "13px", "fontWeight": "500"}),
                dcc.RangeSlider(id="temp-slider",
                    min=int(df["pl_eqt"].min()),
                    max=int(df["pl_eqt"].max()),
                    step=1,
                    value=[int(df["pl_eqt"].min()), int(df["pl_eqt"].max())],
                    marks={200: "200K", 280: "280K", 330: "330K"},
                    tooltip={"placement": "bottom", "always_visible": False})
            ]),
            html.Div([
                html.Label("Discovery Method", style={"fontSize": "13px", "fontWeight": "500"}),
                dcc.Dropdown(id="method-dropdown",
                    options=[{"label": m, "value": m} for m in
                             ["All"] + sorted(df["discoverymethod"].dropna().unique().tolist())],
                    value="All", clearable=False)
            ]),
        ])
    ]),

    # ── Charts ────────────────────────────────
    html.Div(style={"width": "88%", "margin": "0 auto 28px"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            html.Div(dcc.Graph(id="scatter-plot"), style={
                "backgroundColor": "white", "borderRadius": "14px",
                "padding": "8px", "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"}),
            html.Div(dcc.Graph(id="histogram"), style={
                "backgroundColor": "white", "borderRadius": "14px",
                "padding": "8px", "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"}),
        ]),
        html.Div(dcc.Graph(id="top10-bar"), style={
            "backgroundColor": "white", "borderRadius": "14px",
            "padding": "8px", "marginTop": "20px",
            "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"})
    ]),

    # ── ML Prediction card ────────────────────
    html.Div(style={
        "width": "88%", "margin": "0 auto",
        "backgroundColor": "white", "borderRadius": "14px",
        "padding": "28px 32px", "boxShadow": "0 2px 12px rgba(0,0,0,0.07)"
    }, children=[
        html.H3("Predict Habitability for a Custom Planet",
                style={"margin": "0 0 6px", "fontSize": "18px", "fontWeight": "600"}),
        html.P("Enter physical parameters and click Predict.",
               style={"fontSize": "13px", "opacity": ".55", "margin": "0 0 22px"}),

        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr auto", "gap": "16px", "alignItems": "end"}, children=[
            html.Div([
                html.Label("Radius (R⊕)", style={"fontSize": "13px", "fontWeight": "500", "display": "block", "marginBottom": "6px"}),
                dcc.Input(id="input-radius", type="number", value=1.0, step=0.1, min=0.1, max=3.0,
                          style={"width": "100%", "padding": "9px 12px", "borderRadius": "8px",
                                 "border": "1px solid #ddd", "fontSize": "14px", "boxSizing": "border-box"})
            ]),
            html.Div([
                html.Label("Mass (M⊕)", style={"fontSize": "13px", "fontWeight": "500", "display": "block", "marginBottom": "6px"}),
                dcc.Input(id="input-mass", type="number", value=1.0, step=0.1, min=0.1, max=10.0,
                          style={"width": "100%", "padding": "9px 12px", "borderRadius": "8px",
                                 "border": "1px solid #ddd", "fontSize": "14px", "boxSizing": "border-box"})
            ]),
            html.Div([
                html.Label("Temp (K)", style={"fontSize": "13px", "fontWeight": "500", "display": "block", "marginBottom": "6px"}),
                dcc.Input(id="input-temp", type="number", value=288, step=1, min=50, max=500,
                          style={"width": "100%", "padding": "9px 12px", "borderRadius": "8px",
                                 "border": "1px solid #ddd", "fontSize": "14px", "boxSizing": "border-box"})
            ]),
            html.Button("Predict", id="predict-btn", n_clicks=0, style={
                "padding": "10px 28px", "backgroundColor": "#203a43", "color": "white",
                "border": "none", "borderRadius": "8px", "fontSize": "14px",
                "cursor": "pointer", "fontWeight": "600", "whiteSpace": "nowrap"
            }),
        ]),

        html.Div(id="prediction-output", style={"marginTop": "20px"})
    ]),
])

# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
@app.callback(
    Output("scatter-plot", "figure"),
    Output("histogram", "figure"),
    Output("top10-bar", "figure"),
    Output("summary-stats", "children"),
    Input("score-slider", "value"),
    Input("radius-slider", "value"),
    Input("temp-slider", "value"),
    Input("method-dropdown", "value"),
)
def update_graphs(score_range, radius_range, temp_range, method):
    fdf = df[
        df["habitability_score"].between(*score_range) &
        df["pl_rade"].between(*radius_range) &
        df["pl_eqt"].between(*temp_range)
    ].copy()

    if method != "All":
        fdf = fdf[fdf["discoverymethod"] == method]

    n        = len(fdf)
    avg_hab  = round(fdf["habitability_score"].mean(), 3) if n else 0
    avg_rad  = round(fdf["pl_rade"].mean(), 2) if n else 0
    avg_mass = round(fdf["pl_bmasse"].mean(), 2) if n else 0

    summary = html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
        stat_card("Filtered planets",       str(n),        "#e3f2fd", "#1565c0"),
        stat_card("Avg habitability score", str(avg_hab),  "#e8f5e9", "#2e7d32"),
        stat_card("Avg radius (R⊕)",        str(avg_rad),  "#ede7f6", "#4527a0"),
        stat_card("Avg mass (M⊕)",          str(avg_mass), "#fce4ec", "#880e4f"),
    ])

    # Scatter
    plot_df = fdf.copy()
    if n > 0 and "pl_bmasse" in plot_df.columns:
        max_m = plot_df["pl_bmasse"].max()
        plot_df["_sz"] = plot_df["pl_bmasse"].apply(
            lambda x: max(3, min(x / max_m * 18, 18)) if max_m > 0 else 6
        )

    scatter = px.scatter(
        plot_df, x="pl_rade", y="pl_eqt",
        size="_sz" if "_sz" in plot_df.columns and n > 0 else None,
        color="habitability_score",
        hover_data=["pl_name", "pl_bmasse", "pl_eqt"],
        color_continuous_scale="Viridis",
        labels={"pl_rade": "Radius (R⊕)", "pl_eqt": "Temp (K)", "habitability_score": "Score"},
        title="Radius vs Equilibrium Temperature",
    )
    scatter.update_layout(margin=dict(t=44, b=20, l=20, r=20), plot_bgcolor="white")

    # Histogram
    hist = px.histogram(
        fdf, x="habitability_score", nbins=20,
        title="Habitability Score Distribution",
        labels={"habitability_score": "Score"},
        marginal="box", color_discrete_sequence=["#4f86c6"]
    )
    hist.update_layout(margin=dict(t=44, b=20, l=20, r=20), plot_bgcolor="white")

    # Top 10 bar
    top10 = fdf.nlargest(10, "habitability_score")
    bar = px.bar(
        top10, x="habitability_score", y="pl_name", orientation="h",
        title="Top 10 Most Habitable Exoplanets",
        labels={"habitability_score": "Score", "pl_name": "Planet"},
        text="habitability_score",
        color="habitability_score", color_continuous_scale="Viridis"
    )
    bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    bar.update_layout(yaxis={"categoryorder": "total ascending"},
                      margin=dict(t=44, b=20, l=20, r=100), plot_bgcolor="white")

    return scatter, hist, bar, summary


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("input-radius", "value"),
    State("input-mass", "value"),
    State("input-temp", "value"),
    prevent_initial_call=True,
)
def predict_habitability(n_clicks, rade, bmasse, eqt):
    if not n_clicks or None in (rade, bmasse, eqt):
        return ""

    planet = pd.DataFrame([{"pl_rade": rade, "pl_bmasse": bmasse, "pl_eqt": eqt}])
    proba  = model.predict_proba(planet)[0][1]
    pred   = int(proba >= THRESHOLD)
    conf   = proba if pred == 1 else 1 - proba
    reason = explain_prediction(rade, bmasse, eqt)

    if pred == 1:
        bg, border, title_color, verdict = "#f0fdf4", "#86efac", "#166534", "Habitable ✓"
    else:
        bg, border, title_color, verdict = "#fef2f2", "#fca5a5", "#991b1b", "Not Habitable ✗"

    return html.Div(style={
        "backgroundColor": bg,
        "border": f"1px solid {border}",
        "borderRadius": "10px",
        "padding": "18px 22px",
    }, children=[
        html.H4(f"Prediction: {verdict}",
                style={"margin": "0 0 6px", "color": title_color, "fontSize": "16px"}),
        html.P(f"Model confidence: {conf:.0%}  (threshold: {THRESHOLD})",
               style={"margin": "0 0 6px", "fontSize": "13px", "opacity": ".7"}),
        html.P(f"Analysis: {reason}", style={"margin": 0, "fontSize": "13px"}),
    ])


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{port}/")
    app.run(debug=True, port=port)