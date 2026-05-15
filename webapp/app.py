"""TP4 - Phase 7 : Streamlit MonEnergie - design dashboard pro

Dashboard energie avec Plotly 3D, donut, gauge, radar, KPI cards et custom CSS.

3 onglets :
- Forecast 24h : LSTM time series + 3D surface conso heure x jour
- Profil energetique : LSTM 4 classes + donut probabilites + radar 24h
- Sentiment fournisseur : Bi-LSTM Allocine + gauge sentiment

Lancement :  streamlit run webapp/app.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from log_inference import log_inference  # type: ignore
except ImportError:  # pragma: no cover
    def log_inference(*args, **kwargs):
        return None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MonEnergie - Forecast IA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEARCH_PATHS = [
    Path("."),
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
]


def find_file(name: str):
    for base in SEARCH_PATHS:
        p = base / name
        if p.exists():
            return p
    return None


PROFILS = ["Nocturne", "Equilibre nuit", "Equilibre soir", "Vespertine"]
PROFIL_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
PROFIL_RECOS = {
    "Nocturne": "Vous consommez surtout la nuit. **Reco** : tarif Heures Creuses HC/HP, vous economiserez 10-15%.",
    "Equilibre nuit": "Conso plutot nocturne avec quelques pics jour. **Reco** : tarif Heures Creuses ou Tempo.",
    "Equilibre soir": "Conso plutot diurne avec quelques pics soir. **Reco** : tarif Base ou Tempo si vous savez decaler.",
    "Vespertine": "Vous consommez surtout en soiree. **Reco** : tarif Base ou EJP (eviter HC, vous serez en HP).",
}


# ============================================================
# CSS
# ============================================================
CUSTOM_CSS = """
<style>
    /* fade in animation */
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(8px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .main .block-container { animation: fadeIn 0.5s ease-out; padding-top: 2rem; }

    /* hero header */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #0ea5e9 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    /* KPI cards via st.metric */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #10b981;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04), 0 4px 12px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(15, 23, 42, 0.08), 0 12px 24px rgba(15, 23, 42, 0.08);
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.7rem !important;
        font-weight: 700;
    }

    /* tabs */
    [role="tab"] {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        background: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
    }

    /* buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }

    /* recommendation card */
    .reco-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f0f9ff 0%, #ecfdf5 100%);
        border-left: 4px solid var(--reco-color, #10b981);
        animation: fadeIn 0.6s ease-out;
    }
    .reco-card h3 { margin: 0 0 0.5rem 0; color: var(--reco-color, #10b981); }
    .reco-card p { margin: 0; color: #1e293b; line-height: 1.5; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# LOADERS (cached)
# ============================================================
@st.cache_resource
def load_forecast_model():
    p = find_file("energie_lstm.keras")
    s = find_file("energie_scalers.joblib")
    if p is None or s is None:
        return None, None
    return keras.models.load_model(p), joblib.load(s)


@st.cache_resource
def load_profil_model():
    p = find_file("profil_conso.keras")
    m = find_file("profil_conso_meta.joblib")
    if p is None or m is None:
        return None, None
    return keras.models.load_model(p), joblib.load(m)


@st.cache_resource
def load_sentiment_model():
    p = find_file("allocine_sentiment.keras")
    t = find_file("allocine_tokenizer.joblib")
    if p is None or t is None:
        return None, None
    return keras.models.load_model(p), joblib.load(t)


# ============================================================
# DATA GENERATORS (pour demo sans CSV)
# ============================================================
def generate_sample_week(profile: str = "vespertine") -> pd.DataFrame:
    """Genere 168h fictives de consommation pour tester la webapp."""
    rng = np.random.default_rng(seed=42)
    timestamps = pd.date_range(end=datetime.now(), periods=168, freq="h")
    hours = np.array([t.hour for t in timestamps])
    dow = np.array([t.dayofweek for t in timestamps])
    is_weekend = (dow >= 5).astype(float)

    # Pattern de base : pic matin (7-9) + pic soir (18-21)
    base = 0.4 + 0.3 * np.sin((hours - 6) * np.pi / 12) ** 2

    if profile == "nocturne":
        # Forte conso la nuit (ballon ECS)
        base += 0.6 * np.exp(-((hours - 3) ** 2) / 8)
    elif profile == "vespertine":
        # Forte conso le soir
        base += 0.8 * np.exp(-((hours - 19) ** 2) / 6)
    elif profile == "equilibre":
        base += 0.3 * np.sin(hours * np.pi / 12)

    base += rng.normal(0, 0.08, 168)
    base = np.clip(base, 0.1, 5.0)

    df = pd.DataFrame(
        {
            "active_kw": base,
            "sub1_wh": rng.uniform(0, 50, 168) * (hours >= 6) * (hours <= 22),  # cuisine
            "sub2_wh": rng.uniform(0, 30, 168) * (hours >= 8) * (hours <= 20),  # laverie
            "sub3_wh": rng.uniform(20, 80, 168),  # chauffage/ECS
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            "is_weekend": is_weekend,
        },
        index=timestamps,
    )
    return df


# ============================================================
# VIZ HELPERS
# ============================================================
def make_kpi_line_chart(df: pd.DataFrame, forecast_value: float) -> go.Figure:
    """Line chart 7j historique + forecast."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["active_kw"],
            mode="lines",
            line=dict(color="#0ea5e9", width=2),
            fill="tozeroy",
            fillcolor="rgba(14, 165, 233, 0.08)",
            name="Historique 7j",
            hovertemplate="<b>%{x|%a %d %b %Hh}</b><br>%{y:.2f} kW<extra></extra>",
        )
    )

    # Forecast point + zone projetee
    last_time = df.index[-1]
    future_time = last_time + timedelta(hours=24)
    fig.add_trace(
        go.Scatter(
            x=[last_time, future_time],
            y=[df["active_kw"].iloc[-1], forecast_value],
            mode="lines+markers",
            line=dict(color="#10b981", width=3, dash="dot"),
            marker=dict(size=12, color="#10b981", symbol="diamond"),
            name=f"Forecast moy 24h",
            hovertemplate="<b>Forecast</b><br>%{y:.2f} kW<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Puissance (kW)", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def make_3d_consumption_surface(df: pd.DataFrame) -> go.Figure:
    """Surface 3D : conso par heure x jour de la semaine."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["day"] = df.index.dayofyear
    df["day_idx"] = pd.factorize(df["day"])[0]

    pivot = df.pivot_table(values="active_kw", index="day_idx", columns="hour", aggfunc="mean")

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot.values,
                x=list(pivot.columns),
                y=[f"J{i}" for i in pivot.index],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="kW", thickness=15, len=0.7),
                hovertemplate="<b>%{y}</b><br>%{x}h<br>%{z:.2f} kW<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(title="Heure", gridcolor="#e2e8f0"),
            yaxis=dict(title="Jour", gridcolor="#e2e8f0"),
            zaxis=dict(title="Conso (kW)", gridcolor="#e2e8f0"),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            aspectmode="cube",
        ),
        scene_aspectratio=dict(x=1, y=1, z=0.5),
    )
    return fig


def make_breakdown_donut(df: pd.DataFrame) -> go.Figure:
    """Donut sub_metering : cuisine / laverie / chauffage."""
    total1 = df["sub1_wh"].sum()
    total2 = df["sub2_wh"].sum()
    total3 = df["sub3_wh"].sum()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Cuisine", "Laverie", "Chauffage / ECS"],
                values=[total1, total2, total3],
                hole=0.55,
                marker=dict(
                    colors=["#f59e0b", "#0ea5e9", "#ef4444"],
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>%{value:.0f} Wh<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{(total1 + total2 + total3) / 1000:.0f}</b><br><span style='font-size:0.8em;color:#64748b'>kWh</span>",
                showarrow=False,
                font=dict(size=22, color="#0f172a"),
            )
        ],
    )
    return fig


def make_polar_hourly(df: pd.DataFrame) -> go.Figure:
    """Radar 24h : profil horaire moyen."""
    by_hour = df.groupby(df.index.hour)["active_kw"].mean()
    # Boucler pour fermer le radar
    hours = list(by_hour.index) + [by_hour.index[0]]
    values = list(by_hour.values) + [by_hour.values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=[h * 15 for h in hours],
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.25)",
            line=dict(color="#10b981", width=2),
            marker=dict(size=8, color="#10b981"),
            name="Profil horaire",
            hovertemplate="<b>%{theta}°</b> (%{customdata}h)<br>%{r:.2f} kW<extra></extra>",
            customdata=hours,
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=40, r=40, t=20, b=40),
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#e2e8f0"),
            angularaxis=dict(
                tickmode="array",
                tickvals=[h * 15 for h in range(0, 24, 3)],
                ticktext=[f"{h}h" for h in range(0, 24, 3)],
                direction="clockwise",
                rotation=90,
                gridcolor="#e2e8f0",
            ),
        ),
        showlegend=False,
    )
    return fig


def make_profile_donut(probas, predicted_idx: int) -> go.Figure:
    """Donut 4 profils avec le predit en pull out."""
    pulls = [0.12 if i == predicted_idx else 0 for i in range(4)]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=PROFILS,
                values=probas * 100,
                hole=0.45,
                pull=pulls,
                marker=dict(
                    colors=PROFIL_COLORS,
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{PROFILS[predicted_idx]}</b><br><span style='font-size:0.7em;color:#64748b'>predit</span>",
                showarrow=False,
                font=dict(size=14, color=PROFIL_COLORS[predicted_idx]),
            )
        ],
    )
    return fig


def make_sentiment_gauge(score: float, label: str) -> go.Figure:
    """Gauge -100 a +100 pour sentiment."""
    color = "#10b981" if score > 0.5 else "#ef4444"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number=dict(suffix=" / 100", font=dict(size=32, color=color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94a3b8"),
                bar=dict(color=color, thickness=0.3),
                bgcolor="#f1f5f9",
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color="#fee2e2"),
                    dict(range=[30, 50], color="#fef3c7"),
                    dict(range=[50, 70], color="#dcfce7"),
                    dict(range=[70, 100], color="#a7f3d0"),
                ],
                threshold=dict(line=dict(color="black", width=3), thickness=0.75, value=50),
            ),
            title=dict(text=f"<b>{label}</b>", font=dict(size=20, color=color)),
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


# ============================================================
# TAB : FORECAST
# ============================================================
def render_forecast_tab():
    forecast_model, scalers = load_forecast_model()
    if forecast_model is None:
        st.error("Modele forecast introuvable. Lance phase 1+2 sur Kaggle et push energie_lstm.keras + energie_scalers.joblib")
        return

    # Source de donnees : upload ou exemple
    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Uploadez votre courbe (CSV 168 lignes, colonnes : active_kw, sub1_wh, sub2_wh, sub3_wh, hour_sin, hour_cos, is_weekend)",
            type=["csv"],
            key="forecast_upload",
        )
    with col_demo:
        st.write("")
        st.write("")
        use_demo = st.button("⚡ Charger un exemple", use_container_width=True)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
        else:
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq="h")
    elif use_demo or st.session_state.get("forecast_demo", False):
        st.session_state["forecast_demo"] = True
        df = generate_sample_week(profile="vespertine")
    else:
        st.info("📂 Uploadez votre CSV ou cliquez **Charger un exemple** pour une demo.")
        return

    df = df.tail(168)
    features = scalers["features"]
    if any(f not in df.columns for f in features):
        st.error(f"Colonnes manquantes. Attendu : {features}")
        return

    # Prediction
    arr_scaled = scalers["scaler_X"].transform(df[features].values)
    pred_scaled = forecast_model.predict(arr_scaled.reshape(1, 168, len(features)), verbose=0)
    pred_kw = float(scalers["scaler_y"].inverse_transform(pred_scaled)[0, 0])

    avg_kw = float(df["active_kw"].mean())
    delta_pct = (pred_kw - avg_kw) / max(avg_kw, 1e-6) * 100
    cost_eur = pred_kw * 24 * 0.25  # tarif reglemente bleu

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Conso moy 7j", f"{avg_kw:.2f} kW")
    with col2:
        st.metric(
            "Forecast moy 24h",
            f"{pred_kw:.2f} kW",
            f"{delta_pct:+.1f}% vs 7j",
        )
    with col3:
        st.metric("Cout estime jour", f"{cost_eur:.2f} €")
    with col4:
        annual = cost_eur * 365
        st.metric("Extrapolation annuelle", f"{annual:.0f} €")

    st.write("")

    # Charts
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("##### 📈 Historique 7 jours + forecast 24h")
        st.plotly_chart(make_kpi_line_chart(df, pred_kw), use_container_width=True)
    with col_right:
        st.markdown("##### 🍩 Repartition par usage")
        st.plotly_chart(make_breakdown_donut(df), use_container_width=True)

    st.markdown("##### 🌐 Pattern hebdomadaire 3D")
    st.caption("Surface interactive : faites pivoter avec la souris pour voir les pics horaires par jour.")
    st.plotly_chart(make_3d_consumption_surface(df), use_container_width=True)


# ============================================================
# TAB : PROFIL
# ============================================================
def render_profil_tab():
    model_profil, meta = load_profil_model()
    if model_profil is None:
        st.error("Modele profil introuvable. Lance phase 6 sur Kaggle et push profil_conso.keras + profil_conso_meta.joblib")
        return

    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Uploadez votre courbe (CSV 168 lignes)",
            type=["csv"],
            key="profil_upload",
        )
    with col_demo:
        st.write("")
        st.write("")
        col_a, col_b = st.columns(2)
        with col_a:
            demo_v = st.button("🌙 Demo Nocturne", use_container_width=True)
        with col_b:
            demo_n = st.button("🌆 Demo Vespertine", use_container_width=True)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"])).drop(columns=["timestamp"])
        else:
            df.index = pd.date_range(end=datetime.now(), periods=len(df), freq="h")
    elif demo_v:
        df = generate_sample_week(profile="nocturne")
        st.session_state["profil_demo"] = "nocturne"
    elif demo_n:
        df = generate_sample_week(profile="vespertine")
        st.session_state["profil_demo"] = "vespertine"
    elif st.session_state.get("profil_demo"):
        df = generate_sample_week(profile=st.session_state["profil_demo"])
    else:
        st.info("📂 Uploadez votre CSV ou choisissez une demo.")
        return

    df = df.tail(168)
    features = meta["features"]
    if any(f not in df.columns for f in features):
        st.error(f"Colonnes manquantes : {features}")
        return

    # Prediction
    arr_scaled = meta["scaler_X"].transform(df[features].values)
    proba = model_profil.predict(arr_scaled.reshape(1, 168, len(features)), verbose=0)[0]
    predicted_idx = int(np.argmax(proba))
    profil = PROFILS[predicted_idx]
    confidence = float(proba[predicted_idx])

    # Compute ratio peak/offpeak pour KPI
    peak_hours = {18, 19, 20}
    off_hours = {1, 2, 3, 4, 5}
    hours_arr = df.index.hour.values
    peak_mean = df.loc[np.isin(hours_arr, list(peak_hours)), "active_kw"].mean()
    off_mean = df.loc[np.isin(hours_arr, list(off_hours)), "active_kw"].mean()
    ratio = peak_mean / max(off_mean, 1e-6)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Profil predit", profil)
    with col2:
        st.metric("Confiance", f"{confidence * 100:.1f}%")
    with col3:
        st.metric("Ratio peak/offpeak", f"{ratio:.2f}")
    with col4:
        st.metric("Conso moy 7j", f"{df['active_kw'].mean():.2f} kW")

    st.write("")

    # Reco card
    color = PROFIL_COLORS[predicted_idx]
    st.markdown(
        f'<div class="reco-card" style="--reco-color: {color}; border-left-color: {color};">'
        f"<h3 style='color: {color}'>🎯 Recommandation tarifaire</h3>"
        f"<p>{PROFIL_RECOS[profil]}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Charts
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("##### 🍩 Probabilites par profil")
        st.plotly_chart(make_profile_donut(proba, predicted_idx), use_container_width=True)
    with col_right:
        st.markdown("##### 🎯 Profil horaire (radar 24h)")
        st.plotly_chart(make_polar_hourly(df), use_container_width=True)


# ============================================================
# TAB : SENTIMENT
# ============================================================
def render_sentiment_tab():
    model_sentiment, tok_meta = load_sentiment_model()
    if model_sentiment is None:
        st.error("Modele sentiment introuvable. Lance phase 4+5 et push allocine_sentiment.keras + allocine_tokenizer.joblib")
        return

    st.markdown(
        "Tapez ou collez un avis sur un fournisseur d'energie (EDF, Engie, TotalEnergies, etc). "
        "Le Bi-LSTM analyse le sentiment **en francais**."
    )

    user_text = st.text_area(
        "Votre avis",
        height=160,
        placeholder="Exemple : Service client EDF tres reactif, je recommande...",
    )

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run = st.button("🔍 Analyser", type="primary", use_container_width=True)
    with col_clear:
        st.write("")

    if not run:
        # Show example reviews to test
        st.markdown("##### 💡 Avis exemples a tester")
        examples = [
            "EDF nous a augmente la facture sans prevenir et le service client est introuvable, scandaleux",
            "Tres satisfait du conseiller TotalEnergies, reactif et le prix est competitif",
            "Encore 4h de coupure aujourd'hui, j'en peux plus de ce fournisseur catastrophique",
            "Installation Linky chez Engie nickel, technicien rapide et professionnel",
        ]
        for i, ex in enumerate(examples):
            if st.button(f"📝 {ex[:80]}...", key=f"ex_{i}"):
                st.session_state["sentiment_text"] = ex
                st.rerun()
        return

    if not user_text.strip():
        st.warning("Tapez un avis avant de cliquer.")
        return

    # Prediction
    from keras.utils import pad_sequences

    tokenizer = tok_meta["tokenizer"]
    max_len = tok_meta["max_len"]
    seq = tokenizer.texts_to_sequences([user_text])
    pad = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    proba = float(model_sentiment.predict(pad, verbose=0)[0, 0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    confidence = proba if label == "Positif" else 1 - proba

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sentiment", label)
    with col2:
        st.metric("Confiance", f"{confidence * 100:.1f}%")
    with col3:
        st.metric("Tokens analyses", f"{len(user_text.split())}")

    st.write("")

    # Gauge + analysis
    col_gauge, col_meta = st.columns([2, 1])
    with col_gauge:
        st.plotly_chart(make_sentiment_gauge(proba, label), use_container_width=True)
    with col_meta:
        st.markdown("##### Detail analyse")
        st.write(f"- **Score brut sigmoid** : `{proba:.4f}`")
        st.write(f"- **Seuil de decision** : `0.5`")
        st.write(f"- **Modele** : Bi-LSTM 64 units, entraine sur 200k avis Allocine FR")
        st.write(f"- **Transfert** : modele generaliste FR, applicable aux avis Trustpilot fournisseurs energie")

        if label == "Negatif" and confidence > 0.8:
            st.warning("Sentiment fortement negatif - probablement reclamation client.")
        elif label == "Positif" and confidence > 0.8:
            st.success("Sentiment fortement positif - probablement recommandation.")

    log_inference(user_text, label, confidence)


# ============================================================
# MAIN
# ============================================================
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# ⚡ MonEnergie")
        st.markdown("**Dashboard IA pour foyers français**")
        st.markdown("---")
        st.markdown("### À propos")
        st.markdown(
            "3 modèles RNN/LSTM entraînés sur des données françaises ouvertes :"
        )
        st.markdown(
            """
            - **LSTM forecast** sur Household Power FR
            - **LSTM 4 classes** : profil temporel
            - **Bi-LSTM** : sentiment FR (Allociné 200k)
            """
        )
        st.markdown("---")
        st.markdown("### Open Data")
        st.markdown(
            """
            - [Enedis Open Data](https://data.enedis.fr)
            - [RTE Open Data](https://opendata.reseaux-energies.fr)
            - [ADEME ObsDPE](https://odre.opendatasoft.com)
            """
        )
        st.markdown("---")
        st.caption("🔒 Aucune donnée personnelle envoyée. Inference on-device.")
        st.caption("📦 [Source code GitHub](https://github.com/Hakim78/tp4-monenergie)")

    # Hero
    st.markdown('<h1 class="hero-title">⚡ MonEnergie Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Forecast 24h • Profil énergétique • Avis fournisseurs - powered by RNN/LSTM</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["📊 Forecast 24h", "🎯 Profil énergétique", "💬 Sentiment fournisseur"])
    with tab1:
        render_forecast_tab()
    with tab2:
        render_profil_tab()
    with tab3:
        render_sentiment_tab()


if __name__ == "__main__":
    main()
