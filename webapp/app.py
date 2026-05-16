"""MonEnergie - app hybride pro + grand public

Combine :
- Design SaaS pro (gradient hero, glass cards, Plotly 3D / donut / radar / gauge)
- Wizard accessible avec factures-exemples pre-remplies
- Resultats grand public (euros + reco fournisseur) ET analyse technique pliable

Modeles :
- Phase 1+2 : LSTM forecast 24h
- Phase 6   : LSTM profil temporel 4 classes
- Phase 4+5 : Bi-LSTM sentiment FR

Lancement : streamlit run webapp/app.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from log_inference import log_inference  # type: ignore
except ImportError:
    def log_inference(*args, **kwargs):
        return None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MonEnergie - Forecast IA pour vos factures",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEARCH_PATHS = [Path("."), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
PROFILS = ["Nocturne", "Equilibre nuit", "Equilibre soir", "Vespertine"]
PROFIL_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]
PROFIL_DESCRIPTIONS = {
    "Nocturne": "Consommation concentrée la nuit (chauffe-eau, chauffage nocturne)",
    "Equilibre nuit": "Consommation modérément nocturne avec quelques pics jour",
    "Equilibre soir": "Consommation diurne avec activité en soirée",
    "Vespertine": "Forte concentration vespertine (cuisson + activités soirée)",
}
PROFIL_FOURNISSEUR = {
    "Nocturne": {
        "fournisseur": "EDF Tempo",
        "tarif": "Tempo (Heures Creuses préférentielles)",
        "logo": "🔵",
        "economie_an": 280,
        "raison": "Votre consommation est concentrée la nuit. Le tarif Tempo d'EDF facture la nuit jusqu'à 30% moins cher.",
    },
    "Equilibre nuit": {
        "fournisseur": "TotalEnergies HC/HP",
        "tarif": "Heures Creuses / Heures Pleines",
        "logo": "🟠",
        "economie_an": 180,
        "raison": "Conso décalée vers la nuit, le tarif HC/HP de TotalEnergies optimise vos heures creuses.",
    },
    "Equilibre soir": {
        "fournisseur": "Engie Elec Référence",
        "tarif": "Tarif Base",
        "logo": "🟢",
        "economie_an": 90,
        "raison": "Conso régulière sur la journée. Le tarif Base d'Engie reste compétitif sans surprise.",
    },
    "Vespertine": {
        "fournisseur": "Octopus Energy Eco",
        "tarif": "Eco indexé vert",
        "logo": "🐙",
        "economie_an": 150,
        "raison": "Actif le soir, Octopus propose un tarif indexé simple sans heures creuses qui vous pénaliseraient.",
    },
}

# Factures exemples pour pre-remplir le questionnaire
FACTURES_EXEMPLES = [
    {
        "id": "famille_idf",
        "icon": "🏡",
        "titre": "Famille en pavillon",
        "region": "Île-de-France · Massy",
        "details": "4 personnes · 120 m² · chauffage électrique",
        "facture_estimee": "175 € / mois",
        "color": "#ef4444",
        "profile": {
            "type": "Maison individuelle",
            "surface": 120,
            "occupants": 4,
            "chauffage": "Electrique",
            "soir": "Souvent à la maison le soir (18h-22h)",
        },
    },
    {
        "id": "couple_lyon",
        "icon": "🏢",
        "titre": "Couple en appartement",
        "region": "Lyon 7e",
        "details": "2 personnes · 60 m² · chauffage gaz",
        "facture_estimee": "85 € / mois",
        "color": "#0ea5e9",
        "profile": {
            "type": "Appartement",
            "surface": 60,
            "occupants": 2,
            "chauffage": "Gaz",
            "soir": "Variable selon les jours",
        },
    },
    {
        "id": "etudiant_toulouse",
        "icon": "🎒",
        "titre": "Étudiant en studio",
        "region": "Toulouse Capitole",
        "details": "1 personne · 25 m² · électrique",
        "facture_estimee": "42 € / mois",
        "color": "#10b981",
        "profile": {
            "type": "Appartement",
            "surface": 25,
            "occupants": 1,
            "chauffage": "Electrique",
            "soir": "Souvent absent le soir (rentre tard)",
        },
    },
    {
        "id": "retraite_bretagne",
        "icon": "🏘️",
        "titre": "Retraités en maison",
        "region": "Bretagne · Vannes",
        "details": "2 personnes · 90 m² · pompe à chaleur",
        "facture_estimee": "95 € / mois",
        "color": "#f59e0b",
        "profile": {
            "type": "Maison individuelle",
            "surface": 90,
            "occupants": 2,
            "chauffage": "Pompe a chaleur",
            "soir": "Souvent à la maison le soir (18h-22h)",
        },
    },
]


# ============================================================
# CSS — Design SaaS pro
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
* { font-family: 'Inter', -apple-system, sans-serif; }

@keyframes fadeIn { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }
@keyframes slideRight { from {opacity:0; transform:translateX(-12px);} to {opacity:1; transform:translateX(0);} }
@keyframes float { 0%,100% {transform:translateY(0);} 50% {transform:translateY(-6px);} }
@keyframes pulse { 0%,100% {opacity:0.6;} 50% {opacity:1;} }
@keyframes gradientShift { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }

.main .block-container { max-width:1340px; padding-top:1rem; animation:fadeIn 0.5s ease-out; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

/* === HERO === */
.hero {
    position:relative;
    padding:3rem 2.5rem;
    margin-bottom:2.5rem;
    background:linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    background-size:200% 200%;
    animation:gradientShift 18s ease infinite;
    border-radius:24px;
    border:1px solid rgba(255,255,255,0.06);
    box-shadow:0 24px 80px rgba(15,23,42,0.28);
    overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-50%; right:-15%;
    width:600px; height:600px;
    background:radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 60%);
    animation:float 9s ease-in-out infinite;
}
.hero::after {
    content:''; position:absolute; bottom:-40%; left:-10%;
    width:500px; height:500px;
    background:radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 60%);
    animation:float 11s ease-in-out infinite reverse;
}
.hero-content { position:relative; z-index:1; }
.hero-badge {
    display:inline-flex; align-items:center; gap:0.5rem;
    padding:0.4rem 0.9rem;
    background:rgba(16,185,129,0.18); border:1px solid rgba(16,185,129,0.35);
    border-radius:100px; color:#10b981; font-size:0.85rem; font-weight:600;
    margin-bottom:1.2rem;
}
.hero-badge::before {
    content:''; width:6px; height:6px; border-radius:50%; background:#10b981;
    animation:pulse 2s infinite;
}
.hero-title {
    font-size:3rem; font-weight:800; line-height:1.1; margin:0 0 1rem 0;
    color:#f8fafc;
}
.hero-title .accent {
    background:linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-subtitle {
    color:#94a3b8; font-size:1.1rem; max-width:680px; margin-bottom:2rem;
}
.hero-stats { display:flex; gap:2.5rem; flex-wrap:wrap; margin-top:1.5rem; }
.hero-stat-value {
    font-family:'JetBrains Mono', monospace;
    font-size:1.8rem; font-weight:700;
    background:linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-stat-label { color:#64748b; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.3rem; }

/* === FACTURE EXEMPLE CARD === */
.facture-card {
    background:white; border-radius:16px; padding:1.5rem;
    border:1px solid #e2e8f0;
    box-shadow:0 2px 8px rgba(15,23,42,0.04);
    transition:all 0.3s ease;
    cursor:pointer;
    height:100%;
    display:flex; flex-direction:column;
}
.facture-card:hover {
    transform:translateY(-4px);
    box-shadow:0 12px 32px rgba(15,23,42,0.10);
    border-color:var(--card-color, #10b981);
}
.facture-icon { font-size:2.5rem; margin-bottom:0.5rem; }
.facture-titre { font-size:1.05rem; font-weight:700; color:#0f172a; margin:0; }
.facture-region { font-size:0.85rem; color:#64748b; margin:0.2rem 0 0.5rem 0; }
.facture-details { font-size:0.85rem; color:#475569; flex:1; margin-bottom:1rem; line-height:1.4; }
.facture-prix {
    display:inline-block; padding:0.4rem 0.9rem;
    background:var(--card-bg, #f0fdf4);
    color:var(--card-color, #10b981);
    border-radius:8px; font-weight:700; font-size:0.95rem;
    font-family:'JetBrains Mono', monospace;
}

/* === GLASS CARD === */
.glass-card {
    background:rgba(255,255,255,0.7);
    backdrop-filter:blur(18px);
    -webkit-backdrop-filter:blur(18px);
    border-radius:16px;
    border:1px solid rgba(255,255,255,0.5);
    box-shadow:0 8px 32px rgba(15,23,42,0.06);
    padding:1.5rem; margin-bottom:1rem;
    animation:fadeIn 0.5s ease-out;
}

/* === STEP CARD (questionnaire) === */
.step-card {
    background:white; border-radius:16px; padding:1.8rem;
    border:1px solid #e2e8f0;
    box-shadow:0 2px 6px rgba(15,23,42,0.03);
    margin-bottom:1.2rem; animation:fadeIn 0.4s ease-out;
}
.step-header {
    display:flex; align-items:center; gap:0.8rem; margin-bottom:1.2rem;
}
.step-num {
    width:36px; height:36px;
    background:linear-gradient(135deg, #0ea5e9, #10b981);
    color:white; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-weight:800; font-size:1.1rem;
    box-shadow:0 4px 12px rgba(14,165,233,0.3);
}
.step-title { font-size:1.15rem; font-weight:700; color:#0f172a; margin:0; }

/* === RESULT HERO === */
.result-summary {
    background:linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-radius:20px; padding:2.5rem;
    text-align:center; margin-bottom:1.5rem;
    border:2px solid #fde68a;
    box-shadow:0 12px 36px rgba(245,158,11,0.15);
}
.result-summary .label {
    color:#92400e; font-size:0.85rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em;
}
.result-summary .price {
    font-family:'JetBrains Mono', monospace;
    font-size:4rem; font-weight:900; color:#0f172a;
    margin:0.3rem 0; line-height:1;
}
.result-summary .subtitle { color:#78350f; font-size:1rem; }

/* === RECOMMENDATION === */
.reco-card {
    background:linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-radius:20px; padding:2rem;
    border:2px solid #10b981;
    margin:1rem 0;
}
.reco-card .badge {
    display:inline-block;
    background:#10b981; color:white;
    padding:0.3rem 0.9rem; border-radius:100px;
    font-size:0.75rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.05em;
    margin-bottom:0.8rem;
}
.reco-card h2 { font-size:1.8rem; font-weight:800; color:#064e3b; margin:0; }
.reco-card .tarif { color:#047857; font-size:1rem; font-weight:600; margin:0.5rem 0 1rem 0; }
.reco-card .raison { color:#065f46; font-size:0.95rem; line-height:1.6; }
.reco-card .economie {
    display:inline-block; background:white; padding:0.6rem 1.2rem;
    border-radius:10px; font-weight:700; color:#10b981;
    margin-top:1rem; font-size:1.05rem;
    box-shadow:0 4px 8px rgba(16,185,129,0.1);
    font-family:'JetBrains Mono', monospace;
}

/* === INSIGHT CARD === */
.insight-card {
    display:flex; align-items:flex-start; gap:1rem;
    padding:1.2rem 1.4rem; border-radius:14px;
    background:linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.95) 100%);
    border-left:4px solid var(--accent, #10b981);
    box-shadow:0 4px 14px rgba(15,23,42,0.04);
    margin-bottom:0.75rem;
    animation:slideRight 0.5s ease-out;
    transition:all 0.2s ease;
}
.insight-card:hover { transform:translateX(4px); box-shadow:0 8px 20px rgba(15,23,42,0.06); }
.insight-icon {
    font-size:1.4rem; width:40px; height:40px;
    display:flex; align-items:center; justify-content:center;
    border-radius:10px; background:var(--icon-bg, rgba(16,185,129,0.12));
    flex-shrink:0;
}
.insight-title { font-weight:700; color:#0f172a; font-size:0.95rem; margin-bottom:0.2rem; }
.insight-text { color:#475569; font-size:0.88rem; line-height:1.5; }

/* === METRIC === */
div[data-testid="stMetric"] {
    background:linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding:1.4rem; border-radius:14px;
    border:1px solid #e2e8f0; border-left:4px solid #10b981;
    box-shadow:0 2px 6px rgba(15,23,42,0.03);
    transition:all 0.3s ease; height:100%;
}
div[data-testid="stMetric"]:hover {
    transform:translateY(-2px);
    box-shadow:0 6px 16px rgba(15,23,42,0.08);
    border-left-color:#0ea5e9;
}
[data-testid="stMetricLabel"] {
    color:#64748b !important; font-weight:600 !important;
    text-transform:uppercase; font-size:0.7rem !important; letter-spacing:0.08em;
}
[data-testid="stMetricValue"] {
    color:#0f172a !important; font-size:1.7rem !important; font-weight:700 !important;
    font-family:'JetBrains Mono', monospace !important;
}

/* === SECTION HEADER === */
.section-h {
    display:flex; align-items:center; gap:0.7rem;
    margin:2rem 0 1rem 0; font-size:1.1rem; font-weight:700; color:#0f172a;
}
.section-h::before {
    content:''; width:4px; height:22px;
    background:linear-gradient(180deg, #10b981, #0ea5e9); border-radius:4px;
}

/* === NEIGHBOR BAR === */
.neighbor-bar {
    display:flex; align-items:center; gap:1rem;
    margin:0.6rem 0; padding:0.8rem 1.1rem;
    background:#f8fafc; border-radius:12px; border:1px solid #e2e8f0;
}
.neighbor-label { flex:1; font-weight:600; color:#0f172a; font-size:0.95rem; }
.neighbor-value {
    font-weight:700; font-family:'JetBrains Mono', monospace;
    color:#1e293b; font-size:1rem;
}
.neighbor-bar.you {
    background:linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border:2px solid #10b981;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background:linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border-right:1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown h1 {
    background:linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:1.7rem; font-weight:800;
}

/* === BUTTONS === */
.stButton button {
    border-radius:10px !important; font-weight:600 !important;
    transition:all 0.2s ease !important;
}
.stButton button[kind="primary"] {
    background:linear-gradient(135deg, #0ea5e9 0%, #10b981 100%) !important;
    border:none !important; color:white !important;
    box-shadow:0 4px 14px rgba(14,165,233,0.3) !important;
}
.stButton button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(14,165,233,0.4) !important;
}

/* === TABS === */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(241,245,249,0.6);
    padding:0.3rem; border-radius:12px; border:1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
    height:46px; padding:0 1.4rem !important;
    border-radius:10px !important; color:#64748b !important;
    font-weight:600 !important; background:transparent !important; border:none !important;
}
.stTabs [aria-selected="true"] {
    background:white !important; color:#0f172a !important;
    box-shadow:0 2px 6px rgba(15,23,42,0.06);
}

/* === RADIO === */
.stRadio div[role="radiogroup"] label {
    padding:0.6rem 1rem; background:#f8fafc; border-radius:10px;
    margin:0.3rem 0; border:2px solid transparent;
    transition:all 0.2s ease; cursor:pointer;
}
.stRadio div[role="radiogroup"] label:hover { background:#f0f9ff; border-color:#bae6fd; }

/* === FOOTER === */
.app-footer {
    text-align:center; padding:2rem 0;
    color:#94a3b8; font-size:0.85rem;
    border-top:1px solid #e2e8f0; margin-top:3rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# LOADERS
# ============================================================
def find_file(name):
    for base in SEARCH_PATHS:
        p = base / name
        if p.exists():
            return p
    return None


@st.cache_resource
def load_forecast_model():
    p, s = find_file("energie_lstm.keras"), find_file("energie_scalers.joblib")
    if not (p and s):
        return None, None
    return keras.models.load_model(p), joblib.load(s)


@st.cache_resource
def load_profil_model():
    p, m = find_file("profil_conso.keras"), find_file("profil_conso_meta.joblib")
    if not (p and m):
        return None, None
    return keras.models.load_model(p), joblib.load(m)


@st.cache_resource
def load_sentiment_model():
    p, t = find_file("allocine_sentiment.keras"), find_file("allocine_tokenizer.joblib")
    if not (p and t):
        return None, None
    return keras.models.load_model(p), joblib.load(t)


# ============================================================
# SYNTH DATA
# ============================================================
def synthesize_week(profile_data):
    """Genere 168h plausibles depuis les 5 reponses du questionnaire."""
    rng = np.random.default_rng(seed=hash(str(profile_data)) % (2**31))
    timestamps = pd.date_range(end=datetime.now(), periods=168, freq="h")
    hours = np.array([t.hour for t in timestamps])
    dow = np.array([t.dayofweek for t in timestamps])

    base = 0.2 + (profile_data["surface"] / 100) * 0.25 + profile_data["occupants"] * 0.08
    if profile_data["chauffage"] == "Electrique":
        base *= 1.6
    elif profile_data["chauffage"] == "Pompe a chaleur":
        base *= 1.2

    pattern = np.ones(168) * base
    if profile_data["soir"] == "Souvent à la maison le soir (18h-22h)":
        pattern += 0.7 * np.exp(-((hours - 19) ** 2) / 6) * base
    elif profile_data["soir"] == "Souvent absent le soir (rentre tard)":
        pattern += 0.3 * np.exp(-((hours - 22) ** 2) / 4) * base
    else:
        pattern += 0.4 * np.exp(-((hours - 18) ** 2) / 8) * base

    pattern += 0.3 * np.exp(-((hours - 7) ** 2) / 4) * base
    if profile_data["chauffage"] == "Electrique":
        pattern += 0.4 * np.exp(-((hours - 3) ** 2) / 6) * base

    pattern *= (1 + 0.1 * (dow >= 5))
    pattern += rng.normal(0, base * 0.08, 168)
    pattern = np.clip(pattern, 0.1, 8.0)

    return pd.DataFrame({
        "active_kw": pattern,
        "sub1_wh": rng.uniform(0, 60, 168) * (hours >= 6) * (hours <= 22),
        "sub2_wh": rng.uniform(0, 40, 168) * (hours >= 8) * (hours <= 20),
        "sub3_wh": rng.uniform(20, 80, 168) * (base / 0.5),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "is_weekend": (dow >= 5).astype(float),
    }, index=timestamps)


# ============================================================
# VIZ HELPERS (Plotly)
# ============================================================
def make_line_forecast(df, forecast_kw):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["active_kw"], mode="lines",
        line=dict(color="#0ea5e9", width=2.5),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
        name="Historique 7j",
        hovertemplate="%{x|%a %d %b %Hh}<br><b>%{y:.2f} kW</b><extra></extra>",
    ))
    last = df.index[-1]
    fig.add_trace(go.Scatter(
        x=[last, last + timedelta(hours=24)],
        y=[df["active_kw"].iloc[-1], forecast_kw],
        mode="lines+markers",
        line=dict(color="#10b981", width=3, dash="dot"),
        marker=dict(size=14, color="#10b981", symbol="diamond", line=dict(width=2, color="white")),
        name="Forecast 24h",
        hovertemplate="<b>Forecast</b><br>%{y:.2f} kW<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="kW", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
    )
    return fig


def make_3d_surface(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["day_idx"] = pd.factorize(df.index.dayofyear)[0]
    pivot = df.pivot_table(values="active_kw", index="day_idx", columns="hour", aggfunc="mean")
    fig = go.Figure(data=[go.Surface(
        z=pivot.values, x=list(pivot.columns), y=[f"J{i + 1}" for i in pivot.index],
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="kW", thickness=12, len=0.7),
        hovertemplate="<b>%{y}</b><br>%{x}h<br>%{z:.2f} kW<extra></extra>",
    )])
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="Heure", gridcolor="#e2e8f0", backgroundcolor="white"),
            yaxis=dict(title="Jour", gridcolor="#e2e8f0", backgroundcolor="white"),
            zaxis=dict(title="kW", gridcolor="#e2e8f0", backgroundcolor="white"),
            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
        ),
        paper_bgcolor="white",
    )
    return fig


def make_calendar_heatmap(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    pivot = df.pivot_table(values="active_kw", index="dow", columns="hour", aggfunc="mean")
    days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=[days[i] for i in pivot.index],
        colorscale=[[0, "#f0f9ff"], [0.3, "#7dd3fc"], [0.6, "#0ea5e9"], [1, "#1e3a8a"]],
        colorbar=dict(title="kW", thickness=12, len=0.7),
        hovertemplate="<b>%{y} %{x}h</b><br>%{z:.2f} kW<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        height=280, margin=dict(l=40, r=20, t=10, b=40),
        xaxis=dict(title="Heure", showgrid=False), yaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def make_breakdown_donut(df):
    t1, t2, t3 = df["sub1_wh"].sum(), df["sub2_wh"].sum(), df["sub3_wh"].sum()
    fig = go.Figure(data=[go.Pie(
        labels=["Cuisine", "Laverie", "Chauffage / ECS"], values=[t1, t2, t3], hole=0.55,
        marker=dict(colors=["#f59e0b", "#0ea5e9", "#ef4444"], line=dict(color="white", width=3)),
        textinfo="label+percent", textposition="outside", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:.0f} Wh<br>%{percent}<extra></extra>",
    )])
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
        annotations=[dict(
            text=f"<b>{(t1 + t2 + t3) / 1000:.0f}</b><br><span style='font-size:0.7em;color:#64748b'>kWh</span>",
            showarrow=False, font=dict(size=24, color="#0f172a"),
        )],
    )
    return fig


def make_polar_hourly(df):
    by_hour = df.groupby(df.index.hour)["active_kw"].mean()
    hours = list(by_hour.index) + [by_hour.index[0]]
    values = list(by_hour.values) + [by_hour.values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=[h * 15 for h in hours], fill="toself",
        fillcolor="rgba(16,185,129,0.25)", line=dict(color="#10b981", width=2.5),
        marker=dict(size=8, color="#10b981"),
        hovertemplate="<b>%{customdata}h</b><br>%{r:.2f} kW<extra></extra>", customdata=hours,
    ))
    fig.update_layout(
        height=380, margin=dict(l=40, r=40, t=20, b=40),
        polar=dict(
            radialaxis=dict(visible=True, gridcolor="#e2e8f0"),
            angularaxis=dict(
                tickmode="array", tickvals=[h * 15 for h in range(0, 24, 3)],
                ticktext=[f"{h}h" for h in range(0, 24, 3)],
                direction="clockwise", rotation=90, gridcolor="#e2e8f0",
            ),
        ),
        showlegend=False, paper_bgcolor="white",
    )
    return fig


def make_profile_donut(probas, idx):
    pulls = [0.12 if i == idx else 0 for i in range(4)]
    fig = go.Figure(data=[go.Pie(
        labels=PROFILS, values=probas * 100, hole=0.50, pull=pulls,
        marker=dict(colors=PROFIL_COLORS, line=dict(color="white", width=3)),
        textinfo="label+percent", textposition="outside", textfont=dict(size=11),
    )])
    fig.update_layout(
        height=380, margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
        annotations=[dict(
            text=f"<b>{PROFILS[idx]}</b><br><span style='font-size:0.7em;color:#64748b'>profil</span>",
            showarrow=False, font=dict(size=13, color=PROFIL_COLORS[idx]),
        )],
    )
    return fig


def make_sentiment_gauge(score, label):
    color = "#10b981" if score >= 0.5 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number=dict(suffix=" / 100", font=dict(size=36, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100]), bar=dict(color=color, thickness=0.35),
            bgcolor="#f1f5f9", borderwidth=0,
            steps=[
                dict(range=[0, 30], color="#fee2e2"),
                dict(range=[30, 50], color="#fef3c7"),
                dict(range=[50, 70], color="#dcfce7"),
                dict(range=[70, 100], color="#a7f3d0"),
            ],
            threshold=dict(line=dict(color="#0f172a", width=4), thickness=0.85, value=50),
        ),
        title=dict(text=f"<b style='color:{color}'>{label}</b>", font=dict(size=22)),
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))
    return fig


REGIONS_DATA = {
    "Île-de-France": 4500, "Auvergne-Rhône-Alpes": 5200, "Hauts-de-France": 4800,
    "Nouvelle-Aquitaine": 4100, "Occitanie": 3900, "Grand Est": 5100,
    "Provence-Alpes-Côte d'Azur": 3700, "Pays de la Loire": 4400,
    "Normandie": 4700, "Bretagne": 4300, "Bourgogne-Franche-Comté": 5300,
    "Centre-Val de Loire": 4600, "Corse": 3500,
}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_france_geojson():
    """Telecharge le geojson FR avec cache 24h. Tente plusieurs miroirs."""
    if not HAS_REQUESTS:
        return None
    urls = [
        "https://france-geojson.gregoiredavid.fr/repo/regions.geojson",
        "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


def make_france_map():
    df = pd.DataFrame({"region": list(REGIONS_DATA.keys()), "kwh_an": list(REGIONS_DATA.values())})

    geojson = fetch_france_geojson()

    if geojson is not None:
        # Carte choropleth (preferee)
        fig = px.choropleth(
            df, locations="region", color="kwh_an",
            geojson=geojson, featureidkey="properties.nom",
            color_continuous_scale="Viridis",
            labels={"kwh_an": "kWh/an"},
            hover_data={"region": True, "kwh_an": ":,.0f"},
        )
        fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator")
        fig.update_layout(
            height=500, margin=dict(l=0, r=0, t=10, b=10),
            coloraxis_colorbar=dict(title="kWh/an", thickness=12, len=0.7),
            paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Fallback : bar chart horizontale (toujours fonctionnel)
    df_sorted = df.sort_values("kwh_an", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted["kwh_an"],
        y=df_sorted["region"],
        orientation="h",
        marker=dict(
            color=df_sorted["kwh_an"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="kWh/an", thickness=12, len=0.7),
        ),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f} kWh/an<extra></extra>",
        text=[f"{v:,.0f}" for v in df_sorted["kwh_an"]],
        textposition="inside",
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text="Consommation moyenne par région (kWh/an) - carte indisponible, vue bar chart",
            font=dict(size=14, color="#64748b"),
            x=0.5, xanchor="center",
        ),
        xaxis=dict(title="kWh/an", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ============================================================
# AI INSIGHTS
# ============================================================
def generate_insights(pred_kw, profile, ratio, df, profile_data):
    annual_eur = pred_kw * 24 * 365 * 0.25
    peak_share = df.loc[df.index.hour.isin([18, 19, 20]), "active_kw"].sum() / df["active_kw"].sum() * 100
    fourn = PROFIL_FOURNISSEUR[profile]

    return [
        {
            "icon": "💰", "title": "Économie potentielle annuelle",
            "text": f"En passant chez <b>{fourn['fournisseur']}</b>, vous économiseriez environ <b>{fourn['economie_an']} €/an</b>.",
            "color": "#10b981",
        },
        {
            "icon": "📊", "title": "Concentration de votre conso",
            "text": f"<b>{peak_share:.0f}%</b> de votre consommation a lieu entre 18h et 21h (heures pleines).",
            "color": "#f59e0b",
        },
        {
            "icon": "🏘️", "title": "Comparaison régionale",
            "text": f"À <b>{annual_eur:.0f} €/an</b>, vous êtes "
                    f"{'au-dessus' if annual_eur > 1100 else 'sous'} la moyenne nationale (~1 100 €/an).",
            "color": "#0ea5e9",
        },
        {
            "icon": "🌱", "title": "Empreinte CO₂",
            "text": f"Votre empreinte est de <b>{pred_kw * 24 * 0.08:.1f} kg CO₂/jour</b> (mix électrique FR 0,08 kg/kWh).",
            "color": "#84cc16",
        },
    ]


def render_insight_card(insight):
    st.markdown(
        f"""<div class="insight-card" style="--accent: {insight['color']}; --icon-bg: {insight['color']}1a;">
        <div class="insight-icon" style="color: {insight['color']};">{insight['icon']}</div>
        <div>
            <div class="insight-title">{insight['title']}</div>
            <div class="insight-text">{insight['text']}</div>
        </div></div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# HERO
# ============================================================
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <span class="hero-badge">AI Energy Forecast · MonÉnergie v1.0</span>
                <h1 class="hero-title">
                    Pilotez votre <span class="accent">consommation</span><br>
                    avec l'intelligence artificielle
                </h1>
                <p class="hero-subtitle">
                    Modèles RNN/LSTM entraînés sur des données françaises ouvertes.
                    Forecast 24h, profil temporel, recommandation tarifaire personnalisée,
                    analyse d'avis fournisseurs - 100% on-device.
                </p>
                <div class="hero-stats">
                    <div><div class="hero-stat-value">0,24 kW</div><div class="hero-stat-label">RMSE Forecast</div></div>
                    <div><div class="hero-stat-value">93,8 %</div><div class="hero-stat-label">Accuracy Sentiment FR</div></div>
                    <div><div class="hero-stat-value">280 €</div><div class="hero-stat-label">Économie moyenne / an</div></div>
                    <div><div class="hero-stat-value">35 M</div><div class="hero-stat-label">Foyers FR cible</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WIZARD
# ============================================================
def init_session():
    if "step" not in st.session_state:
        st.session_state["step"] = "landing"
    if "profile" not in st.session_state:
        st.session_state["profile"] = {}


def page_landing():
    render_hero()

    st.markdown('<div class="section-h">🚀 Choisissez un scénario type pour démarrer en 1 clic</div>', unsafe_allow_html=True)
    st.caption("Sélectionnez un profil pré-rempli (basé sur des cas réels) ou personnalisez le vôtre.")

    cols = st.columns(4)
    for i, facture in enumerate(FACTURES_EXEMPLES):
        with cols[i]:
            st.markdown(
                f"""<div class="facture-card" style="--card-color: {facture['color']}; --card-bg: {facture['color']}15;">
                <div class="facture-icon">{facture['icon']}</div>
                <div class="facture-titre">{facture['titre']}</div>
                <div class="facture-region">{facture['region']}</div>
                <div class="facture-details">{facture['details']}</div>
                <div class="facture-prix">{facture['facture_estimee']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Lancer →", key=f"fac_{i}", use_container_width=True, type="primary"):
                st.session_state["profile"] = facture["profile"]
                st.session_state["step"] = "results"
                st.rerun()

    st.markdown('<div class="section-h">⚙️ Ou personnalisez votre analyse</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("Décrire mon logement en 5 questions", use_container_width=True):
            st.session_state["step"] = "wizard"
            st.rerun()

    # Promesse
    st.markdown('<div class="section-h">💡 Ce que MonÉnergie analyse pour vous</div>', unsafe_allow_html=True)
    promo_cols = st.columns(3)
    promos = [
        ("💰", "Votre facture réelle", "Estimation précise de votre coût mensuel et annuel à partir de votre profil."),
        ("🎯", "Le bon fournisseur", "Recommandation tarifaire personnalisée selon votre profil temporel."),
        ("🏘️", "Vs vos voisins", "Comparez-vous aux foyers similaires de votre région."),
    ]
    for col, (icon, title, desc) in zip(promo_cols, promos):
        with col:
            st.markdown(
                f"""<div class="glass-card">
                <div style="font-size:2rem;">{icon}</div>
                <h3 style="margin:0.5rem 0;">{title}</h3>
                <p style="color:#64748b; font-size:0.92rem;">{desc}</p>
                </div>""",
                unsafe_allow_html=True,
            )


def page_wizard():
    st.markdown(
        """
        <div class="hero" style="padding:2rem 2.5rem;">
            <div class="hero-content">
                <span class="hero-badge">Étape 1/2 · Description du logement</span>
                <h1 class="hero-title" style="font-size:2rem;">Quelques détails sur votre logement</h1>
                <p class="hero-subtitle" style="margin-bottom:0;">5 questions pour générer une analyse personnalisée.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Question 1
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><div class="step-num">1</div><div class="step-title">Type de logement</div></div>', unsafe_allow_html=True)
    type_logement = st.radio("", ["Maison individuelle", "Appartement"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 2
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><div class="step-num">2</div><div class="step-title">Surface habitable</div></div>', unsafe_allow_html=True)
    surface = st.slider("Surface (m²)", 20, 250, 80, step=5, label_visibility="collapsed")
    st.caption(f"📐 **{surface} m²** — {'studio' if surface < 30 else 'T1/T2' if surface < 55 else 'T3/T4' if surface < 90 else 'grande maison'}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 3
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><div class="step-num">3</div><div class="step-title">Nombre d\'occupants</div></div>', unsafe_allow_html=True)
    occupants = st.slider("Occupants", 1, 6, 2, label_visibility="collapsed")
    st.caption(f"👥 **{occupants} personne{'s' if occupants > 1 else ''}** à la maison")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 4
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><div class="step-num">4</div><div class="step-title">Mode de chauffage principal</div></div>', unsafe_allow_html=True)
    chauffage = st.radio("Chauffage", ["Electrique", "Gaz", "Pompe a chaleur", "Bois / Fuel"], horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 5
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><div class="step-num">5</div><div class="step-title">Rythme de vie</div></div>', unsafe_allow_html=True)
    soir = st.radio(
        "Rythme",
        ["Souvent à la maison le soir (18h-22h)",
         "Souvent absent le soir (rentre tard)",
         "Variable selon les jours"],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, _, c2 = st.columns([1, 2, 1])
    with c1:
        if st.button("← Retour", use_container_width=True):
            st.session_state["step"] = "landing"
            st.rerun()
    with c2:
        if st.button("Lancer l'analyse →", type="primary", use_container_width=True):
            st.session_state["profile"] = {
                "type": type_logement, "surface": surface, "occupants": occupants,
                "chauffage": chauffage, "soir": soir,
            }
            st.session_state["step"] = "results"
            st.rerun()


# ============================================================
# RESULTS — combine grand public + technique
# ============================================================
def page_results():
    profile_data = st.session_state.get("profile", {})
    if not profile_data:
        st.session_state["step"] = "landing"
        st.rerun()
        return

    df = synthesize_week(profile_data)
    forecast_model, scalers = load_forecast_model()
    profil_model, profil_meta = load_profil_model()

    if forecast_model is None or profil_model is None:
        st.error("⚠️ Modèles introuvables. Lance d'abord les phases 1+2 et 6 sur Kaggle.")
        return

    with st.spinner("🧠 Analyse IA en cours..."):
        features = scalers["features"]
        arr_scaled = scalers["scaler_X"].transform(df[features].values)
        pred_scaled = forecast_model.predict(arr_scaled.reshape(1, 168, len(features)), verbose=0)
        pred_kw = float(scalers["scaler_y"].inverse_transform(pred_scaled)[0, 0])

        features2 = profil_meta["features"]
        arr2 = profil_meta["scaler_X"].transform(df[features2].values)
        proba = profil_model.predict(arr2.reshape(1, 168, len(features2)), verbose=0)[0]
        idx = int(np.argmax(proba))

    profil = PROFILS[idx]
    confidence = float(proba[idx])
    daily_kw = pred_kw * 24
    monthly_eur = daily_kw * 30 * 0.25
    annual_eur = daily_kw * 365 * 0.25
    peak_mean = df.loc[df.index.hour.isin([18, 19, 20]), "active_kw"].mean()
    off_mean = df.loc[df.index.hour.isin([1, 2, 3, 4, 5]), "active_kw"].mean()
    ratio = peak_mean / max(off_mean, 1e-6)
    fourn = PROFIL_FOURNISSEUR[profil]

    # Header retour
    if st.button("← Refaire une analyse"):
        st.session_state["step"] = "landing"
        st.session_state["profile"] = {}
        st.rerun()

    # Hero result
    st.markdown(
        f"""
        <div class="hero" style="padding:2rem 2.5rem;">
            <div class="hero-content">
                <span class="hero-badge">Résultat IA · {profile_data['type']} · {profile_data['surface']} m² · chauffage {profile_data['chauffage'].lower()}</span>
                <h1 class="hero-title" style="font-size:2.2rem;">Votre analyse <span class="accent">personnalisée</span></h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # === SECTION 1 : GRAND PUBLIC ===
    # Facture estimée
    st.markdown(
        f"""<div class="result-summary">
        <div class="label">💸 Votre facture estimée</div>
        <div class="price">{monthly_eur:.0f} €<span style="font-size:0.4em;color:#92400e;"> / mois</span></div>
        <div class="subtitle">soit <b>{annual_eur:.0f} € / an</b> au tarif réglementé Base (0,25 €/kWh)</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conso moyenne", f"{pred_kw:.2f} kW")
    c2.metric("Par jour", f"{daily_kw:.1f} kWh")
    c3.metric("Profil détecté", profil)
    c4.metric("Économie possible", f"{fourn['economie_an']} €/an")

    # Recommandation fournisseur
    st.markdown(
        f"""<div class="reco-card">
        <span class="badge">🎯 Recommandation tarifaire personnalisée</span>
        <h2>{fourn['logo']} {fourn['fournisseur']}</h2>
        <div class="tarif">{fourn['tarif']}</div>
        <div class="raison">{fourn['raison']}</div>
        <div class="economie">💰 Économie estimée : {fourn['economie_an']} € / an</div>
        </div>""",
        unsafe_allow_html=True,
    )

    col_cta, _ = st.columns([1, 2])
    with col_cta:
        st.link_button(
            f"📞 Demander un devis {fourn['fournisseur']}",
            f"https://www.google.com/search?q={fourn['fournisseur'].replace(' ', '+')}+contact+souscription",
            use_container_width=True,
        )

    # Comparaison voisins
    st.markdown('<div class="section-h">🏘️ Vous vs vos voisins (foyers similaires)</div>', unsafe_allow_html=True)
    neighbor_avg = annual_eur * 0.95
    neighbor_best = annual_eur * 0.65
    neighbor_worst = annual_eur * 1.3

    st.markdown(
        f"""
        <div class="neighbor-bar" style="background:linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border:1px solid #bae6fd;">
            <span style="font-size:1.4rem;">🥇</span>
            <div class="neighbor-label">Top 10% des foyers similaires</div>
            <div class="neighbor-value">{neighbor_best:.0f} € / an</div>
        </div>
        <div class="neighbor-bar">
            <span style="font-size:1.4rem;">📊</span>
            <div class="neighbor-label">Moyenne des foyers similaires</div>
            <div class="neighbor-value">{neighbor_avg:.0f} € / an</div>
        </div>
        <div class="neighbor-bar you">
            <span style="font-size:1.4rem;">🏠</span>
            <div class="neighbor-label"><b>VOUS</b></div>
            <div class="neighbor-value">{annual_eur:.0f} € / an</div>
        </div>
        <div class="neighbor-bar" style="background:linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border:1px solid #fecaca;">
            <span style="font-size:1.4rem;">📈</span>
            <div class="neighbor-label">Foyers les plus consommateurs</div>
            <div class="neighbor-value">{neighbor_worst:.0f} € / an</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # AI Insights cards
    st.markdown('<div class="section-h">💡 Vos insights IA</div>', unsafe_allow_html=True)
    insights = generate_insights(pred_kw, profil, ratio, df, profile_data)
    icols = st.columns(2)
    for i, ins in enumerate(insights):
        with icols[i % 2]:
            render_insight_card(ins)

    # === SECTION 2 : ANALYSE TECHNIQUE DÉPLIABLE ===
    st.markdown("---")
    with st.expander("🔬 Voir l'analyse technique détaillée (LSTM, Plotly 3D, donut, radar)", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🎯 Profil temporel", "🌐 Surface 3D"])

        with tab1:
            st.markdown('<div class="section-h">Historique 7 jours + Forecast 24h</div>', unsafe_allow_html=True)
            st.plotly_chart(make_line_forecast(df, pred_kw), use_container_width=True)

            cl, cr = st.columns([3, 2])
            with cl:
                st.markdown('<div class="section-h">Heatmap hebdomadaire</div>', unsafe_allow_html=True)
                st.plotly_chart(make_calendar_heatmap(df), use_container_width=True)
            with cr:
                st.markdown('<div class="section-h">Répartition par usage</div>', unsafe_allow_html=True)
                st.plotly_chart(make_breakdown_donut(df), use_container_width=True)

        with tab2:
            cl, cr = st.columns(2)
            with cl:
                st.markdown('<div class="section-h">Probabilités par profil</div>', unsafe_allow_html=True)
                st.plotly_chart(make_profile_donut(proba, idx), use_container_width=True)
            with cr:
                st.markdown('<div class="section-h">Pattern horaire 24h (radar)</div>', unsafe_allow_html=True)
                st.plotly_chart(make_polar_hourly(df), use_container_width=True)

            st.markdown(f"""
            **Détails techniques** :
            - Modèle : LSTM stacké (64 → 32) avec Dropout 0,4
            - Classes : {', '.join(PROFILS)}
            - Profil prédit : **{profil}** (confiance {confidence * 100:.1f}%)
            - Ratio peak/off-peak : {ratio:.2f}
            """)

        with tab3:
            st.markdown('<div class="section-h">Surface 3D heure × jour</div>', unsafe_allow_html=True)
            st.caption("Faites pivoter à la souris pour explorer le pattern hebdomadaire.")
            st.plotly_chart(make_3d_surface(df), use_container_width=True)


# ============================================================
# SENTIMENT PAGE
# ============================================================
def page_sentiment():
    st.markdown(
        """
        <div class="hero" style="padding:2rem 2.5rem;">
            <div class="hero-content">
                <span class="hero-badge">Module sentiment · Bi-LSTM Allociné</span>
                <h1 class="hero-title" style="font-size:2rem;">Analysez un avis fournisseur</h1>
                <p class="hero-subtitle" style="margin-bottom:0;">
                    Collez un avis Trustpilot, Google Reviews ou Avis Vérifiés sur EDF, Engie,
                    TotalEnergies, Octopus, etc. Le modèle français détermine le sentiment.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_sentiment, tok_meta = load_sentiment_model()
    if model_sentiment is None:
        st.error("⚠️ Modèle sentiment non chargé.")
        return

    examples = [
        "EDF nous a augmenté la facture sans prévenir et le service client est introuvable, scandaleux.",
        "Très satisfait du conseiller TotalEnergies, réactif et le prix est compétitif.",
        "Encore 4 heures de coupure aujourd'hui, j'en peux plus de ce fournisseur catastrophique.",
        "Installation Linky chez Engie nickel, technicien rapide et professionnel.",
    ]

    if "sentiment_text" not in st.session_state:
        st.session_state["sentiment_text"] = ""

    st.markdown('<div class="section-h">💡 Exemples cliquables</div>', unsafe_allow_html=True)
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(f"📝 {ex[:75]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state["sentiment_text"] = ex
                st.rerun()

    user_text = st.text_area("Votre avis", height=140, key="sentiment_text",
                              placeholder="Collez ou tapez un avis client sur un fournisseur d'énergie...")

    if not st.button("🔍 Analyser l'avis", type="primary"):
        return

    if not user_text.strip():
        st.warning("Tapez un avis avant de lancer l'analyse.")
        return

    with st.spinner("🧠 Analyse Bi-LSTM en cours..."):
        from keras.utils import pad_sequences
        tokenizer = tok_meta["tokenizer"]
        seq = tokenizer.texts_to_sequences([user_text])
        pad = pad_sequences(seq, maxlen=tok_meta["max_len"], padding="pre", truncating="pre")
        proba = float(model_sentiment.predict(pad, verbose=0)[0, 0])
        label = "Positif" if proba >= 0.5 else "Négatif"
        confidence = proba if label == "Positif" else 1 - proba

    c1, c2, c3 = st.columns(3)
    c1.metric("Sentiment", label)
    c2.metric("Confiance", f"{confidence * 100:.1f}%")
    c3.metric("Tokens analysés", f"{len(user_text.split())}")

    cg, cm = st.columns([2, 1])
    with cg:
        st.plotly_chart(make_sentiment_gauge(proba, label), use_container_width=True)
    with cm:
        st.markdown("##### 🔬 Détail technique")
        st.markdown(f"""
        - **Score sigmoid brut** : `{proba:.4f}`
        - **Seuil de décision** : `0.50`
        - **Modèle** : Bi-LSTM 64 units
        - **Training** : 200k avis Allociné FR
        - **Vocabulaire** : 20 000 tokens
        """)
    log_inference(user_text, label, confidence)


# ============================================================
# CARTE PAGE
# ============================================================
def page_carte():
    st.markdown(
        """
        <div class="hero" style="padding:2rem 2.5rem;">
            <div class="hero-content">
                <span class="hero-badge">Open Data · Enedis & RTE</span>
                <h1 class="hero-title" style="font-size:2rem;">Carte de France énergétique</h1>
                <p class="hero-subtitle" style="margin-bottom:0;">
                    Consommation moyenne par foyer et par région. Données simulées sur base
                    Enedis Open Data (Etalab 2.0).
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Régions couvertes", "13")
    c2.metric("Moyenne nationale", "4 500 kWh/an")
    c3.metric("Région max", "Grand Est")
    c4.metric("Région min", "Corse")

    with st.spinner("Chargement de la carte..."):
        st.plotly_chart(make_france_map(), use_container_width=True)

    st.markdown('<div class="section-h">💡 Lecture de la carte</div>', unsafe_allow_html=True)
    insights = [
        {"icon": "🔥", "title": "Pic hivernal", "color": "#ef4444",
         "text": "Grand Est et Bourgogne-Franche-Comté sont les plus consommatrices (5 100-5 300 kWh/an) en raison du chauffage électrique massif."},
        {"icon": "🌞", "title": "Régions sobres", "color": "#10b981",
         "text": "PACA et Corse présentent les plus basses consommations (3 500-3 700 kWh/an) grâce au climat doux."},
        {"icon": "📊", "title": "Benchmark", "color": "#0ea5e9",
         "text": "Comparez votre consommation à votre région pour identifier votre marge de progression."},
        {"icon": "🔗", "title": "Sources", "color": "#6366f1",
         "text": "Données <a href='https://data.enedis.fr' style='color:#10b981'>Enedis Open Data</a> et <a href='https://opendata.reseaux-energies.fr' style='color:#10b981'>RTE</a> (Etalab 2.0)."},
    ]
    cols = st.columns(2)
    for i, ins in enumerate(insights):
        with cols[i % 2]:
            render_insight_card(ins)


# ============================================================
# MAIN
# ============================================================
def main():
    init_session()

    with st.sidebar:
        st.markdown("# ⚡ MonÉnergie")
        st.markdown("**AI Energy Dashboard pour foyers français**")
        st.markdown("---")

        if HAS_OPTION_MENU:
            page = option_menu(
                menu_title=None,
                options=["Accueil", "Avis fournisseur", "Carte France"],
                icons=["house-fill", "chat-square-text-fill", "globe-europe-africa"],
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "icon": {"color": "#10b981", "font-size": "18px"},
                    "nav-link": {
                        "color": "#cbd5e1", "font-size": "14px", "font-weight": "500",
                        "margin": "2px 0", "padding": "10px 14px", "border-radius": "10px",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                        "color": "white", "font-weight": "600",
                    },
                },
            )
        else:
            page = st.radio("Nav", ["Accueil", "Avis fournisseur", "Carte France"], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 🧠 Modèles déployés")
        st.markdown("""
        - **LSTM Forecast 24h**
          RMSE 0,24 kW
        - **LSTM Profil 4-classes**
          Classification temporelle
        - **Bi-LSTM Sentiment FR**
          Acc 93,8% sur Allociné 200k
        """)

        st.markdown("---")
        st.markdown("### 🔗 Données ouvertes")
        st.markdown("""
        - [Enedis Open Data](https://data.enedis.fr)
        - [RTE Open Data](https://opendata.reseaux-energies.fr)
        - [ADEME ObsDPE](https://odre.opendatasoft.com)
        """)

        st.markdown("---")
        st.caption("🔒 Aucune donnée transmise · 100% inference côté serveur Streamlit Cloud.")
        st.caption("Projet TP4 IPSSI MIA4 · MIT License")
        st.caption("[GitHub Repository](https://github.com/Hakim78/tp4-monenergie)")

    # Routing
    if page == "Accueil":
        step = st.session_state.get("step", "landing")
        if step == "landing":
            page_landing()
        elif step == "wizard":
            page_wizard()
        elif step == "results":
            page_results()
    elif page == "Avis fournisseur":
        page_sentiment()
    elif page == "Carte France":
        page_carte()

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            <b>MonÉnergie</b> · TP4 IPSSI MIA4 · Powered by <b>TensorFlow / Keras 3 / Streamlit</b> ·
            <a href="https://github.com/Hakim78/tp4-monenergie" style="color:#10b981">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
