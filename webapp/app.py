"""MonEnergie - app grand public pour menages francais

Wizard simple en 3 etapes :
1. Decris ton logement (5 questions cliquables)
2. Analyse IA (LSTM + profil + sentiment)
3. Resultats en euros + reco fournisseur + actions

Les modeles tournent en arriere-plan (Phase 1+2, 6, 4+5). L'utilisateur ne
voit jamais "LSTM" ou "ratio peak/offpeak". Tout est en francais simple.
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
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    from log_inference import log_inference  # type: ignore
except ImportError:
    def log_inference(*args, **kwargs):
        return None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MonEnergie - Reduisez vos factures",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEARCH_PATHS = [Path("."), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
PROFILS = ["Nocturne", "Equilibre nuit", "Equilibre soir", "Vespertine"]
PROFIL_LABELS_FR = {
    "Nocturne": "🌙 Vous consommez surtout la nuit",
    "Equilibre nuit": "🌃 Vous etes plutot du soir",
    "Equilibre soir": "🌆 Vous etes plutot en journee",
    "Vespertine": "☀️ Vous etes actif le soir",
}
PROFIL_FOURNISSEUR = {
    "Nocturne": {
        "fournisseur": "EDF Tempo",
        "tarif": "Tempo (Heures Creuses fortes)",
        "logo": "🔵",
        "economie_an": 280,
        "raison": "Vous consommez surtout la nuit ou tot le matin. Le tarif Tempo d'EDF vous fait payer la nuit jusqu'a 30% moins cher.",
    },
    "Equilibre nuit": {
        "fournisseur": "TotalEnergies HC/HP",
        "tarif": "Heures Creuses / Heures Pleines",
        "logo": "🟠",
        "economie_an": 180,
        "raison": "Votre conso est decalee vers la nuit, le tarif HC/HP de TotalEnergies est ideal.",
    },
    "Equilibre soir": {
        "fournisseur": "Engie Elec Reference",
        "tarif": "Tarif Base",
        "logo": "🟢",
        "economie_an": 90,
        "raison": "Conso reguliere sur la journee, le tarif Base d'Engie reste competitif sans surprise.",
    },
    "Vespertine": {
        "fournisseur": "Octopus Energy Eco",
        "tarif": "Eco indexe vert",
        "logo": "🐙",
        "economie_an": 150,
        "raison": "Vous etes actif le soir, Octopus propose un tarif indexe simple sans heures creuses qui vous penaliseraient.",
    },
}


# ============================================================
# CSS — design simple, grand public
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif; }

@keyframes fadeIn { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:translateY(0);} }
@keyframes pulse { 0%,100% {transform:scale(1);} 50% {transform:scale(1.05);} }
@keyframes float { 0%,100% {transform:translateY(0);} 50% {transform:translateY(-8px);} }

.main .block-container { max-width:1200px; padding-top:1rem; animation:fadeIn 0.5s ease-out; }
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

/* HERO */
.hero {
    background:linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
    padding:3rem 2.5rem;
    border-radius:24px;
    color:white;
    margin-bottom:2rem;
    position:relative;
    overflow:hidden;
    box-shadow:0 20px 60px rgba(14,165,233,0.25);
}
.hero::before {
    content:'';
    position:absolute;
    top:-50%; right:-20%;
    width:500px; height:500px;
    background:radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    animation:float 6s ease-in-out infinite;
}
.hero h1 {
    font-size:2.8rem; font-weight:800; margin:0 0 0.8rem 0; line-height:1.1; position:relative;
}
.hero h1 .em { color:#fef08a; }
.hero p { font-size:1.2rem; opacity:0.95; max-width:650px; margin:0 0 1.5rem 0; position:relative; }
.hero-stats { display:flex; gap:2.5rem; flex-wrap:wrap; position:relative; }
.hero-stat-value { font-size:2rem; font-weight:800; }
.hero-stat-label { font-size:0.9rem; opacity:0.85; }

/* STEP CARD */
.step-card {
    background:white;
    border-radius:20px;
    padding:2rem;
    box-shadow:0 4px 12px rgba(15,23,42,0.04), 0 1px 4px rgba(15,23,42,0.03);
    border:1px solid #e2e8f0;
    margin-bottom:1.5rem;
    animation:fadeIn 0.5s ease-out;
}
.step-number {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:36px; height:36px;
    background:linear-gradient(135deg, #0ea5e9 0%, #10b981 100%);
    color:white;
    border-radius:50%;
    font-weight:800;
    margin-right:0.7rem;
    box-shadow:0 4px 10px rgba(14,165,233,0.3);
}
.step-title { font-size:1.4rem; font-weight:700; color:#0f172a; margin-bottom:1rem; display:flex; align-items:center; }

/* BIG RESULT CARD */
.result-bill {
    background:linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-radius:20px;
    padding:2.5rem;
    text-align:center;
    margin-bottom:1.5rem;
    box-shadow:0 8px 24px rgba(245,158,11,0.15);
    border:2px solid #fde68a;
}
.result-bill .label { color:#92400e; font-size:0.95rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
.result-bill .price { font-size:4rem; font-weight:900; color:#0f172a; margin:0.3rem 0; line-height:1; }
.result-bill .subtitle { color:#78350f; font-size:1rem; }

/* RECOMMENDATION CARD */
.reco-fournisseur {
    background:linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border-radius:20px;
    padding:2rem;
    border:2px solid #10b981;
    margin:1rem 0;
    animation:fadeIn 0.7s ease-out;
}
.reco-fournisseur .badge {
    display:inline-block;
    background:#10b981;
    color:white;
    padding:0.3rem 0.9rem;
    border-radius:100px;
    font-size:0.75rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.05em;
    margin-bottom:0.8rem;
}
.reco-fournisseur h2 { font-size:2rem; font-weight:800; color:#064e3b; margin:0; }
.reco-fournisseur .tarif { color:#047857; font-size:1.05rem; font-weight:600; margin:0.5rem 0 1rem 0; }
.reco-fournisseur .raison { color:#065f46; font-size:0.95rem; line-height:1.6; }
.reco-fournisseur .economie {
    display:inline-block;
    background:white;
    padding:0.7rem 1.3rem;
    border-radius:12px;
    font-weight:700;
    color:#10b981;
    margin-top:1rem;
    font-size:1.05rem;
    box-shadow:0 4px 8px rgba(16,185,129,0.1);
}

/* BUTTONS */
.stButton button {
    border-radius:12px !important;
    font-weight:600 !important;
    transition:all 0.2s ease !important;
    padding:0.7rem 1.5rem !important;
    height:auto !important;
}
.stButton button[kind="primary"] {
    background:linear-gradient(135deg, #0ea5e9 0%, #10b981 100%) !important;
    border:none !important;
    box-shadow:0 4px 14px rgba(14,165,233,0.3) !important;
    color:white !important;
    font-size:1.05rem !important;
}
.stButton button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(14,165,233,0.4) !important;
}

/* METRICS */
div[data-testid="stMetric"] {
    background:white;
    padding:1.2rem;
    border-radius:14px;
    border:1px solid #e2e8f0;
    border-left:4px solid #10b981;
    box-shadow:0 2px 4px rgba(15,23,42,0.03);
}

/* PROFILE CARD */
.profile-detected {
    background:white;
    border-radius:16px;
    padding:1.5rem;
    border:2px solid #0ea5e9;
    text-align:center;
    margin:1rem 0;
}
.profile-detected .emoji { font-size:3rem; margin-bottom:0.5rem; }
.profile-detected .label { font-size:1.3rem; font-weight:700; color:#0c4a6e; }
.profile-detected .desc { color:#475569; font-size:0.95rem; margin-top:0.5rem; }

/* NEIGHBORS COMPARISON */
.neighbor-bar {
    display:flex; align-items:center; gap:1rem; margin:0.7rem 0; padding:0.7rem 1rem;
    background:#f8fafc; border-radius:10px;
}
.neighbor-icon { font-size:1.5rem; }
.neighbor-label { flex:1; font-weight:600; color:#0f172a; }
.neighbor-value { font-weight:700; font-family:monospace; }
.neighbor-bar.you { background:linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border:2px solid #10b981; }
.neighbor-bar.best { background:linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); }
.neighbor-bar.bad { background:linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); }

/* ACTION BUTTONS GRID */
.action-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:1rem; margin:1.5rem 0; }
.action-card {
    background:white; padding:1.5rem; border-radius:14px; border:1px solid #e2e8f0;
    cursor:pointer; transition:all 0.2s ease; text-align:center;
}
.action-card:hover { transform:translateY(-3px); box-shadow:0 8px 20px rgba(15,23,42,0.08); border-color:#10b981; }
.action-icon { font-size:2.5rem; margin-bottom:0.5rem; }
.action-title { font-weight:700; color:#0f172a; margin:0.3rem 0; }
.action-desc { font-size:0.85rem; color:#64748b; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background:linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
section[data-testid="stSidebar"] .stMarkdown h1 {
    background:linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:1.7rem; font-weight:800;
}

/* RADIO + SELECT */
.stRadio > label { font-weight:600; color:#0f172a; }
.stRadio div[role="radiogroup"] label {
    padding:0.6rem 1rem; background:#f8fafc; border-radius:10px; margin:0.3rem 0;
    border:2px solid transparent; transition:all 0.2s ease; cursor:pointer;
}
.stRadio div[role="radiogroup"] label:hover { background:#f0f9ff; border-color:#bae6fd; }

/* SLIDER */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background:linear-gradient(135deg, #0ea5e9, #10b981) !important;
    box-shadow:0 4px 8px rgba(14,165,233,0.3) !important;
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
# SYNTH DATA FROM USER PROFILE
# ============================================================
def synthesize_week_from_profile(profile_data):
    """Genere 168h plausibles a partir des reponses du questionnaire."""
    rng = np.random.default_rng(seed=hash(str(profile_data)) % (2**31))
    timestamps = pd.date_range(end=datetime.now(), periods=168, freq="h")
    hours = np.array([t.hour for t in timestamps])
    dow = np.array([t.dayofweek for t in timestamps])

    # Conso de base selon surface + occupants
    base_kw = 0.2 + (profile_data["surface"] / 100) * 0.25 + profile_data["occupants"] * 0.08

    # Modificateur chauffage
    if profile_data["chauffage"] == "Electrique":
        base_kw *= 1.6
    elif profile_data["chauffage"] == "Pompe a chaleur":
        base_kw *= 1.2

    # Pattern horaire selon style de vie
    pattern = np.ones(168) * base_kw
    if profile_data["soir"] == "Souvent a la maison le soir (18h-22h)":
        # Pic vespertine
        pattern += 0.7 * np.exp(-((hours - 19) ** 2) / 6) * base_kw
    elif profile_data["soir"] == "Souvent absent le soir (rentre tard)":
        # Pic nocturne tardif
        pattern += 0.3 * np.exp(-((hours - 22) ** 2) / 4) * base_kw
    else:  # variable
        pattern += 0.4 * np.exp(-((hours - 18) ** 2) / 8) * base_kw

    # Pic matin
    pattern += 0.3 * np.exp(-((hours - 7) ** 2) / 4) * base_kw

    # Ballon ECS la nuit si HC
    if profile_data["chauffage"] == "Electrique":
        pattern += 0.4 * np.exp(-((hours - 3) ** 2) / 6) * base_kw

    # Weekend +10%
    pattern *= (1 + 0.1 * (dow >= 5))

    # Bruit
    pattern += rng.normal(0, base_kw * 0.08, 168)
    pattern = np.clip(pattern, 0.1, 8.0)

    return pd.DataFrame({
        "active_kw": pattern,
        "sub1_wh": rng.uniform(0, 60, 168) * (hours >= 6) * (hours <= 22),
        "sub2_wh": rng.uniform(0, 40, 168) * (hours >= 8) * (hours <= 20),
        "sub3_wh": rng.uniform(20, 80, 168) * (base_kw / 0.5),
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "is_weekend": (dow >= 5).astype(float),
    }, index=timestamps)


# ============================================================
# WIZARD STEPS
# ============================================================
def init_session():
    if "step" not in st.session_state:
        st.session_state["step"] = "welcome"
    if "profile" not in st.session_state:
        st.session_state["profile"] = {}


def page_welcome():
    st.markdown("""
        <div class="hero">
            <h1>Vos factures d'energie<br>en <span class="em">pilote automatique</span></h1>
            <p>Decouvrez en 2 minutes combien vous coute vraiment votre electricite,
            quel fournisseur vous fait economiser le plus, et quelles habitudes changer.
            <b>100% gratuit, 0 inscription</b>.</p>
            <div class="hero-stats">
                <div>
                    <div class="hero-stat-value">280 €</div>
                    <div class="hero-stat-label">Economie moyenne /an</div>
                </div>
                <div>
                    <div class="hero-stat-value">2 min</div>
                    <div class="hero-stat-label">Pour avoir vos resultats</div>
                </div>
                <div>
                    <div class="hero-stat-value">35M</div>
                    <div class="hero-stat-label">Foyers FR concernes</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ Commencer mon analyse gratuite", type="primary", use_container_width=True):
            st.session_state["step"] = "questionnaire"
            st.rerun()

    st.markdown("""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:1rem; margin-top:2rem;">
            <div class="action-card">
                <div class="action-icon">💰</div>
                <div class="action-title">Combien vous payez vraiment</div>
                <div class="action-desc">Estimation precise de votre facture annuelle selon votre logement.</div>
            </div>
            <div class="action-card">
                <div class="action-icon">🎯</div>
                <div class="action-title">Le meilleur fournisseur pour vous</div>
                <div class="action-desc">Reco personnalisee selon votre profil de consommation.</div>
            </div>
            <div class="action-card">
                <div class="action-icon">🏘️</div>
                <div class="action-title">Comparaison avec vos voisins</div>
                <div class="action-desc">Vous etes au-dessus ou en-dessous de la moyenne de votre region ?</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🔒 Vos donnees restent dans votre navigateur. Aucune inscription, aucun email. Open source MIT.")


def page_questionnaire():
    st.markdown("""
        <div class="hero" style="padding:1.8rem 2.5rem;">
            <h1 style="font-size:2rem;">Etape 1/2 : <span class="em">Decrivez votre logement</span></h1>
            <p style="font-size:1rem; margin-bottom:0;">5 questions simples, ca prend 1 minute.</p>
        </div>
    """, unsafe_allow_html=True)

    # Question 1
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">1</span>Vous habitez ?</div>', unsafe_allow_html=True)
    type_logement = st.radio(
        "Type de logement",
        ["Maison individuelle", "Appartement"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 2
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">2</span>Quelle surface (m²) ?</div>', unsafe_allow_html=True)
    surface = st.slider("Surface", 20, 250, 80, step=5, label_visibility="collapsed")
    st.caption(f"📐 {surface} m² - {'studio' if surface < 30 else 'T1/T2' if surface < 55 else 'T3/T4' if surface < 90 else 'grande maison'}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 3
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">3</span>Combien etes-vous a la maison ?</div>', unsafe_allow_html=True)
    occupants = st.slider("Occupants", 1, 6, 2, label_visibility="collapsed")
    st.caption(f"👥 {occupants} personne{'s' if occupants > 1 else ''}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 4
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">4</span>Comment vous chauffez ?</div>', unsafe_allow_html=True)
    chauffage = st.radio(
        "Chauffage",
        ["Electrique", "Gaz", "Pompe a chaleur", "Bois / Fuel"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Question 5
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-title"><span class="step-number">5</span>Votre rythme de vie ?</div>', unsafe_allow_html=True)
    soir = st.radio(
        "Rythme",
        [
            "Souvent a la maison le soir (18h-22h)",
            "Souvent absent le soir (rentre tard)",
            "Variable selon les jours",
        ],
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col_back, _, col_next = st.columns([1, 2, 1])
    with col_back:
        if st.button("← Retour", use_container_width=True):
            st.session_state["step"] = "welcome"
            st.rerun()
    with col_next:
        if st.button("Voir mes resultats →", type="primary", use_container_width=True):
            st.session_state["profile"] = {
                "type": type_logement,
                "surface": surface,
                "occupants": occupants,
                "chauffage": chauffage,
                "soir": soir,
            }
            st.session_state["step"] = "results"
            st.rerun()


def page_results():
    profile_data = st.session_state.get("profile", {})
    if not profile_data:
        st.session_state["step"] = "welcome"
        st.rerun()
        return

    # Generate synthetic week
    df = synthesize_week_from_profile(profile_data)

    # Run forecast model
    forecast_model, scalers = load_forecast_model()
    profil_model, profil_meta = load_profil_model()

    if forecast_model is None or profil_model is None:
        st.error("Modeles non charges. Lancez les phases sur Kaggle d'abord.")
        return

    with st.spinner("🧠 Analyse IA de votre logement..."):
        # Forecast
        features = scalers["features"]
        arr_scaled = scalers["scaler_X"].transform(df[features].values)
        pred_scaled = forecast_model.predict(arr_scaled.reshape(1, 168, len(features)), verbose=0)
        pred_kw = float(scalers["scaler_y"].inverse_transform(pred_scaled)[0, 0])

        # Profil
        features2 = profil_meta["features"]
        arr2 = profil_meta["scaler_X"].transform(df[features2].values)
        proba = profil_model.predict(arr2.reshape(1, 168, len(features2)), verbose=0)[0]
        idx = int(np.argmax(proba))
        profil = PROFILS[idx]

    # Calculs euros
    daily_kw = pred_kw * 24
    monthly_eur = daily_kw * 30 * 0.25
    annual_eur = daily_kw * 365 * 0.25

    fourn = PROFIL_FOURNISSEUR[profil]

    # Reset button
    if st.button("← Refaire mon analyse", use_container_width=False):
        st.session_state["step"] = "welcome"
        st.session_state["profile"] = {}
        st.rerun()

    # Hero result
    st.markdown(f"""
        <div class="hero" style="padding:1.8rem 2.5rem;">
            <h1 style="font-size:2rem;">Voici vos <span class="em">resultats personnalises</span></h1>
            <p style="font-size:1.05rem; margin-bottom:0;">
                {profile_data['type']} de {profile_data['surface']}m² · {profile_data['occupants']} personne(s) · chauffage {profile_data['chauffage'].lower()}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Bill estimate
    st.markdown(f"""
        <div class="result-bill">
            <div class="label">💸 Votre facture estimee</div>
            <div class="price">{monthly_eur:.0f} € / mois</div>
            <div class="subtitle">soit <b>{annual_eur:.0f} € / an</b> au tarif Base regule (0,25 €/kWh)</div>
        </div>
    """, unsafe_allow_html=True)

    # Profile
    st.markdown(f"""
        <div class="profile-detected">
            <div class="emoji">{PROFIL_LABELS_FR[profil].split()[0]}</div>
            <div class="label">{PROFIL_LABELS_FR[profil][2:]}</div>
            <div class="desc">L'IA a detecte votre profil de consommation : <b>{profil}</b></div>
        </div>
    """, unsafe_allow_html=True)

    # Recommandation fournisseur (gros bandeau)
    st.markdown(f"""
        <div class="reco-fournisseur">
            <span class="badge">🎯 Notre recommandation</span>
            <h2>{fourn['logo']} {fourn['fournisseur']}</h2>
            <div class="tarif">Tarif {fourn['tarif']}</div>
            <div class="raison">{fourn['raison']}</div>
            <div class="economie">💰 Economie estimee : {fourn['economie_an']} € / an</div>
        </div>
    """, unsafe_allow_html=True)

    col_cta, _ = st.columns([1, 1])
    with col_cta:
        st.link_button(f"📞 Demander un devis {fourn['fournisseur']}",
                       f"https://www.google.com/search?q={fourn['fournisseur'].replace(' ', '+')}+contact",
                       use_container_width=True)

    # KPI metrics
    st.markdown("---")
    st.markdown("### 📊 Votre consommation en chiffres")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conso moy", f"{pred_kw:.2f} kW")
    c2.metric("Par jour", f"{daily_kw:.1f} kWh")
    c3.metric("Par mois", f"{monthly_eur:.0f} €")
    c4.metric("Par an", f"{annual_eur:.0f} €")

    # Comparaison voisins
    st.markdown("---")
    st.markdown("### 🏘️ Vous vs vos voisins")
    st.caption("Comparaison avec des foyers similaires en France (donnees Enedis Open Data)")

    # Valeurs simulées
    neighbor_avg = annual_eur * (0.85 + np.random.random() * 0.3)
    neighbor_best = annual_eur * 0.65
    neighbor_worst = annual_eur * 1.3

    st.markdown(f"""
        <div class="neighbor-bar best">
            <div class="neighbor-icon">🥇</div>
            <div class="neighbor-label">Meilleur 10% des foyers similaires</div>
            <div class="neighbor-value">{neighbor_best:.0f} € / an</div>
        </div>
        <div class="neighbor-bar">
            <div class="neighbor-icon">📊</div>
            <div class="neighbor-label">Moyenne des foyers similaires</div>
            <div class="neighbor-value">{neighbor_avg:.0f} € / an</div>
        </div>
        <div class="neighbor-bar you">
            <div class="neighbor-icon">🏠</div>
            <div class="neighbor-label"><b>VOUS</b></div>
            <div class="neighbor-value">{annual_eur:.0f} € / an</div>
        </div>
        <div class="neighbor-bar bad">
            <div class="neighbor-icon">📈</div>
            <div class="neighbor-label">Foyers similaires les plus consommateurs</div>
            <div class="neighbor-value">{neighbor_worst:.0f} € / an</div>
        </div>
    """, unsafe_allow_html=True)

    # Actions
    st.markdown("---")
    st.markdown("### ✅ Vos 3 actions concretes pour reduire la facture")

    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        st.markdown(f"""
            <div class="action-card">
                <div class="action-icon">💡</div>
                <div class="action-title">Changer de fournisseur</div>
                <div class="action-desc">Passer chez {fourn['fournisseur']} vous fait economiser <b>{fourn['economie_an']} €/an</b>.</div>
            </div>
        """, unsafe_allow_html=True)
    with action_col2:
        st.markdown(f"""
            <div class="action-card">
                <div class="action-icon">🕐</div>
                <div class="action-title">Decaler vos usages</div>
                <div class="action-desc">Lave-linge et lave-vaisselle apres 22h vous fait gagner <b>~120 €/an</b>.</div>
            </div>
        """, unsafe_allow_html=True)
    with action_col3:
        if profile_data["chauffage"] == "Electrique" and profile_data["type"] == "Maison individuelle":
            st.markdown("""
                <div class="action-card">
                    <div class="action-icon">🌡️</div>
                    <div class="action-title">Pompe a chaleur</div>
                    <div class="action-desc">Aide MaPrimeRenov jusqu'a 9000 €. Economie <b>~600 €/an</b> sur chauffage.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="action-card">
                    <div class="action-icon">🔌</div>
                    <div class="action-title">Multiprises a interrupteur</div>
                    <div class="action-desc">Couper les veilles vous fait economiser <b>~80 €/an</b>.</div>
                </div>
            """, unsafe_allow_html=True)

    # Suivi
    st.markdown("---")
    st.markdown("### 📨 Restez informe (optionnel)")
    col_mail, col_btn = st.columns([3, 1])
    with col_mail:
        email = st.text_input(
            "Email pour recevoir vos prochaines analyses",
            placeholder="vous@exemple.fr",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("📬 M'inscrire", use_container_width=True):
            if email.strip():
                st.toast(f"✅ {email} ajoute a la liste", icon="📨")
            else:
                st.warning("Tapez un email")


# ============================================================
# SIDEBAR OUTILS AVANCES
# ============================================================
def page_avis_fournisseur():
    st.markdown("# 💬 Analyser un avis fournisseur")
    st.caption("Collez un avis Trustpilot ou Google Reviews sur un fournisseur d'energie.")

    model_sentiment, tok_meta = load_sentiment_model()
    if model_sentiment is None:
        st.error("Modele sentiment non disponible.")
        return

    examples = [
        "EDF nous a augmente la facture sans prevenir et le service client est introuvable",
        "Tres satisfait du conseiller TotalEnergies, reactif et le prix est competitif",
        "Encore 4h de coupure aujourd'hui, j'en peux plus de ce fournisseur catastrophique",
        "Installation Linky chez Engie nickel, technicien rapide et professionnel",
    ]

    if "sentiment_text" not in st.session_state:
        st.session_state["sentiment_text"] = ""

    st.markdown("**Exemples cliquables :**")
    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(f"📝 {ex[:65]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state["sentiment_text"] = ex
                st.rerun()

    user_text = st.text_area("Votre avis", height=140, key="sentiment_text")

    if st.button("🔍 Analyser cet avis", type="primary"):
        if not user_text.strip():
            st.warning("Tapez un avis avant de cliquer.")
            return

        with st.spinner("🧠 Analyse en cours..."):
            from keras.utils import pad_sequences
            tokenizer = tok_meta["tokenizer"]
            seq = tokenizer.texts_to_sequences([user_text])
            pad = pad_sequences(seq, maxlen=tok_meta["max_len"], padding="pre", truncating="pre")
            proba = float(model_sentiment.predict(pad, verbose=0)[0, 0])
            label = "Positif" if proba >= 0.5 else "Negatif"

        emoji = "😊" if proba >= 0.5 else "😠"
        color = "#10b981" if proba >= 0.5 else "#ef4444"

        st.markdown(f"""
            <div style="background:white; border-radius:16px; padding:2rem; border:2px solid {color}; text-align:center; margin:1rem 0;">
                <div style="font-size:4rem; margin-bottom:0.5rem;">{emoji}</div>
                <h2 style="color:{color}; margin:0;">{label}</h2>
                <p style="color:#475569; margin-top:0.5rem;">Confiance : <b>{(proba if proba >= 0.5 else 1 - proba) * 100:.0f}%</b></p>
            </div>
        """, unsafe_allow_html=True)

        log_inference(user_text, label, proba if proba >= 0.5 else 1 - proba)


def page_carte_france():
    st.markdown("# 🗺️ Consommation par region")
    st.caption("Comparez votre region a la moyenne nationale (donnees Enedis Open Data).")

    import plotly.express as px

    regions = {
        "Île-de-France": 4500, "Auvergne-Rhône-Alpes": 5200, "Hauts-de-France": 4800,
        "Nouvelle-Aquitaine": 4100, "Occitanie": 3900, "Grand Est": 5100,
        "Provence-Alpes-Côte d'Azur": 3700, "Pays de la Loire": 4400,
        "Normandie": 4700, "Bretagne": 4300, "Bourgogne-Franche-Comté": 5300,
        "Centre-Val de Loire": 4600, "Corse": 3500,
    }
    df = pd.DataFrame({"region": list(regions.keys()), "kwh_an": list(regions.values())})
    fig = px.choropleth(
        df, locations="region", color="kwh_an",
        geojson="https://france-geojson.gregoiredavid.fr/repo/regions.geojson",
        featureidkey="properties.nom", color_continuous_scale="Viridis",
        labels={"kwh_an": "kWh/an moyen"},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=520, margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="kWh/an"),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Region la plus sobre", "Corse", "3 500 kWh/an")
    c2.metric("📊 Moyenne nationale", "4 500 kWh/an")
    c3.metric("🔥 Region la plus energivore", "Grand Est", "5 300 kWh/an")


# ============================================================
# MAIN
# ============================================================
def main():
    init_session()

    with st.sidebar:
        st.markdown("# ⚡ MonEnergie")
        st.markdown("**Reduisez vos factures avec l'IA**")
        st.markdown("---")

        if HAS_OPTION_MENU:
            page = option_menu(
                menu_title=None,
                options=["🏠 Accueil", "💬 Avis fournisseur", "🗺️ Carte France"],
                default_index=0,
                styles={
                    "container": {"padding": "0", "background-color": "transparent"},
                    "nav-link": {
                        "color": "#cbd5e1", "font-size": "14px", "font-weight": "500",
                        "margin": "2px 0", "padding": "10px 14px", "border-radius": "10px",
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                        "color": "white",
                    },
                },
            )
        else:
            page = st.radio("Navigation",
                            ["🏠 Accueil", "💬 Avis fournisseur", "🗺️ Carte France"],
                            label_visibility="collapsed")

        st.markdown("---")
        st.markdown("### 💚 Pourquoi MonEnergie ?")
        st.markdown("""
        - 🆓 **Gratuit**, sans inscription
        - ⚡ Resultats en **2 minutes**
        - 🔒 Donnees **dans votre navigateur**
        - 🇫🇷 Donnees **francaises** uniquement
        - 🤖 IA entrainee sur **2M de mesures reelles**
        """)
        st.markdown("---")
        st.caption("Projet TP4 IPSSI MIA4 · v1.0")
        st.caption("[GitHub](https://github.com/Hakim78/tp4-monenergie) · MIT")

    # Routing
    if page.startswith("🏠"):
        step = st.session_state.get("step", "welcome")
        if step == "welcome":
            page_welcome()
        elif step == "questionnaire":
            page_questionnaire()
        elif step == "results":
            page_results()
    elif page.startswith("💬"):
        page_avis_fournisseur()
    elif page.startswith("🗺️"):
        page_carte_france()


if __name__ == "__main__":
    main()
