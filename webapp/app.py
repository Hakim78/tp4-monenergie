"""TP4 - Phase 7 : Streamlit MonEnergie

WebApp qui consomme les trois modeles entraines :

1. energie_lstm.keras (phase 1+2) : forecast conso 24h depuis 7 jours d'historique
2. profil_conso.keras (phase 6) : classification profil consommateur (4 classes)
3. allocine_sentiment.keras (phase 4+5) : sentiment FR pour analyse d'avis fournisseur

L'utilisateur peut :
- Uploader sa courbe de conso CSV (export Linky) et voir le forecast + son profil
- Coller un avis utilisateur sur un fournisseur (Trustpilot) et voir le sentiment
- Voir des recommandations contextuelles selon son profil

Lancement local :  streamlit run webapp/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import keras
import joblib

try:
    from log_inference import log_inference  # type: ignore
except ImportError:  # pragma: no cover
    def log_inference(*args, **kwargs):
        return None


# Fichiers attendus (root du repo ou cote de app.py)
SEARCH_PATHS = [
    Path("."),
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
]


def find_file(name: str) -> Path | None:
    for base in SEARCH_PATHS:
        p = base / name
        if p.exists():
            return p
    return None


PROFILS = ["Econome", "Standard", "Energivore", "Critique"]
PROFIL_COLORS = ["#16a34a", "#0ea5e9", "#f59e0b", "#dc2626"]
PROFIL_RECOS = {
    "Econome": "Bravo, vos habitudes sont deja efficientes. Verifiez votre DPE et envisagez un fournisseur a prix indexe pour beneficier des baisses.",
    "Standard": "Vous etes dans la moyenne. Quelques gestes ciblees (programmer le ballon, regler le thermostat a 19C) peuvent vous faire economiser 10-15%.",
    "Energivore": "Votre conso est elevee. Une audit thermique pro pourrait reveler des points faibles (combles, fenetres). Penser au tarif Heures Creuses.",
    "Critique": "Conso tres importante. Faites un audit obligatoire et comparez tous les fournisseurs. Une isolation peut faire baisser la facture de 30-40%.",
}


st.set_page_config(
    page_title="MonEnergie",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def load_forecast_model():
    p = find_file("energie_lstm.keras")
    if p is None:
        return None, None
    scalers_p = find_file("energie_scalers.joblib")
    if scalers_p is None:
        return None, None
    return keras.models.load_model(p), joblib.load(scalers_p)


@st.cache_resource
def load_profil_model():
    p = find_file("profil_conso.keras")
    if p is None:
        return None, None
    meta_p = find_file("profil_conso_meta.joblib")
    if meta_p is None:
        return None, None
    return keras.models.load_model(p), joblib.load(meta_p)


@st.cache_resource
def load_sentiment_model():
    p = find_file("allocine_sentiment.keras")
    if p is None:
        return None, None
    tok_p = find_file("allocine_tokenizer.joblib")
    if tok_p is None:
        return None, None
    return keras.models.load_model(p), joblib.load(tok_p)


def render_header():
    st.title("⚡ MonEnergie")
    st.write(
        "Forecast de consommation, analyse de profil et avis fournisseurs - "
        "tout dans le navigateur, modeles RNN/LSTM entraines sur des donnees francaises."
    )


def render_forecast_tab():
    st.subheader("Forecast 24h depuis votre historique")
    st.caption(
        "Uploadez un CSV avec 168 lignes (7 jours d'historique horaire). "
        "Colonnes : active_kw, sub1_wh, sub2_wh, sub3_wh, hour_sin, hour_cos, is_weekend."
    )

    forecast_model, scalers = load_forecast_model()
    if forecast_model is None:
        st.error(
            "Modele forecast introuvable. Lancez d'abord phase1_2_energie_lstm.py "
            "et placez energie_lstm.keras + energie_scalers.joblib a la racine."
        )
        return

    uploaded = st.file_uploader("Votre courbe de conso (CSV 168 lignes)", type=["csv"])
    if uploaded is None:
        st.info("Pas encore de fichier. Un exemple test est disponible dans /data/sample_week.csv.")
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Lecture CSV impossible : {e}")
        return

    features = scalers["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes dans le CSV : {missing}")
        return

    if len(df) < 168:
        st.warning(
            f"Fichier trop court : {len(df)} lignes au lieu de 168. "
            "On pad avec la moyenne mais le forecast sera degrade."
        )

    arr = df[features].tail(168).values
    if len(arr) < 168:
        pad_rows = 168 - len(arr)
        pad = np.tile(arr.mean(axis=0), (pad_rows, 1))
        arr = np.vstack([pad, arr])

    arr_scaled = scalers["scaler_X"].transform(arr)
    arr_seq = arr_scaled.reshape(1, 168, len(features))

    pred_scaled = forecast_model.predict(arr_seq, verbose=0)
    pred_kw = float(scalers["scaler_y"].inverse_transform(pred_scaled)[0, 0])

    avg_kw_input = float(df["active_kw"].tail(168).mean())
    delta_pct = (pred_kw - avg_kw_input) / max(avg_kw_input, 1e-6) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Conso moy. 7 derniers jours", f"{avg_kw_input:.2f} kW")
    col2.metric("Forecast moy. demain", f"{pred_kw:.2f} kW", f"{delta_pct:+.1f}%")
    cost_eur = pred_kw * 24 * 0.25  # estimation 0.25 EUR / kWh, tarif reglemente FR
    col3.metric("Cout estime du jour", f"{cost_eur:.2f} EUR", help="Estime sur tarif reglemente bleu 0.25 EUR/kWh")

    st.line_chart(df.tail(168)["active_kw"].reset_index(drop=True))


def render_profil_tab():
    st.subheader("Votre profil energetique")
    st.caption("Analyse votre semaine et vous classe en 4 categories (Econome, Standard, Energivore, Critique).")

    model_profil, meta = load_profil_model()
    if model_profil is None:
        st.error("Modele profil introuvable. Lancez phase6_profil_consommateur.py.")
        return

    uploaded = st.file_uploader("Courbe de conso (CSV 168 lignes)", type=["csv"], key="profil")
    if uploaded is None:
        st.info("Uploadez votre courbe pour decouvrir votre profil.")
        return

    df = pd.read_csv(uploaded)
    features = meta["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Colonnes manquantes : {missing}")
        return

    arr = df[features].tail(168).values
    if len(arr) < 168:
        pad = np.tile(arr.mean(axis=0), (168 - len(arr), 1))
        arr = np.vstack([pad, arr])
    arr_scaled = meta["scaler_X"].transform(arr).reshape(1, 168, len(features))

    proba = model_profil.predict(arr_scaled, verbose=0)[0]
    profil_idx = int(np.argmax(proba))
    profil = PROFILS[profil_idx]

    st.markdown(
        f"<div style='padding:1rem; border-radius:0.5rem; background-color:{PROFIL_COLORS[profil_idx]}22; "
        f"border-left:6px solid {PROFIL_COLORS[profil_idx]}'>"
        f"<h3 style='margin:0; color:{PROFIL_COLORS[profil_idx]}'>Profil : {profil}</h3>"
        f"<p style='margin-top:0.5rem'>{PROFIL_RECOS[profil]}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("**Probabilites detaillees** :")
    proba_df = pd.DataFrame({"Profil": PROFILS, "Probabilite": proba})
    st.bar_chart(proba_df.set_index("Profil"))


def render_sentiment_tab():
    st.subheader("Analyse d'avis fournisseur")
    st.caption(
        "Le modele Bi-LSTM entraine sur 200k avis Allocine generalise au sentiment FR. "
        "Collez un avis sur EDF, Engie, TotalEnergies ou autre."
    )

    sentiment_model, tok_meta = load_sentiment_model()
    if sentiment_model is None:
        st.error("Modele sentiment introuvable. Lancez phase4_5_allocine_bilstm.py.")
        return

    user_text = st.text_area(
        "Votre avis",
        height=180,
        placeholder="Exemple : Le service client EDF m'a tres bien aide pour ma souscription...",
    )
    if not st.button("Analyser", type="primary"):
        return

    if not user_text.strip():
        st.warning("Tapez un avis avant de cliquer.")
        return

    tokenizer = tok_meta["tokenizer"]
    max_len = tok_meta["max_len"]

    # Keras 3 compat : pad_sequences est dans keras.utils
    from keras.utils import pad_sequences
    seq = tokenizer.texts_to_sequences([user_text])
    pad = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")
    proba = float(sentiment_model.predict(pad, verbose=0)[0, 0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    confidence = proba if label == "Positif" else 1 - proba

    if label == "Positif":
        st.success(f"Sentiment : **Positif** ({confidence * 100:.1f}% confiance)")
    else:
        st.error(f"Sentiment : **Negatif** ({confidence * 100:.1f}% confiance)")

    with st.expander("Detail technique"):
        st.write(f"Score sigmoid brut : `{proba:.4f}`")
        st.write(f"Tokens en entree : `{len(user_text.split())}`")
        st.write(f"Tokens analyses post-pad : `{max_len}`")

    log_inference(user_text, label, confidence)


def main() -> None:
    render_header()

    tab1, tab2, tab3 = st.tabs(["Forecast 24h", "Profil consommateur", "Avis fournisseur"])

    with tab1:
        render_forecast_tab()

    with tab2:
        render_profil_tab()

    with tab3:
        render_sentiment_tab()

    with st.sidebar:
        st.subheader("MonEnergie - TP4")
        st.write(
            "Trois modeles RNN/LSTM entraines sur des donnees francaises ouvertes :"
        )
        st.markdown(
            """
            - **Forecast** : LSTM multivarie sur Household Power Consumption (Sceaux)
            - **Profil** : LSTM 4 classes (Econome/Standard/Energivore/Critique)
            - **Sentiment** : Bi-LSTM Allocine 200k avis FR
            """
        )
        st.divider()
        st.caption("Donnees Open Data : Enedis, RTE, GRDF, ADEME, INSEE")
        st.caption("Aucune information personnelle envoyee. Analyse on-device + appel modele.")


if __name__ == "__main__":
    main()
