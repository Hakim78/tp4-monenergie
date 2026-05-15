"""TP4 - Phase 8 Piste B : logging d'inference.

Loggue chaque prediction dans un CSV local. Robuste aux retours-ligne et
aux virgules dans le texte d'entree (echappes par csv.writer).

En production : remplacer par une DB (Postgres, SQLite) ou un service de
monitoring (Sentry, Logfire, OpenTelemetry).
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "inference_log.csv"
MAX_INPUT_PREVIEW = 200


def log_inference(
    input_text: str,
    prediction: str,
    confidence: float,
    log_file: Path | str = LOG_FILE,
) -> None:
    """Ajoute une ligne au CSV d'audit.

    Colonnes : timestamp_iso, input_preview, prediction, confidence_pct
    """
    log_path = Path(log_file)
    is_new = not log_path.exists()
    preview = input_text.strip().replace("\n", " ")[:MAX_INPUT_PREVIEW]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if is_new:
            writer.writerow(
                ["timestamp_iso", "input_preview", "prediction", "confidence_pct"]
            )
        writer.writerow(
            [
                dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                preview,
                prediction,
                f"{confidence * 100:.2f}",
            ]
        )


if __name__ == "__main__":
    log_inference(
        "this movie was truly amazing, with great acting and soundtrack",
        "Positif",
        0.92,
    )
    log_inference(
        "absolutely terrible\nwith commas, that should be escaped properly",
        "Negatif",
        0.84,
    )
    print(f"Deux entrees ajoutees a {LOG_FILE}")
