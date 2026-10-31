"""
app.py — Pipeline de producción para el análisis de sentimiento de reseñas.

Patrón del template:
    - `explore.ipynb` = la historia (EDA + exploración + conclusiones).
    - `app.py`         = la versión limpia y reutilizable para producción.

Este script:
    1. Carga el modelo de sentimiento fijo (nlptown/bert-base-multilingual-uncased-sentiment).
    2. Lee las reseñas crudas desde data/raw/reviews.csv.
    3. Ejecuta la inferencia UNA sola vez (el modelo se carga una única vez).
    4. Traduce las estrellas predichas a la banda de negocio (Negativo/Neutral/Positivo).
    5. Escribe la salida enriquecida en data/processed/reviews_with_sentiment.csv.

Modelo fijado explícitamente (no "latest") según el brief del cliente (CONTEXT.md):
    1-2 estrellas -> Negativo | 3 estrellas -> Neutral | 4-5 estrellas -> Positivo
"""

import os

import pandas as pd
from transformers import AutoTokenizer, pipeline

from utils import db_connect

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# Modelo FIJO (no "latest"): fine-tuned en reseñas de producto 1-5 estrellas.
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "reviews.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reviews_with_sentiment.csv")

# Mapeo de estrellas -> banda de sentimiento de negocio.
BAND_MAP = {1: "Negativo", 2: "Negativo", 3: "Neutral", 4: "Positivo", 5: "Positivo"}


def parse_prediction(result):
    """Convierte la salida del pipeline ('4 stars', 0.87) en (4, 0.87)."""
    stars = int(result["label"].split()[0])
    return stars, float(result["score"])


def sentiment_band(stars):
    """Traduce estrellas predichas (1-5) a la banda de negocio."""
    return BAND_MAP.get(stars, "Neutral")


def load_classifier(model_name=MODEL_NAME):
    """Carga el modelo UNA sola vez y devuelve un pipeline listo para inferir."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
    )


def run_inference(df, classifier, text_col="review_text"):
    """Ejecuta la inferencia sobre el DataFrame y añade las columnas enriquecidas."""
    texts = df[text_col].astype(str).tolist()
    predictions = classifier(texts)

    df["text_len"] = df[text_col].astype(str).str.len()
    df["pred_stars"] = [parse_prediction(p)[0] for p in predictions]
    df["pred_score"] = [parse_prediction(p)[1] for p in predictions]
    df["sentiment"] = df["pred_stars"].apply(sentiment_band)
    return df


def main():
    # (Opcional) conexión a base de datos del template. Se conserva el patrón,
    # pero el pipeline de inferencia funciona desde CSV, así que la conexión es
    # best-effort: si no hay DATABASE_URL o driver, no detenemos el pipeline.
    try:
        engine = db_connect()  # noqa: F841
    except Exception as exc:  # noqa: BLE001
        print(f"ℹ️  DB no disponible (se continúa desde CSV): {exc}")

    print(f"Cargando modelo: {MODEL_NAME}")
    classifier = load_classifier()
    print("✅ Modelo cargado correctamente")

    df = pd.read_csv(RAW_PATH)
    print(f"Reseñas cargadas: {len(df)}")

    df = run_inference(df, classifier)
    print("✅ Inferencia completada")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Guardado: {OUTPUT_PATH} ({len(df)} filas × {df.shape[1]} columnas)")

    band_counts = df["sentiment"].value_counts()
    total = len(df)
    for band in ["Positivo", "Neutral", "Negativo"]:
        count = int(band_counts.get(band, 0))
        print(f"  {band}: {count} ({count / total * 100:.1f}%)")

    print(f"Media humana: {df['rating'].mean():.2f} | Media modelo: {df['pred_stars'].mean():.2f}")
    return df


if __name__ == "__main__":
    main()
