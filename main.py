"""
FIFA Scout API
==============
Backend liviano en FastAPI que expone el mismo modelo entrenado
(Gradient Boosting Optimizado) y el mismo dataset que la app original,
para que un frontend (Lovable / React) pueda consumirlo sin correr Python.

No re-entrena ni modifica el análisis: carga el mismo .joblib y aplica
exactamente la misma lógica de features, filtros y umbrales.
"""
import io
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# CONSTANTES DEL MODELO — idénticas a app.py
# ============================================================
MODEL_PATH = "modelo_gb_pipeline.joblib"
DATA_PATH = "fifa_players_model_ready.csv"

FEATURES_BASE = [
    "age", "overall_rating",
    "vision", "agility", "standing_tackle", "strength",
    "international_reputation(1-5)", "weak_foot(1-5)", "skill_moves(1-5)",
]
FEATURES_TP1 = [
    "attack_score", "defense_score", "playmaking_score", "physical_score",
    "has_release_clause",
]
FEATURES_NUEVAS = ["preferred_foot_enc", "nationality_freq"]
POSITION_COLS = [
    "pos_Arquero", "pos_Defensor Central", "pos_Delantero Centro",
    "pos_Extremo", "pos_Lateral", "pos_Mediocampista Defensivo",
    "pos_Mediocampista Ofensivo",
]
ALL_FEATURES = FEATURES_BASE + FEATURES_TP1 + FEATURES_NUEVAS + POSITION_COLS
TARGET = "potential"
POSICIONES = [c.replace("pos_", "") for c in POSITION_COLS]
NUMERIC_RANGE_FEATURES = [
    "age", "overall_rating", "vision", "agility", "standing_tackle", "strength",
    "attack_score", "defense_score", "playmaking_score", "physical_score",
]

COMPARATIVA_MODELOS = [
    {"modelo": "Regresión Lineal", "r2_test": 0.8328, "mae": 1.9558, "rmse": 2.4794, "gap_pct": 0.05},
    {"modelo": "Árbol de Decisión", "r2_test": 0.9200, "mae": 1.1234, "rmse": 1.7151, "gap_pct": 0.98},
    {"modelo": "KNN (7 features, K=9)", "r2_test": 0.9029, "mae": 1.3427, "rmse": 1.8898, "gap_pct": 2.14},
    {"modelo": "Random Forest (base)", "r2_test": 0.9301, "mae": 1.0205, "rmse": 1.6033, "gap_pct": 5.99},
    {"modelo": "GB Optimizado (GridSearchCV) — elegido", "r2_test": 0.9355, "mae": 0.9973, "rmse": 1.5402, "gap_pct": 2.23},
]
HIPERPARAMETROS_GB = {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 300, "subsample": 0.8}


def clasificar_potencial(p: float) -> str:
    if p < 65:
        return "Jugador de relleno"
    if p < 75:
        return "Suplente confiable"
    if p < 82:
        return "Titular sólido"
    if p < 88:
        return "Jugador de alto nivel"
    return "Élite mundial"


def resolve_display_columns(df: pd.DataFrame):
    short_col = next((c for c in ["short_name", "name"] if c in df.columns), df.columns[0])
    long_col = next((c for c in ["long_name", "full_name"] if c in df.columns), short_col)
    club_col = next((c for c in ["club_name", "club"] if c in df.columns), None)
    position_col = next((c for c in ["specific_position", "player_positions", "positions"] if c in df.columns), None)
    return short_col, long_col, club_col, position_col


# ============================================================
# CARGA DEL MODELO Y DATASET (una sola vez al arrancar)
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró '{MODEL_PATH}'. Copiá el .joblib junto a main.py.")

modelo = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
df["potential_predicho"] = np.round(modelo.predict(df[ALL_FEATURES]), 2)
SHORT_COL, LONG_COL, CLUB_COL, POSITION_COL = resolve_display_columns(df)

app = FastAPI(title="FIFA Scout API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir al dominio de Lovable en producción
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helpers de filtrado (misma lógica que Explorador de Joyas)
# ============================================================
def aplicar_filtros(
    posiciones: Optional[List[str]] = None,
    edad_min: int = 15,
    edad_max: int = 45,
    valor_min: float = 0.0,
    valor_max: float = 999.0,
    nacionalidades: Optional[List[str]] = None,
    overall_min: int = 50,
) -> pd.DataFrame:
    out = df.copy()
    if POSITION_COL and posiciones:
        out = out[out[POSITION_COL].isin(posiciones)]
    out = out[(out["age"] >= edad_min) & (out["age"] <= edad_max)]
    out = out[(out["value_euro"] >= valor_min * 1_000_000) & (out["value_euro"] <= valor_max * 1_000_000)]
    if nacionalidades:
        out = out[out["nationality"].isin(nacionalidades)]
    out = out[out["overall_rating"] >= overall_min]
    return out


def calcular_joyas(data: pd.DataFrame, edad_tope: int, percentil: float, potencial_min: float) -> pd.DataFrame:
    if data.empty:
        return data
    umbral_valor = data["value_euro"].quantile(percentil)
    return data[
        (data["potential_predicho"] >= potencial_min)
        & (data["value_euro"] <= umbral_valor)
        & (data["age"] <= edad_tope)
    ]


def jugador_a_dict(row) -> dict:
    return {
        "nombre": row[SHORT_COL],
        "nacionalidad": row["nationality"],
        "club": row[CLUB_COL] if CLUB_COL else None,
        "edad": int(row["age"]),
        "overall": int(row["overall_rating"]),
        "potencial_predicho": float(row["potential_predicho"]),
        "valor_euros": float(row["value_euro"]),
        "posicion": row[POSITION_COL] if POSITION_COL else None,
    }


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "jugadores_en_dataset": len(df)}


@app.get("/filters/options")
def filtros_disponibles():
    return {
        "posiciones": POSICIONES,
        "nacionalidades": sorted(df["nationality"].dropna().unique().tolist()),
        "valor_maximo_millones": round(float(df["value_euro"].max()) / 1_000_000, 1),
    }


@app.get("/features/ranges")
def rangos_features():
    """Min/max reales del dataset para cada slider del Predictor Individual."""
    out = {}
    for feat in NUMERIC_RANGE_FEATURES:
        out[feat] = {"min": float(df[feat].min()), "max": float(df[feat].max())}
    return out


@app.get("/players")
def listar_jugadores(
    posiciones: Optional[List[str]] = Query(None),
    edad_min: int = 15,
    edad_max: int = 45,
    valor_min: float = 0.0,
    valor_max: float = 999.0,
    nacionalidades: Optional[List[str]] = Query(None),
    overall_min: int = 50,
    limit: int = Query(500, le=2000),
):
    filtrado = aplicar_filtros(posiciones, edad_min, edad_max, valor_min, valor_max, nacionalidades, overall_min)
    kpis = {
        "jugadores_filtrados": len(filtrado),
        "total_dataset": len(df),
        "potencial_promedio": round(float(filtrado["potential_predicho"].mean()), 1) if not filtrado.empty else None,
    }
    muestra = filtrado.sort_values("potential_predicho", ascending=False).head(limit)
    return {
        "kpis": kpis,
        "jugadores": [jugador_a_dict(r) for _, r in muestra.iterrows()],
        "scatter": [
            {
                "valor_millones": round(float(r["value_euro"]) / 1_000_000, 2),
                "potencial_predicho": float(r["potential_predicho"]),
                "edad": int(r["age"]),
                "overall": int(r["overall_rating"]),
                "nombre": r[SHORT_COL],
            }
            for _, r in filtrado.iterrows()
        ],
    }


@app.get("/players/hidden-gems")
def joyas_ocultas(
    posiciones: Optional[List[str]] = Query(None),
    edad_min: int = 15,
    edad_max: int = 45,
    valor_min: float = 0.0,
    valor_max: float = 999.0,
    nacionalidades: Optional[List[str]] = Query(None),
    overall_min: int = 50,
):
    filtrado = aplicar_filtros(posiciones, edad_min, edad_max, valor_min, valor_max, nacionalidades, overall_min)
    criterios_relajados = None
    joyas = calcular_joyas(filtrado, 23, 0.25, 80)
    if joyas.empty:
        joyas = calcular_joyas(filtrado, 25, 0.25, 78)
        criterios_relajados = "edad ≤ 25 y potencial ≥ 78"
    if joyas.empty:
        joyas = calcular_joyas(filtrado, 25, 0.40, 75)
        criterios_relajados = "percentil de valor 40% y potencial ≥ 75"
    if joyas.empty and not filtrado.empty:
        joyas = filtrado.sort_values("potential_predicho", ascending=False).head(20)
        criterios_relajados = "sin restricción de edad/valor — top 20 por potencial predicho"

    joyas = joyas.sort_values("potential_predicho", ascending=False).head(20)
    return {
        "criterios_relajados": criterios_relajados,
        "jugadores": [jugador_a_dict(r) for _, r in joyas.iterrows()],
    }


@app.get("/players/position-stats")
def estadisticas_por_posicion(posiciones: List[str] = Query(...)):
    """
    Promedios, histogramas y top jugadores para una o más posiciones,
    calculados sobre TODO el subconjunto filtrado (no sobre una muestra
    truncada por `limit`, como hace /players).
    """
    invalidas = [p for p in posiciones if p not in POSICIONES]
    if invalidas:
        raise HTTPException(400, f"Posición inválida: {invalidas}. Opciones: {POSICIONES}")

    filtrado = df[df[POSITION_COL].isin(posiciones)] if POSITION_COL else df.iloc[0:0]

    if filtrado.empty:
        return {
            "cantidad": 0,
            "overall_promedio": None,
            "potencial_promedio": None,
            "edad_promedio": None,
            "valor_promedio": None,
            "nacionalidad_top": None,
            "club_top": None,
            "overall_histograma": None,
            "potencial_histograma": None,
            "edad_histograma": None,
            "top_potencial": [],
        }

    def moda(serie: pd.Series):
        m = serie.dropna().mode()
        return m.iloc[0] if not m.empty else None

    def histograma(serie: pd.Series, bins: int, rango: tuple):
        conteos, bordes = np.histogram(serie.dropna(), bins=bins, range=rango)
        return {"conteos": conteos.tolist(), "bordes": [round(float(b), 1) for b in bordes]}

    top = filtrado.sort_values("potential_predicho", ascending=False).head(3)

    return {
        "cantidad": len(filtrado),
        "overall_promedio": round(float(filtrado["overall_rating"].mean()), 1),
        "potencial_promedio": round(float(filtrado["potential_predicho"].mean()), 1),
        "edad_promedio": round(float(filtrado["age"].mean()), 1),
        "valor_promedio": round(float(filtrado["value_euro"].mean()), 2),
        "nacionalidad_top": moda(filtrado["nationality"]),
        "club_top": moda(filtrado[CLUB_COL]) if CLUB_COL else None,
        "overall_histograma": histograma(filtrado["overall_rating"], 12, (40, 100)),
        "potencial_histograma": histograma(filtrado["potential_predicho"], 12, (40, 100)),
        "edad_histograma": histograma(filtrado["age"], 10, (15, 45)),
        "top_potencial": [
            {"nombre": r[SHORT_COL], "potencial_predicho": float(r["potential_predicho"])}
            for _, r in top.iterrows()
        ],
    }


class PrediccionInput(BaseModel):
    age: int
    overall_rating: int
    vision: int
    agility: int
    standing_tackle: int
    strength: int
    international_reputation: int = Field(..., ge=1, le=5)
    weak_foot: int = Field(..., ge=1, le=5)
    skill_moves: int = Field(..., ge=1, le=5)
    attack_score: float
    defense_score: float
    playmaking_score: float
    physical_score: float
    has_release_clause: bool
    preferred_foot: str  # "Derecho" | "Izquierdo"
    nacionalidad: str
    posicion: str  # uno de POSICIONES


@app.post("/predict")
def predecir(payload: PrediccionInput):
    if payload.posicion not in POSICIONES:
        raise HTTPException(400, f"Posición inválida. Opciones: {POSICIONES}")

    nationality_freq = int(df["nationality"].value_counts().get(payload.nacionalidad, 1))
    fila = {
        "age": payload.age, "overall_rating": payload.overall_rating,
        "vision": payload.vision, "agility": payload.agility,
        "standing_tackle": payload.standing_tackle, "strength": payload.strength,
        "international_reputation(1-5)": payload.international_reputation,
        "weak_foot(1-5)": payload.weak_foot, "skill_moves(1-5)": payload.skill_moves,
        "attack_score": payload.attack_score, "defense_score": payload.defense_score,
        "playmaking_score": payload.playmaking_score, "physical_score": payload.physical_score,
        "has_release_clause": int(payload.has_release_clause),
        "preferred_foot_enc": 1 if payload.preferred_foot == "Derecho" else 0,
        "nationality_freq": nationality_freq,
    }
    for pos in POSICIONES:
        fila[f"pos_{pos}"] = 1 if pos == payload.posicion else 0

    X = pd.DataFrame([fila])[ALL_FEATURES]
    potencial = float(modelo.predict(X)[0])

    top_features = None
    try:
        import shap
        scaler = modelo.named_steps["scaler"]
        arbol = modelo.named_steps["model"]
        X_scaled = scaler.transform(X)
        explainer = shap.TreeExplainer(arbol)
        shap_values = explainer.shap_values(X_scaled, check_additivity=False)
        serie = pd.Series(shap_values[0], index=ALL_FEATURES).sort_values(key=np.abs, ascending=False)
        top_features = [{"feature": k, "impacto": round(float(v), 3)} for k, v in serie.head(8).items()]
    except ImportError:
        pass

    return {
        "potencial_predicho": round(potencial, 1),
        "clasificacion": clasificar_potencial(potencial),
        "top_features": top_features,
    }


@app.get("/template.csv")
def descargar_plantilla():
    from fastapi.responses import StreamingResponse
    plantilla = pd.DataFrame(columns=ALL_FEATURES)
    buf = io.StringIO()
    plantilla.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=plantilla_fifa_scout.csv"},
    )


@app.post("/predict-batch")
async def predecir_lote(archivo: UploadFile = File(...)):
    try:
        contenido = await archivo.read()
        df_lote = pd.read_csv(io.BytesIO(contenido))
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer el archivo: {e}")

    faltantes = [c for c in ALL_FEATURES if c not in df_lote.columns]
    if faltantes:
        raise HTTPException(400, f"Al CSV le faltan {len(faltantes)} columna(s): {', '.join(faltantes)}")

    df_pred = df_lote.copy()
    for col in ALL_FEATURES:
        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

    filas_invalidas = int(df_pred[ALL_FEATURES].isnull().any(axis=1).sum())
    df_validas = df_pred.dropna(subset=ALL_FEATURES).copy()

    if df_validas.empty:
        raise HTTPException(400, "Ninguna fila del CSV tiene datos válidos para predecir.")

    df_validas["potential_predicho"] = np.round(modelo.predict(df_validas[ALL_FEATURES]), 2)

    return {
        "filas_invalidas_excluidas": filas_invalidas,
        "jugadores_procesados": len(df_validas),
        "potencial_promedio": round(float(df_validas["potential_predicho"].mean()), 1),
        "potencial_maximo": round(float(df_validas["potential_predicho"].max()), 1),
        "potencial_minimo": round(float(df_validas["potential_predicho"].min()), 1),
        "resultados": df_validas.to_dict(orient="records"),
    }


@app.get("/model-info")
def info_modelo():
    gb_step = modelo.named_steps["model"]
    importancias = sorted(
        [{"variable": f, "importancia": float(i)} for f, i in zip(ALL_FEATURES, gb_step.feature_importances_)],
        key=lambda x: x["importancia"], reverse=True,
    )
    residuos = (df[TARGET] - df["potential_predicho"]).round(2)
    hist_counts, hist_edges = np.histogram(residuos, bins=60)

    posiciones_conteo = None
    if POSITION_COL:
        vc = df[POSITION_COL].value_counts()
        posiciones_conteo = [{"posicion": k, "cantidad": int(v)} for k, v in vc.items()]

    return {
        "comparativa_modelos": COMPARATIVA_MODELOS,
        "hiperparametros_gb": HIPERPARAMETROS_GB,
        "importancia_variables": importancias,
        "residuos_histograma": {
            "conteos": hist_counts.tolist(),
            "bordes": [round(float(e), 2) for e in hist_edges],
        },
        "dataset": {
            "total_jugadores": len(df),
            "edad_min": int(df["age"].min()),
            "edad_max": int(df["age"].max()),
            "nacionalidades_distintas": int(df["nationality"].nunique()),
        },
        "distribucion_posiciones": posiciones_conteo,
    }


@app.get("/classifier/status")
def estado_clasificador_posicion():
    return {"disponible": False, "mensaje": "Esta funcionalidad está en desarrollo y se habilitará próximamente."}
