from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from ai_analyst import generate_match_ai_response

load_dotenv()

app = FastAPI(title="API Predicción Liga MX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el modelo y el dataset al iniciar
datos_exportados = joblib.load("modelo_ligamx.joblib")
modelo = datos_exportados["modelo"]
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No se encontró la variable de entorno DATABASE_URL")

engine = create_engine(DATABASE_URL)

# Cargar y ordenar cronológicamente por fecha para evitar tomar filas desordenadas
df_maestro = pd.read_sql("SELECT * FROM dataset_maestro", con=engine)
if "Date" in df_maestro.columns:
    df_maestro["Date_parsed"] = pd.to_datetime(df_maestro["Date"], errors="coerce")
    df_maestro = df_maestro.sort_values("Date_parsed", na_position="last").reset_index(drop=True)


class MatchRequest(BaseModel):
    Home: str
    Away: str


class RachaRequest(BaseModel):
    Home: str = Field(..., description="Nombre del equipo local")
    Away: str = Field(..., description="Nombre del equipo visitante")
    days: Optional[int] = Field(default=5, description="Número de partidos a consultar")


class AIAnalysisRequest(BaseModel):
    home: str
    away: str
    prob_local: Optional[float] = None
    prob_empate: Optional[float] = None
    prob_visitante: Optional[float] = None
    descanso_local: Optional[Any] = None
    descanso_visitante: Optional[Any] = None
    racha_local: Optional[str] = None
    racha_visitante: Optional[str] = None
    h2h: Optional[str] = None
    action: str = "summary"  # "summary" | "question" | "executive_brief" | "model_limits"
    question: Optional[str] = ""


def extraer_probabilidades(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convención del proyecto:
      0 -> gana local (home_win)
      1 -> empate (draw)
      2 -> gana visitante (away_win)
    """
    prediccion = modelo.predict(features_df)[0]
    probabilidades = modelo.predict_proba(features_df)[0]

    probs_por_rol = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

    for clase, prob in zip(modelo.classes_, probabilidades):
        pct = round(float(prob) * 100, 2)
        clase_str = str(clase).strip()

        if clase_str in ("0", "H", "L"):
            probs_por_rol["home_win"] = pct
        elif clase_str in ("1", "D"):
            probs_por_rol["draw"] = pct
        elif clase_str in ("2", "A"):
            probs_por_rol["away_win"] = pct

    pred_str = str(prediccion).strip()
    if pred_str in ("0", "H", "L"):
        prediction_label = "home_win"
    elif pred_str in ("1", "D"):
        prediction_label = "draw"
    elif pred_str in ("2", "A"):
        prediction_label = "away_win"
    else:
        prediction_label = pred_str

    return {
        "prediction": prediction_label,
        "raw_prediction": pred_str,
        "confidence_percentage": round(float(np.max(probabilidades)) * 100, 2),
        "probabilities": probs_por_rol,
    }


def obtener_ultimas_estadisticas(Home: str, Away: str, df: pd.DataFrame) -> pd.DataFrame:
    filtro = (df["Home"] == Home) & (df["Away"] == Away)
    partido_historico = df[filtro].tail(1)

    if partido_historico.empty:
        raise ValueError("No se encontraron estadísticas previas para estos equipos.")

    columnas_requeridas = modelo.feature_names_in_
    return partido_historico[columnas_requeridas]


def obtener_historial_equipo(df: pd.DataFrame, equipo: str, days: int) -> Dict[str, Any]:
    partidos = df[(df["Home"] == equipo) | (df["Away"] == equipo)].tail(days)

    resultados = []
    goles_favor = 0
    goles_contra = 0

    for _, fila in partidos.iterrows():
        es_local = fila["Home"] == equipo
        gf = int(fila["HG"]) if es_local else int(fila["AG"])
        gc = int(fila["AG"]) if es_local else int(fila["HG"])

        goles_favor += gf
        goles_contra += gc

        if gf > gc:
            resultados.append("G")
        elif gf < gc:
            resultados.append("P")
        else:
            resultados.append("E")

    total_partidos = len(partidos)
    prom_gf = round(goles_favor / total_partidos, 2) if total_partidos > 0 else 0.0
    prom_gc = round(goles_contra / total_partidos, 2) if total_partidos > 0 else 0.0

    return {
        "equipo": equipo,
        "days_consultados": days,
        "racha_resultados": resultados,
        "promedio_gf": prom_gf,
        "promedio_gc": prom_gc,
    }


@app.get("/teams")
def obtener_equipos():
    try:
        lista_equipos = sorted(df_maestro["Home"].dropna().unique().tolist())
        return {"equipos": lista_equipos}
    except Exception:
        raise HTTPException(status_code=500, detail="Error al leer los equipos")


@app.post("/predict", summary="Predecir resultado y probabilidades", tags=["Predicciones"])
def predict_match(match: MatchRequest):
    try:
        features_df = obtener_ultimas_estadisticas(match.Home, match.Away, df_maestro)
        resultado = extraer_probabilidades(features_df)

        return {
            "Home": match.Home,
            "Away": match.Away,
            "prediction": resultado["prediction"],
            "confidence_percentage": resultado["confidence_percentage"],
            "probabilities": resultado["probabilities"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/stats/fatigue", summary="Obtener días de descanso", tags=["Estadísticas"])
def obtener_fatiga(match: MatchRequest):
    filtro = (df_maestro["Home"] == match.Home) & (df_maestro["Away"] == match.Away)
    partido = df_maestro[filtro].tail(1)

    if partido.empty:
        raise HTTPException(status_code=404, detail="No hay historial para estos equipos")

    return {
        "equipo_local": match.Home,
        "equipo_visitante": match.Away,
        "dias_descanso_local": int(partido["loc_descanso"].values[0]),
        "dias_descanso_visitante": int(partido["vis_descanso"].values[0]),
    }


@app.post("/stats/h2h", summary="Obtener historial directo (H2H)", tags=["Estadísticas"])
def obtener_h2h(match: MatchRequest):
    filtro = (df_maestro["Home"] == match.Home) & (df_maestro["Away"] == match.Away)
    partido = df_maestro[filtro].tail(1)

    if partido.empty:
        raise HTTPException(status_code=404, detail="No hay historial para estos equipos")

    return {
        "equipo_local": match.Home,
        "equipo_visitante": match.Away,
        "victorias_local": int(partido["h2h_victorias_local"].values[0]),
        "empates": int(partido["h2h_empates"].values[0]),
        "victorias_visitante": int(partido["h2h_victorias_visitante"].values[0]),
    }


@app.post("/stats/streak", summary="Obtener racha detallada y promedios", tags=["Estadísticas"])
def obtener_racha(request: RachaRequest):
    if request.days is None or request.days <= 0:
        raise HTTPException(status_code=400, detail="El número de días debe ser mayor a 0")

    local_data = obtener_historial_equipo(df_maestro, request.Home, request.days)
    visitante_data = obtener_historial_equipo(df_maestro, request.Away, request.days)

    return {
        "local": local_data,
        "visitante": visitante_data,
    }


@app.get("/stats/table", summary="Obtener tabla general de la Liga MX por temporada y torneo", tags=["Estadísticas"])
def obtener_tabla_general(season: Optional[str] = None, tournament: Optional[str] = None):
    """
    Devuelve la tabla general filtrada por temporada y opcionalmente por torneo (Apertura / Clausura).
    - Apertura: Partidos de Julio a Diciembre.
    - Clausura: Partidos de Enero a Junio.
    Si no se especifica temporada, toma la más reciente.
    """
    try:
        if not season:
            season = str(df_maestro["Season"].dropna().max())

        df_temporada = df_maestro[df_maestro["Season"] == season].copy()

        if df_temporada.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron datos para la temporada '{season}'",
            )

        df_temporada["Date_dt"] = pd.to_datetime(df_temporada["Date"], errors="coerce")
        df_temporada["Month"] = df_temporada["Date_dt"].dt.month

        if tournament:
            t_lower = tournament.strip().lower()
            if t_lower == "apertura":
                df_temporada = df_temporada[df_temporada["Month"].isin([7, 8, 9, 10, 11, 12])]
            elif t_lower == "clausura":
                df_temporada = df_temporada[df_temporada["Month"].isin([1, 2, 3, 4, 5, 6])]
            else:
                raise HTTPException(
                    status_code=400,
                    detail="El parámetro 'tournament' debe ser 'Apertura' o 'Clausura'",
                )

        if df_temporada.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No hay partidos registrados para el torneo {tournament} en la temporada {season}",
            )

        equipos = df_temporada["Home"].dropna().unique()
        tabla_datos = []

        for equipo in equipos:
            partidos_local = df_temporada[df_temporada["Home"] == equipo]
            partidos_visitante = df_temporada[df_temporada["Away"] == equipo]

            jj = len(partidos_local) + len(partidos_visitante)
            if jj == 0:
                continue

            jg, je, jp, gf, gc, pts = 0, 0, 0, 0, 0, 0

            for _, row in partidos_local.iterrows():
                goles_favor = int(row["HG"])
                goles_contra = int(row["AG"])
                gf += goles_favor
                gc += goles_contra

                if goles_favor > goles_contra:
                    jg += 1
                    pts += 3
                elif goles_favor == goles_contra:
                    je += 1
                    pts += 1
                else:
                    jp += 1

            for _, row in partidos_visitante.iterrows():
                goles_favor = int(row["AG"])
                goles_contra = int(row["HG"])
                gf += goles_favor
                gc += goles_contra

                if goles_favor > goles_contra:
                    jg += 1
                    pts += 3
                elif goles_favor == goles_contra:
                    je += 1
                    pts += 1
                else:
                    jp += 1

            dif = gf - gc

            tabla_datos.append({
                "team": equipo,
                "played": jj,
                "won": jg,
                "drawn": je,
                "lost": jp,
                "goals_for": gf,
                "goals_against": gc,
                "goal_difference": dif,
                "points": pts,
            })

        tabla_ordenada = sorted(
            tabla_datos,
            key=lambda x: (x["points"], x["goal_difference"], x["goals_for"]),
            reverse=True,
        )

        for index, row in enumerate(tabla_ordenada, start=1):
            row["position"] = index

        ultima_fecha_ts = df_temporada["Date_dt"].dropna().max()
        last_updated = (
            ultima_fecha_ts.strftime("%Y-%m-%d")
            if pd.notna(ultima_fecha_ts)
            else None
        )

        return {
            "season": season,
            "tournament": tournament if tournament else "Completo (Anual)",
            "last_updated": last_updated,
            "standings": tabla_ordenada,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al calcular la tabla general: {str(e)}",
        )


@app.post("/ai/match-analysis", summary="Análisis estructurado y consultas con IA", tags=["IA"])
def ai_match_analysis(payload: AIAnalysisRequest):
    match_data = payload.model_dump(exclude={"action", "question"})

    if None in (
        match_data.get("prob_local"),
        match_data.get("prob_empate"),
        match_data.get("prob_visitante"),
    ):
        try:
            features_df = obtener_ultimas_estadisticas(payload.home, payload.away, df_maestro)
            resultado = extraer_probabilidades(features_df)
            probs = resultado["probabilities"]
            match_data["prob_local"] = (
                match_data.get("prob_local")
                if match_data.get("prob_local") is not None
                else probs["home_win"]
            )
            match_data["prob_empate"] = (
                match_data.get("prob_empate")
                if match_data.get("prob_empate") is not None
                else probs["draw"]
            )
            match_data["prob_visitante"] = (
                match_data.get("prob_visitante")
                if match_data.get("prob_visitante") is not None
                else probs["away_win"]
            )
        except Exception:
            match_data["prob_local"] = match_data.get("prob_local") or 0.0
            match_data["prob_empate"] = match_data.get("prob_empate") or 0.0
            match_data["prob_visitante"] = match_data.get("prob_visitante") or 0.0

    filtro = (df_maestro["Home"] == payload.home) & (df_maestro["Away"] == payload.away)
    partido = df_maestro[filtro].tail(1)

    if not partido.empty:
        if match_data.get("descanso_local") is None and "loc_descanso" in partido.columns:
            match_data["descanso_local"] = int(partido["loc_descanso"].values[0])
        if match_data.get("descanso_visitante") is None and "vis_descanso" in partido.columns:
            match_data["descanso_visitante"] = int(partido["vis_descanso"].values[0])
        if match_data.get("h2h") is None and {
            "h2h_victorias_local",
            "h2h_empates",
            "h2h_victorias_visitante",
        }.issubset(partido.columns):
            match_data["h2h"] = (
                f"{payload.home} {int(partido['h2h_victorias_local'].values[0])}, "
                f"Empates {int(partido['h2h_empates'].values[0])}, "
                f"{payload.away} {int(partido['h2h_victorias_visitante'].values[0])}"
            )

    if match_data.get("racha_local") is None:
        h_hist = obtener_historial_equipo(df_maestro, payload.home, 5)
        match_data["racha_local"] = (
            f"{', '.join(h_hist['racha_resultados'])} "
            f"({h_hist['promedio_gf']} GF, {h_hist['promedio_gc']} GC)"
        )

    if match_data.get("racha_visitante") is None:
        a_hist = obtener_historial_equipo(df_maestro, payload.away, 5)
        match_data["racha_visitante"] = (
            f"{', '.join(a_hist['racha_resultados'])} "
            f"({a_hist['promedio_gf']} GF, {a_hist['promedio_gc']} GC)"
        )

    for key in ("descanso_local", "descanso_visitante", "racha_local", "racha_visitante", "h2h"):
        if match_data.get(key) is None:
            match_data[key] = "N/D"

    return generate_match_ai_response(
        match_data=match_data,
        action=payload.action,
        question=payload.question or "",
    )