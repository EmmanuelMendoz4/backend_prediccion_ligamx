from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="API Predicción Liga MX")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar el modelo y el dataset al iniciar
datos_exportados = joblib.load('modelo_ligamx.joblib')
modelo = datos_exportados['modelo']
df_maestro = pd.read_csv('dataset_maestro.csv')

class MatchRequest(BaseModel):
    Home: str
    Away: str

class RachaRequest(BaseModel):
    Home: str = Field(..., description="Nombre del equipo local")
    Away: str = Field(..., description="Nombre del equipo visitante")
    days: Optional[int] = Field(default=5, description="Número de partidos a consultar")

# Función para extraer las 27 características
def obtener_ultimas_estadisticas(Home, Away, df):
    # Filtramos la última vez que estos dos equipos se enfrentaron
    filtro = (df['Home'] == Home) & (df['Away'] == Away)
    partido_historico = df[filtro].tail(1)
    
    if partido_historico.empty:
        raise ValueError("No se encontraron estadísticas previas para estos equipos.")

    # El modelo escanea tus encabezados y extrae solo lo que necesita
    columnas_requeridas = modelo.feature_names_in_
    features = partido_historico[columnas_requeridas]
    
    return features

def obtener_historial_equipo(df, equipo, days):
    # Filtra los últimos N partidos donde el equipo haya jugado como Local o Visitante
    partidos = df[(df['Home'] == equipo) | (df['Away'] == equipo)].tail(days)
    
    resultados = []
    goles_favor = 0
    goles_contra = 0
    
    for _, fila in partidos.iterrows():
        es_local = fila['Home'] == equipo
        # Ajusta 'HomeGoals' y 'AwayGoals' si en tu CSV se llaman distinto (ej: 'GolesLocal', 'FTHG')
        gf = int(fila['HG']) if es_local else int(fila['AG'])
        gc = int(fila['AG']) if es_local else int(fila['HG'])
        
        goles_favor += gf
        goles_contra += gc
        
        if gf > gc:
            resultados.append("G")
        elif gf < gc:
            resultados.append("P")
        else:
            resultados.append("E")
            
    prom_gf = round(goles_favor / len(partidos), 2) if len(partidos) > 0 else 0.0
    prom_gc = round(goles_contra / len(partidos), 2) if len(partidos) > 0 else 0.0
    
    return {
        "equipo": equipo,
        "days_consultados": days,
        "racha_resultados": resultados, # Arreglo exacto para los cuadritos visuales ["G", "E", "P"]
        "promedio_gf": prom_gf,
        "promedio_gc": prom_gc
    }

@app.get("/teams")
def obtener_equipos():
    try:
        # Extrae los nombres únicos de los equipos de la columna 'Home' y los ordena alfabéticamente
        lista_equipos = sorted(df_maestro['Home'].dropna().unique().tolist())
        return {"equipos": lista_equipos}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al leer los equipos")

@app.post("/predict")
def predict_match(match: MatchRequest):
    try:
        # 1. Extraer los datos (DataFrame de 1 fila y 27 columnas)
        features_df = obtener_ultimas_estadisticas(match.Home, match.Away, df_maestro)
        
        # 2. Hacer la predicción real
        prediccion = modelo.predict(features_df)[0] # Ej: 'H', 'D', 'A'
        
        # 3. Obtener la probabilidad (confianza) de esa predicción
        probabilidades = modelo.predict_proba(features_df)[0]
        confianza_maxima = round(float(np.max(probabilidades)) * 100, 2)
        
        return {
            "Home": match.Home,
            "Away": match.Away,
            "prediction": str(prediccion), 
            "confidence_percentage": confianza_maxima
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/stats/fatigue", summary="Obtener días de descanso", tags=["Estadísticas"])
def obtener_fatiga(match: MatchRequest):
    """
    Devuelve los días de descanso con los que llega cada equipo al partido.
    """
    filtro = (df_maestro['Home'] == match.Home) & (df_maestro['Away'] == match.Away)
    partido = df_maestro[filtro].tail(1)
    
    if partido.empty:
        raise HTTPException(status_code=404, detail="No hay historial para estos equipos")
        
    return {
        "equipo_local": match.Home,
        "equipo_visitante": match.Away,
        "dias_descanso_local": int(partido['loc_descanso'].values[0]),
        "dias_descanso_visitante": int(partido['vis_descanso'].values[0])
    }

@app.post("/stats/h2h", summary="Obtener historial directo (H2H)", tags=["Estadísticas"])
def obtener_h2h(match: MatchRequest):
    """
    Devuelve el historial directo de enfrentamientos (H2H) entre el equipo local y el visitante
    para alimentar la gráfica de barras comparativa.
    """
    filtro = (df_maestro['Home'] == match.Home) & (df_maestro['Away'] == match.Away)
    partido = df_maestro[filtro].tail(1)
    
    if partido.empty:
        raise HTTPException(status_code=404, detail="No hay historial para estos equipos")
        
    return {
        "equipo_local": match.Home,
        "equipo_visitante": match.Away,
        # Si tu Data Lead les puso otro nombre, solo cámbialo aquí adentro de los corchetes.
        "victorias_local": int(partido['h2h_victorias_local'].values[0]),
        "empates": int(partido['h2h_empates'].values[0]),
        "victorias_visitante": int(partido['h2h_victorias_visitante'].values[0])
    }

@app.post("/stats/streak", summary="Obtener racha detallada y promedios", tags=["Estadísticas"])
def obtener_racha(request: RachaRequest):
    """
    Devuelve la racha de resultados (G, E, P) y promedios de goles de los últimos N partidos
    para cada equipo de manera dinámica.
    """
    if request.days <= 0:
        raise HTTPException(status_code=400, detail="El número de días debe ser mayor a 0")
        
    local_data = obtener_historial_equipo(df_maestro, request.Home, request.days)
    visitante_data = obtener_historial_equipo(df_maestro, request.Away, request.days)
    
    return {
        "local": local_data,
        "visitante": visitante_data
    }