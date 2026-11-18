from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pydantic import BaseModel, Field
import os
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv

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
datos_exportados = joblib.load('modelo_ligamx.joblib')
modelo = datos_exportados['modelo']
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No se encontró la variable de entorno DATABASE_URL")

# Crear el motor de conexión a PostgreSQL
engine = create_engine(DATABASE_URL)

# Cargar los datos directo de la tabla en Supabase al iniciar la API
df_maestro = pd.read_sql("SELECT * FROM dataset_maestro", con=engine)

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

@app.post("/predict", summary="Predecir resultado y probabilidades", tags=["Predicciones"])
def predict_match(match: MatchRequest):
    try:
        features_df = obtener_ultimas_estadisticas(match.Home, match.Away, df_maestro)
        
        prediccion = modelo.predict(features_df)[0]
        probabilidades = modelo.predict_proba(features_df)[0]
        
        # Empareja cada clase del modelo con su probabilidad redondeada
        mapa_probs = {
            str(clase): round(float(prob) * 100, 2)
            for clase, prob in zip(modelo.classes_, probabilidades)
        }
        
        # Nota: Si tus clases en el modelo son '0', '1', '2' en lugar de 'H', 'D', 'A', 
        # ajusta las llaves dentro de mapa_probs.get(...)
        home_prob = mapa_probs.get("H", mapa_probs.get("1", 0.0))
        draw_prob = mapa_probs.get("D", mapa_probs.get("0", 0.0))
        away_prob = mapa_probs.get("A", mapa_probs.get("2", 0.0))
        
        return {
            "Home": match.Home,
            "Away": match.Away,
            "prediction": str(prediccion),
            "confidence_percentage": round(float(np.max(probabilidades)) * 100, 2),
            "probabilities": {
                "home_win": home_prob,
                "draw": draw_prob,
                "away_win": away_prob
            }
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
    
from typing import Optional

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
            season = df_maestro['Season'].dropna().max()
            
        # 1. Filtrar por temporada base
        df_temporada = df_maestro[df_maestro['Season'] == season].copy()
        
        if df_temporada.empty:
            raise HTTPException(status_code=404, detail=f"No se encontraron datos para la temporada '{season}'")
        
        # 2. Asegurar formato de fecha para extraer el mes
        df_temporada['Date'] = pd.to_datetime(df_temporada['Date'])
        df_temporada['Month'] = df_temporada['Date'].dt.month
        
        # 3. Aplicar filtro de Torneo si el usuario lo solicita
        if tournament:
            t_lower = tournament.lower()
            if t_lower == 'apertura':
                # Apertura: Meses 7 a 12 (Julio - Diciembre)
                df_temporada = df_temporada[df_temporada['Month'].isin([7, 8, 9, 10, 11, 12])]
            elif t_lower == 'clausura':
                # Clausura: Meses 1 a 6 (Enero - Junio)
                df_temporada = df_temporada[df_temporada['Month'].isin([1, 2, 3, 4, 5, 6])]
            else:
                raise HTTPException(status_code=400, detail="El parámetro 'tournament' debe ser 'Apertura' o 'Clausura'")

        if df_temporada.empty:
            raise HTTPException(status_code=404, detail=f"No hay partidos registrados para el torneo {tournament} en la temporada {season}")

        equipos = df_temporada['Home'].dropna().unique()
        tabla_datos = []
        
        for equipo in equipos:
            partidos_local = df_temporada[df_temporada['Home'] == equipo]
            partidos_visitante = df_temporada[df_temporada['Away'] == equipo]
            
            jj = len(partidos_local) + len(partidos_visitante)
            if jj == 0:
                continue
                
            jg, je, jp, gf, gc, pts = 0, 0, 0, 0, 0, 0
            
            for _, row in partidos_local.iterrows():
                goles_favor = int(row['HG'])
                goles_contra = int(row['AG'])
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
                goles_favor = int(row['AG'])
                goles_contra = int(row['HG'])
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
                "points": pts
            })
            
        tabla_ordenada = sorted(tabla_datos, key=lambda x: (x['points'], x['goal_difference']), reverse=True)
        
        for index, row in enumerate(tabla_ordenada, start=1):
            row["position"] = index
            
        return {
            "season": season,
            "tournament": tournament if tournament else "Completo (Anual)",
            "standings": tabla_ordenada
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al calcular la tabla general: {str(e)}")