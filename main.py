from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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