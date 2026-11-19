# API de Predicción Deportiva - Liga MX

Backend desarrollado con FastAPI, Pandas y Scikit-learn para procesar datos históricos y servir predicciones de partidos de la Liga MX a través de una API REST.

---

## Endpoints Disponibles

### 1. Obtener Equipos
* **Ruta:** `/teams`
* **Método:** `GET`
* **Descripción:** Devuelve una lista alfabética de todos los equipos disponibles en el dataset histórico para poblar selectores en la interfaz de usuario.
* **Ejemplo de respuesta (`200 OK`):**
  ```json
  {
    "equipos": [
      "Atlas",
      "Club America",
      "Cruz Azul",
      "Guadalajara Chivas",
      "Monterrey",
      "Tigres UANL"
    ]
  }
  ```

---

### 2. Consultar Fatiga (Días de Descanso)
* **Ruta:** `/stats/fatigue`
* **Método:** `POST`
* **Descripción:** Consulta el último registro del enfrentamiento para calcular los días de descanso de ambos equipos.
* **Cuerpo de la petición (JSON):**
  ```json
  {
    "Home": "Guadalajara Chivas",
    "Away": "Club America"
  }
  ```
* **Ejemplo de respuesta (`200 OK`):**
  ```json
  {
    "equipo_local": "Guadalajara Chivas",
    "equipo_visitante": "Club America",
    "dias_descanso_local": 7,
    "dias_descanso_visitante": 4
  }
  ```

---

### 3. Predecir Partido y Desglose de Probabilidades
* **RUTA:** `/predict`
* **Método:** `POST`
* **Descripción:** Ejecuta el modelo de Machine Learning sobre el contexto histórico de ambos equipos y devuelve tanto el resultado más probable como el desglose porcentual de los tres escenarios posibles (Local, Empate, Visitante).
* **Cuerpo de la Petición (JSON Body):**
  ```json
  {
    "Home": "Guadalajara Chivas",
    "Away": "Club America"
  }
  ```
* **Ejemplo de respuesta (`200 OK`):**
  ```json
  {
    "Home": "Guadalajara Chivas",
    "Away": "Club America",
    "prediction": "H",
    "confidence_percentage": 48.0,
    "probabilities": {
      "home_win": 48.0,
      "draw": 28.0,
      "away_win": 24.0
    }
  }
  ```

---

### 4. Obtener Historial Directo (H2H)
* **Ruta:** `/stats/h2h`
* **Método:** `POST`
* **Descripción:** Devuelve el historial acumulado de enfrentamientos previos (victorias local, empates, victorias visitante) entre ambos equipos.
* **Cuerpo de la Petición (JSON Body):**
  ```json
  {
    "Home": "Guadalajara Chivas",
    "Away": "Club America"
  }
  ```
* **Ejemplo de respuesta (`200 OK`):**
  ```json
  {
    "equipo_local": "Guadalajara Chivas",
    "equipo_visitante": "Club America",
    "victorias_local": 1,
    "empates": 2,
    "victorias_visitante": 2
  }
  ```

---

### 5. Obtener Racha Personalizada y Resultados (Últimos N Partidos)
* **URL:** `/stats/streak`
* **Método:** `POST`
* **Descripción:** Devuelve el historial de resultados individuales (`G` = Ganado, `E` = Empatado, `P` = Perdido) y promedios de goles en un rango configurable de días o partidos previos (`days`).
* **Cuerpo de la Petición (JSON Body):**
  ```json
  {
    "Home": "Guadalajara Chivas",
    "Away": "Club America",
    "days": 5
  }
  ```

 * **Ejemplo de respuesta (`200 OK`):**
  ```json
  {
    "local": {
        "equipo": "Guadalajara Chivas",
        "days_consultados": 5,
        "racha_resultados": [
            "G",
            "E",
            "G",
            "G",
            "E"
        ],
        "promedio_gf": 2.8,
        "promedio_gc": 1
    },
    "visitante": {
        "equipo": "Club America",
        "days_consultados": 5,
        "racha_resultados": [
            "G",
            "G",
            "G",
            "P",
            "E"
        ],
        "promedio_gf": 2.4,
        "promedio_gc": 1.4
    }
  }
  ```

---

### 6. Obtener Tabla General de Posiciones (Con filtros de temporada y torneo)
* **RUTA:** `/stats/table`
* **Método:** `GET`
* **Parámetros de consulta (Query Params):** 
  * `season` (opcional): Especifica la temporada a consultar (Ej. `2023/2024`). Si se omite, devuelve automáticamente la más reciente del dataset.
  * `tournament` (opcional): Filtra por torneo corto (`Apertura` o `Clausura`). Si se omite, calcula el acumulado completo o anual disponible.
* **Descripción:** Calcula y devuelve el estado actual de la tabla general con estadísticas acumuladas (partidos jugados, ganados, empatados, perdidos, goles a favor, goles en contra, diferencia y puntos), ordenada por rendimiento.
* **Ejemplo de Petición:** `GET /stats/table?season=2023/2024&tournament=Apertura`
* **Ejemplo de Respuesta (`200 OK`):**
  ```json
  {
    "season": "2023/2024",
    "tournament": "Apertura",
    "standings": [
      {
        "position": 1,
        "team": "Club America",
        "played": 17,
        "won": 12,
        "drawn": 4,
        "lost": 1,
        "goals_for": 37,
        "goals_against": 14,
        "goal_difference": 23,
        "points": 40
      }
    ]
  }
  ```

---

### 7. Análisis Estructurado y Consultas con IA
* **Ruta:** `/ai/match-analysis`
* **Método:** `POST`
* **Descripción:** Genera un análisis estadístico estructurado (`pronostico`, `factores`, `alcance` y `answer`) o responde consultas específicas sobre el cruce de Liga MX dentro de **Oráculo MX**, apoyándose en el modelo predictivo y `dataset_maestro` en Supabase. Incluye caché en memoria para consultas repetidas, filtro de pertinencia (*guardrail*) para evitar preguntas fuera del alcance del sistema, cascada de modelos rápidos de Gemini (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.6-flash`) con límite de espera y respaldo determinista automático ante saturación o fallos de red.
  * **Campos opcionales de métricas:** Si no se envían `prob_local`, `prob_empate`, `prob_visitante`, `descanso_local`, `descanso_visitante`, `racha_local`, `racha_visitante` o `h2h`, el endpoint los completa automáticamente desde el modelo y el historial disponible para el cruce.
  * **Acciones (`action`):** Soporta `"summary"` (lectura estructurada general del partido), `"executive_brief"` (síntesis breve y directa), `"question"` (respuesta dirigida a `question` validando el alcance del tema) y `"model_limits"` (explicación de los alcances y limitaciones del modelo y los datos históricos).
  * **Variables de entorno:** Requiere `GEMINI_API_KEY` y permite ajustar opcionalmente `GEMINI_MODEL`, `GEMINI_TIMEOUT` y `GEMINI_HTTP_TIMEOUT_MS`.
* **Cuerpo de la petición (JSON):**
  ```json
  {
    "home": "Atlas",
    "away": "UNAM Pumas",
    "prob_local": 28.29,
    "prob_empate": 32.94,
    "prob_visitante": 38.76,
    "descanso_local": 7,
    "descanso_visitante": 6,
    "racha_local": "G, P, E, P, E (1.4 GF, 2.0 GC)",
    "racha_visitante": "E, P, G, P, E (1.0 GF, 1.6 GC)",
    "h2h": "Atlas 1, Empates 3, UNAM Pumas 1",
    "action": "summary",
    "question": ""
  }
  ```

* **Ejemplo de Respuesta (`200 OK`):**
  ```json
  {
    "home": "Atlas",
    "away": "UNAM Pumas",
    "action": "summary",
    "cached": false,
    "source": "gemini",
    "model_used": "gemini-3.5-flash-lite",
    "sections": {
      "pronostico": "El análisis estadístico otorga a UNAM Pumas una probabilidad de victoria de 38.76%, superando el 28.29% asignado a Atlas y el 32.94% de empate.",
      "factores": "UNAM Pumas llega con 6 días de descanso frente a los 7 de Atlas, registrando una racha de E, P, G, P, E con 1.0 GF y 1.6 GC, mientras que Atlas presenta una racha de G, P, E, P, E con 1.4 GF y 2.0 GC. El historial directo (H2H) muestra paridad con 1 triunfo para Atlas, 3 empates y 1 victoria para Pumas.",
      "alcance": "El modelo de Oráculo MX evalúa las probabilidades de triunfo, descanso, rachas recientes y antecedentes directos entre ambos equipos para este encuentro de la Liga MX."
      },
    "answer": "UNAM Pumas parte como favorito con un 38.76% de probabilidad de triunfo frente al 28.29% de Atlas, en un duelo respaldado por el historial de enfrentamientos directos y las estadísticas recientes de ambos clubes."
  }
  ```

---

## Documentación Interactiva
Puedes probar los endpoints directamente desde la interfaz automática de Swagger visitando la ruta `/docs` de tu servidor en ejecución.