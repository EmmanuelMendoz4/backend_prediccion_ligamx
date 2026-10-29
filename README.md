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

### 3. Predecir Partido
* **Ruta:** `/predict`
* **Método:** `POST`
* **Descripción:** Procesa el contexto histórico y ejecuta el modelo de Machine Learning para calcular la probabilidad del resultado.
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
    "Home": "Guadalajara Chivas",
    "Away": "Club America",
    "prediction": "0",
    "confidence_percentage": 38.44
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
  

## Documentación Interactiva
Puedes probar los endpoints directamente desde la interfaz automática de Swagger visitando la ruta `/docs` de tu servidor en ejecución.