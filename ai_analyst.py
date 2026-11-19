import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types

MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
]
TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT", "10.0"))
HTTP_TIMEOUT_MS = int(float(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "10000")))

_CLIENT: Optional[genai.Client] = None
_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

_OUT_OF_SCOPE_KEYWORDS = (
    "receta",
    "cocina",
    "tarea de",
    "código de",
    "programar en",
    "política",
    "presidente",
    "elecciones",
    "bitcoin",
    "cripto",
    "apuesta segura",
    "parlay",
    "chiste",
    "poema",
    "película",
    "serie",
)

_DOMAIN_KEYWORDS = (
    "partido",
    "pronóstico",
    "pronostico",
    "probabilidad",
    "empate",
    "victoria",
    "local",
    "visitante",
    "descanso",
    "fatiga",
    "racha",
    "gol",
    "goles",
    "h2h",
    "historial",
    "liga mx",
    "oráculo",
    "oraculo",
    "modelo",
    "límite",
    "limite",
    "alcance",
    "dato",
    "datos",
)


def _get_client() -> Optional[genai.Client]:
    global _CLIENT
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if _CLIENT is None:
        _CLIENT = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
        )
    return _CLIENT


def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_in_scope(question: str, home: str, away: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return True
    if home.lower() in q or away.lower() in q:
        return True
    if any(word in q for word in _DOMAIN_KEYWORDS):
        return True
    if any(bad in q for bad in _OUT_OF_SCOPE_KEYWORDS):
        return False
    return len(q.split()) <= 10


def _build_fallback(match_data: Dict[str, Any], action: str, question: str = "") -> Dict[str, Any]:
    home = match_data.get("home", "Equipo Local")
    away = match_data.get("away", "Equipo Visitante")
    prob_home = match_data.get("prob_local", 0.0)
    prob_draw = match_data.get("prob_empate", 0.0)
    prob_away = match_data.get("prob_visitante", 0.0)
    rest_home = match_data.get("descanso_local", "N/D")
    rest_away = match_data.get("descanso_visitante", "N/D")
    form_home = str(match_data.get("racha_local", "N/D")).strip()
    form_away = str(match_data.get("racha_visitante", "N/D")).strip()
    h2h = match_data.get("h2h", "N/D")

    pronostico = (
        f"Las probabilidades estimadas ubican a {away} en {prob_away}%, el empate en {prob_draw}% "
        f"y a {home} en {prob_home}%, reflejando el balance estadístico del cruce."
    )
    factores = (
        f"En días de descanso registran {home}: {rest_home} y {away}: {rest_away}. "
        f"En sus últimos cinco partidos se reporta {home} [{form_home}] frente a {away} [{form_away}], "
        f"con historial directo registrado en: {h2h}."
    )
    alcance = (
        "Este análisis se deriva exclusivamente de las métricas y registros históricos disponibles "
        "en Oráculo MX; no contempla lesiones recientes, alineaciones confirmadas ni factores externos."
    )

    if action == "executive_brief":
        answer = (
            f"{home} vs. {away}: {away} ({prob_away}%), empate ({prob_draw}%) y {home} ({prob_home}%). "
            f"Descanso: {rest_home} vs. {rest_away}; H2H: {h2h}."
        )
    elif action == "model_limits":
        answer = (
            f"{alcance} El sistema evalúa probabilidades históricas y variables de contexto del dataset, "
            "sin garantizar resultados ni incorporar eventos no cargados en la base."
        )
    elif action == "question":
        answer = f"Sobre tu consulta ('{question.strip()}'), {pronostico} {factores} {alcance}"
    else:
        answer = f"{pronostico} {factores} {alcance}"

    return {
        "home": home,
        "away": away,
        "action": action,
        "cached": False,
        "source": "fallback",
        "sections": {
            "pronostico": pronostico,
            "factores": factores,
            "alcance": alcance,
        },
        "answer": answer,
    }


def _call_gemini_structured(
    match_data: Dict[str, Any],
    action: str,
    question: str = "",
) -> Optional[Dict[str, Any]]:
    client = _get_client()
    if client is None:
        return None

    home = match_data.get("home", "Equipo Local")
    away = match_data.get("away", "Equipo Visitante")

    system_rules = (
        "Eres el analista oficial de Oráculo MX para la Liga MX. "
        "Responde SIEMPRE en español, con tono profesional, claro y conciso. "
        "REGLA ESTRICTA: Usa ÚNICAMENTE los datos numéricos y estadísticos proporcionados en el contexto. "
        "Nunca inventes lesiones, fichajes, técnicos, árbitros, clima ni marcadores no incluidos. "
        "Devuelve ÚNICAMENTE un objeto JSON válido con las claves exactas: "
        '{"pronostico": "...", "factores": "...", "alcance": "...", "answer": "..."}. '
        "Cada sección debe ser breve (1 a 2 oraciones)."
    )

    action_instruction = {
        "summary": "Genera la lectura estructurada completa del partido en pronostico, factores y alcance; en answer coloca un resumen ejecutivo de 2 oraciones.",
        "executive_brief": "Enfoca answer en un mensaje breve y directo (máximo 2 oraciones) listo para compartir o leer rápido, llenando también pronostico, factores y alcance.",
        "model_limits": "Enfatiza en alcance y answer los límites reales del modelo y del dataset histórico (qué no evalúa el sistema).",
        "question": f"Responde de forma precisa y acotada a esta pregunta del usuario sobre el partido o la app: '{question.strip()}'. Refleja la respuesta en answer y completa pronostico, factores y alcance.",
    }.get(action, "Genera la lectura estructurada del partido.")

    prompt = (
        f"{system_rules}\n\n"
        f"Instrucción de acción: {action_instruction}\n\n"
        f"Datos del partido:\n{json.dumps(match_data, ensure_ascii=False, indent=2)}"
    )

    raw_text = ""
    used_model = None

    for model_name in dict.fromkeys(MODEL_CANDIDATES):
        def _run(m: str = model_name) -> str:
            response = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=300,
                ),
            )
            return (response.text or "").strip()

        inicio = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                while True:
                    restante = TIMEOUT_SECONDS - (time.time() - inicio)
                    if restante <= 0:
                        raise FuturesTimeoutError()
                    try:
                        raw_text = future.result(timeout=min(2.0, restante))
                        break
                    except FuturesTimeoutError:
                        print(f"[ai_analyst] ({model_name}) Esperando respuesta... ({time.time() - inicio:.1f}s)", flush=True)

            if raw_text:
                used_model = model_name
                print(f"[ai_analyst] {model_name} respondió en {time.time() - inicio:.2f}s", flush=True)
                break
        except (FuturesTimeoutError, Exception) as exc:
            print(
                f"[ai_analyst] {model_name} falló tras {time.time() - inicio:.2f}s "
                f"({type(exc).__name__}: {exc}), probando siguiente...",
                flush=True,
            )
            raw_text = ""
    else:
        print("[ai_analyst] Respaldo determinista activado tras agotar modelos.", flush=True)
        return None

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE).strip()

    try:
        parsed = json.loads(cleaned)
        fallback = _build_fallback(match_data, action, question)
        sections = {
            "pronostico": str(parsed.get("pronostico") or fallback["sections"]["pronostico"]).strip(),
            "factores": str(parsed.get("factores") or fallback["sections"]["factores"]).strip(),
            "alcance": str(parsed.get("alcance") or fallback["sections"]["alcance"]).strip(),
        }
        answer = str(parsed.get("answer") or f"{sections['pronostico']} {sections['factores']}").strip()
        return {
            "home": home,
            "away": away,
            "action": action,
            "cached": False,
            "source": "gemini",
            "model_used": used_model,
            "sections": sections,
            "answer": answer,
        }
    except json.JSONDecodeError:
        fallback = _build_fallback(match_data, action, question)
        fallback["source"] = "gemini"
        fallback["model_used"] = used_model
        fallback["answer"] = cleaned
        return fallback

def generate_match_ai_response(
    match_data: Dict[str, Any],
    action: str = "summary",
    question: str = "",
) -> Dict[str, Any]:
    home = str(match_data.get("home", "")).strip()
    away = str(match_data.get("away", "")).strip()
    norm_action = _normalize_text(action) or "summary"
    norm_question = _normalize_text(question)

    if norm_action == "question" and norm_question and not _is_in_scope(norm_question, home, away):
        base = _build_fallback(match_data, "question", question)
        base["source"] = "guardrail"
        base["answer"] = (
            f"Solo puedo responder consultas relacionadas con el cruce {home} vs. {away}, "
            "las métricas y probabilidades del modelo o los alcances de Oráculo MX. "
            "Intenta preguntar por descanso, racha, historial H2H o qué factores no considera el sistema."
        )
        return base

    cache_key = (
        _normalize_text(home),
        _normalize_text(away),
        norm_action,
        norm_question,
    )

    if cache_key in _CACHE:
        cached_item = dict(_CACHE[cache_key])
        cached_item["sections"] = dict(_CACHE[cache_key]["sections"])
        cached_item["cached"] = True
        print(f"[ai_analyst] Cache HIT para {cache_key}", flush=True)
        return cached_item

    print(f"[ai_analyst] Cache MISS -> consultando IA para {cache_key}...", flush=True)
    result = _call_gemini_structured(match_data, norm_action, question)
    if result is None:
        result = _build_fallback(match_data, norm_action, question)

    _CACHE[cache_key] = dict(result)
    _CACHE[cache_key]["sections"] = dict(result["sections"])
    return result