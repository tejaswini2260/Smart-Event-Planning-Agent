"""
agents.py - Multi-agent architecture for the Smart Event Planning Agent.

Agents (each a separate class with run() -> AgentResult):
  1. RequirementAnalysisAgent   deterministic validation
  2. KnowledgeResearchAgent     RAG: Ollama embeddings + NumPy cosine similarity
  3. BudgetPlanningAgent        deterministic maths (utils/budget_utils.py)
  4. VenueLogisticsAgent        deterministic capacity/logistics rules
  5. SchedulePlanningAgent      deterministic timeline + preparation tasks
  6. EventPlanWriterAgent       LLM (Ollama llama3.2:3b) with deterministic fallback
  7. QualityAssuranceAgent      deterministic verification (PASS/FAIL)

EventPlanningOrchestrator calls them in order and applies the DETERMINISTIC
ORCHESTRATION RULES listed in ORCHESTRATION_RULES. These routing decisions are
fixed Python rules, not autonomous LLM decisions.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests

from utils import budget_utils as bu

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KB_PATH = PROJECT_ROOT / "knowledge_base" / "event_packages.txt"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

EVENT_TYPES: List[str] = list(bu.EVENT_TYPES)
FOOD_PREFERENCES: List[str] = list(bu.FOOD_MULTIPLIERS.keys())
REQUIRED_FIELDS = [
    "event_type", "event_date", "location", "guest_count",
    "total_budget", "venue_capacity", "food_preference",
]
MIN_GUESTS, MAX_GUESTS, MAX_TEXT_LENGTH = 1, 5000, 500

SUCCESS, WARNING, FAILED, SKIPPED = "SUCCESS", "WARNING", "FAILED", "SKIPPED"

ORCHESTRATION_RULES = [
    ("R1", "If requirement validation fails, stop. No plan is generated and the status is INVALID_INPUT."),
    ("R2", "If the knowledge base is missing/empty or embeddings fail, continue with keyword fallback or no context and record a warning."),
    ("R3", "If the budget agent fails, stop (the plan cannot be built without a valid budget)."),
    ("R4", "If guests exceed venue capacity or the catering budget is overrun, still draft the plan but flag the problem; the final status can never be VALIDATED."),
    ("R5", "If the LLM fails or returns unusable text, use the deterministic fallback plan and record a warning."),
    ("R6", "The final status is VALIDATED only if the QA Agent returns PASS and no critical constraint failed."),
]

DISCLAIMER = (
    "\n\n---\n**Important:** This is a planning aid. All prices and budget splits are illustrative "
    "estimates, not vendor quotations. This application does not reserve venues, contact vendors "
    "or process payments; please confirm every detail with real vendors in writing."
)


# --------------------------------------------------------------------------- #
# Shared data structures
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    agent: str
    action: str
    status: str
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseAgent:
    name = "Base Agent"
    responsibility = ""
    inputs = ""
    outputs = ""
    error_handling = ""

    def describe(self) -> Dict[str, str]:
        return {
            "Agent": self.name,
            "Responsibility": self.responsibility,
            "Inputs": self.inputs,
            "Outputs": self.outputs,
            "Error handling": self.error_handling,
        }


def parse_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or number != int(number):
        return None
    return int(number)


# --------------------------------------------------------------------------- #
# Ollama client (REST API via requests)
# --------------------------------------------------------------------------- #
class OllamaError(Exception):
    """Any problem talking to the local Ollama server (message is user-friendly)."""


class OllamaClient:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, llm_model: str = LLM_MODEL,
                 embed_model: str = EMBED_MODEL, timeout: int = 600):
        self.host = host.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict] = None,
                 model: Optional[str] = None, timeout: Optional[int] = None) -> dict:
        url = f"{self.host}{path}"
        try:
            resp = requests.request(method, url, json=payload, timeout=timeout or self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                "Cannot connect to Ollama. Install it from ollama.com and make sure it is running "
                "(open the Ollama app or run 'ollama serve')."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError("Ollama took too long to respond. Try again, or use a smaller request.") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Request to Ollama failed: {exc}") from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            if resp.status_code == 404 or "not found" in str(detail).lower():
                name = model or "the required model"
                raise OllamaError(f"Model '{name}' is not available in Ollama. Download it with: ollama pull {name}")
            raise OllamaError(f"Ollama returned an error ({resp.status_code}): {detail}")
        try:
            return resp.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned an unreadable response.") from exc

    @staticmethod
    def _model_installed(model: str, installed: List[str]) -> bool:
        if model in installed:
            return True
        return ":" not in model and f"{model}:latest" in installed

    def check_status(self) -> Dict[str, Any]:
        try:
            data = self._request("GET", "/api/tags", timeout=5)
        except OllamaError as exc:
            return {"running": False, "installed_models": [],
                    "missing_models": [self.llm_model, self.embed_model], "message": str(exc)}
        installed = [m.get("name", "") for m in data.get("models", [])]
        missing = [m for m in (self.llm_model, self.embed_model) if not self._model_installed(m, installed)]
        message = "Ollama is running and both models are installed." if not missing else (
            "Ollama is running but models are missing. Run: " + " ; ".join(f"ollama pull {m}" for m in missing))
        return {"running": True, "installed_models": installed, "missing_models": missing, "message": message}

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            data = self._request("POST", "/api/embed", {"model": self.embed_model, "input": batch},
                                 model=self.embed_model)
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch) or any(not e for e in embeddings):
                raise OllamaError("Ollama returned incomplete embeddings. Update Ollama and re-run: "
                                  f"ollama pull {self.embed_model}")
            vectors.extend(embeddings)
        return vectors

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.4,
                 num_predict: int = 1500) -> str:
        payload = {
            "model": self.llm_model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 8192},
        }
        if system:
            payload["system"] = system
        data = self._request("POST", "/api/generate", payload, model=self.llm_model)
        text = str(data.get("response", "")).strip()
        if not text:
            raise OllamaError("The language model returned an empty answer.")
        return text


# --------------------------------------------------------------------------- #
# Agent 1: Requirement Analysis
# --------------------------------------------------------------------------- #
class RequirementAnalysisAgent(BaseAgent):
    name = "Requirement Analysis Agent"
    responsibility = "Validate the user's event details and produce a clean, structured requirement object."
    inputs = "Raw form fields (event type, date, location, guests, budget, capacity, theme, food, special requirements)."
    outputs = "Normalised requirements dict, or a list of errors/warnings."
    error_handling = "Never raises; every problem becomes a readable error. Any error stops the workflow (rule R1)."

    def run(self, raw: Dict[str, Any], today: Optional[date] = None) -> AgentResult:
        today = today or date.today()
        raw = raw or {}
        errors: List[str] = []
        warnings: List[str] = []
        req: Dict[str, Any] = {}

        missing = [f for f in REQUIRED_FIELDS if raw.get(f) is None or str(raw.get(f)).strip() == ""]
        if missing:
            errors.append("Missing required field(s): " + ", ".join(missing) + ".")

        if "event_type" not in missing:
            event_type = str(raw["event_type"]).strip()
            if event_type in EVENT_TYPES:
                req["event_type"] = event_type
            else:
                errors.append(f"Unsupported event type '{event_type}'. Choose one of: {', '.join(EVENT_TYPES)}.")

        if "event_date" not in missing:
            event_date = parse_date(raw["event_date"])
            if event_date is None:
                errors.append("Event date is not a valid date (use YYYY-MM-DD).")
            elif event_date < today:
                errors.append("Event date is in the past. Please choose today or a future date.")
            else:
                req["event_date"] = event_date.isoformat()
                if (event_date - today).days < 14:
                    warnings.append("The event is less than 14 days away; short lead times make planning harder "
                                    "(general guidance).")

        if "location" not in missing:
            location = str(raw["location"]).strip()
            if len(location) < 2:
                errors.append("Location must contain at least 2 characters.")
            else:
                req["location"] = location

        if "guest_count" not in missing:
            guests = _to_int(raw["guest_count"])
            if guests is None:
                errors.append("Guest count must be a whole number.")
            elif not (MIN_GUESTS <= guests <= MAX_GUESTS):
                errors.append(f"Guest count must be between {MIN_GUESTS} and {MAX_GUESTS}.")
            else:
                req["guest_count"] = guests

        if "total_budget" not in missing:
            ok, amount, message = bu.validate_budget(raw["total_budget"])
            if ok:
                req["total_budget"] = amount
            else:
                errors.append(message)

        if "venue_capacity" not in missing:
            capacity = _to_int(raw["venue_capacity"])
            if capacity is None or capacity < 1:
                errors.append("Venue capacity must be a whole number of at least 1.")
            else:
                req["venue_capacity"] = capacity

        if "food_preference" not in missing:
            food = str(raw["food_preference"]).strip()
            if food in FOOD_PREFERENCES:
                req["food_preference"] = food
            else:
                errors.append(f"Unsupported food preference '{food}'. Choose one of: {', '.join(FOOD_PREFERENCES)}.")

        for key, default in (("theme", "Not specified"), ("special_requirements", "None")):
            text = str(raw.get(key) or "").strip()
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
                warnings.append(f"'{key}' was longer than {MAX_TEXT_LENGTH} characters and was shortened.")
            req[key] = text or default

        if errors:
            return AgentResult(self.name, "Validate event inputs", FAILED,
                               f"{len(errors)} validation error(s) found.",
                               data={"requirements": {}, "missing_fields": missing},
                               errors=errors, warnings=warnings)
        status = WARNING if warnings else SUCCESS
        return AgentResult(self.name, "Validate event inputs", status,
                           "All required fields are valid.",
                           data={"requirements": req, "missing_fields": []}, warnings=warnings)


# --------------------------------------------------------------------------- #
# Agent 2: Knowledge Research (RAG)
# --------------------------------------------------------------------------- #
class KnowledgeBaseError(Exception):
    """Knowledge base file is missing, unreadable or empty."""


STOPWORDS = {"the", "and", "for", "with", "that", "this", "are", "you", "from", "per", "not", "can", "has",
             "have", "will", "any", "all", "its", "our", "your", "but", "into", "than", "then", "also"}


def cosine_similarity(query_vec, matrix) -> np.ndarray:
    """Cosine similarity between one query vector and each row of a matrix."""
    q = np.asarray(query_vec, dtype=float)
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    denom = np.linalg.norm(q) * np.linalg.norm(m, axis=1)
    denom = np.where(denom == 0, 1e-12, denom)
    return (m @ q) / denom


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in STOPWORDS]


def keyword_scores(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    q = set(_tokens(query))
    scores = []
    for c in chunks:
        if not q:
            scores.append(0.0)
            continue
        body, title = set(_tokens(c["text"])), set(_tokens(c["title"]))
        scores.append((len(q & body) + 2 * len(q & title)) / len(q))
    return scores


class KnowledgeResearchAgent(BaseAgent):
    name = "Knowledge Research Agent"
    responsibility = "Retrieve the most relevant knowledge sections using embeddings and cosine similarity (RAG)."
    inputs = "Validated requirements; knowledge_base/event_packages.txt."
    outputs = "Top-ranked knowledge chunks with scores/sources and a context string."
    error_handling = ("Missing/empty file -> warning, no context. Embedding failure -> keyword fallback + warning (rule R2).")

    def __init__(self, client: Any, kb_path: Optional[Path] = None, per_query: int = 2, max_results: int = 6):
        self.client = client
        self.kb_path = Path(kb_path) if kb_path else DEFAULT_KB_PATH
        self.per_query = per_query
        self.max_results = max_results
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._mtime: Optional[float] = None

    # 1. LOAD: split the text file into sections that start with '## '
    def load_chunks(self) -> List[Dict[str, Any]]:
        if not self.kb_path.exists():
            raise KnowledgeBaseError(f"Knowledge base file not found: {self.kb_path.name}")
        try:
            text = self.kb_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise KnowledgeBaseError(f"Could not read the knowledge base: {exc}") from exc
        if not text:
            raise KnowledgeBaseError("The knowledge base file is empty.")
        chunks: List[Dict[str, Any]] = []
        title, lines = "Introduction", []

        def flush() -> None:
            body = "\n".join(lines).strip()
            if body:
                chunks.append({"id": len(chunks) + 1, "title": title, "text": body,
                               "source": f"{self.kb_path.name}#{title}"})

        for line in text.splitlines():
            if line.startswith("## "):
                flush()
                title, lines = line[3:].strip(), []
            else:
                lines.append(line)
        flush()
        if not chunks:
            raise KnowledgeBaseError("The knowledge base has no readable sections.")
        return chunks

    # 2. EMBED: one vector per section (cached until the file changes)
    def build_index(self) -> None:
        chunks = self.load_chunks()
        mtime = self.kb_path.stat().st_mtime
        if self._embeddings is not None and self._mtime == mtime:
            return
        vectors = self.client.embed([f"search_document: {c['title']}\n{c['text']}" for c in chunks])
        matrix = np.array(vectors, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise OllamaError("Embedding output had an unexpected shape.")
        self._chunks, self._embeddings, self._mtime = chunks, matrix, mtime

    def _merge(self, score_lists, chunks, per_query, max_results, method, drop_zero=False):
        best: Dict[int, float] = {}
        for scores in score_lists:
            order = np.argsort(-np.asarray(scores, dtype=float))[:per_query]
            for idx in order:
                idx, s = int(idx), float(scores[int(idx)])
                if drop_zero and s <= 0:
                    continue
                if idx not in best or s > best[idx]:
                    best[idx] = s
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:max_results]
        return [{**chunks[i], "score": round(s, 4), "method": method} for i, s in ranked]

    # 3. SEARCH (semantic): embed queries, rank sections by cosine similarity
    def retrieve(self, queries, per_query: Optional[int] = None, max_results: Optional[int] = None):
        if isinstance(queries, str):
            queries = [queries]
        self.build_index()
        qvecs = self.client.embed([f"search_query: {q}" for q in queries])
        score_lists = [cosine_similarity(v, self._embeddings) for v in qvecs]
        return self._merge(score_lists, self._chunks, per_query or self.per_query,
                           max_results or self.max_results, "semantic")

    def keyword_retrieve(self, queries, per_query: Optional[int] = None, max_results: Optional[int] = None):
        if isinstance(queries, str):
            queries = [queries]
        chunks = self.load_chunks()
        score_lists = [keyword_scores(q, chunks) for q in queries]
        return self._merge(score_lists, chunks, per_query or self.per_query,
                           max_results or self.max_results, "keyword", drop_zero=True)

    @staticmethod
    def build_queries(req: Dict[str, Any]) -> List[str]:
        return [
            f"{req['event_type']} package for {req['guest_count']} guests budget INR {req['total_budget']}",
            f"decoration theme ideas: {req.get('theme', '')} {req['event_type']}",
            f"catering menu options {req['food_preference']} cost per guest",
            f"venue capacity guidelines, budget planning and preparation timeline for {req['event_type']}. "
            f"{req.get('special_requirements', '')}",
        ]

    @staticmethod
    def _context(chunks: List[Dict[str, Any]]) -> str:
        text = "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)
        return text[:6000]

    def run(self, req: Dict[str, Any]) -> AgentResult:
        action = "Retrieve relevant knowledge (RAG)"
        queries = self.build_queries(req)
        try:
            chunks = self.retrieve(queries)
            return AgentResult(self.name, action, SUCCESS,
                               f"Retrieved {len(chunks)} section(s) using {EMBED_MODEL} + cosine similarity.",
                               data={"chunks": chunks, "context": self._context(chunks),
                                     "method": "semantic (embeddings + cosine similarity)", "queries": queries})
        except KnowledgeBaseError as exc:
            return AgentResult(self.name, action, WARNING,
                               "No knowledge base available; planning continues without retrieved context.",
                               data={"chunks": [], "context": "", "method": "none", "queries": queries},
                               warnings=[str(exc)])
        except OllamaError as exc:
            try:
                chunks = self.keyword_retrieve(queries)
            except KnowledgeBaseError as kb_exc:
                return AgentResult(self.name, action, WARNING, "Knowledge retrieval unavailable.",
                                   data={"chunks": [], "context": "", "method": "none", "queries": queries},
                                   warnings=[str(kb_exc)])
            return AgentResult(self.name, action, WARNING,
                               f"Embeddings unavailable; used keyword fallback ({len(chunks)} section(s)).",
                               data={"chunks": chunks, "context": self._context(chunks),
                                     "method": "keyword fallback", "queries": queries},
                               warnings=[f"Embedding step failed: {exc}"])


# --------------------------------------------------------------------------- #
# Agent 3: Budget Planning
# --------------------------------------------------------------------------- #
class BudgetPlanningAgent(BaseAgent):
    name = "Budget Planning Agent"
    responsibility = "Allocate the budget across categories with deterministic maths and detect overruns."
    inputs = "Total budget, guest count, event type, food preference."
    outputs = "Category allocations (sum == total), catering estimate, overrun status, suggestions."
    error_handling = "Invalid budget -> FAILED (rule R3). Overrun -> WARNING and critical flag (rule R4)."

    def run(self, req: Dict[str, Any]) -> AgentResult:
        action = "Allocate budget and check for overruns"
        try:
            analysis = bu.analyze_budget(req["total_budget"], req["guest_count"],
                                         req["event_type"], req["food_preference"])
        except (KeyError, ValueError, TypeError) as exc:
            return AgentResult(self.name, action, FAILED, "Budget could not be calculated.",
                               errors=[f"Budget calculation failed: {exc}"])
        if not analysis["sum_matches_total"]:
            return AgentResult(self.name, action, FAILED, "Allocations do not add up to the total budget.",
                               errors=["Internal budget error: allocations do not equal the total."])
        if analysis["status"] == "OVERRUN":
            return AgentResult(
                self.name, action, WARNING,
                f"Allocations sum to {bu.format_inr(req['total_budget'])}, but catering is short by "
                f"{bu.format_inr(analysis['catering_shortfall'])} (estimate).",
                data=analysis,
                warnings=[f"Budget overrun (estimate): catering needs about "
                          f"{bu.format_inr(analysis['catering_required_estimate'])} but only "
                          f"{bu.format_inr(analysis['allocations']['Catering'])} is allocated."])
        return AgentResult(self.name, action, SUCCESS,
                           f"Allocations sum exactly to {bu.format_inr(req['total_budget'])}; catering estimate is covered.",
                           data=analysis)


# --------------------------------------------------------------------------- #
# Agent 4: Venue and Logistics
# --------------------------------------------------------------------------- #
class VenueLogisticsAgent(BaseAgent):
    name = "Venue and Logistics Agent"
    responsibility = "Compare guests with venue capacity and estimate basic logistics."
    inputs = "Guest count, venue capacity, event date, special requirements."
    outputs = "Capacity decision (OK / crowded / too large / exceeded), logistics estimates, suggestions."
    error_handling = "Guests > capacity -> FAILED with suggestions (critical flag, rule R4)."

    def run(self, req: Dict[str, Any]) -> AgentResult:
        action = "Validate venue capacity and logistics"
        try:
            guests, capacity = int(req["guest_count"]), int(req["venue_capacity"])
        except (KeyError, TypeError, ValueError):
            return AgentResult(self.name, action, FAILED, "Guest count or capacity missing.",
                               errors=["Cannot check venue capacity without guest count and capacity."])
        utilization = guests / capacity
        recommended = int(math.ceil(guests * 1.10))
        logistics = {
            "tables_needed": int(math.ceil(guests / 10)),
            "parking_spaces_estimate": int(math.ceil(guests / 4)),
            "service_staff_estimate": max(2, int(math.ceil(guests / 25))),
            "utilization_percent": round(utilization * 100, 1),
            "recommended_min_capacity": recommended,
        }
        notes: List[str] = []
        event_date = parse_date(req.get("event_date"))
        if event_date and event_date.weekday() >= 5:
            notes.append("The event falls on a weekend, when venues are usually busier; secure the venue early.")
        if "outdoor" in str(req.get("special_requirements", "")).lower():
            notes.append("Outdoor setting requested: arrange a covered or indoor weather backup.")
        suggestions: List[str] = []
        errors: List[str] = []
        warnings: List[str] = []

        if guests > capacity:
            status, capacity_ok = FAILED, False
            errors.append(f"Guest count ({guests}) exceeds venue capacity ({capacity}) by {guests - capacity}.")
            suggestions.append(f"Choose a venue with capacity of at least {recommended} (guests + 10% comfort buffer).")
            suggestions.append(f"Or reduce the guest count to {int(capacity * 0.9)} or fewer for comfortable space.")
            summary = f"REJECTED: {guests} guests exceed capacity {capacity}."
        else:
            capacity_ok = True
            if utilization > 0.90:
                status = WARNING
                warnings.append(f"Venue will be {logistics['utilization_percent']}% full; it may feel crowded.")
                suggestions.append(f"Consider capacity of at least {recommended} for comfort.")
                summary = f"Capacity OK but crowded ({logistics['utilization_percent']}% full)."
            elif utilization < 0.40 and guests >= 20:
                status = WARNING
                warnings.append(f"Venue is much larger than needed ({logistics['utilization_percent']}% full); "
                                "a smaller venue could save budget.")
                suggestions.append("Consider a smaller venue to reduce cost.")
                summary = f"Capacity OK but venue is oversized ({logistics['utilization_percent']}% full)."
            else:
                status = SUCCESS
                summary = f"Capacity OK: {guests} guests in a venue for {capacity} ({logistics['utilization_percent']}% full)."

        return AgentResult(self.name, action, status, summary,
                           data={"capacity_ok": capacity_ok, "guest_count": guests, "venue_capacity": capacity,
                                 "logistics": logistics, "notes": notes, "suggestions": suggestions,
                                 "note": "Logistics numbers are rule-of-thumb estimates."},
                           errors=errors, warnings=warnings)


# --------------------------------------------------------------------------- #
# Agent 5: Schedule Planning
# --------------------------------------------------------------------------- #
# (offset in minutes from guest arrival, activity, notes)
SCHEDULE_TEMPLATES: Dict[str, List[tuple]] = {
    "Wedding": [
        (-240, "Venue set-up and decoration finishing", "Decorators and helpers complete the stage and entrance."),
        (-90, "Sound, lighting and photography team arrival", "Do a quick sound check."),
        (0, "Guest arrival and welcome drinks", "Seat elderly guests first."),
        (45, "Main wedding ceremony and rituals", "Follow the family's traditions."),
        (135, "Couple and family photography session", "Share a shot list with the photographer."),
        (180, "Dinner service", "Menu preference: {food}."),
        (270, "Music, dance and entertainment", "Agree the end time with the venue."),
        (345, "Cake cutting / blessings and farewell", ""),
        (390, "Closing, guest departure and venue handover", "Check belongings and vendor payments."),
    ],
    "Birthday": [
        (-120, "Venue set-up and decoration", "Balloons, backdrop and cake table."),
        (-30, "Final checks with helpers", ""),
        (0, "Guest arrival and welcome", ""),
        (30, "Games and activities", "Suitable for the guests' age group."),
        (90, "Cake cutting and photos", ""),
        (120, "Meal service", "Menu preference: {food}."),
        (180, "Music and dance", ""),
        (225, "Return gifts and closing", ""),
    ],
    "Engagement": [
        (-180, "Venue set-up and stage decoration", ""),
        (-60, "Vendor arrival and sound check", ""),
        (0, "Guest arrival and welcome drinks", ""),
        (40, "Ring ceremony", "Keep rings with a trusted family member."),
        (90, "Photography with family and friends", ""),
        (120, "Dinner service", "Menu preference: {food}."),
        (180, "Music and entertainment", ""),
        (225, "Closing and guest departure", ""),
    ],
    "Anniversary": [
        (-150, "Venue set-up and lighting", ""),
        (-45, "Final checks and sound test", ""),
        (0, "Guest arrival and welcome", ""),
        (40, "Toast and memory slideshow", "Prepare photos/videos in advance."),
        (80, "Cake cutting", ""),
        (105, "Dinner service", "Menu preference: {food}."),
        (165, "Music and entertainment", ""),
        (210, "Closing and farewell", ""),
    ],
    "Naming Ceremony": [
        (-120, "Venue set-up and ritual area preparation", ""),
        (-30, "Final preparation with family", ""),
        (0, "Guest arrival and welcome", ""),
        (30, "Naming ceremony rituals", "Coordinate with the family priest or elders."),
        (75, "Blessings and family photos", ""),
        (105, "Lunch service", "Menu preference: {food}."),
        (165, "Family time and return gifts", ""),
        (210, "Closing and departure", ""),
    ],
    "Corporate Event": [
        (-120, "Venue and audio-visual set-up", "Test projector, microphones and internet."),
        (-30, "Registration desk preparation", ""),
        (0, "Registration and networking tea", ""),
        (30, "Opening address", ""),
        (75, "Main session / presentations", ""),
        (150, "Break and networking", ""),
        (180, "Team activity or awards", ""),
        (240, "Lunch", "Menu preference: {food}."),
        (300, "Closing remarks and feedback", ""),
        (330, "Venue handover", ""),
    ],
    "Other": [
        (-120, "Venue set-up and decoration", ""),
        (-30, "Final checks with helpers", ""),
        (0, "Guest arrival and welcome", ""),
        (45, "Main programme", ""),
        (120, "Meal service", "Menu preference: {food}."),
        (180, "Entertainment and social time", ""),
        (240, "Closing and departure", ""),
    ],
}

DEFAULT_START_TIMES = {
    "Wedding": "17:00", "Birthday": "17:00", "Engagement": "17:30", "Anniversary": "19:00",
    "Naming Ceremony": "11:00", "Corporate Event": "10:00", "Other": "17:00",
}

COMMON_PREP = [
    (90, "Finalise the total budget and confirm it with all organisers"),
    (90, "Prepare a draft guest list and confirm the expected guest count"),
    (75, "Shortlist venues and compare capacity, rules and availability yourself"),
    (60, "Request written quotations from at least three caterers, decorators and photographers"),
    (45, "Finalise theme, menu and decoration plan"),
    (30, "Send invitations and track RSVPs"),
    (21, "Confirm bookings in writing and agree advance payments through your own channels"),
    (14, "Prepare the run-sheet and share the schedule with helpers"),
    (7, "Confirm the final guest count with the caterer and venue"),
    (3, "Reconfirm all vendors and prepare an emergency contact list"),
    (1, "Final venue walk-through and set-up check"),
    (0, "Event day: follow the schedule and assign one coordinator per area"),
]
EXTRA_PREP = {
    "Wedding": [(90, "Fix the auspicious date and ceremony requirements with the family"),
                (45, "Plan outfits, accessories and guest accommodation")],
    "Birthday": [(30, "Plan the cake, games and return gifts")],
    "Engagement": [(45, "Plan outfits, rings and the ceremony programme")],
    "Anniversary": [(30, "Prepare the memory slideshow and any surprise elements")],
    "Naming Ceremony": [(30, "Confirm ritual requirements with the family priest or elders")],
    "Corporate Event": [(45, "Confirm audio-visual equipment, agenda and speaker list"),
                        (14, "Send the agenda and joining instructions to attendees")],
}


class SchedulePlanningAgent(BaseAgent):
    name = "Schedule Planning Agent"
    responsibility = "Create the event-day timeline and a dated preparation task list."
    inputs = "Event type, date, food preference and special requirements."
    outputs = "Timeline rows (Phase, Time, Activity, Notes) and preparation tasks with due dates."
    error_handling = "Invalid date/type -> FAILED. Short lead time -> tasks are compressed and flagged."

    def run(self, req: Dict[str, Any], today: Optional[date] = None) -> AgentResult:
        action = "Generate event timeline and preparation tasks"
        today = today or date.today()
        event_date = parse_date(req.get("event_date"))
        event_type = req.get("event_type")
        if event_date is None or event_type not in SCHEDULE_TEMPLATES:
            return AgentResult(self.name, action, FAILED, "Cannot build a schedule.",
                               errors=["A valid event type and date are required to build a schedule."])

        special = str(req.get("special_requirements", "")).lower() + " " + str(req.get("theme", "")).lower()
        start_text = DEFAULT_START_TIMES[event_type]
        if "morning" in special:
            start_text = "10:00"
        elif "afternoon" in special or "lunch" in special:
            start_text = "12:30"
        elif "evening" in special or "night" in special:
            start_text = "18:30"
        start_dt = datetime.combine(event_date, datetime.strptime(start_text, "%H:%M").time())

        food = req.get("food_preference", "")
        timeline = []
        for offset, activity, notes in SCHEDULE_TEMPLATES[event_type]:
            moment = start_dt + timedelta(minutes=offset)
            timeline.append({
                "Phase": "Preparation" if offset < 0 else "Event",
                "Time": moment.strftime("%H:%M"),
                "Activity": activity,
                "Notes": notes.replace("{food}", food),
                "offset_minutes": offset,
            })

        warnings: List[str] = []
        tasks = []
        compressed = False
        for days, text in sorted(COMMON_PREP + EXTRA_PREP.get(event_type, []), key=lambda t: -t[0]):
            due = event_date - timedelta(days=days)
            if due < today:
                due, text, compressed = today, text + " (ASAP - short lead time)", True
            tasks.append({"task": text, "due_date": due.isoformat(), "days_before": days})
        extra = str(req.get("special_requirements", "None")).strip()
        if extra and extra.lower() != "none":
            due = max(today, event_date - timedelta(days=21))
            tasks.append({"task": f"Plan for special requirement: {extra[:150]}",
                          "due_date": due.isoformat(), "days_before": 21})
        tasks.sort(key=lambda t: t["due_date"])
        if compressed:
            warnings.append("The event is close; some preparation tasks were compressed to start today.")

        return AgentResult(self.name, action, WARNING if warnings else SUCCESS,
                           f"Built {len(timeline)} timeline entries starting at {start_text} and {len(tasks)} preparation tasks.",
                           data={"timeline": timeline, "preparation_tasks": tasks, "start_time": start_text},
                           warnings=warnings)


# --------------------------------------------------------------------------- #
# Agent 6: Event Plan Writer (LLM)
# --------------------------------------------------------------------------- #
class EventPlanWriterAgent(BaseAgent):
    name = "Event Plan Writer Agent"
    responsibility = "Write the personalised event plan with the local LLM, using only the verified facts."
    inputs = "Requirements, retrieved knowledge, budget, venue decision, schedule, constraint issues."
    outputs = "Markdown plan text and its source ('llm' or 'fallback')."
    error_handling = "LLM failure or unusable output -> deterministic fallback plan + warning (rule R5)."

    SYSTEM = ("You are a careful event planning assistant. Use ONLY the facts supplied. Never invent vendor names, "
              "confirmed bookings, availability or exact prices. Call all money figures 'estimated'. "
              "Never change the guest count, date, location or budget.")

    def __init__(self, client: Any):
        self.client = client

    def _build_prompt(self, req, knowledge, budget, venue, schedule, issues) -> str:
        alloc = "\n".join(f"- {c}: {bu.format_inr(a)} (estimate)" for c, a in budget["allocations"].items())
        sched = "\n".join(f"- {r['Time']} {r['Activity']}" for r in schedule.get("timeline", []))
        context = (knowledge or {}).get("context", "") or "No retrieved knowledge available."
        issue_text = "\n".join(f"- {i}" for i in issues) if issues else "None"
        return f"""Write a personalised event plan in Markdown.

EVENT FACTS (do not change):
- Event type: {req['event_type']}
- Date: {req['event_date']}
- Location: {req['location']}
- Guests: {req['guest_count']}
- Total budget: {bu.format_inr(req['total_budget'])}
- Venue capacity: {req['venue_capacity']}
- Theme: {req['theme']}
- Food preference: {req['food_preference']}
- Special requirements: {req['special_requirements']}

BUDGET ALLOCATION (estimates):
{alloc}

VENUE CHECK: {venue.get('capacity_ok') and 'Capacity is sufficient.' or 'CAPACITY EXCEEDED.'} Logistics estimates: {venue.get('logistics', {})}

SCHEDULE (shown separately; refer to it, do not repeat it as a table):
{sched}

RETRIEVED KNOWLEDGE (illustrative estimates, not real vendor offers):
{context}

CONSTRAINT ISSUES TO MENTION:
{issue_text}

Use exactly these Markdown headings:
## 1. Event Overview
## 2. Theme and Decoration
## 3. Catering and Menu
## 4. Activities and Entertainment
## 5. Venue and Logistics
## 6. Budget Notes
## 7. Preparation Timeline
{"## 8. Constraints Needing Attention" if issues else ""}
Be specific and practical. Do not claim anything is booked or confirmed."""

    def _fallback_plan(self, req, knowledge, budget, venue, schedule, issues) -> str:
        titles = [c["title"] for c in (knowledge or {}).get("chunks", [])]
        lines = [
            "## 1. Event Overview",
            f"Event type: {req['event_type']} on {req['event_date']} in {req['location']} for {req['guest_count']} guests. "
            f"Theme: {req['theme']}. Special requirements: {req['special_requirements']}.",
            "", "## 2. Theme and Decoration",
            f"Build the decoration around the theme '{req['theme']}'. Compare light-based and floral options with "
            f"decorators; the estimated decoration budget is {bu.format_inr(budget['allocations']['Decoration'])}.",
            "", "## 3. Catering and Menu",
            f"Menu preference: {req['food_preference']}. Estimated catering cost is about "
            f"{bu.format_inr(budget['catering_required_estimate'])} (about {bu.format_inr(budget['catering_per_guest_estimate'])} per guest) "
            f"against an allocation of {bu.format_inr(budget['allocations']['Catering'])}. Plan food for 5-10% extra guests.",
            "", "## 4. Activities and Entertainment",
            "Activities follow the schedule table: " + "; ".join(
                r["Activity"] for r in schedule.get("timeline", []) if r["Phase"] == "Event") + ".",
            "", "## 5. Venue and Logistics",
            f"Venue capacity is {req['venue_capacity']} for {req['guest_count']} guests "
            f"({venue.get('logistics', {}).get('utilization_percent', 'n/a')}% full). Estimated needs: "
            f"{venue.get('logistics', {}).get('tables_needed', '?')} tables, about "
            f"{venue.get('logistics', {}).get('parking_spaces_estimate', '?')} parking spaces and "
            f"{venue.get('logistics', {}).get('service_staff_estimate', '?')} service staff.",
            "", "## 6. Budget Notes",
            f"Total budget {bu.format_inr(req['total_budget'])}, split as estimates: " + ", ".join(
                f"{c} {bu.format_inr(a)}" for c, a in budget["allocations"].items()) + ".",
            "", "## 7. Preparation Timeline",
        ]
        for t in schedule.get("preparation_tasks", [])[:10]:
            lines.append(f"- {t['due_date']}: {t['task']}")
        if titles:
            lines += ["", "Knowledge sections consulted: " + ", ".join(titles) + "."]
        if issues:
            lines += ["", "## 8. Constraints Needing Attention"] + [f"- {i}" for i in issues]
        return "\n".join(lines)

    def run(self, req, knowledge, budget, venue, schedule, issues: Optional[List[str]] = None) -> AgentResult:
        action = "Write the personalised event plan"
        issues = issues or []
        try:
            prompt = self._build_prompt(req, knowledge, budget, venue, schedule, issues)
        except (KeyError, TypeError, AttributeError) as exc:
            return AgentResult(self.name, action, FAILED, "Missing inputs for plan writing.",
                               errors=[f"Plan writer received incomplete inputs: {exc}"])
        try:
            text = self.client.generate(prompt, system=self.SYSTEM)
            if len(text.strip()) < 300:
                raise OllamaError("The model's answer was too short to be a usable plan.")
            return AgentResult(self.name, action, SUCCESS,
                               f"Generated a {len(text.split())}-word plan with {LLM_MODEL}.",
                               data={"plan_text": text.strip() + DISCLAIMER, "source": "llm"})
        except OllamaError as exc:
            plan = self._fallback_plan(req, knowledge, budget, venue, schedule, issues)
            return AgentResult(self.name, action, WARNING,
                               "LLM unavailable; produced a deterministic fallback plan instead.",
                               data={"plan_text": plan + DISCLAIMER, "source": "fallback"},
                               warnings=[f"Plan generation fell back to a template: {exc}"])


# --------------------------------------------------------------------------- #
# Agent 7: Quality Assurance
# --------------------------------------------------------------------------- #
class QualityAssuranceAgent(BaseAgent):
    name = "Quality Assurance Agent"
    responsibility = "Independently re-verify budget, capacity, schedule and plan text; return PASS/FAIL with reasons."
    inputs = "Requirements, knowledge, budget, venue, schedule data and the plan text."
    outputs = "PASS/FAIL, list of checks (critical or not) with explanations."
    error_handling = "Any failed CRITICAL check -> FAIL. Non-critical failures -> PASS with warnings."

    SECTION_GROUPS = [["overview"], ["decoration", "theme"], ["catering", "menu"],
                      ["logistics", "venue"], ["timeline", "schedule", "preparation"], ["budget"]]
    CLAIM_PATTERN = re.compile(
        r"\b(confirmed booking|booking confirmed|is booked|has been booked|have been booked|"
        r"guaranteed availability|already reserved|payment received)\b", re.I)

    def run(self, requirements, knowledge=None, budget=None, venue=None, schedule=None,
            plan_text: str = "") -> AgentResult:
        action = "Verify constraints and plan quality"
        req = requirements or {}
        budget, venue, schedule = budget or {}, venue or {}, schedule or {}
        plan_text = plan_text or ""
        checks: List[Dict[str, Any]] = []

        def add(name: str, passed: bool, critical: bool, detail: str) -> None:
            checks.append({"check": name, "passed": bool(passed), "critical": critical, "detail": detail})

        # 1. Required fields
        missing = [f for f in REQUIRED_FIELDS if req.get(f) in (None, "")]
        add("Required fields present", not missing, True,
            "All required fields are present." if not missing else f"Missing: {', '.join(missing)}.")

        # 2. Budget validity
        ok, total, msg = bu.validate_budget(req.get("total_budget"))
        add("Total budget is valid", ok, True, msg)

        # 3. Allocation sums to total
        allocs = budget.get("allocations")
        if isinstance(allocs, dict) and ok:
            s = sum(v for v in allocs.values() if isinstance(v, (int, float)))
            add("Allocations sum to total budget", s == total, True,
                f"Allocations add up to {bu.format_inr(s)}; total budget is {bu.format_inr(total)}.")
        else:
            add("Allocations sum to total budget", False, True, "No budget allocation was produced.")

        # 4. Categories complete and non-negative
        if isinstance(allocs, dict):
            good = set(allocs) == set(bu.CATEGORIES) and all(
                isinstance(v, (int, float)) and v >= 0 for v in allocs.values())
            add("Budget categories complete and non-negative", good, True,
                "All six categories are present with non-negative amounts." if good else "Categories missing or negative.")
        else:
            add("Budget categories complete and non-negative", False, True, "No allocation to check.")

        # 5. Venue capacity (independent recomputation + consistency with the venue agent)
        guests, capacity = _to_int(req.get("guest_count")), _to_int(req.get("venue_capacity"))
        if guests is not None and capacity is not None:
            fits = guests <= capacity
            consistent = (venue.get("capacity_ok") == fits) if "capacity_ok" in venue else True
            add("Guest count within venue capacity", fits and consistent, True,
                f"{guests} guests vs capacity {capacity}: " + ("fits." if fits else "EXCEEDS capacity.")
                + ("" if consistent else " The venue agent's result disagrees with this check."))
        else:
            add("Guest count within venue capacity", False, True, "Guest count or capacity is missing.")

        # 6. Budget overrun
        if isinstance(allocs, dict) and guests is not None and "Catering" in allocs and req.get("event_type"):
            needed = bu.estimate_catering_cost(guests, req["event_type"], req.get("food_preference", "Vegetarian"))
            add("Catering budget covers estimated menu cost", allocs["Catering"] >= needed, True,
                f"Estimated catering need {bu.format_inr(needed)} vs allocation {bu.format_inr(allocs['Catering'])}.")
        else:
            add("Catering budget covers estimated menu cost", False, True, "Cannot compute catering need.")

        # 7. Schedule present and chronological
        timeline = schedule.get("timeline") or []
        offsets = [r.get("offset_minutes") for r in timeline]
        chrono = bool(timeline) and all(isinstance(o, int) for o in offsets) and offsets == sorted(offsets)
        add("Schedule exists and is chronological", chrono, False,
            f"{len(timeline)} entries in order." if chrono else "Schedule is empty or out of order.")

        # 8. Plan is non-trivial
        plan_ok = len(plan_text.strip()) >= 300
        add("Plan text generated", plan_ok, True,
            f"Plan has {len(plan_text.split())} words." if plan_ok else "Plan is missing or too short.")

        # 9. Addresses event type
        event_type = str(req.get("event_type", ""))
        keyword = event_type.split()[0].lower() if event_type else ""
        addresses = event_type == "Other" or (keyword != "" and keyword in plan_text.lower())
        add("Plan addresses the event type", addresses, False,
            f"Plan mentions '{keyword}'." if addresses else f"Plan never mentions the event type '{event_type}'.")

        # 10. Expected sections
        lower = plan_text.lower()
        found = sum(1 for group in self.SECTION_GROUPS if any(k in lower for k in group))
        add("Plan has the expected sections", found >= 4, False,
            f"{found} of {len(self.SECTION_GROUPS)} expected topics found.")

        # 11. Unsupported booking/availability claims
        claims = self.CLAIM_PATTERN.findall(plan_text)
        add("No unsupported booking/availability claims", not claims, False,
            "No confirmed-booking language found." if not claims else f"Suspicious phrases: {sorted(set(c.lower() for c in claims))}.")

        # 12. Guest-count contradictions
        if guests is not None:
            allowed = {guests, capacity, int(math.ceil(guests * 1.1))}
            numbers = {int(n.replace(",", "")) for n in re.findall(
                r"(\d[\d,]*)\s*(?:guests|attendees|people)\b(?!\s*per\b)", plan_text, re.I) if n.replace(",", "").isdigit()}
            odd = sorted(numbers - allowed)
            add("No guest-count contradictions", not odd, False,
                "Guest numbers in the plan match the requirements." if not odd
                else f"Plan mentions guest numbers {odd} that differ from {guests}; please verify.")

        # 13. Knowledge used
        used = bool((knowledge or {}).get("chunks"))
        add("Retrieved knowledge was available", used, False,
            "Knowledge sections were retrieved." if used else "No knowledge context was available.")

        critical_fail = [c for c in checks if c["critical"] and not c["passed"]]
        minor_fail = [c for c in checks if not c["critical"] and not c["passed"]]
        result = "FAIL" if critical_fail else "PASS"
        reasons = [f"[CRITICAL] {c['check']}: {c['detail']}" for c in critical_fail] + \
                  [f"[Warning] {c['check']}: {c['detail']}" for c in minor_fail]
        status = FAILED if critical_fail else (WARNING if minor_fail else SUCCESS)
        summary = (f"{result}: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed; "
                   f"{len(critical_fail)} critical failure(s).")
        return AgentResult(self.name, action, status, summary,
                           data={"result": result, "checks": checks, "reasons": reasons})


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class EventPlanningOrchestrator:
    """Calls the seven agents in order and applies the deterministic rules R1-R6."""

    def __init__(self, client: Any = None, kb_path: Optional[Path] = None):
        self.client = client or OllamaClient()
        self.requirement_agent = RequirementAnalysisAgent()
        self.knowledge_agent = KnowledgeResearchAgent(self.client, kb_path)
        self.budget_agent = BudgetPlanningAgent()
        self.venue_agent = VenueLogisticsAgent()
        self.schedule_agent = SchedulePlanningAgent()
        self.writer_agent = EventPlanWriterAgent(self.client)
        self.qa_agent = QualityAssuranceAgent()

    def describe_agents(self) -> List[Dict[str, str]]:
        return [a.describe() for a in (
            self.requirement_agent, self.knowledge_agent, self.budget_agent, self.venue_agent,
            self.schedule_agent, self.writer_agent, self.qa_agent)]

    @staticmethod
    def _execute(step: int, agent: BaseAgent, action: str, fn: Callable[[], AgentResult],
                 trace: List[Dict[str, Any]]) -> AgentResult:
        started = time.perf_counter()
        try:
            result = fn()
            if not isinstance(result, AgentResult) or not isinstance(result.data, dict):
                raise TypeError("agent returned an invalid output object")
        except Exception as exc:  # unexpected bug -> recorded, never crashes the app
            result = AgentResult(agent.name, action, FAILED, f"Unexpected error: {exc}",
                                 errors=[f"Unexpected error in {agent.name}: {exc}"])
        trace.append({
            "step": step, "agent": agent.name, "action": result.action or action, "status": result.status,
            "summary": result.summary, "duration_ms": int((time.perf_counter() - started) * 1000),
            "errors": list(result.errors), "warnings": list(result.warnings),
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        return result

    @staticmethod
    def _skip(trace: List[Dict[str, Any]], step: int, agent: BaseAgent, reason: str) -> None:
        trace.append({"step": step, "agent": agent.name, "action": "Skipped", "status": SKIPPED,
                      "summary": reason, "duration_ms": 0, "errors": [], "warnings": [],
                      "time": datetime.now().strftime("%H:%M:%S")})

    def run(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        errors: List[str] = []
        warnings: List[str] = []
        result: Dict[str, Any] = {
            "validation_status": "INVALID_INPUT", "requirements": {}, "knowledge": None, "budget": None,
            "venue": None, "schedule": None, "plan_text": "", "plan_source": None, "qa": None,
            "trace": trace, "errors": errors, "warnings": warnings,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        def collect(res: AgentResult) -> None:
            errors.extend(res.errors)
            warnings.extend(res.warnings)

        # Step 1 - Requirement Analysis (rule R1)
        req_res = self._execute(1, self.requirement_agent, "Validate event inputs",
                                lambda: self.requirement_agent.run(raw_input), trace)
        collect(req_res)
        if req_res.status == FAILED:
            for i, agent in enumerate([self.knowledge_agent, self.budget_agent, self.venue_agent,
                                       self.schedule_agent, self.writer_agent, self.qa_agent], start=2):
                self._skip(trace, i, agent, "Skipped because requirement validation failed (rule R1).")
            return result
        req = req_res.data["requirements"]
        result["requirements"] = req

        # Step 2 - Knowledge Research (rule R2: failures are non-fatal)
        know_res = self._execute(2, self.knowledge_agent, "Retrieve relevant knowledge (RAG)",
                                 lambda: self.knowledge_agent.run(req), trace)
        collect(know_res)
        knowledge = know_res.data or {"chunks": [], "context": "", "method": "none"}
        result["knowledge"] = knowledge

        # Step 3 - Budget Planning (rule R3)
        bud_res = self._execute(3, self.budget_agent, "Allocate budget and check for overruns",
                                lambda: self.budget_agent.run(req), trace)
        collect(bud_res)
        if bud_res.status == FAILED:
            for i, agent in enumerate([self.venue_agent, self.schedule_agent, self.writer_agent, self.qa_agent], start=4):
                self._skip(trace, i, agent, "Skipped because the budget could not be calculated (rule R3).")
            return result
        budget = bud_res.data
        result["budget"] = budget

        # Step 4 - Venue and Logistics (rule R4)
        ven_res = self._execute(4, self.venue_agent, "Validate venue capacity and logistics",
                                lambda: self.venue_agent.run(req), trace)
        collect(ven_res)
        venue = ven_res.data or {"capacity_ok": False}
        result["venue"] = venue

        # Step 5 - Schedule Planning
        sch_res = self._execute(5, self.schedule_agent, "Generate event timeline and preparation tasks",
                                lambda: self.schedule_agent.run(req), trace)
        collect(sch_res)
        schedule = sch_res.data or {"timeline": [], "preparation_tasks": []}
        result["schedule"] = schedule

        # Step 6 - Plan Writer (rules R4/R5): issues are passed to the writer
        issues = list(ven_res.errors) + (budget.get("suggestions", []) if budget.get("status") == "OVERRUN" else [])
        if budget.get("status") == "OVERRUN":
            issues.insert(0, f"Estimated catering shortfall of {bu.format_inr(budget['catering_shortfall'])}.")
        if ven_res.status == FAILED:
            issues += venue.get("suggestions", [])
        wr_res = self._execute(6, self.writer_agent, "Write the personalised event plan",
                               lambda: self.writer_agent.run(req, knowledge, budget, venue, schedule, issues), trace)
        collect(wr_res)
        result["plan_text"] = wr_res.data.get("plan_text", "")
        result["plan_source"] = wr_res.data.get("source")

        # Step 7 - Quality Assurance (rule R6)
        qa_res = self._execute(7, self.qa_agent, "Verify constraints and plan quality",
                               lambda: self.qa_agent.run(req, knowledge, budget, venue, schedule, result["plan_text"]),
                               trace)
        result["qa"] = qa_res.data or {"result": "FAIL", "checks": [], "reasons": qa_res.errors}

        critical_flag = ven_res.status == FAILED or budget.get("status") == "OVERRUN"
        validated = result["qa"].get("result") == "PASS" and not critical_flag
        result["validation_status"] = "VALIDATED" if validated else "NOT_VALIDATED"
        return result