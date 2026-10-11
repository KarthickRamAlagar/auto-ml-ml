"""Provider presets: pick the most capable available chat model for each LLM provider automatically.

For every provider whose API key is present in the environment we ask its OpenAI-compatible ``GET /models``
endpoint what is actually available to that key, then choose the highest-ranked model from a capability-ordered
preference list. Model catalogues change often, so nothing here is pinned to one exact ID: if discovery fails we
use the provider's known-good default.

Environment (comma-separate to give several keys to one provider):
    GROQ_API_KEY / GROQ_API_KEYS        CEREBRAS_API_KEY(S)     GEMINI_API_KEY(S)
    MISTRAL_API_KEY(S)                  OPENROUTER_API_KEY(S)

Each provider becomes one router group: [best model @ key1, best model @ key2, ..., next-best model @ key1, ...]
(at most 5 entries = 1 primary + 4 backups). Only the model names and scores are ever sent as prompts.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    env: str  # prefix: <env>_API_KEY or <env>_API_KEYS
    prefs: tuple[str, ...]  # regexes, most capable first
    default: str
    exclude: str = r"embed|whisper|tts|guard|moderation|image|imagen|veo|audio|live|vision|rerank|aqa|transcribe"
    only: str | None = None  # id must match (e.g. OpenRouter free tier)


PROVIDERS: tuple[Provider, ...] = (
    Provider("groq", "https://api.groq.com/openai/v1", "GROQ",
             (r"gpt-oss-120b", r"kimi-k2", r"llama-3\.3-70b", r"qwen3-32b", r"llama-4-maverick", r"llama-4-scout",
              r"gpt-oss-20b", r"llama-3\.1-8b"), "llama-3.3-70b-versatile"),
    Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS",
             (r"gpt-oss-120b", r"qwen-3-235b", r"llama-3\.3-70b", r"llama3\.1-8b"), "llama-3.3-70b"),
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI",
             # Flash models first: they have a free quota; the pro models usually do not, and the app only needs short prose.
             (r"^gemini-3(\.\d+)?-flash$", r"^gemini-2\.5-flash$", r"^gemini-2\.0-flash$", r"^gemini-3(\.\d+)?-pro",
              r"^gemini-2\.5-pro"), "gemini-2.5-flash",
             exclude=r"embed|tts|image|imagen|veo|audio|live|vision|aqa|lite|gemma|robotics|computer-use|learnlm"),
    Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL",
             (r"^mistral-large-latest$", r"^mistral-large", r"^mistral-medium-latest$", r"^mistral-medium",
              r"^mistral-small-latest$", r"^mistral-small"), "mistral-small-latest",
             exclude=r"embed|moderation|ocr|codestral|voxtral|pixtral|ministral|saba|devstral"),
    Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER",
             (r"deepseek.*(r1|v3)", r"gpt-oss-120b", r"qwen3.*235b", r"llama-3\.3-70b", r"gemma-3-27b",
              r"mistral-small", r"gpt-oss-20b", r"qwen3", r"glm", r"nemotron", r".*"), "meta-llama/llama-3.3-70b-instruct:free",
             only=r":free$"),
)
BY_NAME = {p.name: p for p in PROVIDERS}


def provider_keys(p: Provider, environ=os.environ) -> list[str]:
    raw = environ.get(f"{p.env}_API_KEYS", "") + "," + environ.get(f"{p.env}_API_KEY", "")
    seen, out = set(), []
    for k in (x.strip() for x in raw.split(",")):
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def rank_models(p: Provider, ids: list[str]) -> list[str]:
    """Available chat models, best first, according to the provider's preference list."""
    clean = [i.split("/", 1)[1] if i.startswith("models/") else i for i in ids]
    ok = [i for i in clean if not re.search(p.exclude, i, re.I) and (not p.only or re.search(p.only, i))]
    ranked: list[tuple[int, str]] = []
    for i in ok:
        for pos, pat in enumerate(p.prefs):
            if re.search(pat, i, re.I):
                ranked.append((pos, i))
                break
    ranked.sort(key=lambda t: (t[0], len(t[1]), t[1]))  # prefer the shorter alias (e.g. "-latest") on ties
    return [i for _, i in ranked]


def discover(p: Provider, key: str, get=httpx.get) -> list[str]:
    r = get(f"{p.base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=6)
    r.raise_for_status()
    data = r.json()
    items = data.get("data", data) if isinstance(data, dict) else data
    return [str(m.get("id") or m.get("name") or "") for m in items if isinstance(m, dict)]


_cache: dict[str, tuple[float, list[str]]] = {}
_lock = threading.Lock()
_TTL = 6 * 3600


def best_models(p: Provider, key: str, get=httpx.get, n: int = 3) -> list[str]:
    """Top ``n`` models for this provider (cached for 6 h); falls back to the provider default."""
    with _lock:
        hit = _cache.get(p.name)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1][:n]
    try:
        ranked = rank_models(p, discover(p, key, get))
    except Exception:  # noqa: BLE001 - discovery is best effort
        ranked = []
    if not ranked:
        ranked = [p.default]
        ttl_ok = False
    else:
        ttl_ok = True
    if ttl_ok:
        with _lock:
            _cache[p.name] = (time.time(), ranked)
    return ranked[:n]


def build_groups(environ=os.environ, get=httpx.get) -> list[dict]:
    groups = []
    for p in PROVIDERS:
        keys = provider_keys(p, environ)
        if not keys:
            continue
        models = best_models(p, keys[0], get)
        specs: list[dict] = []
        for m in models:  # best model on every key first, then the runner-up models
            for k in keys:
                specs.append({"base_url": p.base_url, "model": m, "api_key": k, "provider": p.name})
        groups.append({"group": p.name, "models": specs[:12]})
    return groups


def describe(environ=os.environ, get=httpx.get) -> list[dict]:
    """Safe-to-show summary for the admin page (never includes keys)."""
    out = []
    for p in PROVIDERS:
        keys = provider_keys(p, environ)
        if not keys:
            out.append({"provider": p.name, "configured": False, "keys": 0, "model": None, "runner_up": None})
            continue
        m = best_models(p, keys[0], get)
        out.append({"provider": p.name, "configured": True, "keys": len(keys), "model": m[0],
                    "runner_up": m[1] if len(m) > 1 else None})
    return out
