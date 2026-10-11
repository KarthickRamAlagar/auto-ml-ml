"""Lecture videos per model.

With YOUTUBE_API_KEY set, the best embeddable tutorial videos are looked up through the YouTube Data API
(search.list, ~100 quota units per call, so good results are cached for a week per model and failures for a
minute). Without a key, or if the API fails, curated IDs from videos.json are used, and finally a plain YouTube
search link. When a lookup fails, the reason is reported (a short code for users, details for the admin test).
"""
from __future__ import annotations

import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote_plus

import httpx

from .config import get_settings

_CURATED: dict = json.loads((Path(__file__).parent / "videos.json").read_text(encoding="utf-8"))
_TTL = 7 * 24 * 3600
_FAIL_TTL = 60
_cache: dict[str, tuple[float, list[dict]]] = {}
_fail: dict[str, tuple] = {}
_lock = threading.Lock()
_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Google error reasons -> (short code shown to users, hint shown to the admin)
_HINTS = {
    "keyInvalid": ("key_invalid", "The API key is not valid. Copy it again from Google Cloud Console > Credentials."),
    "API_KEY_INVALID": ("key_invalid", "The API key is not valid. Copy it again from Google Cloud Console > Credentials."),
    "accessNotConfigured": ("api_disabled", "YouTube Data API v3 is not enabled for the project that owns this key. Enable it in Google Cloud Console > APIs & Services > Library."),
    "SERVICE_DISABLED": ("api_disabled", "YouTube Data API v3 is not enabled for the project that owns this key. Enable it in Google Cloud Console > APIs & Services > Library."),
    "quotaExceeded": ("quota", "The daily quota (10,000 units, about 100 searches) is used up. It resets at midnight Pacific time."),
    "rateLimitExceeded": ("quota", "Too many requests at once. Try again in a minute."),
    "ipRefererBlocked": ("key_restricted", "The key is restricted to websites (HTTP referrers). A server key must have no restriction or an IP restriction."),
    "API_KEY_HTTP_REFERRER_BLOCKED": ("key_restricted", "The key is restricted to websites (HTTP referrers). A server key must have no restriction or an IP restriction."),
    "API_KEY_IP_ADDRESS_BLOCKED": ("key_restricted", "The key is restricted to other IP addresses. Remove the restriction or add this server's IP."),
    "API_KEY_SERVICE_BLOCKED": ("key_restricted", "The key is restricted to other APIs. Add YouTube Data API v3 to its allowed APIs."),
    "forbidden": ("forbidden", "Google refused the request. Check the key's API and application restrictions."),
}


class LookupError_(Exception):
    def __init__(self, code: str, hint: str):
        super().__init__(hint)
        self.code, self.hint = code, hint


def _explain(resp: httpx.Response) -> LookupError_:
    reason, msg = "", ""
    try:
        err = resp.json().get("error", {})
        msg = str(err.get("message", ""))[:200]
        reason = (err.get("errors") or [{}])[0].get("reason", "") or next(
            (d.get("reason", "") for d in err.get("details", []) if isinstance(d, dict) and d.get("reason")), "")
    except Exception:  # noqa: BLE001
        pass
    code, hint = _HINTS.get(reason, ("error", f"YouTube returned HTTP {resp.status_code}" + (f": {msg}" if msg else ".")))
    if code == "forbidden" and msg:
        hint = f"{hint} Google said: {msg}"
    return LookupError_(code, hint)


LANGS = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali",
         "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "es": "Spanish", "fr": "French", "de": "German",
         "pt": "Portuguese", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese"}


def _keys(raw: str) -> list[str]:
    """YOUTUBE_API_KEY / YOUTUBE_API_KEYS may hold several comma-separated keys."""
    seen, out = set(), []
    for k in (x.strip() for x in (raw or "").split(",")):
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _search_api(query: str, key: str, limit: int = 3, get=httpx.get, strict: bool = True, lang: str = "en") -> list[dict]:
    """Search with the first working key; a refused, invalid or over-quota key falls through to the next one."""
    keys = _keys(key)
    last: LookupError_ | None = None
    for k in keys:
        try:
            return _search_one(query, k, limit, get, strict, lang)
        except LookupError_ as exc:
            last = exc
    if last:
        raise last
    raise LookupError_("no_key", "No YouTube API key is configured.")


def _search_one(query: str, key: str, limit: int, get, strict: bool, lang: str) -> list[dict]:
    params = {"part": "snippet", "type": "video", "q": query, "maxResults": limit, "videoEmbeddable": "true",
              "relevanceLanguage": lang, "order": "relevance", "key": key}
    if strict:
        params.update({"safeSearch": "strict", "videoDuration": "medium"})
    r = get("https://www.googleapis.com/youtube/v3/search", timeout=8, params=params)
    if r.status_code >= 400:
        raise _explain(r)
    out = []
    for it in r.json().get("items", []):
        vid = (it.get("id") or {}).get("videoId", "")
        sn = it.get("snippet") or {}
        if _ID.match(vid):
            out.append({"id": vid, "title": html.unescape(sn.get("title", ""))[:120], "channel": html.unescape(sn.get("channelTitle", ""))[:60]})
    return out


def diagnose(get=httpx.get) -> dict:
    """Admin test: one real search per configured key, with a plain-language verdict for each."""
    keys = _keys(get_settings().youtube_api_key)
    if not keys:
        return {"ok": False, "code": "no_key", "keys": [], "message": "YOUTUBE_API_KEY is not set in the environment of the running backend. "
                "Export it in the same terminal before starting uvicorn, then restart."}
    rows = []
    for i, k in enumerate(keys, 1):
        try:
            vids = _search_one("decision tree machine learning tutorial", k, 2, get, False, "en")
            rows.append({"key": i, "ok": bool(vids), "code": "ok" if vids else "empty",
                         "message": f"Working. Found {len(vids)} video(s), for example “{vids[0]['title']}”." if vids else "The key works but the search returned nothing."})
        except LookupError_ as exc:
            rows.append({"key": i, "ok": False, "code": exc.code, "message": exc.hint})
        except Exception as exc:  # noqa: BLE001
            rows.append({"key": i, "ok": False, "code": "network", "message": f"Could not reach YouTube ({type(exc).__name__})."})
    good = [r for r in rows if r["ok"]]
    first = good[0] if good else rows[0]
    msg = first["message"] if len(rows) == 1 else (f"{len(good)} of {len(rows)} keys work. " + " ".join(f"Key {r['key']}: {r['message']}" for r in rows if not r["ok"]) + (f" {good[0]['message']}" if good else ""))
    return {"ok": bool(good), "code": first["code"], "keys": rows, "message": msg.strip()}


def _curated(model_id: str) -> list[dict]:
    out = []
    for v in _CURATED.get(model_id, []):
        if isinstance(v, str) and _ID.match(v):
            out.append({"id": v, "title": "", "channel": ""})
        elif isinstance(v, dict) and _ID.match(str(v.get("id", ""))):
            out.append({"id": v["id"], "title": str(v.get("title", ""))[:120], "channel": str(v.get("channel", ""))[:60]})
    return out


def find_lectures(model_id: str, model_name: str, lang: str = "en", get=httpx.get) -> dict:
    """Up to 3 videos for a model in the chosen language: the first is the main one, the other two are 'similar'."""
    lang = lang if lang in LANGS else "en"
    where = "" if lang == "en" else f" in {LANGS[lang]}"
    search_url = "https://www.youtube.com/results?search_query=" + quote_plus(f"{model_name} explained machine learning{where}")
    videos: list[dict] = []
    status, detail = "curated", ""
    key = get_settings().youtube_api_key
    ck = f"{model_id}:{lang}"
    if not key:
        status = "no_key"
    else:
        with _lock:
            hit = _cache.get(ck)
            bad = _fail.get(ck)
        if hit and time.time() - hit[0] < _TTL:
            videos, status = hit[1], "ok"
        elif bad and time.time() - bad[0] < _FAIL_TTL:
            status, detail = bad[1], bad[2] if len(bad) > 2 else ""
        else:
            try:
                q = f"{model_name} machine learning explained tutorial{where}"
                videos = _search_api(q, key, get=get, lang=lang) or _search_api(q, key, get=get, strict=False, lang=lang)
                if videos:
                    status = "ok"
                    with _lock:
                        _cache[ck] = (time.time(), videos)
                else:
                    status = "empty"
            except LookupError_ as exc:
                status, detail = exc.code, exc.hint
            except Exception:  # noqa: BLE001 - network problems fall back to curated IDs
                status = "network"
            if status != "ok":
                with _lock:
                    _fail[ck] = (time.time(), status, detail)
    note = ""
    if not videos and lang == "en":
        videos = _curated(model_id)
    elif not videos:
        videos = _curated(model_id)
        if videos:
            note = f"No {LANGS[lang]} videos could be fetched, so English ones are shown."
    return {"model_id": model_id, "name": model_name, "videos": videos[:3], "video_ids": [v["id"] for v in videos[:3]],
            "search_url": search_url, "lang": lang, "note": note, "reason": "" if status == "ok" else status,
            "status": status if not videos or status == "ok" else "curated", "detail": detail if status != "ok" else ""}


def find_many(specs: list[tuple[str, str]], lang: str = "en", get=httpx.get) -> list[dict]:
    """Look several models up at once so one slow answer does not stack up."""
    if not specs:
        return []
    with ThreadPoolExecutor(max_workers=min(6, len(specs))) as pool:
        return list(pool.map(lambda s: find_lectures(s[0], s[1], lang=lang, get=get), specs))
