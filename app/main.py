"""Auto ML API. Run locally:  uvicorn app.main:app --reload --port 7860"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import quote, urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import datasets as ds_mod
from . import limits, pricing
from . import fastload
from . import methods as methods_mod
from .auth import AuthUser, current_user, require_admin
from .config import get_settings
from .datasets import DatasetError, datasets
from . import eda as eda_mod
from .jobs import job_view, jobs, stream_events
from .llm_router import get_router, narrate
from .models_registry import MODELS, suggest_models
from .notify import notify_admin_payment, notify_admin_reminder
from .pipeline import JobConfig, make_verdict
from .quiz import QuizError, quizzes
from .store import get_store

settings = get_settings()
app = FastAPI(title="Auto ML API", version="1.0.0",
              docs_url=None if settings.is_production else "/docs", redoc_url=None,
              openapi_url=None if settings.is_production else "/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.frontend_origins), allow_credentials=False,
                   allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Authorization", "Content-Type"])

IST = timezone(timedelta(hours=5, minutes=30))
_state_lock = threading.RLock()
_hits: dict[tuple, deque] = defaultdict(deque)
_now = time.time  # patched in tests


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.exception_handler(DatasetError)
async def _dataset_error(_: Request, exc: DatasetError):
    return JSONResponse({"detail": exc.message}, status_code=exc.status)


@app.exception_handler(QuizError)
async def _quiz_error(_: Request, exc: QuizError):
    return JSONResponse({"detail": exc.message}, status_code=exc.status)


def rate_limit(uid: str, name: str, limit: int, per: int = 60) -> None:
    now = _now()
    q = _hits[(uid, name)]
    while q and now - q[0] > per:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Too many requests. Please wait a moment.")
    q.append(now)


def load_user(u: AuthUser) -> dict:
    doc = get_store().get_user(u.uid) or limits.new_user(u.uid, u.email, _now())
    doc["email"] = u.email
    return doc


def usage_for(u: AuthUser) -> dict:
    with _state_lock:
        doc = load_user(u)
        snap = limits.snapshot(doc, _now(), unlimited=u.is_vip)
        get_store().put_user(u.uid, doc)
    return snap


def get_job(job_id: str, u: AuthUser):
    try:
        return jobs.get(job_id, u.uid)
    except KeyError:
        raise HTTPException(404, "Run not found.") from None


def job_quote(job, uid: str) -> dict:
    n = len(job.cfg.models)
    if job.vip:
        return pricing.quote(models=n, free_remaining=n, video_models=0)
    return pricing.quote(models=n, free_remaining=job.free_remaining_before,
                         video_models=n if job.cfg.videos else 0, team=job.cfg.team,
                         quiz_correct=quizzes.score_for_job(uid, job.id), methods=job.cfg.preprocess.methods_count)


# --------------------------------------------------------------------------- basic
@app.get("/api/health")
def health():
    return {"ok": True, "env": settings.app_env, "auth": settings.auth_mode, "kaggle": ds_mod.kaggle_enabled()}


@app.get("/api/me")
def me(u: AuthUser = Depends(current_user)):
    return {"uid": u.uid, "email": u.email, "is_admin": u.is_admin, "is_vip": u.is_vip, "usage": usage_for(u),
            "limits": {"free": pricing.FREE_MODELS, "paid": pricing.PAID_MODELS, "quiz_seconds": 75,
                       "report_price": int(pricing.REPORT_PRICE), "free_reports": pricing.FREE_REPORTS,
                       "free_methods": pricing.FREE_METHODS, "method_price": int(pricing.METHOD_PRICE)}}


@app.get("/api/usage")
def usage(u: AuthUser = Depends(current_user)):
    return usage_for(u)


@app.get("/api/models")
def list_models(task: Literal["classification", "regression"], domain: str = "Other", u: AuthUser = Depends(current_user)):
    return {"models": suggest_models(task, domain)}


class ModelParseIn(BaseModel):
    task: Literal["classification", "regression"]
    text: str = Field(min_length=2, max_length=300)


@app.post("/api/models/parse")
def parse_models(body: ModelParseIn, u: AuthUser = Depends(current_user)):
    """Turn 'random forest, decision tree and an ANN' into model ids. Plain rules first, then the LLM if configured."""
    from . import advisor
    from .llm_router import get_router

    rate_limit(u.uid, "model-parse", 20)
    res = advisor.parse_request(body.text, body.task)
    res["source"] = "rules"
    router = get_router()
    if not res["model_ids"] and router:
        avail = {m.id: m.name for m in MODELS.values() if body.task in m.classes}
        try:
            out, _ = router.complete([
                {"role": "system", "content": "Map the user's text to model ids from this list and answer with ONLY a JSON array of ids. "
                                              "Ignore anything that is not a model name. Ids: " + json.dumps(avail)},
                {"role": "user", "content": body.text}], max_tokens=60)
            ids = [i for i in json.loads(out[out.index("["): out.rindex("]") + 1]) if i in avail]
            if ids:
                res.update(model_ids=list(dict.fromkeys(ids)), source="llm")
        except Exception:  # noqa: BLE001
            pass
    return res


class PlanIn(BaseModel):
    dataset_id: str = Field(max_length=200)
    target: str = Field(max_length=200)
    task: Literal["classification", "regression"]
    model_ids: list[str] = Field(min_length=1, max_length=12)


@app.post("/api/models/plan")
def models_plan(body: PlanIn, u: AuthUser = Depends(current_user)):
    """Recommended settings per chosen model, measured on a small sample of the dataset (never stored)."""
    from . import advisor
    from .llm_router import get_router

    rate_limit(u.uid, "model-plan", 12)
    df = datasets.get(body.dataset_id, u.uid).df
    if body.target not in df.columns:
        raise HTTPException(422, "Target column not found in the dataset.")
    ids = [i for i in dict.fromkeys(body.model_ids) if i in MODELS and body.task in MODELS[i].classes]
    if not ids:
        raise HTTPException(422, "None of these models are available for this task.")
    res = advisor.build_plan(df, body.target, body.task, ids)
    res["summary"] = advisor.summarise(res["facts"], res["plans"], get_router())
    return res


# --------------------------------------------------------------------------- datasets
class SelectIn(BaseModel):
    id: str = Field(max_length=200)


class UrlIn(BaseModel):
    url: str = Field(max_length=1000)


def _register(u: AuthUser, df, name: str, source: str, default_target: str | None = None, ref: str | None = None) -> dict:
    stored = datasets.put(u.uid, name, df, source, default_target, ref=ref)
    prof = ds_mod.profile(df)
    tgt = default_target if default_target in df.columns else df.columns[-1]
    return {"dataset_id": stored.id, "name": name, "source": source, "ref": ref, "profile": prof, "suggested_target": tgt,
            "datetime_columns": [c["name"] for c in prof["columns"] if c["kind"] == "datetime"]}


@app.get("/api/datasets/suggest")
def datasets_suggest(domain: str = "Other", task: str = "", q: str = "", live: bool = True,
                     source: Literal["recommended", "kaggle", "huggingface", "uci"] = "recommended",
                     include_small: bool = False, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "suggest", 30)
    return ds_mod.suggest(domain[:40], task if task in ("classification", "regression") else "", q[:80], live,
                          source=source, include_small=include_small)


@app.post("/api/datasets/select")
def datasets_select(body: SelectIn, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "dataset", 10)
    if body.id.startswith("builtin:"):
        df, item = ds_mod.load_builtin(body.id)
        return _register(u, df, item.name, "builtin", item.default_target, ref=body.id)
    df, name, tgt = ds_mod.load_external(body.id)
    return _register(u, df, name, body.id.split(":", 1)[0], tgt, ref=body.id)


@app.post("/api/datasets/upload")
def datasets_upload(file: UploadFile = File(...), u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "dataset", 10)
    cap = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(cap + 1)
    if len(data) > cap:
        raise DatasetError(f"File is larger than {settings.max_upload_mb} MB.", 413)
    df = fastload.parse_any(data, file.filename or "")
    name = re.sub(r"[^\w .-]", "", (file.filename or "uploaded.csv"))[:60] or "uploaded.csv"
    return _register(u, df, name, "upload")


@app.post("/api/datasets/from-url")
def datasets_from_url(body: UrlIn, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "dataset", 10)
    raw = ds_mod.fetch_bytes(body.url, settings.max_upload_mb * 1024 * 1024)
    df = fastload.parse_any(raw, urlparse(body.url).path)
    return _register(u, df, "linked-dataset", "link", ref=body.url if len(body.url) <= 300 else None)


class EdaColsIn(BaseModel):
    dataset_id: str = Field(max_length=200)
    target: str = Field(max_length=200)
    task: Literal["classification", "regression"]


@app.post("/api/datasets/eda-columns")
def eda_columns(body: EdaColsIn, u: AuthUser = Depends(current_user)):
    """For wide datasets: statistically rank feature columns so the user can choose what to explore."""
    rate_limit(u.uid, "eda-cols", 20)
    df = datasets.get(body.dataset_id, u.uid).df
    if body.target not in df.columns:
        raise HTTPException(422, "Target column not found in the dataset.")
    return eda_mod.recommend_eda_columns(df, body.target, body.task)


class EdaIn(BaseModel):
    dataset_id: str = Field(max_length=200)
    target: str = Field(max_length=200)
    task: Literal["classification", "regression"]
    columns: list[str] | None = Field(default=None, max_length=60)


@app.post("/api/datasets/eda")
def dataset_eda(body: EdaIn, u: AuthUser = Depends(current_user)):
    """EDA before the run, so balancing and feature-selection advice can be given after the user has seen it."""
    rate_limit(u.uid, "eda", 20)
    df = datasets.get(body.dataset_id, u.uid).df
    if body.target not in df.columns:
        raise HTTPException(422, "Target column not found in the dataset.")
    if body.columns and any(c not in df.columns for c in body.columns):
        raise HTTPException(422, "Unknown column in the EDA selection.")
    if len(df) > settings.max_train_rows:
        df = df.sample(settings.max_train_rows, random_state=42)
    return eda_mod.build_eda(df, body.target, body.task, body.columns)


@app.get("/api/datasets/{dataset_id}/session")
def dataset_session(dataset_id: str, u: AuthUser = Depends(current_user)):
    """The same summary /select returns, for a dataset that is still held in memory (second ML type on one dataset)."""
    s = datasets.get(dataset_id, u.uid)
    prof = ds_mod.profile(s.df)
    tgt = s.default_target if s.default_target in s.df.columns else s.df.columns[-1]
    return {"dataset_id": s.id, "name": s.name, "source": s.source, "ref": s.ref, "profile": prof, "suggested_target": tgt,
            "datetime_columns": [c["name"] for c in prof["columns"] if c["kind"] == "datetime"]}


class AdviceIn(BaseModel):
    dataset_id: str = Field(max_length=200)
    target: str = Field(max_length=200)
    task: Literal["classification", "regression"]


@app.post("/api/datasets/method-advice")
def method_advice(body: AdviceIn, u: AuthUser = Depends(current_user)):
    """After EDA: is balancing required, which balancing / feature-selection methods do we recommend?"""
    rate_limit(u.uid, "advice", 30)
    df = datasets.get(body.dataset_id, u.uid).df
    if body.target not in df.columns:
        raise HTTPException(422, "Target column not found in the dataset.")
    out = methods_mod.advise(df, body.target, body.task)
    out["free_methods"] = pricing.FREE_METHODS
    out["method_price"] = int(pricing.METHOD_PRICE)
    return out


# --------------------------------------------------------------------------- pricing preview
class QuoteIn(BaseModel):
    models: int = Field(ge=0, le=12)
    methods: int = Field(0, ge=0, le=10)
    videos: bool = False
    team: bool = False


@app.post("/api/quote")
def quote_preview(body: QuoteIn, u: AuthUser = Depends(current_user)):
    snap = usage_for(u)
    free_left = body.models if u.is_vip else snap["free_remaining"]
    q = pricing.quote(models=body.models, free_remaining=free_left, video_models=0 if u.is_vip else (body.models if body.videos else 0),
                      team=body.team and not u.is_vip,
                      methods=0 if u.is_vip else body.methods)
    q["within_cap"] = u.is_vip or body.models <= snap["total_remaining"]
    q["usage"] = snap
    return q


# --------------------------------------------------------------------------- jobs
_CHAT_ID = re.compile(r"^[a-z0-9]{8,24}$")


@app.post("/api/jobs")
def create_job(cfg: JobConfig, chat_id: str | None = Query(None, pattern=r"^[a-z0-9]{8,24}$"),
               group_id: str | None = Query(None, pattern=r"^[a-z0-9]{8,24}$"),
               parent_id: str | None = Query(None, pattern=r"^[a-z0-9]{8,24}$"), u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "job", 8)
    st = get_store()
    main = None
    if parent_id:
        main = st.get_project(parent_id)
        if not main or main.get("uid") != u.uid or main.get("kind") != "main":
            raise HTTPException(404, "Main chat not found.")
        tasks = [(st.get_project(c) or {}).get("task") for c in main.get("children", [])]
        if len(main.get("children", [])) >= 2 or cfg.task in tasks:
            raise HTTPException(409, "This dataset already has a run of that type. Each dataset can have one classification and one regression run.")
    if cfg.task == "regression" and cfg.preprocess.balancing:
        cfg.preprocess.balancing = []  # balancing is a classification-only idea
    if chat_id and st.get_project(chat_id):
        raise HTTPException(409, "This chat already has a run. Start a new chat for another run.")
    if group_id:
        g = st.get_group(group_id)
        if not g or g["uid"] != u.uid:
            raise HTTPException(404, "Project not found.")
    max_queue = int(os.getenv("MAX_QUEUED_JOBS", "30"))
    if jobs.queued_count() >= max_queue:  # refuse BEFORE any quota is reserved
        raise HTTPException(503, {"message": f"The analysis engine is busy ({max_queue} runs are waiting). Please try again in a few minutes. Nothing was charged."})
    stored = datasets.get(cfg.dataset_id, u.uid)
    df = stored.df
    cols = set(df.columns)
    if cfg.target not in cols:
        raise HTTPException(422, "Target column not found in the dataset.")
    if cfg.time_column and cfg.time_column not in cols:
        raise HTTPException(422, "Time column not found in the dataset.")
    if any(c not in cols for c in cfg.preprocess.drop_columns):
        raise HTTPException(422, "Unknown column in drop list.")
    if cfg.eda_columns and any(c not in cols for c in cfg.eda_columns):
        raise HTTPException(422, "Unknown column in the EDA selection.")
    ids = [m.id for m in cfg.models]
    if len(set(ids)) != len(ids):
        raise HTTPException(422, "Each model can be selected once per run.")
    if any(i not in MODELS or cfg.task not in MODELS[i].classes for i in ids):
        raise HTTPException(422, f"One or more models are not available for {cfg.task}.")
    with _state_lock:
        doc = load_user(u)
        try:
            res = limits.reserve(doc, len(ids), _now(), unlimited=u.is_vip)
        except limits.LimitError as exc:
            raise HTTPException(429, {"message": exc.message, "resets_at": exc.resets_at, "code": exc.code}) from None
        get_store().put_user(u.uid, doc)
    get_store().add_event({"ts": _now(), "uid": u.uid, "domain": cfg.domain, "task": cfg.task, "models": ids,
                           "dataset": stored.name})
    job = jobs.create(u.uid, cfg, stored, res["free_remaining_before"], u.is_vip, df.copy(),
                      reserved=(res["free"], res["paid"]) if not u.is_vip else (0, 0), chat_id=chat_id, group_id=group_id,
                      parent_id=parent_id)
    if main is not None:
        main["children"] = [*main.get("children", []), job.project_id]
        main["updated_at"] = _now()
        get_store().put_project(main["id"], main)
    return {"job_id": job.id, "project_id": job.project_id, "needs_payment": job.needs_payment,
            "paid_models": sorted(job.paid_ids), "paid_methods": job.paid_methods, "free_remaining_before": res["free_remaining_before"],
            "usage": limits.snapshot(doc, _now(), unlimited=u.is_vip)}


@app.get("/api/jobs/{job_id}")
def get_job_view(job_id: str, u: AuthUser = Depends(current_user)):
    return job_view(get_job(job_id, u))


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str, start: int = Query(0, ge=0, alias="from"), u: AuthUser = Depends(current_user)):
    job = get_job(job_id, u)
    return StreamingResponse(stream_events(job, start), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{job_id}/quote")
def job_quote_view(job_id: str, u: AuthUser = Depends(current_user)):
    job = get_job(job_id, u)
    q = job_quote(job, u.uid)
    q["needs_payment"] = job.needs_payment and q["total"] > 0 and not job.unlocked
    return q


@app.get("/api/jobs/{job_id}/lectures")
def job_lectures(job_id: str, lang: str = Query("en", max_length=5), u: AuthUser = Depends(current_user)):
    """Lecture videos for every model of this run in the chosen language (main video first, two similar ones after)."""
    from . import videos as vid

    rate_limit(u.uid, "lectures", 30)
    job = get_job(job_id, u)
    if not job.cfg.videos:
        raise HTTPException(404, "Video lectures were not added to this run.")
    if job.videos_locked:
        raise HTTPException(402, "Video lectures unlock together with your results after payment.")
    lang = lang if lang in vid.LANGS else "en"
    return {"lang": lang, "languages": [{"code": c, "name": n} for c, n in vid.LANGS.items()],
            "lectures": vid.find_many([(s.id, MODELS[s.id].name) for s in job.cfg.models], lang=lang)}


_NARRATIVES: dict[str, dict] = {}  # job id -> AI summary already produced (in memory)


@app.post("/api/jobs/{job_id}/narrate")
def job_narrate(job_id: str, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "narrate", 6)
    job = get_job(job_id, u)
    view = job_view(job)
    if not view["verdict"]:
        raise HTTPException(409, "The run is not finished yet.")
    out = narrate(view["verdict"], job.cfg.domain, job.cfg.task)
    _NARRATIVES[job.id] = out  # the report reuses it instead of asking the AI again
    return out


# --------------------------------------------------------------------------- quiz
class QuizStart(BaseModel):
    job_id: str


class AnswerIn(BaseModel):
    qid: str = Field(max_length=20)
    option_id: str | None = Field(default=None, max_length=1)


@app.post("/api/quiz/start")
def quiz_start(body: QuizStart, u: AuthUser = Depends(current_user)):
    get_job(body.job_id, u)
    s = quizzes.start(u.uid, body.job_id)
    return quizzes.current(s)


@app.get("/api/quiz/{quiz_id}/current")
def quiz_current(quiz_id: str, u: AuthUser = Depends(current_user)):
    return quizzes.current(quizzes.get(quiz_id, u.uid))


@app.post("/api/quiz/{quiz_id}/answer")
def quiz_answer(quiz_id: str, body: AnswerIn, u: AuthUser = Depends(current_user)):
    s = quizzes.get(quiz_id, u.uid)
    out = quizzes.answer(s, body.qid, body.option_id)
    out["score"] = s.score
    return out


# --------------------------------------------------------------------------- payments
def _upi_uri(amount: int, ref: str) -> str:
    q = {"pa": settings.upi_id, "pn": settings.upi_name, "am": f"{amount}.00", "cu": "INR", "tn": ref}
    return "upi://pay?" + "&".join(f"{k}={quote(str(v), safe='@')}" for k, v in q.items())


def _new_ref() -> str:
    return "AML-" + "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5))


def _public_payment(p: dict) -> dict:
    keys = ("id", "ref", "kind", "job_id", "amount", "breakdown", "status", "created_at", "claimed_at", "decided_at",
            "reminded_at", "reminders", "project_id")
    ctx = p.get("context") or {}
    return {**{k: p.get(k) for k in keys}, "upi_uri": _upi_uri(p["amount"], p["ref"]), "payee": settings.upi_name,
            "remind_after": settings.remind_after_seconds, "now": _now(),
            "context": {"domain": ctx.get("domain"), "task": ctx.get("task"), "models": ctx.get("models", [])}}


class PaymentIn(BaseModel):
    kind: Literal["models", "report"]
    job_id: str


@app.post("/api/payments")
def create_payment(body: PaymentIn, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "payment", 10)
    if u.is_vip:
        raise HTTPException(400, "Your account has free access; no payment is needed.")
    job = get_job(body.job_id, u)
    st = get_store()
    with _state_lock:
        existing = next((p for p in st.list_payments(uid=u.uid) if p["uid"] == u.uid and p["job_id"] == job.id
                         and p["kind"] == body.kind and p["status"] in ("created", "claimed", "approved")), None)
        if body.kind == "models":
            if not job.needs_payment:
                raise HTTPException(400, "This run has nothing to pay for.")
            q = job_quote(job, u.uid)
            if q["total"] <= 0:
                raise HTTPException(400, "Nothing to pay.")
            amount, breakdown = q["total"], q
        else:
            doc = load_user(u)
            rq = pricing.report_quote(doc.get("reports_used", 0))
            if not rq["needs_payment"]:
                raise HTTPException(400, "You still have a free report; no payment is needed.")
            amount, breakdown = rq["total"], {"lines": [{"label": "Report download", "amount": rq["total"]}], "total": rq["total"]}
        if existing:
            if existing["status"] == "created" and existing["amount"] != amount:
                existing = st.update_payment(existing["id"], {"amount": amount, "breakdown": breakdown})
            return _public_payment(existing)
        p = {"id": uuid.uuid4().hex[:12], "ref": _new_ref(), "uid": u.uid, "user_email": u.email, "kind": body.kind,
             "job_id": job.id, "project_id": job.project_id, "amount": amount, "breakdown": breakdown,
             "status": "created", "created_at": _now(), "claimed_at": None, "decided_at": None,
             "context": {"domain": job.cfg.domain, "task": job.cfg.task, "models": [m.id for m in job.cfg.models],
                         "project_id": job.project_id}}
        st.create_payment(p)
    return _public_payment(p)


@app.post("/api/payments/{pid}/claim")
def claim_payment(pid: str, background: BackgroundTasks, u: AuthUser = Depends(current_user)):
    st = get_store()
    with _state_lock:
        p = st.get_payment(pid)
        if not p or p["uid"] != u.uid:
            raise HTTPException(404, "Payment not found.")
        if p["status"] == "created":
            p = st.update_payment(pid, {"status": "claimed", "claimed_at": _now()})
            try:
                job = jobs.get(p["job_id"], u.uid)
                if p["kind"] == "models":
                    job.unlocked, job.payment_id, job.rejected = True, pid, False
                else:
                    job.report_granted = True
                jobs.persist(job)
            except KeyError:
                pass
            background.add_task(notify_admin_payment, p, u.email, p.get("context", {}))
        elif p["status"] in ("rejected",):
            raise HTTPException(409, "This payment was rejected.")
    return {"payment": _public_payment(p), "chat_message":
            f"Payment of ₹{p['amount']} (ref {p['ref']}) sent to the admin for verification. Your results are unlocked meanwhile."}


@app.get("/api/payments/mine")
def my_payments(page: int = Query(1, ge=1), per_page: int = Query(7, ge=1, le=50), u: AuthUser = Depends(current_user)):
    """The user's payments, newest first, with the chat each one belongs to (7 per page by default)."""
    st = get_store()
    mine = sorted((p for p in st.list_payments(uid=u.uid)), key=lambda p: -p["created_at"])
    total = len(mine)
    pages = max(1, -(-total // per_page))
    page = min(page, pages)
    out = []
    for p in mine[(page - 1) * per_page: page * per_page]:
        chat = st.get_project(p.get("project_id") or "") if p.get("project_id") else None
        out.append({**_public_payment(p), "chat": ({"id": chat["id"], "title": chat.get("title"), "dataset": chat.get("dataset")}
                                                   if chat and chat.get("uid") == u.uid else None)})
    return {"payments": out, "page": page, "pages": pages, "total": total, "per_page": per_page}


@app.get("/api/payments/{pid}")
def get_payment(pid: str, u: AuthUser = Depends(current_user)):
    p = get_store().get_payment(pid)
    if not p or p["uid"] != u.uid:
        raise HTTPException(404, "Payment not found.")
    return _public_payment(p)


@app.post("/api/payments/{pid}/remind")
def remind_admin(pid: str, background: BackgroundTasks, u: AuthUser = Depends(current_user)):
    """After the wait (7 minutes by default) the user can nudge the admin: an email plus a flag on the admin dashboard."""
    st = get_store()
    with _state_lock:
        p = st.get_payment(pid)
        if not p or p["uid"] != u.uid:
            raise HTTPException(404, "Payment not found.")
        if p["status"] != "claimed":
            raise HTTPException(409, f"This payment is {p['status']}; there is nothing to remind about.")
        now = _now()
        since = p.get("reminded_at") or p.get("claimed_at") or p["created_at"]
        wait = settings.remind_after_seconds
        if now - since < wait:
            left = int(wait - (now - since))
            raise HTTPException(429, {"message": f"You can remind the admin in {left // 60}m {left % 60:02d}s.", "seconds_left": left})
        p = st.update_payment(pid, {"reminded_at": now, "reminders": int(p.get("reminders") or 0) + 1})
        waited = int(now - (p.get("claimed_at") or p["created_at"]))
    background.add_task(notify_admin_reminder, p, u.email, p.get("context", {}), waited)
    return _public_payment(p)


# --------------------------------------------------------------------------- reports
@app.post("/api/reports/{job_id}/access")
def report_access(job_id: str, u: AuthUser = Depends(current_user)):
    job = get_job(job_id, u)
    if job.status != "done":
        raise HTTPException(409, "The run is not finished yet.")
    granted = u.is_vip or job.report_granted
    if not granted:
        with _state_lock:
            doc = load_user(u)
            limits.normalize(doc, _now())
            if pricing.report_quote(doc.get("reports_used", 0))["needs_payment"]:
                paid = any(p["uid"] == u.uid and p["job_id"] == job.id and p["kind"] == "report"
                           and p["status"] in ("claimed", "approved") for p in get_store().list_payments(uid=u.uid))
                if not paid:
                    raise HTTPException(402, {"message": "Report downloads cost ₹5 after your 2 free reports.",
                                              "price": int(pricing.REPORT_PRICE)})
            else:
                doc["reports_used"] = doc.get("reports_used", 0) + 1
                get_store().put_user(u.uid, doc)
            job.report_granted = True
            jobs.persist(job)
    view = job_view(job)
    # Never wait for the AI here: opening a report must be instant. Use the AI summary if one was already made, else the rule-based text.
    verdict = view["verdict"]
    view["narrative"] = (_NARRATIVES.get(job.id) or {"text": " ".join(verdict.get("text", [])), "source": "rules"}) if verdict else None
    view["generated_at"] = _now()
    view["user_email"] = u.email
    return view


# --------------------------------------------------------------------------- chats, projects (groups) & feedback
class ChatPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    group_id: str | None = Field(default=None, max_length=24)
    ungroup: bool = False


def _own_chat(cid: str, u: AuthUser) -> dict:
    p = get_store().get_project(cid)
    if not p or p.get("uid") != u.uid:
        raise HTTPException(404, "Chat not found.")
    return p


def _public_chat(p: dict) -> dict:
    out = {k: v for k, v in p.items() if k not in ("uid", "config", "reserved", "refunded", "dataset_id")}
    days = settings.retention_days
    if days > 0:
        out["expires_at"] = (p.get("updated_at") or p.get("created_at") or _now()) + days * 86400
    return out


@app.get("/api/chats")
def list_chats(u: AuthUser = Depends(current_user)):
    items = [jobs.reconcile(p) for p in get_store().list_projects(u.uid)]
    items.sort(key=lambda p: -(p.get("updated_at") or p.get("created_at") or 0))
    return {"chats": [_public_chat(p) for p in items[:300]]}


@app.get("/api/projects")  # backwards-compatible alias
def list_projects(u: AuthUser = Depends(current_user)):
    return {"projects": list_chats(u)["chats"]}


@app.get("/api/chats/{cid}")
def get_chat(cid: str, u: AuthUser = Depends(current_user)):
    return _public_chat(jobs.reconcile(_own_chat(cid, u)))


@app.get("/api/chats/{cid}/rerun")
def chat_rerun(cid: str, u: AuthUser = Depends(current_user)):
    """Everything needed to repeat a run except the data itself, which is never stored."""
    p = _own_chat(cid, u)
    cfg = dict(p.get("config") or {})
    if not cfg:
        raise HTTPException(404, "This chat has no saved settings to repeat.")
    cfg.pop("dataset_id", None)
    return {"config": cfg, "dataset": {"name": p.get("dataset"), "ref": p.get("dataset_ref"), "source": p.get("dataset_source") or ""},
            "title": p.get("title"), "group_id": p.get("group_id")}


def _held_dataset(p: dict, u: AuthUser) -> str | None:
    """The run's dataset id if the dataset is still in memory (it is wiped DATASET_TTL_SECONDS after the last run)."""
    did = p.get("dataset_id")
    if not did:
        return None
    try:
        datasets.keep(datasets.get(did, u.uid).id)
        return did
    except DatasetError:
        return None


@app.post("/api/chats/{cid}/another")
def chat_another_type(cid: str, u: AuthUser = Depends(current_user)):
    """The user wants the other ML type (classification <-> regression) on the same dataset. The finished run
    becomes a sub-chat of a new main chat; the next run is created under that main chat (create_job parent_id)."""
    rate_limit(u.uid, "chat-edit", 60)
    p = _own_chat(cid, u)
    st = get_store()
    if p.get("parent_id"):
        main = st.get_project(p["parent_id"])
        if main and main.get("uid") == u.uid:
            return {"main_id": main["id"], "other_task": "regression" if p.get("task") == "classification" else "classification",
                    "dataset_id": _held_dataset(p, u)}
    if p.get("kind") == "main":
        raise HTTPException(409, "This is already a main chat.")
    if p.get("status") != "done":
        raise HTTPException(409, "Finish this run first.")
    mid = uuid.uuid4().hex[:12]
    now = _now()
    st.put_project(mid, {"id": mid, "uid": u.uid, "kind": "main", "title": p.get("title") or p.get("dataset") or "Dataset",
                         "dataset": p.get("dataset"), "domain": p.get("domain"), "children": [cid], "status": "done",
                         "created_at": now, "updated_at": now, "group_id": p.get("group_id"), "dataset_ref": p.get("dataset_ref"),
                         "dataset_source": p.get("dataset_source"), "task": "both"})
    p["parent_id"] = mid
    st.put_project(cid, p)
    return {"main_id": mid, "other_task": "regression" if p.get("task") == "classification" else "classification",
            "dataset_id": _held_dataset(p, u)}


@app.get("/api/chats/{cid}/runs")
def chat_runs(cid: str, u: AuthUser = Depends(current_user)):
    """A main chat's sub-chats with their job ids, in the order they were run."""
    p = _own_chat(cid, u)
    if p.get("kind") != "main":
        raise HTTPException(404, "Not a main chat.")
    runs = []
    for c in p.get("children", []):
        sub = get_store().get_project(c)
        if sub and sub.get("uid") == u.uid:
            runs.append(_public_chat(jobs.reconcile(sub)))
    return {"main": _public_chat(p), "runs": runs}


@app.patch("/api/chats/{cid}")
def patch_chat(cid: str, body: ChatPatch, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "chat-edit", 60)
    p = _own_chat(cid, u)
    if body.title is not None:
        p["title"] = body.title.strip()
    if body.ungroup:
        p["group_id"] = None
    elif body.group_id:
        g = get_store().get_group(body.group_id)
        if not g or g["uid"] != u.uid:
            raise HTTPException(404, "Project not found.")
        p["group_id"] = body.group_id
    p["updated_at"] = _now()
    get_store().put_project(cid, p)
    return _public_chat(p)


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str, u: AuthUser = Depends(current_user)):
    p = _own_chat(cid, u)
    if p.get("kind") == "main":
        for c in p.get("children", []):
            sub = get_store().get_project(c)
            if sub and sub.get("uid") == u.uid:
                if sub.get("job_id"):
                    jobs.delete(sub["job_id"])
                get_store().delete_project(c)
    elif p.get("parent_id"):
        main = get_store().get_project(p["parent_id"])
        if main and main.get("uid") == u.uid:
            main["children"] = [c for c in main.get("children", []) if c != cid]
            get_store().put_project(main["id"], main)
    if p.get("job_id"):
        jobs.delete(p["job_id"])
    get_store().delete_project(cid)
    return {"ok": True}


class GroupIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)


def _own_group(gid: str, u: AuthUser) -> dict:
    g = get_store().get_group(gid)
    if not g or g.get("uid") != u.uid:
        raise HTTPException(404, "Project not found.")
    return g


@app.get("/api/groups")
def list_groups(u: AuthUser = Depends(current_user)):
    st = get_store()
    chats = st.list_projects(u.uid)
    out = []
    for g in st.list_groups(u.uid):
        mine = [c for c in chats if c.get("group_id") == g["id"]]
        out.append({"id": g["id"], "name": g["name"], "created_at": g["created_at"], "chat_count": len(mine),
                    "domain": mine[0].get("domain") if mine else None, "task": mine[0].get("task") if mine else None})
    return {"groups": out}


@app.post("/api/groups")
def create_group(body: GroupIn, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "group", 20)
    st = get_store()
    if len(st.list_groups(u.uid)) >= 50:
        raise HTTPException(409, "You can have up to 50 projects.")
    g = {"id": uuid.uuid4().hex[:12], "uid": u.uid, "name": body.name.strip(), "created_at": _now()}
    st.put_group(g["id"], g)
    return {"id": g["id"], "name": g["name"], "created_at": g["created_at"], "chat_count": 0, "domain": None, "task": None}


@app.patch("/api/groups/{gid}")
def rename_group(gid: str, body: GroupIn, u: AuthUser = Depends(current_user)):
    g = _own_group(gid, u)
    g["name"] = body.name.strip()
    get_store().put_group(gid, g)
    return {"id": gid, "name": g["name"]}


@app.delete("/api/groups/{gid}")
def delete_group(gid: str, u: AuthUser = Depends(current_user)):
    _own_group(gid, u)
    st = get_store()
    for c in st.list_projects(u.uid):  # chats are kept, just moved out of the project
        if c.get("group_id") == gid:
            c["group_id"] = None
            st.put_project(c["id"], c)
    st.delete_group(gid)
    return {"ok": True}


class FeedbackIn(BaseModel):
    rating: int = Field(ge=1, le=5)
    text: str = Field("", max_length=1000)
    job_id: str | None = Field(default=None, max_length=32)


@app.post("/api/feedback")
def feedback(body: FeedbackIn, u: AuthUser = Depends(current_user)):
    rate_limit(u.uid, "feedback", 5)
    get_store().add_feedback({"ts": _now(), "uid": u.uid, "email": u.email, "rating": body.rating,
                              "text": body.text.strip(), "job_id": body.job_id})
    return {"ok": True}


# --------------------------------------------------------------------------- admin
def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).date().isoformat()


@app.get("/api/admin/summary")
def admin_summary(_: AuthUser = Depends(require_admin)):
    st = get_store()
    payments, users, events = st.list_payments(), st.list_users(), st.list_events()
    approved = [p for p in payments if p["status"] == "approved"]
    claimed = [p for p in payments if p["status"] == "claimed"]
    today = datetime.now(IST).date()
    days = [(today - timedelta(days=i)).isoformat() for i in range(13, -1, -1)]
    rev, jobs_by_day, signups = Counter(), Counter(), Counter()
    for p in approved:
        rev[_day(p.get("decided_at") or p["created_at"])] += p["amount"]
    for e in events:
        jobs_by_day[_day(e["ts"])] += 1
    for usr in users:
        signups[_day(usr["created_at"])] += 1
    dom = Counter(e.get("domain", "Other") for e in events)
    mod = Counter(m for e in events for m in e.get("models", []))
    fb = st.list_feedback(200)
    return {
        "totals": {
            "users": len(users), "paying_users": len({p["uid"] for p in approved}),
            "revenue": sum(p["amount"] for p in approved), "pending_amount": sum(p["amount"] for p in claimed),
            "pending_count": len(claimed), "runs": len(events),
            "models_trained": sum(len(e.get("models", [])) for e in events),
            "avg_rating": round(sum(f["rating"] for f in fb) / len(fb), 2) if fb else None,
        },
        "revenue_by_day": [{"date": d, "amount": rev[d]} for d in days],
        "runs_by_day": [{"date": d, "count": jobs_by_day[d]} for d in days],
        "signups_by_day": [{"date": d, "count": signups[d]} for d in days],
        "domains": [{"name": k, "count": v} for k, v in dom.most_common(8)],
        "models": [{"id": k, "name": MODELS[k].name if k in MODELS else k, "count": v} for k, v in mod.most_common(10)],
    }


@app.get("/api/admin/payments")
def admin_payments(status: str | None = None, _: AuthUser = Depends(require_admin)):
    items = get_store().list_payments(status if status in ("created", "claimed", "approved", "rejected") else None)
    order = {"claimed": 0, "created": 1, "approved": 2, "rejected": 3}
    items.sort(key=lambda p: (order.get(p["status"], 9), -p["created_at"]))
    return {"payments": [{**{k: v for k, v in p.items() if k != "uid"}, "upi_uri": None} for p in items[:300]]}


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]


@app.post("/api/admin/payments/{pid}/decision")
def admin_decision(pid: str, body: DecisionIn, _: AuthUser = Depends(require_admin)):
    st = get_store()
    with _state_lock:
        p = st.get_payment(pid)
        if not p:
            raise HTTPException(404, "Payment not found.")
        if p["status"] not in ("created", "claimed"):
            raise HTTPException(409, f"Payment is already {p['status']}.")
        if body.decision == "approve":
            p = st.update_payment(pid, {"status": "approved", "decided_at": _now()})
            doc = st.get_user(p["uid"])
            if doc:
                doc["verified_spend"] = doc.get("verified_spend", 0) + p["amount"]
                st.put_user(p["uid"], doc)
        else:
            p = st.update_payment(pid, {"status": "rejected", "decided_at": _now()})
            doc = st.get_user(p["uid"])
            if doc:
                doc["paid_blocked"] = True
                st.put_user(p["uid"], doc)
            try:
                job = jobs.get(p["job_id"], p["uid"])
                if p["kind"] == "models":
                    job.unlocked, job.rejected = False, True
                else:
                    job.report_granted = False
                jobs.persist(job)
            except KeyError:
                pass
    return {"payment": {k: v for k, v in p.items() if k != "uid"}}


@app.get("/api/admin/users")
def admin_users(_: AuthUser = Depends(require_admin)):
    users = get_store().list_users()
    users.sort(key=lambda x: -x.get("verified_spend", 0))
    keep = ("email", "created_at", "verified_spend", "free_used", "paid_used", "reports_used", "paid_blocked")
    return {"users": [{k: x.get(k) for k in keep} for x in users[:300]]}


class VipIn(BaseModel):
    email: str = Field(max_length=200)


_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.get("/api/admin/vip")
def vip_list(_: AuthUser = Depends(require_admin)):
    return {"emails": sorted(get_store().get_vip()), "admins": sorted(settings.admin_emails)}


@app.post("/api/admin/vip")
def vip_add(body: VipIn, _: AuthUser = Depends(require_admin)):
    email = body.email.strip().lower()
    if not _EMAIL.match(email):
        raise HTTPException(422, "Enter a valid email address.")
    get_store().add_vip(email)
    return {"emails": sorted(get_store().get_vip())}


@app.delete("/api/admin/vip")
def vip_remove(email: str, _: AuthUser = Depends(require_admin)):
    get_store().remove_vip(email.strip().lower())
    return {"emails": sorted(get_store().get_vip())}


@app.get("/api/admin/feedback")
def admin_feedback(_: AuthUser = Depends(require_admin)):
    return {"feedback": get_store().list_feedback(100)}


@app.get("/api/admin/projects/{pid}")
def admin_project(pid: str, _: AuthUser = Depends(require_admin)):
    p = get_store().get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found.")
    return {k: v for k, v in p.items() if k not in ("uid", "config")}


# --------------------------------------------------------------------------- admin: LLM providers
@app.get("/api/admin/llm")
def admin_llm(_: AuthUser = Depends(require_admin)):
    from .llm_providers import describe

    return {"providers": describe(), "explicit_groups": bool(settings.llm_groups_json),
            "integrations": {"kaggle": ds_mod.kaggle_enabled(), "youtube": bool(settings.youtube_api_key),
                             "email": bool(settings.resend_api_key and settings.admin_notify_email),
                             "firestore": settings.store_backend == "firestore", "auth": settings.auth_mode}}


@app.get("/api/admin/notifications")
def admin_notifications(_: AuthUser = Depends(require_admin)):
    """Payments waiting for a decision, most urgent first (reminded ones on top). The dashboard polls this."""
    now = _now()
    pending = [p for p in get_store().list_payments("claimed")]
    items = [{"id": p["id"], "ref": p["ref"], "user_email": p.get("user_email"), "amount": p["amount"], "kind": p["kind"],
              "claimed_at": p.get("claimed_at"), "reminded_at": p.get("reminded_at"), "reminders": int(p.get("reminders") or 0),
              "waiting_seconds": int(now - (p.get("claimed_at") or p["created_at"])),
              "domain": (p.get("context") or {}).get("domain")} for p in pending]
    items.sort(key=lambda i: (-(i["reminders"] > 0), -(i["reminded_at"] or i["claimed_at"] or 0)))
    return {"items": items, "count": len(items), "reminders": sum(1 for i in items if i["reminders"]), "now": now}


@app.post("/api/admin/youtube/test")
def admin_youtube_test(_: AuthUser = Depends(require_admin)):
    from .videos import diagnose

    return diagnose()


@app.post("/api/admin/llm/test")
def admin_llm_test(_: AuthUser = Depends(require_admin)):
    """Calls every configured key once and reports each result, so one bad key is easy to spot."""
    from .llm_router import get_router, probe

    router = get_router()
    if not router:
        raise HTTPException(409, "No LLM provider is configured.")
    results = probe(router)
    ok = [r for r in results if r["ok"]]
    return {"ok": bool(ok), "working": len(ok), "total": len(results), "results": results,
            "message": (f"{len(ok)} of {len(results)} key/model combinations work. Failed ones are skipped automatically and the next key is used."
                        if ok else "Every key failed. See the reasons below.")}


@app.on_event("startup")
def _warm() -> None:
    """Resolve the best model per LLM provider in the background so the first request is not slow."""
    from .llm_router import get_router

    threading.Thread(target=get_router, daemon=True).start()
    threading.Thread(target=lambda: _safe(ds_mod._uci_index), daemon=True).start()  # builds the UCI catalogue once
    threading.Thread(target=_retention_loop, daemon=True).start()


def _safe(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def _retention_loop() -> None:
    while True:
        _safe(jobs.sweep_retention)
        time.sleep(6 * 3600)
