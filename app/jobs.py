"""Background training jobs, event streaming and entitlement-based redaction.

A job computes ALL selected models. Results for models above the user's free allowance (and the video
lectures) are withheld by the server until a payment has been claimed, so locking is not a UI trick.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import zlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

from . import pricing
from .config import get_settings
from .datasets import datasets
from .models_registry import MODELS
from .pipeline import JobConfig, PipelineError, make_verdict, run_pipeline
from .store import get_store

log = logging.getLogger("automl.jobs")
# Runs in parallel = JOB_WORKERS (set it to the container's vCPU count: 2 for standard-3). Extra runs wait in a queue.
JOB_WORKERS = max(1, int(os.getenv("JOB_WORKERS", "2")))
_executor = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="automl-job")


@dataclass
class Job:
    id: str
    uid: str
    cfg: JobConfig
    dataset_name: str
    paid_ids: set[str]
    free_remaining_before: int
    vip: bool
    project_id: str
    status: str = "queued"  # queued | running | done | error
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    results: dict | None = None
    created: float = field(default_factory=time.time)
    finished_at: float | None = None
    unlocked: bool = False  # a payment was claimed/approved for this job
    payment_id: str | None = None
    rejected: bool = False
    report_granted: bool = False
    reserved: tuple[int, int] = (0, 0)  # (free, paid) model runs reserved from the user's window
    group_id: str | None = None
    dataset_ref: str | None = None  # how to load the same dataset again (builtin:…, kaggle:…, uci:…, hf:…); None for uploads
    dataset_source: str = ""
    parent_id: str | None = None  # main chat this run is a sub-chat of
    dataset_id: str | None = None  # in-memory dataset id (only valid while the dataset is still held)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- entitlements
    @property
    def needs_payment(self) -> bool:
        return not self.vip and (bool(self.paid_ids) or self.cfg.videos or self.paid_methods > 0)

    @property
    def paid_methods(self) -> int:
        return max(0, self.cfg.preprocess.methods_count - pricing.FREE_METHODS)

    @property
    def methods_locked(self) -> bool:
        return self.paid_methods > 0 and not self.vip and not self.unlocked

    def model_locked(self, mid: str) -> bool:
        return not self.vip and mid in self.paid_ids and not self.unlocked

    @property
    def videos_locked(self) -> bool:
        return self.cfg.videos and not self.vip and not self.unlocked

    def add_event(self, ev: dict) -> None:
        with self.lock:
            ev = {**ev, "seq": len(self.events), "ts": time.time()}
            self.events.append(ev)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        ttl = get_settings().job_ttl_seconds
        now = time.time()
        for k in [k for k, j in self._jobs.items() if j.finished_at and now - j.finished_at > ttl]:
            del self._jobs[k]

    def create(self, uid: str, cfg: JobConfig, ds, free_remaining_before: int, vip: bool, df,
               reserved: tuple[int, int] = (0, 0), chat_id: str | None = None, group_id: str | None = None,
               parent_id: str | None = None) -> Job:
        paid_ids = {m.id for m in cfg.models[max(0, free_remaining_before):]} if not vip else set()
        job = Job(id=uuid.uuid4().hex[:16], uid=uid, cfg=cfg, dataset_name=ds.name, paid_ids=paid_ids,
                  free_remaining_before=free_remaining_before, vip=vip, project_id=chat_id or uuid.uuid4().hex[:12],
                  reserved=reserved, group_id=group_id, parent_id=parent_id, dataset_id=ds.id,
                  dataset_ref=getattr(ds, "ref", None), dataset_source=getattr(ds, "source", ""))
        with self._lock:
            self._sweep()
            self._jobs[job.id] = job
        self._save_project(job)  # the chat exists (status "queued") from the moment the run starts, so a reload can reattach
        _executor.submit(self._run, job, df, ds.id)
        return job

    def queued_count(self) -> int:
        """Runs accepted but not started yet (used to refuse new work when the engine is swamped)."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued")

    def queue_ahead(self, job: Job) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued" and j.created < job.created) if job.status == "queued" else 0

    def live(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def get(self, job_id: str, uid: str, allow_admin: bool = False) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:  # finished runs live in the store as an archive (never the dataset)
            job = self._load_archive(job_id)
            if job is not None:
                with self._lock:
                    job = self._jobs.setdefault(job_id, job)
        if not job or (job.uid != uid and not allow_admin):
            raise KeyError(job_id)
        return job

    # ---- archive (results survive restarts and the in-memory TTL)
    def persist(self, job: Job) -> None:
        """Save everything needed to show this run again, including entitlement flags."""
        if job.status not in ("done", "error"):
            return
        try:
            payload = {
                "id": job.id, "uid": job.uid, "cfg": job.cfg.model_dump(), "dataset_name": job.dataset_name,
                "paid_ids": sorted(job.paid_ids), "free_remaining_before": job.free_remaining_before, "vip": job.vip,
                "project_id": job.project_id, "status": job.status, "error": job.error, "created": job.created,
                "finished_at": job.finished_at, "unlocked": job.unlocked, "payment_id": job.payment_id,
                "rejected": job.rejected, "report_granted": job.report_granted, "reserved": list(job.reserved),
                "dataset_ref": job.dataset_ref, "dataset_source": job.dataset_source, "parent_id": job.parent_id,
                "results": job.results, "events": _compact_events(job.events),
            }
            blob = zlib.compress(json.dumps(payload, default=str).encode("utf-8"), 6)
            get_store().put_archive(job.id, {"uid": job.uid, "project_id": job.project_id, "blob": blob,
                                             "saved_at": time.time()})
        except Exception:  # noqa: BLE001 - history is best effort; the live job is unaffected
            log.exception("Could not save the results archive for job %s", job.id)

    def _load_archive(self, job_id: str) -> Job | None:
        try:
            rec = get_store().get_archive(job_id)
            if not rec:
                return None
            d = json.loads(zlib.decompress(bytes(rec["blob"])).decode("utf-8"))
            job = Job(id=d["id"], uid=d["uid"], cfg=JobConfig(**d["cfg"]), dataset_name=d["dataset_name"],
                      paid_ids=set(d["paid_ids"]), free_remaining_before=d["free_remaining_before"], vip=d["vip"],
                      project_id=d["project_id"], status=d["status"], error=d["error"], events=d["events"],
                      results=d["results"], created=d["created"], finished_at=time.time(), unlocked=d["unlocked"],
                      payment_id=d["payment_id"], rejected=d["rejected"], report_granted=d["report_granted"],
                      reserved=tuple(d["reserved"]), dataset_ref=d.get("dataset_ref"),
                      dataset_source=d.get("dataset_source", ""), parent_id=d.get("parent_id"))  # type: ignore[arg-type]
            return job
        except Exception:  # noqa: BLE001
            log.exception("Could not read the results archive for job %s", job_id)
            return None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
        try:
            get_store().delete_archive(job_id)
        except Exception:  # noqa: BLE001
            pass

    # ---- worker
    def _run(self, job: Job, df, dataset_id: str) -> None:
        s = get_settings()
        job.status = "running"
        job.add_event({"type": "job_start"})
        try:
            job.results = run_pipeline(df, job.cfg, job.dataset_name, job.add_event, s.max_train_rows, s.max_job_seconds)
            job.status = "done"
        except PipelineError as exc:
            job.status, job.error = "error", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "error", f"Unexpected error: {str(exc)[:200]}"
        finally:
            datasets.keep(dataset_id)  # held in memory only (never stored) for DATASET_TTL so the other ML type can reuse it
            del df
            job.finished_at = time.time()
            if job.status == "error" and job.results is None:
                self._refund(job)  # nothing was trained, so give the reserved quota back
            job.add_event({"type": "job_end", "status": job.status, "error": job.error})
            self._save_project(job)
            self.persist(job)

    def reconcile(self, project: dict) -> dict:
        """A run that was queued or running when the server restarted can never finish: mark it interrupted and
        give the reserved allowance back (once)."""
        if project.get("status") in ("queued", "running") and not self.live(project.get("job_id", "")):
            project = {**project, "status": "interrupted", "updated_at": time.time()}
            free, paid = (project.get("reserved") or [0, 0])
            if (free or paid) and not project.get("refunded"):
                self._refund_amounts(project["uid"], int(free), int(paid))
                project["refunded"] = True
            try:
                get_store().put_project(project["id"], project)
            except Exception:  # noqa: BLE001
                log.exception("Could not mark chat %s as interrupted", project.get("id"))
        return project

    def _refund_amounts(self, uid: str, free: int, paid: int) -> None:
        try:
            st = get_store()
            doc = st.get_user(uid)
            if doc:
                doc["free_used"] = max(0, doc.get("free_used", 0) - free)
                doc["paid_used"] = max(0, doc.get("paid_used", 0) - paid)
                st.put_user(uid, doc)
        except Exception:  # noqa: BLE001
            log.exception("Refund failed for %s", uid)

    def sweep_retention(self) -> int:
        """Delete chats (and their saved results) that were last touched more than RETENTION_DAYS ago."""
        days = get_settings().retention_days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        n = 0
        try:
            for p in get_store().list_projects_before(cutoff):
                if p.get("status") in ("queued", "running") and self.live(p.get("job_id", "")):
                    continue
                if p.get("job_id"):
                    self.delete(p["job_id"])
                get_store().delete_project(p["id"])
                n += 1
        except Exception:  # noqa: BLE001
            log.exception("Retention sweep failed")
        return n

    def _refund(self, job: Job) -> None:
        free, paid = job.reserved
        if job.vip or not (free or paid):
            return
        try:
            st = get_store()
            doc = st.get_user(job.uid)
            if doc:
                doc["free_used"] = max(0, doc.get("free_used", 0) - free)
                doc["paid_used"] = max(0, doc.get("paid_used", 0) - paid)
                st.put_user(job.uid, doc)
        except Exception:  # noqa: BLE001
            pass

    def _save_project(self, job: Job) -> None:
        try:
            winner = None
            if job.results:
                v = make_verdict(job.cfg.task, job.cfg.domain, list(job.results["models"].values()))
                winner = v["winner"]
            prev = get_store().get_project(job.project_id) or {}  # keep a rename / move made while it was running
            get_store().put_project(job.project_id, {
                "id": job.project_id, "uid": job.uid, "job_id": job.id, "title": prev.get("title") or f"{job.dataset_name}",
                "domain": job.cfg.domain, "task": job.cfg.task, "dataset": job.dataset_name,
                "models": [m.id for m in job.cfg.models], "winner": winner, "status": job.status,
                "created_at": job.created, "updated_at": time.time(), "config": job.cfg.model_dump(),
                "group_id": prev["group_id"] if "group_id" in prev else job.group_id,
                "parent_id": prev.get("parent_id") or job.parent_id, "kind": "run", "dataset_id": job.dataset_id,
                "dataset_ref": job.dataset_ref, "dataset_source": job.dataset_source,
                "reserved": list(job.reserved), "refunded": prev.get("refunded", False),
            })
        except Exception:  # noqa: BLE001 - project history is best effort
            log.exception("Could not save the chat record for job %s", job.id)


def _compact_events(events: list[dict], max_logs: int = 12) -> list[dict]:
    """Keep what the 'ran your pipeline' view needs; drop bulky duplicates (eda / model_result live in results)."""
    out, logs = [], {}
    for ev in events:
        t = ev.get("type")
        if t in ("eda", "model_result"):
            continue
        if t == "log":
            n = logs.get(ev["step"], 0)
            if n >= max_logs:
                continue
            logs[ev["step"]] = n + 1
            ev = {**ev, "text": str(ev["text"])[:300]}
        out.append({k: v for k, v in ev.items() if k != "ts"})
    return out


jobs = JobManager()


# --------------------------------------------------------------------------- redaction & views
def redact_event(job: Job, ev: dict) -> dict | None:
    t = ev.get("type")
    if t == "model_result":
        if job.model_locked(ev["model_id"]):
            r = ev["result"]
            return {"type": "model_result", "model_id": ev["model_id"],
                    "result": {"id": r["id"], "name": r["name"], "status": r.get("status"), "locked": True}}
        return ev
    mid = ev.get("model_id")
    sid = ev.get("step", ev.get("id", ""))
    if mid is None and isinstance(sid, str) and sid.startswith("train:"):
        mid = sid.split(":", 1)[1]
    if mid and job.model_locked(mid):
        if t == "step_start":
            return {**ev, "code": None, "locked": True}
        if t == "log":
            return None
    return ev


def stream_events(job: Job, start: int = 0) -> Iterator[str]:
    i, last = start, time.time()
    while True:
        with job.lock:
            new = job.events[i:]
        for ev in new:
            i += 1
            out = redact_event(job, ev)
            if out is not None:
                yield f"data: {json.dumps(out)}\n\n"
            if ev.get("type") == "job_end":
                return
        if not new:
            time.sleep(0.15)
            if time.time() - last > 15:
                last = time.time()
                yield ": ping\n\n"


def _steps_from_events(job: Job) -> list[dict]:
    steps: dict[str, dict] = {}
    with job.lock:
        events = list(job.events)
    for ev in events:
        red = redact_event(job, ev)
        if red is None:
            continue
        t = red.get("type")
        if t == "step_start":
            steps[red["id"]] = {"id": red["id"], "title": red["title"], "code": red.get("code"), "logs": [],
                                "status": "running", "model_id": red.get("model_id"), "locked": red.get("locked", False)}
        elif t == "step_code" and red["id"] in steps:
            steps[red["id"]]["code"] = red["code"]
        elif t == "log" and red["step"] in steps:
            steps[red["step"]]["logs"].append(red["text"])
        elif t == "step_end" and red["id"] in steps:
            steps[red["id"]].update(status=red["status"], ms=red.get("ms"), error=red.get("error"))
    return list(steps.values())


def visible_models(job: Job) -> tuple[list[dict], int]:
    res = job.results or {"models": {}, "order": []}
    out, locked = [], 0
    for mid in res["order"]:
        m = res["models"].get(mid)
        if not m:
            continue
        if job.model_locked(mid):
            locked += 1
            out.append({"id": mid, "name": m["name"], "status": m.get("status"), "locked": True})
        else:
            out.append(m)
    return out, locked


def lectures(job: Job) -> list[dict]:
    if not job.cfg.videos or job.videos_locked:
        return []
    from .videos import find_many

    return find_many([(spec.id, MODELS[spec.id].name) for spec in job.cfg.models])


def build_script(job: Job) -> str | None:
    res = job.results
    if not res:
        return None
    parts = ["# Auto ML export: run with pandas and scikit-learn installed", res["codes"].get("load", ""),
             res["codes"].get("prep", ""), res["codes"].get("split", "")]
    for mid in res["order"]:
        if job.model_locked(mid):
            continue
        code = res["codes"]["models"].get(mid)
        if code:
            parts.append(f"# ---- {MODELS[mid].name} ----\n{code}")
    return "\n\n".join(p for p in parts if p)


def _meta_view(job: Job, meta: dict | None) -> dict | None:
    """Hide the method-comparison details while extra (paid) methods are still locked."""
    if not meta or not job.methods_locked:
        return meta
    m = {k: v for k, v in meta.items() if k != "methods"}
    m["methods"] = {"locked": True}
    return m


def job_view(job: Job) -> dict:
    models, locked = visible_models(job)
    res = job.results or {}
    done_models = [m for m in models if not m.get("locked")]
    verdict = None
    if job.status == "done":
        verdict = make_verdict(job.cfg.task, job.cfg.domain, done_models, locked, (res.get("meta") or {}).get("imbalance"))
    return {
        "id": job.id, "status": job.status, "queue_ahead": jobs.queue_ahead(job), "error": job.error, "dataset": job.dataset_name,
        "config": {
            "task": job.cfg.task, "domain": job.cfg.domain, "target": job.cfg.target, "split": job.cfg.split,
            "test_size": job.cfg.test_size, "preprocess": job.cfg.preprocess.model_dump(),
            "models": [{"id": m.id, "tuning": m.tuning} for m in job.cfg.models], "eda": job.cfg.eda,
            "videos": job.cfg.videos, "team": job.cfg.team,
        },
        "steps": _steps_from_events(job), "eda": res.get("eda"), "models": models, "verdict": verdict,
        "meta": _meta_view(job, res.get("meta")), "script": build_script(job) if job.status == "done" else None,
        "lectures": lectures(job),
        "locks": {
            "needs_payment": job.needs_payment, "unlocked": job.unlocked or job.vip, "locked_models": locked,
            "videos_locked": job.videos_locked, "methods_locked": job.methods_locked, "paid_methods": job.paid_methods, "payment_id": job.payment_id, "rejected": job.rejected,
            "free_remaining_before": job.free_remaining_before,
        },
        "project_id": job.project_id,
    }
