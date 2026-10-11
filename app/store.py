"""Persistence layer. Only metadata is stored (users, payments, projects, feedback, usage events).
Datasets are NEVER persisted: they live in RAM for a short TTL (see datasets.py).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import threading
import time
from typing import Any, Protocol

from .config import get_settings


class Store(Protocol):
    def get_user(self, uid: str) -> dict | None: ...
    def put_user(self, uid: str, data: dict) -> None: ...
    def list_users(self) -> list[dict]: ...
    def create_payment(self, data: dict) -> None: ...
    def get_payment(self, pid: str) -> dict | None: ...
    def update_payment(self, pid: str, patch: dict) -> dict | None: ...
    def list_payments(self, status: str | None = None, uid: str | None = None) -> list[dict]: ...
    def add_event(self, data: dict) -> None: ...
    def list_events(self, since: float = 0) -> list[dict]: ...
    def get_vip(self) -> set[str]: ...
    def add_vip(self, email: str) -> None: ...
    def remove_vip(self, email: str) -> None: ...
    def add_feedback(self, data: dict) -> None: ...
    def list_feedback(self, limit: int = 100) -> list[dict]: ...
    def put_project(self, pid: str, data: dict) -> None: ...
    def get_project(self, pid: str) -> dict | None: ...
    def list_projects(self, uid: str) -> list[dict]: ...
    def delete_project(self, pid: str) -> None: ...
    def list_projects_before(self, ts: float) -> list[dict]: ...
    def put_archive(self, job_id: str, data: dict) -> None: ...
    def get_archive(self, job_id: str) -> dict | None: ...
    def delete_archive(self, job_id: str) -> None: ...
    def put_group(self, gid: str, data: dict) -> None: ...
    def get_group(self, gid: str) -> dict | None: ...
    def list_groups(self, uid: str) -> list[dict]: ...
    def delete_group(self, gid: str) -> None: ...


class MemoryStore:
    """Thread-safe in-memory store, used for local dev and tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.users: dict[str, dict] = {}
        self.payments: dict[str, dict] = {}
        self.events: list[dict] = []
        self.vip: set[str] = set()
        self.feedback: list[dict] = []
        self.projects: dict[str, dict] = {}
        self.archives: dict[str, dict] = {}
        self.groups: dict[str, dict] = {}

    def get_user(self, uid):
        with self._lock:
            u = self.users.get(uid)
            return copy.deepcopy(u) if u else None

    def put_user(self, uid, data):
        with self._lock:
            self.users[uid] = copy.deepcopy(data)

    def list_users(self):
        with self._lock:
            return [copy.deepcopy(u) for u in self.users.values()]

    def create_payment(self, data):
        with self._lock:
            self.payments[data["id"]] = copy.deepcopy(data)

    def get_payment(self, pid):
        with self._lock:
            p = self.payments.get(pid)
            return copy.deepcopy(p) if p else None

    def update_payment(self, pid, patch):
        with self._lock:
            if pid not in self.payments:
                return None
            self.payments[pid].update(copy.deepcopy(patch))
            return copy.deepcopy(self.payments[pid])

    def list_payments(self, status=None, uid=None):
        with self._lock:
            items = [copy.deepcopy(p) for p in self.payments.values()
                     if status in (None, p.get("status")) and uid in (None, p.get("uid"))]
        return sorted(items, key=lambda p: p.get("created_at", 0), reverse=True)

    def add_event(self, data):
        with self._lock:
            self.events.append(copy.deepcopy(data))

    def list_events(self, since=0):
        with self._lock:
            return [copy.deepcopy(e) for e in self.events if e.get("ts", 0) >= since]

    def get_vip(self):
        with self._lock:
            return set(self.vip)

    def add_vip(self, email):
        with self._lock:
            self.vip.add(email.lower())

    def remove_vip(self, email):
        with self._lock:
            self.vip.discard(email.lower())

    def add_feedback(self, data):
        with self._lock:
            self.feedback.append(copy.deepcopy(data))

    def list_feedback(self, limit=100):
        with self._lock:
            return [copy.deepcopy(f) for f in reversed(self.feedback[-limit:])]

    def put_project(self, pid, data):
        with self._lock:
            self.projects[pid] = copy.deepcopy(data)

    def get_project(self, pid):
        with self._lock:
            p = self.projects.get(pid)
            return copy.deepcopy(p) if p else None

    def list_projects(self, uid):
        with self._lock:
            items = [copy.deepcopy(p) for p in self.projects.values() if p.get("uid") == uid]
        return sorted(items, key=lambda p: p.get("created_at", 0), reverse=True)

    def delete_project(self, pid):
        with self._lock:
            self.projects.pop(pid, None)

    def list_projects_before(self, ts):
        with self._lock:
            return [copy.deepcopy(p) for p in self.projects.values() if (p.get("updated_at") or p.get("created_at") or 0) < ts]

    def put_archive(self, job_id, data):
        with self._lock:
            self.archives[job_id] = copy.deepcopy(data)

    def get_archive(self, job_id):
        with self._lock:
            a = self.archives.get(job_id)
            return copy.deepcopy(a) if a else None

    def delete_archive(self, job_id):
        with self._lock:
            self.archives.pop(job_id, None)

    def put_group(self, gid, data):
        with self._lock:
            self.groups[gid] = copy.deepcopy(data)

    def get_group(self, gid):
        with self._lock:
            g = self.groups.get(gid)
            return copy.deepcopy(g) if g else None

    def list_groups(self, uid):
        with self._lock:
            items = [copy.deepcopy(g) for g in self.groups.values() if g.get("uid") == uid]
        return sorted(items, key=lambda g: g.get("created_at", 0), reverse=True)

    def delete_group(self, gid):
        with self._lock:
            self.groups.pop(gid, None)


class FirestoreStore:
    """Firestore-backed store. Accessed ONLY from the backend (Admin SDK); client rules deny everything."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def _doc(self, col: str, id_: str):
        return self.db.collection(col).document(id_)

    def get_user(self, uid):
        d = self._doc("users", uid).get()
        return d.to_dict() if d.exists else None

    def put_user(self, uid, data):
        self._doc("users", uid).set(data)

    def list_users(self):
        return [d.to_dict() for d in self.db.collection("users").stream()]

    def create_payment(self, data):
        self._doc("payments", data["id"]).set(data)

    def get_payment(self, pid):
        d = self._doc("payments", pid).get()
        return d.to_dict() if d.exists else None

    def update_payment(self, pid, patch):
        ref = self._doc("payments", pid)
        if not ref.get().exists:
            return None
        ref.update(patch)
        return ref.get().to_dict()

    def list_payments(self, status=None, uid=None):
        q = self.db.collection("payments")
        if status:
            q = q.where("status", "==", status)
        if uid:
            q = q.where("uid", "==", uid)
        items = [d.to_dict() for d in q.stream()]
        return sorted(items, key=lambda p: p.get("created_at", 0), reverse=True)

    def add_event(self, data):
        self.db.collection("events").add(data)

    def list_events(self, since=0):
        return [d.to_dict() for d in self.db.collection("events").where("ts", ">=", since).stream()]

    def get_vip(self):
        d = self._doc("settings", "vip").get()
        return set((d.to_dict() or {}).get("emails", [])) if d.exists else set()

    def add_vip(self, email):
        s = self.get_vip()
        s.add(email.lower())
        self._doc("settings", "vip").set({"emails": sorted(s)})

    def remove_vip(self, email):
        s = self.get_vip()
        s.discard(email.lower())
        self._doc("settings", "vip").set({"emails": sorted(s)})

    def add_feedback(self, data):
        self.db.collection("feedback").add(data)

    def list_feedback(self, limit=100):
        docs = self.db.collection("feedback").order_by("ts", direction="DESCENDING").limit(limit).stream()
        return [d.to_dict() for d in docs]

    def put_project(self, pid, data):
        self._doc("projects", pid).set(data)

    def get_project(self, pid):
        d = self._doc("projects", pid).get()
        return d.to_dict() if d.exists else None

    def list_projects(self, uid):
        items = [d.to_dict() for d in self.db.collection("projects").where("uid", "==", uid).stream()]
        return sorted(items, key=lambda p: p.get("created_at", 0), reverse=True)

    def delete_project(self, pid):
        self._doc("projects", pid).delete()

    def list_projects_before(self, ts):
        # created_at is always present; updated_at is checked in Python so a recently touched chat is kept
        old = [d.to_dict() for d in self.db.collection("projects").where("created_at", "<", ts).stream()]
        return [p for p in old if (p.get("updated_at") or p.get("created_at") or 0) < ts]

    # Archives are stored as one compressed blob: Firestore rejects nested arrays (confusion/correlation matrices)
    # and caps documents at 1 MB.
    def put_archive(self, job_id, data):
        self._doc("archives", job_id).set(data)

    def get_archive(self, job_id):
        d = self._doc("archives", job_id).get()
        return d.to_dict() if d.exists else None

    def delete_archive(self, job_id):
        self._doc("archives", job_id).delete()

    def put_group(self, gid, data):
        self._doc("groups", gid).set(data)

    def get_group(self, gid):
        d = self._doc("groups", gid).get()
        return d.to_dict() if d.exists else None

    def list_groups(self, uid):
        items = [d.to_dict() for d in self.db.collection("groups").where("uid", "==", uid).stream()]
        return sorted(items, key=lambda g: g.get("created_at", 0), reverse=True)

    def delete_group(self, gid):
        self._doc("groups", gid).delete()



class RemoteStore:
    """State kept by the Cloudflare Worker (D1 for records, R2 for run archives), reached over a signed internal API.

    Every call is POST {STORE_URL}/internal/store with {"op", "args"}, signed X-Store-Ts / X-Store-Sig =
    HMAC-SHA256(STORE_SECRET, ts + "." + sha256(body)). Bytes travel as {"__b64__": ...}. Only read operations
    and idempotent writes are retried; create_payment is never retried blindly.
    """

    _RETRY_SAFE = {"get_user", "list_users", "get_payment", "list_payments", "list_events", "get_vip", "list_feedback",
                   "get_project", "list_projects", "list_projects_before", "get_archive", "get_group", "list_groups",
                   "put_user", "put_project", "put_group", "put_archive", "delete_project", "delete_group",
                   "delete_archive", "add_vip", "remove_vip"}

    def __init__(self, url: str, secret: str, client: Any = None) -> None:
        import httpx

        self.url = url.rstrip("/") + "/internal/store"
        self.secret = secret.encode()
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(20.0, connect=5.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )

    # -- codec ---------------------------------------------------------------------------------------------
    @classmethod
    def _enc(cls, v: Any) -> Any:
        if isinstance(v, (bytes, bytearray)):
            return {"__b64__": base64.b64encode(bytes(v)).decode("ascii")}
        if isinstance(v, float):
            return v if math.isfinite(v) else None  # JSON has no NaN/Infinity
        if isinstance(v, dict):
            return {str(k): cls._enc(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [cls._enc(x) for x in v]
        return v

    @classmethod
    def _dec(cls, v: Any) -> Any:
        if isinstance(v, dict):
            if set(v) == {"__b64__"}:
                return base64.b64decode(v["__b64__"])
            return {k: cls._dec(x) for k, x in v.items()}
        if isinstance(v, list):
            return [cls._dec(x) for x in v]
        return v

    # -- transport -----------------------------------------------------------------------------------------
    def _call(self, op: str, *args: Any) -> Any:
        body = json.dumps({"op": op, "args": [self._enc(a) for a in args]}, separators=(",", ":"), default=str)
        attempts = 3 if op in self._RETRY_SAFE else 1
        last: Exception | None = None
        for i in range(attempts):
            ts = str(int(time.time()))
            sig = hmac.new(self.secret, f"{ts}.{hashlib.sha256(body.encode()).hexdigest()}".encode(), hashlib.sha256).hexdigest()
            try:
                r = self.client.post(self.url, content=body, headers={
                    "content-type": "application/json", "x-store-ts": ts, "x-store-sig": sig})
                if r.status_code < 500 and r.status_code != 429:
                    if r.status_code >= 400:
                        raise StoreError(r.status_code, _detail(r))
                    return self._dec(r.json().get("result"))
                last = StoreError(r.status_code, _detail(r))
            except StoreError:
                raise
            except Exception as e:  # noqa: BLE001 - network error
                last = e
            if i + 1 < attempts:
                time.sleep(0.15 * (2 ** i))
        raise StoreError(503, f"State service unavailable: {last}")

    # -- Store protocol ------------------------------------------------------------------------------------
    def get_user(self, uid): return self._call("get_user", uid)
    def put_user(self, uid, data): self._call("put_user", uid, data)
    def list_users(self): return self._call("list_users") or []
    def create_payment(self, data): self._call("create_payment", data)
    def get_payment(self, pid): return self._call("get_payment", pid)
    def update_payment(self, pid, patch): return self._call("update_payment", pid, patch)
    def list_payments(self, status=None, uid=None): return self._call("list_payments", status, uid) or []
    def add_event(self, data): self._call("add_event", data)
    def list_events(self, since=0): return self._call("list_events", since) or []
    def get_vip(self): return {str(e).lower() for e in (self._call("get_vip") or [])}
    def add_vip(self, email): self._call("add_vip", email)
    def remove_vip(self, email): self._call("remove_vip", email)
    def add_feedback(self, data): self._call("add_feedback", data)
    def list_feedback(self, limit=100): return self._call("list_feedback", limit) or []
    def put_project(self, pid, data): self._call("put_project", pid, data)
    def get_project(self, pid): return self._call("get_project", pid)
    def list_projects(self, uid): return self._call("list_projects", uid) or []
    def delete_project(self, pid): self._call("delete_project", pid)
    def list_projects_before(self, ts): return self._call("list_projects_before", ts) or []
    def put_archive(self, job_id, data): self._call("put_archive", job_id, data)
    def get_archive(self, job_id): return self._call("get_archive", job_id)
    def delete_archive(self, job_id): self._call("delete_archive", job_id)
    def put_group(self, gid, data): self._call("put_group", gid, data)
    def get_group(self, gid): return self._call("get_group", gid)
    def list_groups(self, uid): return self._call("list_groups", uid) or []
    def delete_group(self, gid): self._call("delete_group", gid)


class StoreError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


def _detail(r: Any) -> str:
    try:
        d = r.json().get("detail")
        return d.get("message") if isinstance(d, dict) else str(d or r.text[:200])
    except Exception:  # noqa: BLE001
        return r.text[:200]


_store: Store | None = None
_store_lock = threading.Lock()


def init_firebase_app():
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:  # already initialised
        return
    sa = get_settings().firebase_sa_json
    if not sa:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON is required for firebase auth/firestore")
    firebase_admin.initialize_app(credentials.Certificate(json.loads(sa)))


def get_store() -> Store:
    global _store
    with _store_lock:
        if _store is None:
            if get_settings().store_backend == "firestore":
                from firebase_admin import firestore

                init_firebase_app()
                _store = FirestoreStore(firestore.client())
            elif get_settings().store_backend == "remote":
                st = get_settings()
                if not st.store_url or len(st.store_secret) < 32:
                    raise RuntimeError("STORE_URL and STORE_SECRET (32+ chars) are required for STORE_BACKEND=remote")
                _store = RemoteStore(st.store_url, st.store_secret)
            else:
                _store = MemoryStore()
        return _store


def set_store(store: Store | None) -> None:
    """Test hook."""
    global _store
    _store = store
