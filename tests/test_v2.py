"""V2: chats/projects, archived results, wide-dataset EDA picker, LLM provider presets, YouTube lookup, Kaggle auth."""
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import llm_providers as lp
from app import videos
from app.eda import build_eda, recommend_eda_columns
from app.jobs import jobs
from app.main import app
from conftest import auth
from test_e2e import run_job, select, job_body, stream, USER, FRIEND

client = TestClient(app)


def test_chat_id_is_used_and_listed_then_survives_memory_expiry():
    ds = select(USER)
    r = client.post("/api/jobs?chat_id=abc12345xy", json=job_body(ds["dataset_id"], ["logreg", "tree"]), headers=auth(USER))
    assert r.status_code == 200 and r.json()["project_id"] == "abc12345xy"
    jid = r.json()["job_id"]
    stream(USER, jid)
    chats = client.get("/api/chats", headers=auth(USER)).json()["chats"]
    assert [c["id"] for c in chats] == ["abc12345xy"] and "config" not in chats[0]
    before = client.get(f"/api/jobs/{jid}", headers=auth(USER)).json()
    jobs._jobs.clear()  # simulate a restart / TTL sweep: only the archive remains
    after = client.get(f"/api/jobs/{jid}", headers=auth(USER)).json()
    assert after["status"] == "done" and after["verdict"]["winner"] == before["verdict"]["winner"]
    assert len(after["models"]) == 2 and after["eda"] and after["script"]
    assert [s["id"] for s in after["steps"]][:2] == [s["id"] for s in before["steps"]][:2]
    # another user cannot read it
    assert client.get(f"/api/jobs/{jid}", headers=auth(FRIEND)).status_code == 404
    # a chat holds exactly one run
    ds2 = select(USER)
    dup = client.post("/api/jobs?chat_id=abc12345xy", json=job_body(ds2["dataset_id"], ["logreg"]), headers=auth(USER))
    assert dup.status_code == 409
    bad = client.post("/api/jobs?chat_id=BAD!", json=job_body(ds2["dataset_id"], ["logreg"]), headers=auth(USER))
    assert bad.status_code == 422


def test_paid_lock_state_survives_archive_reload():
    from test_e2e import seed_usage
    seed_usage(USER, free_used=10)
    job, _ = run_job(USER, ["logreg", "tree"])
    jobs._jobs.clear()
    v = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert v["locks"]["locked_models"] == 2 and all(m.get("locked") for m in v["models"])
    p = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    client.post(f"/api/payments/{p['id']}/claim", headers=auth(USER))
    jobs._jobs.clear()  # claim must have been persisted
    v = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert v["locks"]["unlocked"] is True and not any(m.get("locked") for m in v["models"])


def test_groups_move_rename_delete_flow():
    g = client.post("/api/groups", json={"name": "Loans"}, headers=auth(USER)).json()
    ds = select(USER)
    r = client.post(f"/api/jobs?chat_id=chat1234abcd&group_id={g['id']}", json=job_body(ds["dataset_id"], ["logreg"]), headers=auth(USER))
    assert r.status_code == 200
    stream(USER, r.json()["job_id"])
    groups = client.get("/api/groups", headers=auth(USER)).json()["groups"]
    assert groups[0]["chat_count"] == 1 and groups[0]["task"] == "classification"
    ds = select(USER)
    r2 = client.post("/api/jobs?chat_id=chat5678abcd", json=job_body(ds["dataset_id"], ["tree"]), headers=auth(USER))
    stream(USER, r2.json()["job_id"])
    assert client.patch("/api/chats/chat5678abcd", json={"group_id": g["id"], "title": "Second try"}, headers=auth(USER)).json()["group_id"] == g["id"]
    # other users can neither see nor move into it
    assert client.patch("/api/chats/chat5678abcd", json={"title": "x"}, headers=auth(FRIEND)).status_code == 404
    assert client.patch("/api/chats/chat5678abcd", json={"group_id": g["id"]}, headers=auth(FRIEND)).status_code == 404
    assert client.get("/api/groups", headers=auth(USER)).json()["groups"][0]["chat_count"] == 2
    assert client.patch(f"/api/groups/{g['id']}", json={"name": "Loans 2"}, headers=auth(USER)).json()["name"] == "Loans 2"
    assert client.patch("/api/chats/chat5678abcd", json={"ungroup": True}, headers=auth(USER)).json()["group_id"] is None
    client.delete(f"/api/groups/{g['id']}", headers=auth(USER))
    chats = {c["id"]: c for c in client.get("/api/chats", headers=auth(USER)).json()["chats"]}
    assert set(chats) == {"chat1234abcd", "chat5678abcd"} and chats["chat1234abcd"]["group_id"] is None
    assert client.delete("/api/chats/chat1234abcd", headers=auth(USER)).json() == {"ok": True}
    assert [c["id"] for c in client.get("/api/chats", headers=auth(USER)).json()["chats"]] == ["chat5678abcd"]


def wide_df(n=400, cols=30):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, cols)), columns=[f"f{i}" for i in range(cols)])
    df["y"] = (df["f3"] * 2 + df["f7"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df["id"] = [f"row{i}" for i in range(n)]
    df["const"] = 1
    df["grade"] = np.where(df["y"] == 1, "A", "B")
    return df


def test_recommend_eda_columns_ranks_signal_and_skips_junk():
    rec = recommend_eda_columns(wide_df(), "y", "classification", k=5)
    assert rec["wide"] is True
    assert set(rec["recommended"][:3]) >= {"f3", "grade"} and "f7" in rec["recommended"]
    assert rec["skipped"]["const"] == "constant" and rec["skipped"]["id"] == "looks like an ID"
    eda = build_eda(wide_df(), "y", "classification", columns=rec["recommended"])
    assert eda["overview"]["explored_cols"] == 6
    assert {c["column"] for c in eda["numeric_summary"]} <= set(rec["recommended"])


def test_eda_columns_endpoint_and_job_validation():
    ds = select(USER, "builtin:credit-risk")
    r = client.post("/api/datasets/eda-columns", json={"dataset_id": ds["dataset_id"], "target": "defaulted", "task": "classification"}, headers=auth(USER))
    assert r.status_code == 200 and r.json()["wide"] is False and len(r.json()["recommended"]) <= 12
    bad = job_body(ds["dataset_id"], ["logreg"], target="defaulted", eda_columns=["nope"])
    assert client.post("/api/jobs", json=bad, headers=auth(USER)).status_code == 422
    ok = job_body(ds["dataset_id"], ["logreg"], target="defaulted", eda_columns=r.json()["recommended"][:4])
    j = client.post("/api/jobs", json=ok, headers=auth(USER))
    assert j.status_code == 200
    stream(USER, j.json()["job_id"])
    v = client.get(f"/api/jobs/{j.json()['job_id']}", headers=auth(USER)).json()
    assert v["eda"]["overview"]["explored_cols"] == 5


# ---- LLM providers
GROQ_IDS = ["llama-3.1-8b-instant", "whisper-large-v3", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "meta-llama/llama-guard-4-12b"]


def test_rank_models_picks_most_capable_and_skips_non_chat():
    g = lp.BY_NAME["groq"]
    assert lp.rank_models(g, GROQ_IDS)[:2] == ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
    assert "whisper-large-v3" not in lp.rank_models(g, GROQ_IDS)
    gem = lp.BY_NAME["gemini"]
    ids = ["models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-2.5-flash-lite", "models/gemini-embedding-001", "models/gemini-2.5-flash-image"]
    assert lp.rank_models(gem, ids) == ["gemini-2.5-flash", "gemini-2.5-pro"]
    orr = lp.BY_NAME["openrouter"]
    ids = ["openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct:free", "deepseek/deepseek-r1:free"]
    assert lp.rank_models(orr, ids)[0] == "deepseek/deepseek-r1:free"
    assert lp.rank_models(lp.BY_NAME["mistral"], ["mistral-small-latest", "mistral-large-latest", "mistral-embed", "codestral-latest"]) == ["mistral-large-latest", "mistral-small-latest"]


class FakeResp:
    def __init__(self, data, code=200):
        self._d, self.status_code = data, code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._d


def test_build_groups_from_env_with_discovery_and_default_fallback():
    lp._cache.clear()
    env = {"GROQ_API_KEYS": "k1,k2", "GEMINI_API_KEY": "g1", "CEREBRAS_API_KEY": ""}
    calls = []

    def fake_get(url, headers=None, timeout=0):
        calls.append(url)
        if "groq" in url:
            return FakeResp({"data": [{"id": i} for i in GROQ_IDS]})
        raise RuntimeError("network down")  # gemini discovery fails -> default

    groups = {g["group"]: g for g in lp.build_groups(env, fake_get)}
    assert set(groups) == {"groq", "gemini"}
    gm = groups["groq"]["models"]
    assert [m["model"] for m in gm][:4] == ["openai/gpt-oss-120b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile"]
    assert [m["api_key"] for m in gm][:2] == ["k1", "k2"]
    assert groups["gemini"]["models"][0]["model"] == "gemini-2.5-flash"
    d = lp.describe(env, fake_get)
    assert next(x for x in d if x["provider"] == "groq")["model"] == "openai/gpt-oss-120b"
    assert "api_key" not in str(d) and next(x for x in d if x["provider"] == "cerebras")["configured"] is False


def test_router_accepts_inline_keys_and_fails_over():
    from app.llm_router import LLMRouter
    seen = []

    def transport(base, key, payload, timeout):
        seen.append(key)
        if key == "bad":
            raise RuntimeError("retryable:429")
        return "ok"

    r = LLMRouter([{"group": "a", "models": [{"base_url": "https://x", "model": "m", "api_key": "bad"},
                                             {"base_url": "https://x", "model": "m", "api_key": "good"}]}], transport=transport)
    assert r.complete([{"role": "user", "content": "hi"}])[0] == "ok" and seen == ["bad", "good"]
    r.complete([{"role": "user", "content": "hi"}])
    assert seen[2:] == ["good"]  # the bad key is cooling down


def test_admin_llm_endpoint_is_admin_only_and_hides_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret")
    lp._cache.clear()
    monkeypatch.setattr(lp, "discover", lambda p, k, get=None: GROQ_IDS if p.name == "groq" else [])
    assert client.get("/api/admin/llm", headers=auth(USER)).status_code == 403
    r = client.get("/api/admin/llm", headers=auth("admin@example.com"))
    assert r.status_code == 200 and "super-secret" not in r.text
    assert next(p for p in r.json()["providers"] if p["provider"] == "groq")["model"] == "openai/gpt-oss-120b"


# ---- YouTube
def test_youtube_lookup_caches_and_falls_back(monkeypatch):
    videos._cache.clear()
    monkeypatch.setattr(videos, "get_settings", lambda: type("S", (), {"youtube_api_key": "yt"})())
    n = {"c": 0}

    def fake_get(url, params=None, timeout=0):
        n["c"] += 1
        return FakeResp({"items": [{"id": {"videoId": "dQw4w9WgXcQ"}, "snippet": {"title": "KNN explained", "channelTitle": "Ch"}},
                                   {"id": {"videoId": "short"}, "snippet": {}}]})

    a = videos.find_lectures("knn", "K-Nearest Neighbours", get=fake_get)
    b = videos.find_lectures("knn", "K-Nearest Neighbours", get=fake_get)
    assert a["video_ids"] == ["dQw4w9WgXcQ"] and a["videos"][0]["title"] == "KNN explained" and n["c"] == 1 and b == a

    def boom(*a, **k):
        raise RuntimeError("quota")

    videos._cache.clear()
    c = videos.find_lectures("svm", "SVM", get=boom)
    assert len(c["video_ids"]) >= 2 and c["status"] == "curated" and c["reason"] == "network" and c["search_url"].startswith("https://www.youtube.com/results")


# ---- Kaggle auth
def test_kaggle_token_uses_bearer_and_legacy_uses_basic(monkeypatch):
    from app import datasets as ds
    S = type("S", (), {})
    s = S(); s.kaggle_api_token, s.kaggle_username, s.kaggle_key = "KGAT_x", "", ""
    monkeypatch.setattr(ds, "get_settings", lambda: s)
    assert ds.kaggle_enabled() and ds._kaggle_auth() == ({"Authorization": "Bearer KGAT_x"}, None)
    s2 = S(); s2.kaggle_api_token, s2.kaggle_username, s2.kaggle_key = "", "u", "k"
    monkeypatch.setattr(ds, "get_settings", lambda: s2)
    assert ds._kaggle_auth() == ({}, ("u", "k"))


def test_cors_allows_patch_for_rename_and_move():
    r = client.options("/api/chats/abc12345", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PATCH",
                                                       "Access-Control-Request-Headers": "authorization,content-type"})
    assert r.status_code == 200
    assert "PATCH" in r.headers.get("access-control-allow-methods", "")
