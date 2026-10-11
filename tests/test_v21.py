"""V2.1: size-constrained dataset sources, payment waiting/reminders, interrupted chats, retention, model advisor, tree picture."""
import dataclasses
import io
import json
import time
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import advisor, notify, videos
from app import datasets as ds
from app import main as main_mod
from app.jobs import jobs
from app.main import app
from app.models_registry import MODELS, clean_params
from app.store import get_store
from conftest import auth
from test_e2e import ADMIN, FRIEND, USER, job_body, run_job, seed_usage, select, stream

client = TestClient(app)


# ---------------------------------------------------------------- datasets: size rule and sources
def test_recommended_list_follows_the_size_rule_and_small_sets_are_opt_in():
    r = client.get("/api/datasets/suggest", params={"domain": "Healthcare", "task": "classification"}, headers=auth(USER)).json()
    assert r["rule"] == {"min_rows": 50000, "min_cols": 30} and r["source"] == "recommended"
    assert r["items"] and all(i["rows"] > 50000 and i["cols"] >= 30 for i in r["items"])
    assert r["items"][0]["id"] == "builtin:big-readmission"
    withsmall = client.get("/api/datasets/suggest", params={"domain": "Healthcare", "task": "classification", "include_small": "true"},
                           headers=auth(USER)).json()["items"]
    assert any(i["id"] == "builtin:breast-cancer" for i in withsmall)


def test_every_domain_has_a_large_recommendation_and_it_loads():
    for dom, task in [("Finance", "classification"), ("E-commerce", "regression"), ("Real estate", "regression"),
                      ("Education", "regression"), ("Manufacturing", "classification"), ("Healthcare", "classification")]:
        items = client.get("/api/datasets/suggest", params={"domain": dom, "task": task}, headers=auth(USER)).json()["items"]
        assert items and items[0]["match"], (dom, task)
    d = select(USER, "builtin:big-clv")
    assert d["profile"]["rows"] == 60000 and d["profile"]["cols"] == 34 and d["ref"] == "builtin:big-clv"


def test_kaggle_search_filters_by_size_and_reads_both_field_styles(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(ds, "get_settings", lambda: dataclasses.replace(s, kaggle_api_token="KGAT_x"))
    seen = {}

    class R:
        status_code = 200

        def json(self):
            return [{"ref": "a/big", "titleNullable": "Big one", "totalBytesNullable": 40_000_000, "downloadCount": 9},
                    {"ref": "b/other", "title": "Other", "totalBytes": 30_000_000, "downloadCount": 99}]

        def raise_for_status(self):
            pass

    def fake_get(url, params=None, **kw):
        seen["params"], seen["headers"] = params, kw.get("headers")
        return R()

    monkeypatch.setattr(ds.httpx, "get", fake_get)
    out = ds.suggest("Finance", "classification", source="kaggle")
    assert seen["params"]["minSize"] >= 5_000_000 and seen["headers"] == {"Authorization": "Bearer KGAT_x"}
    assert [i["id"] for i in out["items"]][0] == "kaggle:b/other"  # most downloaded first, de-duplicated across keywords
    assert all(i["source"] == "kaggle" for i in out["items"]) and out["rule"]["min_rows"] == 50000
    assert {i["name"] for i in out["items"]} == {"Big one", "Other"}


def test_kaggle_download_rejects_datasets_below_the_rule(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(ds, "get_settings", lambda: dataclasses.replace(s, kaggle_api_token="KGAT_x"))
    small = pd.DataFrame(np.random.rand(200, 5), columns=list("abcde")).to_csv(index=False).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.csv", small)
    monkeypatch.setattr(ds, "fetch_bytes", lambda *a, **k: buf.getvalue())
    with pytest.raises(ds.DatasetError) as e:
        ds.load_external("kaggle:owner/small")
    assert e.value.status == 422 and "50,000 rows" in e.value.message and "Upload" in e.value.message


def test_uci_index_is_filtered_by_size_task_and_domain(monkeypatch):
    items = [
        {"uci_id": 1, "name": "Covertype", "abstract": "forest cover", "area": "Biology", "tasks": ["Classification"], "rows": 581012,
         "cols": 55, "target": "Cover_Type", "data_url": "u", "page": "p"},
        {"uci_id": 2, "name": "Iris", "abstract": "flowers", "area": "Biology", "tasks": ["Classification"], "rows": 150, "cols": 5,
         "target": "class", "data_url": "u", "page": "p"},
        {"uci_id": 3, "name": "Big regression", "abstract": "money", "area": "Business", "tasks": ["Regression"], "rows": 90000,
         "cols": 40, "target": "y", "data_url": "u", "page": "p"},
    ]
    monkeypatch.setattr(ds, "_uci_index", lambda block=True: items)
    got = ds.uci_search("Finance", "classification")
    assert [g["id"] for g in got] == ["uci:1"] and got[0]["rows"] == 581012 and got[0]["default_target"] == "Cover_Type"
    assert [g["id"] for g in ds.uci_search("Finance", "regression")] == ["uci:3"]
    assert ds.uci_search("", "", "forest")[0]["id"] == "uci:1" and ds.uci_search("", "", "nothing-matches") == []


def test_huggingface_results_need_a_known_shape_that_meets_the_rule(monkeypatch):
    class R:
        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ds.httpx, "get", lambda *a, **k: R([{"id": "o/big", "tags": ["license:mit"]}, {"id": "o/small"}, {"id": "o/noviewer"}]))
    shapes = {"o/big": (80000, 40), "o/small": (500, 8), "o/noviewer": (None, None)}
    monkeypatch.setattr(ds, "_hf_shape", lambda did: shapes[did])
    got = ds.hf_search("credit", task="classification")
    assert [g["id"] for g in got] == ["hf:o/big"] and got[0]["cols"] == 40


def test_source_warnings_are_specific(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(ds, "get_settings", lambda: dataclasses.replace(s, kaggle_api_token="", kaggle_username="", kaggle_key=""))
    out = ds.suggest("Finance", "classification", source="kaggle")
    assert out["items"] == [] and "no Kaggle API token" in out["warnings"][0]
    monkeypatch.setattr(ds, "_uci_index", lambda block=True: [])
    assert "at least 50,000 rows and 30 columns" in ds.suggest("Finance", "classification", source="uci")["warnings"][0]


# ---------------------------------------------------------------- payments
def _paid_job():
    seed_usage(USER, free_used=9, window_start=time.time() - 60)
    job, _ = run_job(USER, ["logreg", "knn", "tree"])
    pay = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    return job, pay


def test_remind_flow_waits_then_emails_and_shows_on_admin_dashboard(monkeypatch):
    job, pay = _paid_job()
    assert client.post(f"/api/payments/{pay['id']}/remind", headers=auth(USER)).status_code == 409  # not claimed yet
    claim = client.post(f"/api/payments/{pay['id']}/claim", headers=auth(USER)).json()["payment"]
    assert claim["remind_after"] == 420 and claim["claimed_at"]
    early = client.post(f"/api/payments/{pay['id']}/remind", headers=auth(USER))
    assert early.status_code == 429 and early.json()["detail"]["seconds_left"] > 400
    assert client.post(f"/api/payments/{pay['id']}/remind", headers=auth(FRIEND)).status_code == 404

    st = get_store()
    st.update_payment(pay["id"], {"claimed_at": time.time() - 8 * 60})  # 8 minutes pass
    notify.OUTBOX.clear()
    ok = client.post(f"/api/payments/{pay['id']}/remind", headers=auth(USER))
    assert ok.status_code == 200 and ok.json()["reminders"] == 1
    assert any("Reminder" in m["subject"] and pay["ref"] in m["subject"] for m in notify.OUTBOX)
    again = client.post(f"/api/payments/{pay['id']}/remind", headers=auth(USER))
    assert again.status_code == 429  # a fresh wait starts after each reminder

    n = client.get("/api/admin/notifications", headers=auth(ADMIN)).json()
    assert n["count"] == 1 and n["reminders"] == 1 and n["items"][0]["ref"] == pay["ref"] and n["items"][0]["waiting_seconds"] >= 480
    assert client.get("/api/admin/notifications", headers=auth(USER)).status_code == 403
    client.post(f"/api/admin/payments/{pay['id']}/decision", json={"decision": "approve"}, headers=auth(ADMIN))
    assert client.get("/api/admin/notifications", headers=auth(ADMIN)).json()["count"] == 0
    assert client.post(f"/api/payments/{pay['id']}/remind", headers=auth(USER)).status_code == 409


def test_my_payments_are_paginated_seven_per_page_with_chat_titles():
    st = get_store()
    me = client.get("/api/me", headers=auth(USER)).json()
    for i in range(10):
        st.create_payment({"id": f"p{i:02d}", "ref": f"AML-{i}", "uid": me["uid"], "user_email": USER, "kind": "models", "job_id": "j",
                           "project_id": "chat1234abcd", "amount": 10 + i, "breakdown": {"lines": [], "total": 10 + i},
                           "status": "approved", "created_at": 1000 + i, "claimed_at": None, "decided_at": None, "context": {"domain": "Finance"}})
    st.put_project("chat1234abcd", {"id": "chat1234abcd", "uid": me["uid"], "title": "Loan run", "dataset": "Loans"})
    p1 = client.get("/api/payments/mine", headers=auth(USER)).json()
    assert p1["per_page"] == 7 and len(p1["payments"]) == 7 and p1["pages"] == 2 and p1["total"] == 10
    assert p1["payments"][0]["id"] == "p09" and p1["payments"][0]["chat"]["title"] == "Loan run"
    p2 = client.get("/api/payments/mine", params={"page": 2}, headers=auth(USER)).json()
    assert len(p2["payments"]) == 3 and client.get("/api/payments/mine", headers=auth(FRIEND)).json()["total"] == 0
    assert client.get("/api/payments/p00", headers=auth(USER)).json()["ref"] == "AML-0"
    assert client.get("/api/payments/p00", headers=auth(FRIEND)).status_code == 404


# ---------------------------------------------------------------- chats: interrupted, retention, re-run
def _seed_chat(uid, cid, status, **extra):
    get_store().put_project(cid, {"id": cid, "uid": uid, "job_id": "gone" + cid, "title": "T", "domain": "Finance", "task": "classification",
                                  "dataset": "D", "models": ["logreg"], "winner": None, "status": status, "created_at": time.time(),
                                  "updated_at": time.time(), "config": {"dataset_id": "x", "task": "classification", "domain": "Finance",
                                  "target": "y", "models": [{"id": "logreg"}]}, "group_id": None, **extra})


def test_a_run_lost_in_a_restart_is_marked_interrupted_and_refunded_once():
    me = client.get("/api/me", headers=auth(USER)).json()
    seed_usage(USER, free_used=4, paid_used=0, window_start=time.time() - 30)
    _seed_chat(me["uid"], "lostchat0001", "running", reserved=[3, 0])
    c = client.get("/api/chats/lostchat0001", headers=auth(USER)).json()
    assert c["status"] == "interrupted" and "config" not in c and "reserved" not in c
    assert client.get("/api/usage", headers=auth(USER)).json()["free_used"] == 1
    client.get("/api/chats", headers=auth(USER))
    assert client.get("/api/usage", headers=auth(USER)).json()["free_used"] == 1  # refunded only once


def test_retention_sweep_deletes_only_chats_older_than_fifteen_days():
    me = client.get("/api/me", headers=auth(USER)).json()
    _seed_chat(me["uid"], "oldchat00001", "done", created_at=time.time() - 16 * 86400, updated_at=time.time() - 16 * 86400)
    _seed_chat(me["uid"], "freshchat001", "done", created_at=time.time() - 14 * 86400, updated_at=time.time() - 14 * 86400)
    get_store().put_archive("goneoldchat00001", {"uid": me["uid"], "blob": b"x"})
    assert jobs.sweep_retention() == 1
    ids = [c["id"] for c in client.get("/api/chats", headers=auth(USER)).json()["chats"]]
    assert ids == ["freshchat001"]
    exp = client.get("/api/chats/freshchat001", headers=auth(USER)).json()["expires_at"]
    assert abs(exp - (time.time() - 14 * 86400 + 15 * 86400)) < 5


def test_rerun_returns_settings_and_a_way_to_reload_the_same_dataset():
    job, _ = run_job(USER, ["logreg"], dataset="builtin:iris")
    cid = job["project_id"]
    r = client.get(f"/api/chats/{cid}/rerun", headers=auth(USER)).json()
    assert r["dataset"]["ref"] == "builtin:iris" and r["dataset"]["source"] == "builtin"
    assert r["config"]["target"] == "species" and "dataset_id" not in r["config"] and r["config"]["models"][0]["id"] == "logreg"
    assert client.get(f"/api/chats/{cid}/rerun", headers=auth(FRIEND)).status_code == 404


# ---------------------------------------------------------------- models: typed request, advisor, tree picture
def test_typed_model_request_is_understood():
    r = client.post("/api/models/parse", json={"task": "classification", "text": "random forest, decision tree and an ANN, also xgboost"},
                    headers=auth(USER)).json()
    assert r["model_ids"] == ["rf", "tree", "mlp"] and r["not_offered"] == ["XGBoost"]
    assert advisor.parse_request("extra trees", "classification")["model_ids"] == ["extra"]  # not read as 'tree'
    reg = client.post("/api/models/parse", json={"task": "regression", "text": "logistic regression and ridge"}, headers=auth(USER)).json()
    assert reg["model_ids"] == ["ridge"] and reg["not_for_task"] == ["Logistic Regression"]


def test_plan_recommends_settings_with_reasons_and_asks_the_right_parameters():
    d = select(USER, "builtin:diabetes")
    body = {"dataset_id": d["dataset_id"], "target": "progression", "task": "regression", "model_ids": ["tree", "mlp", "ridge", "rf"]}
    r = client.post("/api/models/plan", json=body, headers=auth(USER))
    assert r.status_code == 200, r.text
    plans = r.json()["plans"]
    tree = {p["name"]: p for p in plans["tree"]["params"]}
    assert tree["pruning"]["ask"] and tree["pruning"]["recommended"] in ("none", "pre", "post", "both") and tree["ccp_alpha"]["reason"]
    assert tree["criterion"]["options"] == ["squared_error", "friedman_mse", "absolute_error"]  # regression criteria only
    mlp = {p["name"]: p for p in plans["mlp"]["params"]}
    assert {"hidden_layers", "activation", "learning_rate_init", "early_stopping"} <= {n for n, p in mlp.items() if p["ask"]}
    assert mlp["hidden_layers"]["recommended"] in mlp["hidden_layers"]["options"]
    assert plans["ridge"]["measured"] and r.json()["summary"]["text"]
    assert client.post("/api/models/plan", json={**body, "model_ids": ["logreg"]}, headers=auth(USER)).status_code == 422


def test_tree_pruning_and_ann_settings_reach_the_estimator():
    t = MODELS["tree"]
    assert t.kwargs("classification", clean_params("tree", {"pruning": "post", "ccp_alpha": 0.01, "max_depth": 4})) == {"random_state": 42, "ccp_alpha": 0.01}
    assert t.kwargs("classification", clean_params("tree", {"pruning": "pre", "ccp_alpha": 0.01, "max_depth": 4, "min_samples_leaf": 7})) == \
        {"random_state": 42, "max_depth": 4, "min_samples_leaf": 7}
    kw = MODELS["mlp"].kwargs("classification", clean_params("mlp", {"hidden_layers": "128,64", "activation": "tanh", "early_stopping": "yes"}))
    assert kw["hidden_layer_sizes"] == (128, 64) and kw["activation"] == "tanh" and kw["early_stopping"] is True
    assert clean_params("mlp", {"activation": "sigmoid-ish"}) == {}  # unknown choices are dropped


def test_decision_tree_run_returns_a_drawable_tree_and_eda_lists_features():
    ds_ = select(USER, "builtin:iris")
    body = job_body(ds_["dataset_id"], ["tree"])
    body["models"] = [{"id": "tree", "tuning": "custom", "params": {"pruning": "pre", "max_depth": 3, "criterion": "entropy"}}]
    job = client.post("/api/jobs", json=body, headers=auth(USER)).json()
    stream(USER, job["job_id"])
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    tree = view["models"][0]["tree"]
    assert tree["depth"] <= 3 and tree["leaves"] >= 2 and tree["root"]["feature"] and "threshold" in tree["root"] and tree["root"]["n"] > 0
    assert "left" in tree["root"] and "label" in tree["root"]["left"]
    cols = view["eda"]["columns"]
    assert [c["role"] for c in cols].count("target") == 1 and {c["name"] for c in cols} >= {"species"} and all("kind" in c for c in cols)


# ---------------------------------------------------------------- YouTube diagnostics
def _fake_youtube(status, reason=None, items=None):
    class R:
        status_code = status

        def json(self):
            if reason:
                return {"error": {"message": "x", "errors": [{"reason": reason}]}}
            return {"items": items or []}

    return lambda *a, **k: R()


def test_youtube_failures_are_reported_with_a_reason(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(videos, "get_settings", lambda: dataclasses.replace(s, youtube_api_key="K"))
    videos._cache.clear(); videos._fail.clear()
    out = videos.find_lectures("tree", "Decision Tree", get=_fake_youtube(403, "ipRefererBlocked"))
    assert out["status"] == "curated" and out["reason"] == "key_restricted" and len(out["videos"]) >= 2
    assert videos.diagnose(get=_fake_youtube(403, "accessNotConfigured"))["code"] == "api_disabled"
    assert videos.diagnose(get=_fake_youtube(400, "keyInvalid"))["code"] == "key_invalid"
    assert videos.diagnose(get=_fake_youtube(403, "quotaExceeded"))["code"] == "quota"
    vid = [{"id": {"videoId": "abcdefghijk"}, "snippet": {"title": "T", "channelTitle": "C"}}]
    videos._cache.clear(); videos._fail.clear()
    ok = videos.find_lectures("rf", "Random Forest", get=_fake_youtube(200, items=vid))
    assert ok["status"] == "ok" and ok["videos"][0]["id"] == "abcdefghijk"
    monkeypatch.setattr(videos, "get_settings", lambda: dataclasses.replace(s, youtube_api_key=""))
    assert videos.diagnose()["code"] == "no_key" and videos.find_lectures("knn", "KNN")["reason"] == "no_key"
    assert client.post("/api/admin/youtube/test", headers=auth(USER)).status_code == 403
    assert client.post("/api/admin/youtube/test", headers=auth(ADMIN)).json()["code"] == "no_key"


def test_lectures_by_language(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(videos, "get_settings", lambda: dataclasses.replace(s, youtube_api_key="K"))
    videos._cache.clear(); videos._fail.clear()
    seen = {}

    def fake(url, params=None, timeout=0):
        seen.update(params)
        return _fake_youtube(200, items=[{"id": {"videoId": f"abcdefghij{i}"}, "snippet": {"title": f"T{i}", "channelTitle": "C"}} for i in range(3)])()

    out = videos.find_lectures("tree", "Decision Tree", lang="ta", get=fake)
    assert out["lang"] == "ta" and seen["relevanceLanguage"] == "ta" and "Tamil" in seen["q"] and len(out["videos"]) == 3
    assert videos.find_lectures("tree", "Decision Tree", lang="xx", get=fake)["lang"] == "en"
    # a failing language falls back to English curated videos with a note
    bad = videos.find_lectures("knn", "KNN", lang="hi", get=_fake_youtube(403, "quotaExceeded"))
    assert "English" in bad["note"] and bad["reason"] == "quota" and bad["videos"]


def test_chronological_split_falls_back_when_data_is_sorted_by_class():
    import numpy as np
    import pandas as pd
    from app import pipeline

    df = pd.DataFrame(np.random.default_rng(0).normal(size=(1000, 5)), columns=list("abcde"))
    df["y"] = np.r_[np.zeros(800), np.ones(200)].astype(int)
    cfg = pipeline.JobConfig.model_validate({"dataset_id": "x", "task": "classification", "domain": "Finance", "target": "y",
                                             "split": "chronological", "test_size": 0.3, "models": [{"id": "logreg", "tuning": "default", "params": {}}]})
    ev = []
    res = pipeline.run_pipeline(df, cfg, "t", ev.append, 100000, 120)
    assert any("Switching to a random" in str(e) for e in ev) and res["models"]["logreg"]["status"] == "done"


def test_llm_probe_reports_each_key_and_failover_skips_bad_ones():
    from app import llm_router as lr

    def transport(base, key, payload, timeout):
        if key == "bad":
            raise lr.ProviderError(401, "HTTP 401: invalid api key")
        return "ready"

    groups = [{"group": "groq", "models": [{"base_url": "u", "model": "m1", "api_key": k, "provider": "groq"} for k in ("bad", "good1", "good2")]}]
    router = lr.LLMRouter(groups, transport=transport)
    res = lr.probe(router)
    assert [(r["key"], r["ok"]) for r in res] == [(1, False), (2, True), (3, True)] and "invalid api key" in res[0]["error"]
    assert all("good" not in str(r["error"]) for r in res)
    text, meta = router.complete([{"role": "user", "content": "hi"}])  # bad key is skipped, next key answers
    assert text == "ready"
    assert router._cool and max(router._cool.values()) - router.clock() > 500  # a 401 is parked for 10 minutes, not 1


def test_youtube_accepts_several_keys_and_falls_through_bad_ones(monkeypatch):
    s = main_mod.settings
    monkeypatch.setattr(videos, "get_settings", lambda: dataclasses.replace(s, youtube_api_key="bad, good ,bad"))
    vid = [{"id": {"videoId": "abcdefghijk"}, "snippet": {"title": "T", "channelTitle": "C"}}]

    def fake(url, params=None, timeout=0):
        return _fake_youtube(400, "keyInvalid")() if params["key"] == "bad" else _fake_youtube(200, items=vid)()

    videos._cache.clear(); videos._fail.clear()
    out = videos.find_lectures("rf", "Random Forest", get=fake)
    assert out["status"] == "ok" and out["videos"][0]["id"] == "abcdefghijk"
    d = videos.diagnose(get=fake)
    assert d["ok"] and [r["ok"] for r in d["keys"]] == [False, True] and "1 of 2" in d["message"]


def test_reasoning_models_get_room_and_empty_replies_are_errors():
    from app import llm_router as lr

    p = lr.make_payload({"model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1"}, [], 10)
    assert p["max_tokens"] >= 400 and p["reasoning_effort"] == "low"
    q = lr.make_payload({"model": "mistral-medium-latest", "base_url": "https://api.mistral.ai/v1"}, [], 10)
    assert q["max_tokens"] == 10 and "reasoning_effort" not in q


def test_hf_parquet_only_dataset_loads(monkeypatch):
    import io

    import pandas as pd

    from app import datasets as ds

    df = pd.DataFrame({f"c{i}": range(60_000) for i in range(32)})
    buf = io.BytesIO()
    df.to_parquet(buf)
    monkeypatch.setattr(ds, "_hf_parquet_urls", lambda did: ["https://huggingface.co/x.parquet"])
    monkeypatch.setattr(ds, "fetch_bytes", lambda *a, **k: buf.getvalue())

    class R:
        def json(self):
            return {"siblings": [{"rfilename": "data/train-00000.parquet"}]}

    monkeypatch.setattr(ds.httpx, "get", lambda *a, **k: R())
    out, name, _ = ds.load_external("hf:owner/tabular")
    assert out.shape == (60_000, 32) and name == "owner/tabular"


def test_uci_search_reports_warming_instead_of_blocking(monkeypatch):
    from app import datasets as ds

    def warming(block=True):
        raise ds.CatalogWarming("loading")

    monkeypatch.setattr(ds, "_uci_index", warming)
    res = ds.suggest("Finance", "classification", "credit", source="uci")
    assert res["items"] == [] and "still loading" in res["warnings"][0]
