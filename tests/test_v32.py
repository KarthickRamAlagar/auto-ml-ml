import numpy as np
import pandas as pd
import pytest

from app import datasets as d
from app import methods as mth
from app.pipeline import JobConfig, ModelSpec, PipelineError, Preprocess, clean_user_grid, run_pipeline


def _run(df, **kw):
    cfg = JobConfig(dataset_id="x", **kw)
    ev = []
    return cfg, run_pipeline(df, cfg, "t", ev.append, max_rows=50_000, max_seconds=120), ev


def _imbalanced(n=600, p=12):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=n) > 1.4).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["y"] = np.where(y == 1, "yes", "no")
    return df


def test_rich_metrics_and_methods_classification():
    df = _imbalanced()
    cfg, res, ev = _run(df, task="classification", target="y",
                        models=[ModelSpec(id="logreg"), ModelSpec(id="rf", tuning="custom", params={"n_estimators": 30})],
                        preprocess=Preprocess(balancing=["class_weight", "smote", "oversample"], selection=["kbest", "tree"]))
    meta = res["meta"]["methods"]
    assert meta["balancing"]["chosen"] in ("class_weight", "smote", "oversample")
    assert meta["selection"]["chosen"] in ("kbest", "tree") and len(meta["selection"]["tried"]) == 2
    assert res["meta"]["methods_count"] == 5
    m = res["models"]["logreg"]
    assert m["status"] == "done"
    assert m["cv"]["k"] == 3 and len(m["cv"]["folds"]) == 3
    assert m["report"]["rows"][-1]["label"] == "weighted avg"
    assert m["roc"][0]["auc"] > 0.5 and m["roc"][0]["fpr"][0] == 0
    assert "sampler" in res["codes"]["prep"] or "class_weight" in res["codes"]["prep"]


def test_regression_plot_and_selection():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 8))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["t"] = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=400)
    cfg, res, _ = _run(df, task="regression", target="t", models=[ModelSpec(id="ridge"), ModelSpec(id="rf", tuning="custom", params={"n_estimators": 20})],
                       preprocess=Preprocess(selection=["lasso", "mutual_info", "rfe"], balancing=["smote"]))
    r = res["models"]["ridge"]
    assert r["metrics"].keys() >= {"r2", "rmse", "mae"}
    assert len(r["reg_plot"]["actual"]) == len(r["reg_plot"]["pred"]) > 50
    assert r["cv"]["metric"] == "R²"
    assert res["meta"]["methods"]["selection"]["chosen"]
    assert "balancing" not in res["meta"]["methods"]  # regression: balancing skipped


def test_user_grid_search():
    df = _imbalanced()
    cfg, res, _ = _run(df, task="classification", target="y",
                       models=[ModelSpec(id="knn", tuning="grid_user", params={"n_neighbors": [3, 7, 99999, "x"], "weights": ["uniform", "evil"]})])
    k = res["models"]["knn"]
    assert k["tuning"] == "grid_user"
    assert k["user_grid"]["n_neighbors"] == [3, 7, 50]  # clamped; junk dropped
    assert k["best_params"]["n_neighbors"] in (3, 7, 50)
    assert "GridSearchCV" in k["code"]


def test_user_grid_limit():
    with pytest.raises(PipelineError):
        clean_user_grid("rf", "classification", {"n_estimators": [10, 20, 30, 40, 50, 60], "max_depth": [1, 2, 3, 4, 5, 6],
                                                  "min_samples_leaf": [1, 2, 3, 4, 5, 6]})


def test_advice():
    df = _imbalanced()
    a = mth.advise(df, "y", "classification")
    assert a["balancing"]["required"] and "class_weight" in a["balancing"]["recommended"]
    assert mth.advise(df, "f0", "regression")["balancing"]["required"] is False


# ------------------------------------------------------------------ API: methods pricing, advice, multi-type
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from conftest import auth  # noqa: E402
from test_e2e import stream  # noqa: E402

client = TestClient(app)
U = "multi@example.com"


def test_quote_methods():
    r = client.post("/api/quote", json={"models": 2, "methods": 5}, headers=auth(U)).json()
    assert r["paid_methods"] == 2 and r["total"] == 4 and r["needs_payment"]
    assert client.post("/api/quote", json={"models": 2, "methods": 3}, headers=auth(U)).json()["total"] == 0


def test_method_advice_endpoint():
    ds = client.post("/api/datasets/select", json={"id": "builtin:iris"}, headers=auth(U)).json()
    r = client.post("/api/datasets/method-advice", json={"dataset_id": ds["dataset_id"], "target": "species", "task": "classification"}, headers=auth(U))
    assert r.status_code == 200 and "balancing" in r.json() and r.json()["free_methods"] == 3


def test_both_types_become_sub_chats():
    ds = client.post("/api/datasets/select", json={"id": "builtin:housing"}, headers=auth(U)).json()
    tgt = ds["suggested_target"]
    j1 = client.post("/api/jobs", json={"dataset_id": ds["dataset_id"], "task": "regression", "domain": "Real estate", "target": tgt,
                                        "models": [{"id": "ridge"}]}, headers=auth(U)).json()
    stream(U, j1["job_id"])
    cid = j1["project_id"]
    a = client.post(f"/api/chats/{cid}/another", headers=auth(U))
    assert a.status_code == 200, a.text
    main_id = a.json()["main_id"]
    assert a.json()["other_task"] == "classification"
    # same type again is refused
    bad = client.post(f"/api/jobs?parent_id={main_id}", json={"dataset_id": ds["dataset_id"], "task": "regression", "domain": "Real estate",
                                                               "target": tgt, "models": [{"id": "ridge"}]}, headers=auth(U))
    assert bad.status_code == 409
    df_cols = ds["profile"]["columns"]
    cat = next((c["name"] for c in df_cols if c["kind"] in ("categorical", "boolean")), None)
    if cat:
        j2 = client.post(f"/api/jobs?parent_id={main_id}", json={"dataset_id": ds["dataset_id"], "task": "classification", "domain": "Real estate",
                                                                  "target": cat, "models": [{"id": "logreg"}]}, headers=auth(U))
        assert j2.status_code == 200, j2.text
        stream(U, j2.json()["job_id"])
        runs = client.get(f"/api/chats/{main_id}/runs", headers=auth(U)).json()
        assert [r["task"] for r in runs["runs"]] == ["regression", "classification"]
        assert all(r["parent_id"] == main_id for r in runs["runs"])
    assert client.delete(f"/api/chats/{main_id}", headers=auth(U)).status_code == 200
    assert client.get(f"/api/chats/{cid}", headers=auth(U)).status_code == 404


# ------------------------------------------------------------------ multi-format uploads and the fast loader
import io  # noqa: E402
import json as _json  # noqa: E402

from app import fastload  # noqa: E402


def _frame(n=60):
    rng = np.random.default_rng(3)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.integers(0, 5, n), "c": rng.choice(["x", "y"], n)})


@pytest.mark.parametrize("name,maker", [
    ("d.csv", lambda df: df.to_csv(index=False).encode()),
    ("d.tsv", lambda df: df.to_csv(index=False, sep="\t").encode()),
    ("d.parquet", lambda df: (lambda b: (df.to_parquet(b), b.getvalue())[1])(io.BytesIO())),
    ("d.xlsx", lambda df: (lambda b: (df.to_excel(b, index=False), b.getvalue())[1])(io.BytesIO())),
    ("d.json", lambda df: df.to_json(orient="records").encode()),
    ("d.jsonl", lambda df: df.to_json(orient="records", lines=True).encode()),
    ("d.json", lambda df: _json.dumps({"data": df.to_dict("records")}).encode()),
])
def test_parse_any_formats(name, maker):
    out = fastload.parse_any(maker(_frame()), name)
    assert out.shape == (60, 3) and list(out.columns) == ["a", "b", "c"]


def test_upload_endpoint_accepts_xlsx_and_rejects_xls():
    b = io.BytesIO()
    _frame().to_excel(b, index=False)
    r = client.post("/api/datasets/upload", files={"file": ("sales.xlsx", b.getvalue())}, headers=auth(U))
    assert r.status_code == 200 and r.json()["profile"]["columns"]
    bad = client.post("/api/datasets/upload", files={"file": ("old.xls", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"0" * 50)}, headers=auth(U))
    assert bad.status_code == 415


def test_cache_makes_second_load_instant(monkeypatch):
    df = pd.DataFrame({f"c{i}": range(60_000) for i in range(32)})
    buf = io.BytesIO()
    df.to_parquet(buf)
    calls = {"n": 0}

    def fake(url, headers=None):
        calls["n"] += 1
        return pd.read_parquet(io.BytesIO(buf.getvalue()))

    monkeypatch.setattr(d, "_hf_parquet_urls", lambda did: ["https://huggingface.co/x.parquet"])
    monkeypatch.setattr(fastload, "parquet_range", fake)
    monkeypatch.setattr(d.httpx, "get", lambda *a, **k: type("R", (), {"json": lambda self: {"siblings": []}})())
    a = d.load_external("hf:owner/cached")[0]
    b = d.load_external("hf:owner/cached")[0]
    assert a.shape == b.shape == (60_000, 32) and calls["n"] == 1


def test_range_reader_reads_only_what_it_needs(monkeypatch):
    """parquet_range must fetch the footer and the first row groups only, not the whole file."""
    import http.server
    import threading
    import zipfile

    big = pd.DataFrame({f"c{i}": np.random.default_rng(i).normal(size=200_000) for i in range(10)})
    pq_buf = io.BytesIO()
    big.to_parquet(pq_buf, row_group_size=20_000)
    payload = pq_buf.getvalue()
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("big.csv", big.head(5000).to_csv(index=False))
    blobs = {"/f.parquet": payload, "/f.zip": zbuf.getvalue()}
    served = {"bytes": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            data = blobs[self.path]
            rng = self.headers.get("Range")
            if rng:
                a, b = rng.split("=")[1].split("-")
                a, b = int(a), min(int(b), len(data) - 1)
                chunk = data[a:b + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
            else:
                chunk = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            served["bytes"] += len(chunk)
            self.wfile.write(chunk)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setattr(d, "validate_public_url", lambda u: u)
    monkeypatch.setattr(d, "MAX_LOAD_ROWS", 40_000)
    try:
        out = fastload.parquet_range(base + "/f.parquet")
        assert len(out) == 40_000 and out.shape[1] == 10
        assert served["bytes"] < len(payload) * 0.6, (served["bytes"], len(payload))
        served["bytes"] = 0
        raw = fastload.zip_csv_range(base + "/f.zip", None, None)
        assert raw.count(b"\n") >= 5000 and raw.endswith(b"\n")
    finally:
        srv.shutdown()


def test_preview_eda_and_session_endpoints():
    ds = client.post("/api/datasets/select", json={"id": "builtin:credit-risk"}, headers=auth(U)).json()
    tgt = ds["suggested_target"]
    e = client.post("/api/datasets/eda", json={"dataset_id": ds["dataset_id"], "target": tgt, "task": "classification"}, headers=auth(U))
    assert e.status_code == 200 and "overview" in e.json() and "inference" in e.json()
    s = client.get(f"/api/datasets/{ds['dataset_id']}/session", headers=auth(U))
    assert s.status_code == 200 and s.json()["profile"]["rows"] == ds["profile"]["rows"]
    assert client.get(f"/api/datasets/{ds['dataset_id']}/session", headers=auth("other@example.com")).status_code == 404


def test_llm_deadline_stops_a_chain_of_slow_providers():
    import time as _t
    from app.llm_router import AllProvidersFailed, LLMRouter

    groups = [{"group": "g", "models": [{"base_url": f"http://x{i}", "model": "m", "api_key": f"k{i}", "provider": "openai"} for i in range(6)]}]
    calls = []

    def slow(base, key, payload, timeout):
        calls.append(key)
        _t.sleep(0.3)
        raise RuntimeError("down")

    r = LLMRouter(groups, transport=slow)
    t0 = _t.monotonic()
    with pytest.raises(AllProvidersFailed):
        r.complete([{"role": "user", "content": "hi"}], deadline=0.5)
    assert _t.monotonic() - t0 < 1.5 and len(calls) < 6


def test_busy_engine_refuses_new_runs_before_charging(monkeypatch):
    u = "busy@example.com"
    ds = client.post("/api/datasets/select", json={"id": "builtin:iris"}, headers=auth(u)).json()
    before = client.get("/api/me", headers=auth(u)).json()["usage"]["free_used"]
    monkeypatch.setenv("MAX_QUEUED_JOBS", "0")
    r = client.post("/api/jobs", json={"dataset_id": ds["dataset_id"], "task": "classification", "domain": "Other", "target": "species",
                                       "models": [{"id": "logreg"}]}, headers=auth(u))
    assert r.status_code == 503 and "busy" in r.text
    assert client.get("/api/me", headers=auth(u)).json()["usage"]["free_used"] == before  # nothing was reserved


def test_job_view_reports_queue_position():
    u = "queue@example.com"
    ds = client.post("/api/datasets/select", json={"id": "builtin:iris"}, headers=auth(u)).json()
    j = client.post("/api/jobs", json={"dataset_id": ds["dataset_id"], "task": "classification", "domain": "Other", "target": "species",
                                       "models": [{"id": "logreg"}]}, headers=auth(u)).json()
    stream(u, j["job_id"])
    assert client.get(f"/api/jobs/{j['job_id']}", headers=auth(u)).json()["queue_ahead"] == 0
