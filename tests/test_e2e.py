"""End-to-end API tests: auth, free run, streaming, paid gating, payment claim/approve/reject, VIP, reports."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from app import notify
from app.main import app
from app.store import get_store
from conftest import auth

client = TestClient(app)
USER, FRIEND, ADMIN = "karthi@example.com", "friend@example.com", "admin@example.com"


def select(email, dataset="builtin:iris"):
    r = client.post("/api/datasets/select", json={"id": dataset}, headers=auth(email))
    assert r.status_code == 200, r.text
    return r.json()


def job_body(ds_id, models, **kw):
    body = {"dataset_id": ds_id, "task": "classification", "domain": "Education", "target": "species", "eda": True,
            "models": [{"id": m} for m in models]}
    body.update(kw)
    return body


def run_job(email, models, dataset="builtin:iris", target="species", **kw):
    ds = select(email, dataset)
    r = client.post("/api/jobs", json=job_body(ds["dataset_id"], models, target=target, **kw), headers=auth(email))
    assert r.status_code == 200, r.text
    job = r.json()
    events = stream(email, job["job_id"])
    return job, events


def stream(email, job_id):
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events", headers=auth(email)) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                events.append(ev)
                if ev["type"] == "job_end":
                    break
    return events


def seed_usage(email, **fields):
    me = client.get("/api/me", headers=auth(email)).json()
    st = get_store()
    doc = st.get_user(me["uid"])
    doc.update(fields)
    st.put_user(me["uid"], doc)


def test_auth_required_and_dev_tokens_validated():
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    r = client.get("/api/me", headers=auth(USER))
    assert r.status_code == 200 and not r.json()["is_admin"] and r.json()["usage"]["free_remaining"] == 10
    assert client.get("/api/me", headers=auth(ADMIN)).json()["is_admin"] is True


def test_security_headers_present():
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"


def test_dataset_suggestions_and_profile():
    r = client.get("/api/datasets/suggest", params={"domain": "Finance", "task": "classification", "live": "false",
                                                    "include_small": "true"}, headers=auth(USER))
    items = r.json()["items"]
    assert items[0]["id"] == "builtin:big-fraud" and items[0]["match"] is True  # large sets first
    small = next(i for i in items if i["id"] == "builtin:credit-risk")
    assert small["rows"] == 5000 and small["cols"] == 12
    ds = select(USER, "builtin:credit-risk")
    assert ds["suggested_target"] == "defaulted" and ds["datetime_columns"] == ["application_date"]


def test_models_endpoint_orders_by_domain():
    r = client.get("/api/models", params={"task": "classification", "domain": "Finance"}, headers=auth(USER)).json()["models"]
    assert r[0]["id"] == "logreg" and r[0]["recommended"] and any(m["id"] == "knn" and m["params"] for m in r)


def test_free_run_streams_steps_and_updates_usage():
    job, events = run_job(USER, ["logreg", "knn"])
    kinds = [e["type"] for e in events]
    assert kinds[0] == "job_start" and kinds[-1] == "job_end" and events[-1]["status"] == "done"
    assert "eda" in kinds and kinds.count("model_result") == 2
    assert not job["needs_payment"]
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert view["status"] == "done" and view["verdict"]["winner"] and all("metrics" in m for m in view["models"])
    assert view["script"].count("model = Pipeline") == 2 and "GridSearchCV" not in view["script"]
    assert [s["id"] for s in view["steps"]][:4] == ["load", "eda", "prep", "split"]
    usage = client.get("/api/usage", headers=auth(USER)).json()
    assert usage["free_used"] == 2 and usage["resets_at"] is not None


def test_dataset_is_held_briefly_after_run_then_wiped_and_isolated_between_users():
    ds = select(USER)
    r = client.post("/api/jobs", json=job_body(ds["dataset_id"], ["nb"]), headers=auth(FRIEND))
    assert r.status_code == 404  # someone else's dataset
    r = client.post("/api/jobs", json=job_body(ds["dataset_id"], ["nb"]), headers=auth(USER))
    stream(USER, r.json()["job_id"])
    from app.datasets import datasets as store
    store.get(ds["dataset_id"], client.get("/api/me", headers=auth(USER)).json()["uid"])  # still held (for the other ML type)
    store._items[ds["dataset_id"]].created -= 10_000  # ...but only for DATASET_TTL_SECONDS
    again = client.post("/api/jobs", json=job_body(ds["dataset_id"], ["nb"]), headers=auth(USER))
    assert again.status_code == 404
    # another user's job is not readable
    assert client.get(f"/api/jobs/{r.json()['job_id']}", headers=auth(FRIEND)).status_code == 404


def test_validation_errors():
    ds = select(USER)
    h = auth(USER)
    assert client.post("/api/jobs", json=job_body(ds["dataset_id"], ["logreg"], target="nope"), headers=h).status_code == 422
    assert client.post("/api/jobs", json=job_body(ds["dataset_id"], ["logreg", "logreg"]), headers=h).status_code == 422
    assert client.post("/api/jobs", json=job_body(ds["dataset_id"], ["linreg"]), headers=h).status_code == 422  # regression-only
    assert client.post("/api/jobs", json=job_body(ds["dataset_id"], []), headers=h).status_code == 422
    assert client.post("/api/jobs", json=job_body(ds["dataset_id"], ["logreg"], cv_folds=99), headers=h).status_code == 422
    assert client.get("/api/usage", headers=h).json()["free_used"] == 0  # nothing reserved by failed requests


def test_daily_limit_returns_429_with_reset_time():
    seed_usage(USER, free_used=10, paid_used=15, window_start=time.time() - 60)
    ds = select(USER)
    r = client.post("/api/jobs", json=job_body(ds["dataset_id"], ["logreg"]), headers=auth(USER))
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert detail["resets_at"] > time.time() and "limit" in detail["message"].lower()


def test_failed_run_refunds_quota():
    ds = select(USER, "builtin:housing")
    body = job_body(ds["dataset_id"], ["logreg"], target="price_lakhs")  # continuous target, classification task
    job = client.post("/api/jobs", json=body, headers=auth(USER)).json()
    events = stream(USER, job["job_id"])
    assert events[-1]["status"] == "error"
    assert client.get("/api/usage", headers=auth(USER)).json()["free_used"] == 0


def test_upload_csv_and_ssrf_rejected():
    rows = "\n".join(f"{i},{i % 3},{'a' if i % 2 else 'b'},{(i * 7) % 5}" for i in range(80))
    r = client.post("/api/datasets/upload", files={"file": ("my data.csv", ("x1,y,grp,label\n" + rows).encode(), "text/csv")},
                    headers=auth(USER))
    assert r.status_code == 200 and r.json()["profile"]["rows"] == 80 and r.json()["name"] == "my data.csv"
    for url in ("http://169.254.169.254/latest/meta-data", "https://localhost/x.csv", "https://evil.example.com/a.csv"):
        assert client.post("/api/datasets/from-url", json={"url": url}, headers=auth(USER)).status_code == 400


def test_paid_models_are_locked_until_payment_claimed_then_admin_decides():
    seed_usage(USER, free_used=9, window_start=time.time() - 60)  # 1 free model left
    job, events = run_job(USER, ["logreg", "knn", "tree"])
    assert job["needs_payment"] and job["paid_models"] == ["knn", "tree"]

    # server withholds paid results: both in the stream and in the final view
    for ev in events:
        if ev["type"] == "model_result" and ev["model_id"] in ("knn", "tree"):
            assert ev["result"].get("locked") and "metrics" not in ev["result"]
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    by_id = {m["id"]: m for m in view["models"]}
    assert "metrics" in by_id["logreg"] and by_id["knn"]["locked"] and "metrics" not in by_id["knn"]
    assert all(not s["logs"] and s["code"] is None for s in view["steps"] if s["model_id"] == "knn")
    assert "KNeighbors" not in view["script"] and view["locks"]["locked_models"] == 2

    # quiz: server-timed, answers never revealed; score feeds the discount
    q = client.post("/api/quiz/start", json={"job_id": job["job_id"]}, headers=auth(USER)).json()
    assert q["total"] == 7 and q["seconds_left"] > 70 and "answer" not in json.dumps(q).lower()
    out = client.post(f"/api/quiz/{q['quiz_id']}/answer", json={"qid": q["qid"], "option_id": "A"}, headers=auth(USER)).json()
    assert set(out) >= {"correct", "timed_out", "finished", "score"}
    assert client.post(f"/api/quiz/{q['quiz_id']}/answer", json={"qid": q["qid"], "option_id": "A"}, headers=auth(USER)).status_code == 409

    quote = client.get(f"/api/jobs/{job['job_id']}/quote", headers=auth(USER)).json()
    assert quote["paid_models"] == 2 and quote["needs_payment"]
    pay = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    assert pay["amount"] == quote["total"] and pay["ref"].startswith("AML-") and pay["status"] == "created"
    assert pay["upi_uri"].startswith("upi://pay?pa=test@upi") and f"am={pay['amount']}.00" in pay["upi_uri"] and pay["ref"] in pay["upi_uri"]
    again = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    assert again["id"] == pay["id"]  # idempotent

    notify.OUTBOX.clear()
    claim = client.post(f"/api/payments/{pay['id']}/claim", headers=auth(USER)).json()
    assert claim["payment"]["status"] == "claimed" and "sent to the admin" in claim["chat_message"]
    assert len(notify.OUTBOX) == 1
    mail = notify.OUTBOX[0]
    assert pay["ref"] in mail["subject"] and f"/admin?tab=payments&payment={pay['id']}" in mail["text"] and USER in mail["text"]

    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()  # provisionally unlocked
    assert all("metrics" in m for m in view["models"]) and view["locks"]["unlocked"]
    assert "KNeighbors" in view["script"]

    # only admins can decide
    assert client.post(f"/api/admin/payments/{pay['id']}/decision", json={"decision": "approve"}, headers=auth(USER)).status_code == 403
    queue = client.get("/api/admin/payments", headers=auth(ADMIN)).json()["payments"]
    assert queue[0]["id"] == pay["id"] and queue[0]["status"] == "claimed" and "uid" not in queue[0]
    ok = client.post(f"/api/admin/payments/{pay['id']}/decision", json={"decision": "approve"}, headers=auth(ADMIN))
    assert ok.json()["payment"]["status"] == "approved"
    assert client.post(f"/api/admin/payments/{pay['id']}/decision", json={"decision": "reject"}, headers=auth(ADMIN)).status_code == 409
    assert client.get("/api/usage", headers=auth(USER)).json()["verified_spend"] == pay["amount"]

    summary = client.get("/api/admin/summary", headers=auth(ADMIN)).json()
    assert summary["totals"]["revenue"] == pay["amount"] and summary["totals"]["pending_count"] == 0
    assert summary["models"] and summary["domains"][0]["name"] == "Education" and len(summary["revenue_by_day"]) == 14


def test_rejected_payment_relocks_results_and_blocks_paid_features():
    seed_usage(USER, free_used=10, window_start=time.time() - 60)
    job, _ = run_job(USER, ["logreg"])
    pay = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    client.post(f"/api/payments/{pay['id']}/claim", headers=auth(USER))
    client.post(f"/api/admin/payments/{pay['id']}/decision", json={"decision": "reject"}, headers=auth(ADMIN))
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert view["models"][0]["locked"] and view["locks"]["rejected"]
    usage = client.get("/api/usage", headers=auth(USER)).json()
    assert usage["paid_blocked"] is True
    ds = select(USER)
    r = client.post("/api/jobs", json=job_body(ds["dataset_id"], ["nb"]), headers=auth(USER))
    assert r.status_code == 429 and r.json()["detail"]["code"] == "paid_blocked"


def test_free_inside_tier_needs_no_payment_but_video_fee_does():
    job, _ = run_job(USER, ["logreg"], videos=True)
    assert job["needs_payment"] and job["paid_models"] == []
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert view["locks"]["videos_locked"] and view["lectures"] == []
    pay = client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(USER)).json()
    assert pay["amount"] == 2
    client.post(f"/api/payments/{pay['id']}/claim", headers=auth(USER))
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(USER)).json()
    assert view["lectures"][0]["model_id"] == "logreg" and "youtube.com/results" in view["lectures"][0]["search_url"]


def test_team_surcharge_and_quote_preview():
    q = client.post("/api/quote", json={"models": 12, "videos": True, "team": True}, headers=auth(USER)).json()
    assert q["paid_models"] == 2 and q["gross"] == 29.4 and q["total"] == 29 and q["within_cap"] is True  # (4 + 24) + 5%
    q = client.post("/api/quote", json={"models": 12, "videos": True, "team": False}, headers=auth(USER)).json()
    assert q["total"] == 4 + 24  # 2 paid models + 12 videos
    assert client.post("/api/quote", json={"models": 13}, headers=auth(USER)).status_code == 422  # max 12 per run


def test_vip_via_admin_gets_unlimited_free_access():
    assert client.get("/api/admin/vip", headers=auth(USER)).status_code == 403
    r = client.post("/api/admin/vip", json={"email": "Friend@Example.com "}, headers=auth(ADMIN))
    assert r.json()["emails"] == [FRIEND]
    assert client.post("/api/admin/vip", json={"email": "not-an-email"}, headers=auth(ADMIN)).status_code == 422
    assert client.get("/api/me", headers=auth(FRIEND)).json()["is_vip"] is True
    job, _ = run_job(FRIEND, ["logreg", "knn", "tree", "rf", "svm", "gb", "nb", "ada", "extra", "mlp"], videos=True)
    assert not job["needs_payment"]
    view = client.get(f"/api/jobs/{job['job_id']}", headers=auth(FRIEND)).json()
    assert all("metrics" in m for m in view["models"]) and view["locks"]["unlocked"]
    assert client.post("/api/payments", json={"kind": "models", "job_id": job["job_id"]}, headers=auth(FRIEND)).status_code == 400
    assert client.get("/api/usage", headers=auth(FRIEND)).json()["unlimited"] is True
    client.delete("/api/admin/vip", params={"email": FRIEND}, headers=auth(ADMIN))
    assert client.get("/api/me", headers=auth(FRIEND)).json()["is_vip"] is False


def test_reports_first_two_free_then_five_rupees():
    ids = []
    for _ in range(3):
        job, _ = run_job(USER, ["nb"])
        ids.append(job["job_id"])
    assert client.post(f"/api/reports/{ids[0]}/access", headers=auth(USER)).status_code == 200
    assert client.post(f"/api/reports/{ids[0]}/access", headers=auth(USER)).status_code == 200  # same job: no extra charge
    rep = client.post(f"/api/reports/{ids[1]}/access", headers=auth(USER)).json()
    assert rep["verdict"]["winner"] and rep["narrative"]["source"] == "rules" and rep["eda"]
    r = client.post(f"/api/reports/{ids[2]}/access", headers=auth(USER))
    assert r.status_code == 402 and r.json()["detail"]["price"] == 5
    pay = client.post("/api/payments", json={"kind": "report", "job_id": ids[2]}, headers=auth(USER)).json()
    assert pay["amount"] == 5
    client.post(f"/api/payments/{pay['id']}/claim", headers=auth(USER))
    assert client.post(f"/api/reports/{ids[2]}/access", headers=auth(USER)).status_code == 200


def test_projects_feedback_and_admin_views():
    job, _ = run_job(USER, ["logreg"])
    time.sleep(0.2)
    projects = client.get("/api/projects", headers=auth(USER)).json()["projects"]
    assert projects and projects[0]["winner"]["id"] == "logreg" and projects[0]["domain"] == "Education"
    assert client.post("/api/feedback", json={"rating": 5, "text": "Great"}, headers=auth(USER)).status_code == 200
    assert client.post("/api/feedback", json={"rating": 9}, headers=auth(USER)).status_code == 422
    fb = client.get("/api/admin/feedback", headers=auth(ADMIN)).json()["feedback"]
    assert fb[0]["rating"] == 5
    users = client.get("/api/admin/users", headers=auth(ADMIN)).json()["users"]
    assert any(u["email"] == USER for u in users)
    proj = client.get(f"/api/admin/projects/{projects[0]['id']}", headers=auth(ADMIN)).json()
    assert "uid" not in proj and proj["dataset"]
    assert client.get("/api/admin/summary", headers=auth(USER)).status_code == 403
