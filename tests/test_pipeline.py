import pytest

from app import datasets as d
from app.pipeline import JobConfig, ModelSpec, PipelineError, Preprocess, make_verdict, run_pipeline


def run(builtin: str, **kw):
    df, item = d.load_builtin(builtin)
    kw.setdefault("target", item.default_target)
    kw.setdefault("task", item.tasks[0])
    cfg = JobConfig(dataset_id="x", domain=item.domains[0], **kw)
    events = []
    res = run_pipeline(df, cfg, item.name, events.append, max_rows=50_000, max_seconds=120)
    return cfg, res, events


def test_classification_with_tuning_selection_and_pca():
    cfg, res, events = run(
        "builtin:credit-risk",
        models=[ModelSpec(id="logreg"), ModelSpec(id="knn", tuning="grid"),
                ModelSpec(id="rf", tuning="custom", params={"n_estimators": 60, "max_depth": 6, "evil": "__import__('os')"})],
        preprocess=Preprocess(scale="robust", feature_selection="kbest", k_best=8, reduction="pca"),
    )
    assert set(res["models"]) == {"logreg", "knn", "rf"}
    for m in res["models"].values():
        assert m["status"] == "done"
        assert 0.5 < m["metrics"]["accuracy"] <= 1.0
        assert m["metrics"]["roc_auc"] is not None
        assert m["confusion"]["matrix"]
    knn = res["models"]["knn"]
    assert knn["best_params"]["n_neighbors"] in (1, 3, 5, 7, 9, 11, 15, 21)
    assert len(knn["curve"]) == 8 and knn["grid_top"]
    assert "evil" not in res["models"]["rf"]["params"]  # unknown params never reach sklearn
    assert res["models"]["rf"]["params"]["n_estimators"] == 60
    # step events for the live "running code" view
    kinds = {e["type"] for e in events}
    assert {"step_start", "log", "step_end", "eda", "model_result"} <= kinds
    ids = [e["id"] for e in events if e["type"] == "step_start"]
    assert ids == ["load", "eda", "prep", "split", "train:logreg", "train:knn", "train:rf"]
    # the code shown is the code that ran
    assert "GridSearchCV" in res["codes"]["models"]["knn"]
    assert "RandomForestClassifier(random_state=42, n_jobs=1, n_estimators=60, max_depth=6)" in res["codes"]["models"]["rf"]
    assert "SelectKBest" in res["codes"]["prep"] and "PCA" in res["codes"]["prep"]


def test_regression_metrics_and_verdict():
    cfg, res, _ = run("builtin:housing", models=[ModelSpec(id="ridge", tuning="grid"), ModelSpec(id="gb"), ModelSpec(id="linreg")])
    for m in res["models"].values():
        assert m["metrics"]["r2"] > 0.5 and m["metrics"]["rmse"] > 0
    v = make_verdict("regression", "Real estate", list(res["models"].values()))
    assert v["winner"]["id"] in {"ridge", "gb", "linreg"} and len(v["ranking"]) == 3
    assert v["ranking"][0]["score"] >= v["ranking"][1]["score"]


def test_chronological_split_uses_latest_rows_as_test():
    cfg, res, events = run("builtin:churn", models=[ModelSpec(id="logreg")], split="chronological", time_column="signup_date")
    assert res["meta"]["test_rows"] == 800
    assert any("Chronological" in e.get("text", "") for e in events if e["type"] == "log")
    assert {x["column"] for x in res["meta"]["dropped"]} == {"signup_date"}


def test_eda_is_json_ready_and_has_inference():
    cfg, res, _ = run("builtin:credit-risk", models=[ModelSpec(id="nb")])
    import json
    json.dumps(res["eda"])  # no NaN/numpy leaks
    eda = res["eda"]
    assert eda["overview"]["rows"] == 5000 and eda["missing"] and eda["histograms"] and eda["correlation"]["columns"]
    assert eda["target"]["classes"] and any(i["level"] in ("warn", "info", "good") for i in eda["inference"])


def test_drop_rows_imputation_and_drop_columns():
    cfg, res, events = run("builtin:credit-risk", models=[ModelSpec(id="tree")],
                           preprocess=Preprocess(impute="drop_rows", drop_columns=["loan_purpose"]))
    logs = " ".join(e.get("text", "") for e in events if e["type"] == "log")
    assert "containing missing values" in logs and "removed by you" in logs


def test_bad_target_and_wrong_task_fail_cleanly():
    df, item = d.load_builtin("builtin:housing")
    cfg = JobConfig(dataset_id="x", task="classification", target="price_lakhs", models=[ModelSpec(id="rf")])
    with pytest.raises(PipelineError):
        run_pipeline(df, cfg, "h", lambda e: None, 50_000, 60)
    cfg = JobConfig(dataset_id="x", task="regression", target="nope", models=[ModelSpec(id="rf")])
    with pytest.raises(PipelineError):
        run_pipeline(df, cfg, "h", lambda e: None, 50_000, 60)
    cfg = JobConfig(dataset_id="x", task="regression", target="price_lakhs", models=[ModelSpec(id="nb")])
    with pytest.raises(PipelineError):
        run_pipeline(df, cfg, "h", lambda e: None, 50_000, 60)


def test_time_limit_skips_remaining_models():
    df, item = d.load_builtin("builtin:iris")
    cfg = JobConfig(dataset_id="x", task="classification", target="species", eda=False,
                    models=[ModelSpec(id="logreg"), ModelSpec(id="knn")])
    res = run_pipeline(df, cfg, "iris", lambda e: None, 50_000, max_seconds=-1)
    assert all(m["status"] == "skipped" for m in res["models"].values())


def test_large_data_is_sampled():
    df, item = d.load_builtin("builtin:credit-risk")
    cfg = JobConfig(dataset_id="x", task="classification", target="defaulted", eda=False, models=[ModelSpec(id="tree")])
    res = run_pipeline(df, cfg, "c", lambda e: None, max_rows=1000, max_seconds=60)
    assert res["meta"]["train_rows"] + res["meta"]["test_rows"] <= 1000
