import socket

import pytest

from app import datasets as d
from app.datasets import DatasetError
from app.llm_router import AllProvidersFailed, LLMRouter, narrate, set_router


def fake_resolver(ip):
    return lambda host, port, proto=0: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]


def test_rejects_http_and_unknown_hosts():
    with pytest.raises(DatasetError):
        d.validate_public_url("http://raw.githubusercontent.com/x.csv", fake_resolver("140.82.1.1"))
    with pytest.raises(DatasetError):
        d.validate_public_url("https://evil.example.com/x.csv", fake_resolver("93.184.216.34"))


def test_rejects_private_and_loopback_addresses_even_on_allowed_hosts():
    for ip in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "192.168.1.9"):
        with pytest.raises(DatasetError):
            d.validate_public_url("https://raw.githubusercontent.com/x.csv", fake_resolver(ip))


def test_accepts_public_allowed_host():
    url = "https://raw.githubusercontent.com/o/r/main/x.csv"
    assert d.validate_public_url(url, fake_resolver("185.199.108.133")) == url


def test_rejects_credentials_in_url():
    with pytest.raises(DatasetError):
        d.validate_public_url("https://user:pw@huggingface.co/x.csv", fake_resolver("1.2.3.4"))


def test_normalizes_drive_and_github_links():
    assert d.normalize_url("https://drive.google.com/file/d/ABC123/view?usp=sharing") == \
        "https://drive.google.com/uc?export=download&id=ABC123"
    assert d.normalize_url("https://github.com/o/r/blob/main/data/x.csv") == \
        "https://raw.githubusercontent.com/o/r/main/data/x.csv"


def test_parse_csv_detects_semicolons_and_dates():
    rows = "\n".join(f"2024-01-{(i % 28) + 1:02d};{i};{i * 2}" for i in range(40))
    df = d.parse_csv(("order_date;a;b\n" + rows).encode())
    assert list(df.columns) == ["order_date", "a", "b"]
    assert str(df["order_date"].dtype).startswith("datetime64")


def test_parse_csv_rejects_tiny_files():
    with pytest.raises(DatasetError):
        d.parse_csv(b"a,b\n1,2\n")


# ------------------------------------------------------------------ LLM router
GROUPS = [
    {"group": "g1", "models": [
        {"base_url": "https://a", "model": "primary-1", "api_key_env": "K_A"},
        {"base_url": "https://b", "model": "backup-1", "api_key_env": "K_B"}]},
    {"group": "g2", "models": [{"base_url": "https://c", "model": "primary-2", "api_key_env": "K_C"}]},
]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for k in ("K_A", "K_B", "K_C"):
        monkeypatch.setenv(k, "x")


def test_router_falls_back_to_secondary_then_cools_down_primary():
    calls = []

    def transport(base, key, payload, timeout):
        calls.append(payload["model"])
        if payload["model"] == "primary-1":
            raise RuntimeError("retryable:429")
        return "ok from " + payload["model"]

    r = LLMRouter(GROUPS[:1], transport=transport, clock=lambda: 100.0)
    text, meta = r.complete([{"role": "user", "content": "hi"}])
    assert text == "ok from backup-1" and calls == ["primary-1", "backup-1"]
    calls.clear()
    r.complete([{"role": "user", "content": "hi"}])
    assert calls == ["backup-1"]  # primary is cooling down


def test_router_recovers_after_cooldown():
    now = {"t": 0.0}

    def transport(base, key, payload, timeout):
        if payload["model"] == "primary-1" and now["t"] < 50:
            raise RuntimeError("retryable:500")
        return payload["model"]

    r = LLMRouter(GROUPS[:1], transport=transport, clock=lambda: now["t"])
    assert r.complete([])[0] == "backup-1"
    now["t"] = 100.0
    assert r.complete([])[0] == "primary-1"


def test_router_rotates_groups_and_raises_when_all_fail():
    seen = []
    r = LLMRouter(GROUPS, transport=lambda b, k, p, t: seen.append(p["model"]) or "x", clock=lambda: 0.0)
    r.complete([])
    r.complete([])
    assert seen == ["primary-1", "primary-2"]  # load is spread across groups

    def boom(*a, **k):
        raise RuntimeError("down")

    with pytest.raises(AllProvidersFailed):
        LLMRouter(GROUPS, transport=boom, clock=lambda: 0.0).complete([])


def test_narrate_falls_back_to_rules_and_never_sends_column_names():
    verdict = {"winner": {"id": "rf", "name": "Random Forest", "score": 0.91}, "metric": "R²",
               "ranking": [{"name": "Random Forest", "score": 0.91, "id": "rf", "rank": 1}], "warnings": [],
               "text": ["Random Forest is best."]}
    sent = {}

    def transport(base, key, payload, timeout):
        sent["body"] = str(payload)
        return "A short plain-English summary."

    set_router(LLMRouter(GROUPS[:1], transport=transport))
    out = narrate(verdict, "Finance", "regression")
    assert out["source"] == "llm" and "Random Forest" in sent["body"]
    set_router(LLMRouter(GROUPS[:1], transport=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))))
    assert narrate(verdict, "Finance", "regression")["source"] == "rules"
    set_router(None)
    assert narrate(verdict, "Finance", "regression") == {"text": "Random Forest is best.", "source": "rules"}
