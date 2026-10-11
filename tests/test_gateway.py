import json, time
import httpx, pytest
from app.auth import gateway_signature
from app.store import RemoteStore, StoreError

SECRET = "s" * 40


def test_gateway_signature_is_stable_and_binds_fields():
    a = gateway_signature(SECRET, "123", "u1", "a@b.co", "1", "0")
    assert a == gateway_signature(SECRET, "123", "u1", "a@b.co", "1", "0")
    assert a != gateway_signature(SECRET, "123", "u1", "a@b.co", "1", "1")
    assert a != gateway_signature(SECRET, "123", "u2", "a@b.co", "1", "0")


def _store(handler):
    return RemoteStore("https://w.example", SECRET, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_remote_store_signs_and_roundtrips_bytes():
    seen = {}

    def h(req: httpx.Request):
        import hashlib, hmac
        body = req.content.decode()
        ts = req.headers["x-store-ts"]
        exp = hmac.new(SECRET.encode(), f"{ts}.{hashlib.sha256(body.encode()).hexdigest()}".encode(), "sha256").hexdigest()
        assert req.headers["x-store-sig"] == exp and abs(time.time() - int(ts)) < 5
        seen.update(json.loads(body))
        return httpx.Response(200, json={"result": {"blob": {"__b64__": "AQID"}, "n": 1}})

    st = _store(h)
    out = st.get_archive("job1")
    assert out["blob"] == b"\x01\x02\x03"
    st.put_archive("job1", {"blob": b"\x01\x02\x03", "x": float("nan")})
    assert seen["args"][1] == {"blob": {"__b64__": "AQID"}, "x": None}


def test_remote_store_retries_reads_not_creates_and_maps_errors():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        return httpx.Response(503, json={}) if calls["n"] < 2 else httpx.Response(200, json={"result": ["A@B.co"]})

    assert _store(flaky).get_vip() == {"a@b.co"}
    calls["n"] = 0
    with pytest.raises(StoreError):
        _store(lambda r: (calls.__setitem__("n", calls["n"] + 1), httpx.Response(503, json={}))[1]).create_payment({"id": "p"})
    assert calls["n"] == 1
    with pytest.raises(StoreError) as e:
        _store(lambda r: httpx.Response(409, json={"detail": {"message": "Already exists"}})).create_payment({"id": "p"})
    assert e.value.status == 409
