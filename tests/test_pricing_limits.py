import pytest

from app import limits, pricing


def test_free_tier_costs_nothing():
    q = pricing.quote(models=10, free_remaining=10)
    assert q["total"] == 0 and not q["needs_payment"]


def test_models_above_free_tier():
    q = pricing.quote(models=12, free_remaining=10)
    assert q["paid_models"] == 2 and q["total"] == 4


def test_partial_free_remaining():
    q = pricing.quote(models=3, free_remaining=1)
    assert q["paid_models"] == 2 and q["total"] == 4


def test_video_fee_applies_even_inside_free_tier():
    q = pricing.quote(models=3, free_remaining=10, video_models=3)
    assert q["total"] == 6 and q["needs_payment"]


def test_team_surcharge_rounds_half_up():
    q = pricing.quote(models=15, free_remaining=10, team=True)  # 10 + 5% = 10.5
    assert q["gross"] == 10.5 and q["total"] == 11


def test_quiz_discount_per_correct():
    q = pricing.quote(models=15, free_remaining=10, quiz_correct=7)  # 10 - 1.75 = 8.25
    assert q["quiz_discount"] == 1.75 and q["total"] == 8


def test_quiz_discount_capped_at_half_the_bill():
    q = pricing.quote(models=11, free_remaining=10, quiz_correct=7)  # bill 2, cap 1
    assert q["quiz_discount"] == 1.0 and q["total"] == 1


def test_quiz_correct_is_clamped():
    assert pricing.quote(models=15, free_remaining=10, quiz_correct=99)["quiz_correct"] == 7


def test_report_pricing():
    assert pricing.report_quote(0)["total"] == 0
    assert pricing.report_quote(1)["free_left"] == 1
    assert pricing.report_quote(2)["total"] == 5


def test_window_shrinks_with_spend():
    base = limits.window_seconds(0)
    assert base == 20400
    assert limits.window_seconds(50) == base          # must EXCEED 50
    assert limits.window_seconds(51) == int(base * 0.97)
    assert limits.window_seconds(101) == int(base * 0.94)


def test_reserve_free_then_paid_then_cap():
    u = limits.new_user("u", "u@x.com", 0)
    r = limits.reserve(u, 12, now=1000)
    assert r == {"free": 10, "paid": 2, "free_remaining_before": 10}
    r = limits.reserve(u, 13, now=1100)  # 13 more paid = 15 total paid
    assert r["free"] == 0 and r["paid"] == 13
    with pytest.raises(limits.LimitError) as e:
        limits.reserve(u, 1, now=1200)
    assert e.value.resets_at == 1000 + 20400


def test_window_resets_after_expiry():
    u = limits.new_user("u", "u@x.com", 0)
    limits.reserve(u, 10, now=1000)
    snap = limits.snapshot(u, now=1000 + 20400 - 1)
    assert snap["free_remaining"] == 0
    snap = limits.snapshot(u, now=1000 + 20400)
    assert snap["free_remaining"] == 10 and snap["resets_at"] is None


def test_vip_is_unlimited():
    u = limits.new_user("u", "u@x.com", 0)
    for _ in range(5):
        limits.reserve(u, 12, now=1, unlimited=True)
    assert u["free_used"] == 0


def test_paid_blocked_user_cannot_reserve_paid_models():
    u = limits.new_user("u", "u@x.com", 0)
    u["paid_blocked"] = True
    limits.reserve(u, 10, now=1)  # free is fine
    with pytest.raises(limits.LimitError):
        limits.reserve(u, 1, now=2)
