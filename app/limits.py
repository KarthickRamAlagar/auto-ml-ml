"""Per-account usage windows.

A window starts when the user runs their first model and lasts 5h40m. After that the counters reset.
Verified spend above INR 50 shortens the window by 3%, above INR 100 by 6%.
Within a window: 10 free models + 15 paid models = 25 maximum.
"""
from __future__ import annotations

from .pricing import FREE_MODELS, PAID_MODELS

BASE_WINDOW_SECONDS = 5 * 3600 + 40 * 60  # 20,400 s


class LimitError(Exception):
    def __init__(self, message: str, resets_at: float | None = None, code: str = "limit"):
        super().__init__(message)
        self.message = message
        self.resets_at = resets_at
        self.code = code


def new_user(uid: str, email: str, now: float) -> dict:
    return {
        "uid": uid,
        "email": email,
        "created_at": now,
        "window_start": None,
        "free_used": 0,
        "paid_used": 0,
        "verified_spend": 0,
        "reports_used": 0,
        "paid_blocked": False,
    }


def window_seconds(verified_spend: float) -> int:
    if verified_spend > 100:
        return int(BASE_WINDOW_SECONDS * 0.94)
    if verified_spend > 50:
        return int(BASE_WINDOW_SECONDS * 0.97)
    return BASE_WINDOW_SECONDS


def normalize(user: dict, now: float) -> dict:
    """Reset counters if the window has expired. Mutates and returns user."""
    start = user.get("window_start")
    if start is not None and now >= start + window_seconds(user.get("verified_spend", 0)):
        user["window_start"] = None
        user["free_used"] = 0
        user["paid_used"] = 0
    return user


def snapshot(user: dict, now: float, unlimited: bool = False) -> dict:
    normalize(user, now)
    win = window_seconds(user.get("verified_spend", 0))
    start = user.get("window_start")
    resets_at = (start + win) if start is not None else None
    free_left = max(0, FREE_MODELS - user["free_used"])
    paid_left = max(0, PAID_MODELS - user["paid_used"])
    return {
        "unlimited": unlimited,
        "free_limit": FREE_MODELS,
        "paid_limit": PAID_MODELS,
        "free_used": user["free_used"],
        "paid_used": user["paid_used"],
        "free_remaining": free_left,
        "paid_remaining": paid_left,
        "total_remaining": free_left + paid_left,
        "window_seconds": win,
        "window_start": start,
        "resets_at": resets_at,
        "now": now,
        "verified_spend": user.get("verified_spend", 0),
        "reports_used": user.get("reports_used", 0),
        "paid_blocked": bool(user.get("paid_blocked")),
    }


def reserve(user: dict, n: int, now: float, unlimited: bool = False) -> dict:
    """Reserve n model runs. Returns {'free': a, 'paid': b, 'free_remaining_before': r}."""
    normalize(user, now)
    free_left = max(0, FREE_MODELS - user["free_used"])
    paid_left = max(0, PAID_MODELS - user["paid_used"])
    if unlimited:
        return {"free": n, "paid": 0, "free_remaining_before": max(free_left, n)}
    if n < 1:
        raise LimitError("Select at least one model.", code="invalid")
    snap = snapshot(user, now)
    if n > free_left + paid_left:
        raise LimitError(
            f"Daily limit reached: you can run {free_left + paid_left} more model(s) in this window.",
            resets_at=snap["resets_at"],
        )
    free = min(n, free_left)
    paid = n - free
    if paid and user.get("paid_blocked"):
        raise LimitError(
            "Paid features are locked on this account because a payment could not be verified. Contact the admin.",
            code="paid_blocked",
        )
    if user.get("window_start") is None:
        user["window_start"] = now
    user["free_used"] += free
    user["paid_used"] += paid
    return {"free": free, "paid": paid, "free_remaining_before": free_left}
