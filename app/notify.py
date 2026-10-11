"""Admin payment notification email (via Resend's HTTP API). Falls back to logging when unconfigured.
Firebase has no built-in email service, so email goes through the backend.
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone

import httpx

from .config import get_settings

log = logging.getLogger("automl.notify")
OUTBOX: list[dict] = []  # inspected by tests; capped
IST = timezone(timedelta(hours=5, minutes=30))


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%d %b %Y, %I:%M %p IST")


def send_email(to: str, subject: str, html_body: str, text_body: str) -> bool:
    s = get_settings()
    msg = {"to": to, "subject": subject, "text": text_body}
    OUTBOX.append(msg)
    del OUTBOX[:-50]
    if not (s.resend_api_key and to):
        log.info("Email not sent (RESEND_API_KEY or recipient missing): %s", subject)
        return False
    try:
        r = httpx.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {s.resend_api_key}"},
            json={"from": s.mail_from, "to": [to], "subject": subject, "html": html_body, "text": text_body},
            timeout=8,
        )
        r.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Email send failed: %s", exc)
        return False


def notify_admin_payment(payment: dict, user_email: str, context: dict) -> bool:
    s = get_settings()
    admin_url = f"{s.public_app_url}/admin?tab=payments&payment={payment['id']}"
    chat_url = f"{s.public_app_url}/admin?tab=payments&payment={payment['id']}&project={context.get('project_id', '')}"
    rows = [
        ("Reference", payment["ref"]),
        ("User", f"{payment.get('user_name') or user_email} ({user_email})"),
        ("Amount", f"INR {payment['amount']}"),
        ("Type", payment["kind"]),
        ("Claimed at", _fmt_time(payment.get("claimed_at") or payment["created_at"])),
        ("Domain / task", f"{context.get('domain', '-')} / {context.get('task', '-')}"),
        ("Models", ", ".join(context.get("models", [])) or "-"),
        ("Breakdown", "; ".join(f"{ln['label']}: {ln['amount']}" for ln in payment.get("breakdown", {}).get("lines", [])) or "-"),
    ]
    text = "A user marked a payment as paid.\n\n" + "\n".join(f"{k}: {v}" for k, v in rows) + \
        f"\n\nReview and approve: {admin_url}\nChat/project context: {chat_url}\n" \
        "Check your UPI app for a credit matching the reference before approving."
    table = "".join(f"<tr><td style='padding:4px 12px 4px 0;color:#6b6b6b'>{html.escape(k)}</td>"
                    f"<td style='padding:4px 0'><b>{html.escape(str(v))}</b></td></tr>" for k, v in rows)
    body = (f"<div style='font-family:system-ui,sans-serif;max-width:520px'><h2>Payment claimed</h2>"
            f"<table>{table}</table><p><a href='{html.escape(admin_url)}'>Open in admin</a> &middot; "
            f"<a href='{html.escape(chat_url)}'>Open chat context</a></p>"
            "<p style='color:#6b6b6b'>Check your UPI app for a credit that matches the reference before approving.</p></div>")
    return send_email(s.admin_notify_email or next(iter(s.admin_emails), ""), f"[Auto ML] Payment claimed: INR {payment['amount']} ({payment['ref']})", body, text)


def notify_admin_reminder(payment: dict, user_email: str, context: dict, waited_seconds: int) -> bool:
    s = get_settings()
    admin_url = f"{s.public_app_url}/admin?tab=payments&payment={payment['id']}"
    mins = max(1, round(waited_seconds / 60))
    rows = [("Reference", payment["ref"]), ("User", user_email), ("Amount", f"INR {payment['amount']}"),
            ("Waiting for", f"{mins} minute(s)"), ("Reminders sent", str(payment.get("reminders", 1))),
            ("Domain / task", f"{context.get('domain', '-')} / {context.get('task', '-')}")]
    text = "A user is still waiting for a payment to be approved.\n\n" + "\n".join(f"{k}: {v}" for k, v in rows) + \
        f"\n\nReview and approve: {admin_url}\nCheck your UPI app for a credit matching the reference first."
    table = "".join(f"<tr><td style='padding:4px 12px 4px 0;color:#6b6b6b'>{html.escape(k)}</td>"
                    f"<td style='padding:4px 0'><b>{html.escape(str(v))}</b></td></tr>" for k, v in rows)
    body = (f"<div style='font-family:system-ui,sans-serif;max-width:520px'><h2>Reminder: payment waiting</h2>"
            f"<table>{table}</table><p><a href='{html.escape(admin_url)}'>Open in admin</a></p></div>")
    return send_email(s.admin_notify_email or next(iter(s.admin_emails), ""),
                      f"[Auto ML] Reminder: payment waiting {mins} min (INR {payment['amount']}, {payment['ref']})", body, text)
