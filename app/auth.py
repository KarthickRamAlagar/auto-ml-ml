"""Authentication.
- gateway (Cloudflare Worker in front): the Worker signs the caller's identity with a shared secret (HMAC-SHA256, 60 s window).
  The engine trusts nothing else, so it cannot be called directly.
- firebase: Firebase ID tokens verified with the Admin SDK (the earlier setup).
- dev: 'Bearer dev:<email>' tokens (refused when APP_ENV=production).
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException

from .config import get_settings
from .store import get_store, init_firebase_app


@dataclass
class AuthUser:
    uid: str
    email: str
    verified: bool
    admin: bool | None = None  # set by the gateway, which has already checked the verified admin e-mail

    @property
    def is_admin(self) -> bool:
        if self.admin is not None:
            return self.admin
        s = get_settings()
        # In production the email must be verified by Firebase, so a spoofed address can't become admin.
        return self.email in s.admin_emails and (self.verified or s.auth_mode == "dev")

    @property
    def is_vip(self) -> bool:
        """Admins and listed emails get unlimited free access."""
        return self.is_admin or self.email in get_store().get_vip()


def gateway_signature(secret: str, ts: str, uid: str, email: str, verified: str, admin: str) -> str:
    return hmac.new(secret.encode(), f"{ts}.{uid}.{email}.{verified}.{admin}".encode(), hashlib.sha256).hexdigest()


def _gateway_user(uid: str | None, email: str | None, verified: str | None, admin: str | None, ts: str | None, sig: str | None) -> AuthUser:
    if not (uid and email and verified in ("0", "1") and admin in ("0", "1") and ts and sig):
        raise HTTPException(401, "Missing gateway identity")
    try:
        age = abs(time.time() - int(ts))
    except ValueError:
        raise HTTPException(401, "Invalid gateway identity") from None
    expected = gateway_signature(get_settings().gateway_secret, ts, uid, email, verified, admin)
    if age > 60 or not hmac.compare_digest(expected, sig):
        raise HTTPException(401, "Invalid gateway identity")
    return AuthUser(uid=uid, email=email.lower(), verified=verified == "1", admin=admin == "1")


def current_user(authorization: str | None = Header(default=None),
                 x_gateway_uid: str | None = Header(default=None), x_gateway_email: str | None = Header(default=None),
                 x_gateway_verified: str | None = Header(default=None), x_gateway_admin: str | None = Header(default=None),
                 x_gateway_ts: str | None = Header(default=None), x_gateway_sig: str | None = Header(default=None)) -> AuthUser:
    if get_settings().auth_mode == "gateway":
        return _gateway_user(x_gateway_uid, x_gateway_email, x_gateway_verified, x_gateway_admin, x_gateway_ts, x_gateway_sig)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization[7:].strip()
    s = get_settings()
    if s.auth_mode == "dev":
        if not token.startswith("dev:") or "@" not in token:
            raise HTTPException(401, "Invalid dev token")
        email = token[4:].strip().lower()
        uid = "dev_" + hashlib.sha1(email.encode()).hexdigest()[:16]
        return AuthUser(uid=uid, email=email, verified=True)
    try:
        from firebase_admin import auth as fb_auth

        init_firebase_app()
        decoded = fb_auth.verify_id_token(token, check_revoked=False)
    except Exception as exc:  # noqa: BLE001 - any failure means unauthenticated
        raise HTTPException(401, "Invalid or expired token") from exc
    email = (decoded.get("email") or "").lower()
    if not email:
        raise HTTPException(401, "Account has no email")
    return AuthUser(uid=decoded["uid"], email=email, verified=bool(decoded.get("email_verified")))


def require_admin(user: AuthUser = Depends(current_user)) -> AuthUser:
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user
