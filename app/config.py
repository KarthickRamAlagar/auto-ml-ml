"""Central settings, read from environment variables (see backend/.env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _csv(name: str, default: str = "") -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    app_env: str
    auth_mode: str  # "dev" | "firebase" | "gateway"
    frontend_origins: tuple[str, ...]
    admin_emails: frozenset[str]
    store_backend: str  # "memory" | "firestore" | "remote"
    firebase_sa_json: str
    gateway_secret: str
    store_url: str
    store_secret: str
    upi_id: str
    upi_name: str
    admin_notify_email: str
    resend_api_key: str
    mail_from: str
    public_app_url: str
    kaggle_username: str
    kaggle_key: str
    kaggle_api_token: str
    youtube_api_key: str
    hf_token: str
    llm_groups_json: str
    max_upload_mb: int
    max_train_rows: int
    max_job_seconds: int
    dataset_ttl_seconds: int
    job_ttl_seconds: int
    retention_days: int
    min_dataset_rows: int
    min_dataset_cols: int
    max_external_mb: int
    remind_after_seconds: int

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    app_env = os.getenv("APP_ENV", "development").lower()
    auth_mode = os.getenv("AUTH_MODE", "gateway" if app_env == "production" else "dev").lower()
    if app_env == "production" and auth_mode == "dev":
        raise RuntimeError("AUTH_MODE=dev is not allowed when APP_ENV=production")
    if app_env == "production" and os.getenv("STORE_BACKEND", "memory").lower() == "memory":
        raise RuntimeError("STORE_BACKEND=memory would lose all data on restart; use STORE_BACKEND=remote in production")
    if auth_mode not in ("dev", "firebase", "gateway"):
        raise RuntimeError("AUTH_MODE must be dev, firebase or gateway")
    if auth_mode == "gateway" and len(os.getenv("GATEWAY_SECRET", "")) < 32:
        raise RuntimeError("AUTH_MODE=gateway needs GATEWAY_SECRET (at least 32 characters), the same value the Worker has")
    return Settings(
        app_env=app_env,
        auth_mode=auth_mode,
        frontend_origins=tuple(_csv("FRONTEND_ORIGINS", "http://localhost:5173")),
        admin_emails=frozenset(e.lower() for e in _csv("ADMIN_EMAILS")),
        store_backend=os.getenv("STORE_BACKEND", "memory").lower(),
        firebase_sa_json=os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", ""),
        gateway_secret=os.getenv("GATEWAY_SECRET", ""),
        store_url=os.getenv("STORE_URL", "").rstrip("/"),
        store_secret=os.getenv("STORE_SECRET", ""),
        upi_id=os.getenv("UPI_ID", "yourname@bank"),
        upi_name=os.getenv("UPI_PAYEE_NAME", "Auto ML"),
        admin_notify_email=os.getenv("ADMIN_NOTIFY_EMAIL", ""),
        resend_api_key=os.getenv("RESEND_API_KEY", ""),
        mail_from=os.getenv("MAIL_FROM", "Auto ML <onboarding@resend.dev>"),
        public_app_url=os.getenv("PUBLIC_APP_URL", "http://localhost:5173").rstrip("/"),
        kaggle_username=os.getenv("KAGGLE_USERNAME", ""),
        kaggle_key=os.getenv("KAGGLE_KEY", ""),
        kaggle_api_token=os.getenv("KAGGLE_API_TOKEN", ""),
        youtube_api_key=",".join(x for x in (os.getenv("YOUTUBE_API_KEYS", ""), os.getenv("YOUTUBE_API_KEY", "")) if x),
        hf_token=os.getenv("HF_TOKEN", ""),
        llm_groups_json=os.getenv("LLM_GROUPS", ""),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "25")),
        max_train_rows=int(os.getenv("MAX_TRAIN_ROWS", "100000")),
        max_job_seconds=int(os.getenv("MAX_JOB_SECONDS", "600")),
        dataset_ttl_seconds=int(os.getenv("DATASET_TTL_SECONDS", "1800")),
        job_ttl_seconds=int(os.getenv("JOB_TTL_SECONDS", "3600")),
        retention_days=int(os.getenv("RETENTION_DAYS", "15")),
        min_dataset_rows=int(os.getenv("MIN_DATASET_ROWS", "50000")),
        min_dataset_cols=int(os.getenv("MIN_DATASET_COLS", "30")),
        max_external_mb=int(os.getenv("MAX_EXTERNAL_MB", "200")),
        remind_after_seconds=int(os.getenv("REMIND_AFTER_SECONDS", "420")),
    )
