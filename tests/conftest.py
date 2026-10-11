import os
import sys
from pathlib import Path

os.environ.update({
    "APP_ENV": "development", "AUTH_MODE": "dev", "ADMIN_EMAILS": "admin@example.com",
    "STORE_BACKEND": "memory", "UPI_ID": "test@upi", "UPI_PAYEE_NAME": "Auto ML Test",
    "PUBLIC_APP_URL": "http://localhost:5173",
    "DATA_CACHE_DIR": __import__("tempfile").mkdtemp(prefix="automl-test-cache-"),
})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app import quiz as quiz_mod  # noqa: E402
from app.store import MemoryStore, set_store  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    set_store(MemoryStore())
    quiz_mod.quizzes._sessions.clear()
    from app import main  # imported lazily so env vars above are set first

    main._hits.clear()
    import shutil
    from pathlib import Path
    for f in Path(os.environ["DATA_CACHE_DIR"]).glob("*"):
        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
    yield
    set_store(None)


def auth(email: str) -> dict:
    return {"Authorization": f"Bearer dev:{email}"}
