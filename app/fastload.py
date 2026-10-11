"""Fast dataset loading.

Why Kaggle / Hugging Face / UCI felt slow: the old loader downloaded the whole file (up to 200 MB) in one
connection, parsed it, and only then cut it down to the row limit. This module:

  * reads only what is needed: HTTP Range requests for Parquet (footer + the first row groups) and for ZIP files
    (central directory + the start of the biggest CSV), and stops a plain CSV download once enough rows arrived;
  * remembers every dataset it has fetched in a small on-disk cache, so choosing the same dataset again is instant;
  * falls back to the old full download whenever a host does not support Range requests.

Uploads from the user's own machine never touch any of this, which is why they were already fast.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

from . import datasets as D

log = logging.getLogger("automl.fastload")

BLOCK = 1024 * 1024          # bytes per Range request
CSV_HEAD_BYTES = 48 * 1024 * 1024   # a CSV download stops after this much (rows beyond the cap are never used)
CACHE_TTL = 6 * 3600
CACHE_MAX_BYTES = 600 * 1024 * 1024
_cache_lock = threading.Lock()


class NoRange(Exception):
    """The host does not support HTTP Range requests (or answered unexpectedly)."""


# --------------------------------------------------------------------------- disk cache
def _cache_dir() -> Path:
    p = Path(os.getenv("DATA_CACHE_DIR") or Path(tempfile.gettempdir()) / "automl-cache")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(item_id: str) -> str:
    return hashlib.sha1(item_id.encode()).hexdigest()[:24]


def cache_get(item_id: str):
    """Returns (frame, name, default_target) or None."""
    try:
        base = _cache_dir() / _key(item_id)
        meta_p, data_p = base.with_suffix(".json"), base.with_suffix(".parquet")
        if not (meta_p.exists() and data_p.exists()):
            return None
        meta = json.loads(meta_p.read_text())
        if time.time() - meta["saved"] > CACHE_TTL:
            return None
        df = pd.read_parquet(data_p)
        os.utime(data_p)
        return df, meta["name"], meta.get("target")
    except Exception:  # noqa: BLE001 - a broken cache entry is just a miss
        return None


def cache_put(item_id: str, df: pd.DataFrame, name: str, target: str | None) -> None:
    try:
        with _cache_lock:
            base = _cache_dir() / _key(item_id)
            df.to_parquet(base.with_suffix(".parquet"), index=False)
            base.with_suffix(".json").write_text(json.dumps({"name": name, "target": target, "saved": time.time()}))
            files = sorted(_cache_dir().glob("*.parquet"), key=lambda f: f.stat().st_mtime)
            total = sum(f.stat().st_size for f in files)
            while files and total > CACHE_MAX_BYTES:
                f = files.pop(0)
                total -= f.stat().st_size
                f.unlink(missing_ok=True)
                f.with_suffix(".json").unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - caching must never fail a load
        log.warning("dataset cache write failed", exc_info=True)


# --------------------------------------------------------------------------- Range access
class RangeFile(io.RawIOBase):
    """A read-only, seekable file over an https URL, fetched in 1 MB Range blocks. Every hop of the redirect chain
    is validated with the same SSRF rules as the normal downloader."""

    def __init__(self, url: str, headers: dict | None = None, auth: tuple[str, str] | None = None):
        self._client = httpx.Client(follow_redirects=False, timeout=25)
        self._auth, self._headers = auth, headers
        self._blocks: dict[int, bytes] = {}
        self._pos = 0
        self.url = self._resolve(D.normalize_url(url))
        r = self._get(0, 0)
        if r.status_code != 206:
            raise NoRange("server ignored the Range header")
        cr = r.headers.get("content-range", "")
        if "/" not in cr or not cr.rsplit("/", 1)[1].isdigit():
            raise NoRange("unknown size")
        self.size = int(cr.rsplit("/", 1)[1])

    def _resolve(self, url: str) -> str:
        current = url
        for _ in range(5):
            D.validate_public_url(current)
            host = (urlparse(current).hostname or "").lower()
            kaggle = host == "www.kaggle.com"
            r = self._client.get(current, headers={**((self._headers or {}) if kaggle else {}), "Range": "bytes=0-0"},
                                 auth=self._auth if kaggle else None)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    raise NoRange("bad redirect")
                current = httpx.URL(current).join(loc).__str__()
                continue
            if r.status_code >= 400:
                raise D.DatasetError(f"The source returned HTTP {r.status_code}.", 502)
            return current
        raise NoRange("too many redirects")

    def _get(self, start: int, end: int) -> httpx.Response:
        D.validate_public_url(self.url)
        kaggle = (urlparse(self.url).hostname or "").lower() == "www.kaggle.com"
        return self._client.get(self.url, headers={**(self._headers if kaggle and self._headers else {}), "Range": f"bytes={start}-{end}"},
                                auth=self._auth if kaggle else None)

    # io.RawIOBase API
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = {0: offset, 1: self._pos + offset, 2: self.size + offset}[whence]
        self._pos = max(0, min(self._pos, self.size))
        return self._pos

    def _block(self, i: int) -> bytes:
        b = self._blocks.get(i)
        if b is None:
            start = i * BLOCK
            r = self._get(start, min(start + BLOCK, self.size) - 1)
            if r.status_code != 206:
                raise NoRange(f"Range request failed ({r.status_code})")
            b = r.content
            if len(self._blocks) > 64:  # bounded memory: drop the oldest block
                self._blocks.pop(next(iter(self._blocks)))
            self._blocks[i] = b
        return b

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        end = min(self._pos + n, self.size)
        out = []
        pos = self._pos
        while pos < end:
            blk = self._block(pos // BLOCK)
            off = pos % BLOCK
            take = blk[off: off + (end - pos)]
            if not take:
                break
            out.append(take)
            pos += len(take)
        self._pos = pos
        return b"".join(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            super().close()


def _trim_to_newline(data: bytes) -> bytes:
    i = data.rfind(b"\n")
    return data[: i + 1] if i > 0 else data


def parquet_range(url: str, headers: dict | None = None) -> pd.DataFrame:
    """Read the first row groups of a remote Parquet file, stopping at the row/cell cap."""
    import pyarrow.parquet as pq

    f = RangeFile(url, headers=headers)
    try:
        pf = pq.ParquetFile(f)
        ncols = max(1, len(pf.schema_arrow.names))
        cap = max(1000, min(D.MAX_LOAD_ROWS, D.MAX_CELLS // ncols))
        frames, n = [], 0
        for i in range(pf.num_row_groups):  # one row group per request burst; iter_batches would read the whole file
            frames.append(pf.read_row_group(i).to_pandas())
            n += len(frames[-1])
            if n >= cap:
                break
        df = pd.concat(frames, ignore_index=True).head(cap) if frames else pd.DataFrame()
    finally:
        f.close()
    return D._clean_frame(df)


def csv_head(url: str, max_bytes: int, headers: dict | None = None, auth: tuple[str, str] | None = None) -> bytes:
    """Stream a CSV and stop once CSV_HEAD_BYTES have arrived (cut at the last full line). Raises 413 only if the
    host reports a size above max_bytes and gives no way to read a part of it."""
    current = D.normalize_url(url)
    limit = min(max_bytes, CSV_HEAD_BYTES)
    with httpx.Client(follow_redirects=False, timeout=25) as client:
        for _ in range(5):
            D.validate_public_url(current)
            host = (urlparse(current).hostname or "").lower()
            kaggle = host == "www.kaggle.com"
            with client.stream("GET", current, auth=auth if kaggle else None, headers=headers if kaggle else None) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    loc = r.headers.get("location")
                    if not loc:
                        raise D.DatasetError("Bad redirect from source.")
                    current = httpx.URL(current).join(loc).__str__()
                    continue
                if r.status_code >= 400:
                    raise D.DatasetError(f"The source returned HTTP {r.status_code}.", 502)
                chunks, total = [], 0
                for chunk in r.iter_bytes(256 * 1024):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= limit:
                        return _trim_to_newline(b"".join(chunks))
                return b"".join(chunks)
    raise D.DatasetError("Too many redirects.")


def zip_csv_range(url: str, headers: dict | None, auth: tuple[str, str] | None) -> bytes:
    """Biggest CSV inside a remote ZIP, reading only the central directory and the first CSV_HEAD_BYTES of it."""
    f = RangeFile(url, headers=headers, auth=auth)
    try:
        with zipfile.ZipFile(f) as zf:
            best = None
            for info in zf.infolist():
                if info.filename.lower().endswith(".csv") and info.file_size > 0:
                    if best is None or info.file_size > best.file_size:
                        best = info
            if best is None:
                raise D.DatasetError("No CSV file found inside the downloaded archive.")
            with zf.open(best) as member:
                return _trim_to_newline(member.read(CSV_HEAD_BYTES))
    finally:
        f.close()


# --------------------------------------------------------------------------- any-format parsing (uploads and links)
def sniff(data: bytes, filename: str = "") -> str:
    """csv | xlsx | parquet | json | xls — decided from the file's first bytes first, its name second."""
    name = (filename or "").lower()
    if data[:4] == b"PAR1":
        return "parquet"
    if data[:2] == b"PK":
        if name.endswith((".csv", ".zip")):
            return "zip"
        return "xlsx"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "xls"
    head = data[:2048].lstrip(b"\xef\xbb\xbf \t\r\n")
    if name.endswith((".json", ".jsonl", ".ndjson")) or head[:1] in (b"[", b"{"):
        return "json"
    return "csv"


def parse_any(data: bytes, filename: str = "") -> pd.DataFrame:
    """CSV / TSV / TXT, Excel (.xlsx), Parquet, JSON / JSON Lines — and a ZIP that contains one of those."""
    if not data or not data.strip():
        raise D.DatasetError("The file is empty.")
    kind = sniff(data, filename)
    if kind == "parquet":
        return D.parse_parquet(data)
    if kind == "xls":
        raise D.DatasetError("Old .xls files are not supported. Save the sheet as .xlsx or .csv and upload again.", 415)
    if kind == "zip":
        return parse_any(D._download_zip_csv(data), "inner.csv")
    if kind == "xlsx":
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine="openpyxl", nrows=D.MAX_LOAD_ROWS)
        except Exception as exc:  # noqa: BLE001
            raise D.DatasetError(f"Could not read this Excel file: {str(exc)[:120]}", 422) from exc
        sheets = {k: v for k, v in sheets.items() if v.shape[0] >= 1 and v.shape[1] >= 2}
        if not sheets:
            raise D.DatasetError("No sheet with at least 2 columns and some rows was found in this workbook.")
        name = max(sheets, key=lambda k: sheets[k].shape[0] * sheets[k].shape[1])
        return D._clean_frame(sheets[name].dropna(how="all").dropna(axis=1, how="all"))
    if kind == "json":
        text = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):  # {"data": [...]} / {"rows": [...]} or a column-oriented object
                lists = [v for v in obj.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                obj = lists[0] if lists else obj
            df = pd.json_normalize(obj) if isinstance(obj, list) else pd.DataFrame(obj)
        except json.JSONDecodeError:
            try:  # JSON Lines
                df = pd.read_json(io.StringIO(text), lines=True, nrows=D.MAX_LOAD_ROWS)
            except Exception as exc:  # noqa: BLE001
                raise D.DatasetError(f"Could not read this file as JSON: {str(exc)[:120]}", 422) from exc
        except Exception as exc:  # noqa: BLE001
            raise D.DatasetError(f"Could not read this file as JSON: {str(exc)[:120]}", 422) from exc
        return D._clean_frame(df.head(D.MAX_LOAD_ROWS))
    return D.parse_csv(data)
