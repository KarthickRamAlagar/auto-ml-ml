"""Datasets: built-in catalog, live search (Kaggle / Hugging Face), upload parsing, URL import and an
in-memory dataset store with a short TTL. Datasets are never written to disk or to the database.
"""
from __future__ import annotations

import csv
import io
import re
import ipaddress
import socket
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pandas as pd

from .config import get_settings

DOMAINS = ["Finance", "Healthcare", "E-commerce", "Real estate", "Education", "Manufacturing", "Other"]
MAX_LOAD_ROWS = 200_000
MAX_COLUMNS = 500
MAX_CELLS = 25_000_000  # rows x columns kept from a wide parquet table


class CatalogWarming(Exception):
    """The UCI catalogue is still being built in the background."""


class DatasetError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------- built-in datasets
def _dates(rng, n, start="2023-01-01", days=730):
    offs = np.sort(rng.integers(0, days, n))
    return pd.Timestamp(start) + pd.to_timedelta(offs, unit="D")


def _mask_nan(rng, s: pd.Series, frac: float) -> pd.Series:
    s = s.astype(float)
    s[rng.random(len(s)) < frac] = np.nan
    return s


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _synthetic_credit() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 5000
    income = rng.lognormal(10.8, 0.5, n).round(0)
    loan = (income * rng.uniform(0.1, 1.2, n)).round(0)
    score = np.clip(rng.normal(680, 70, n), 300, 900).round()
    years = np.clip(rng.normal(8, 6, n), 0, 40).round(1)
    util = np.clip(rng.beta(2, 3, n), 0, 1).round(3)
    emp = rng.choice(["Salaried", "Self-employed", "Business", "Retired"], n, p=[0.55, 0.2, 0.15, 0.1])
    home = rng.choice(["Rent", "Own", "Mortgage"], n, p=[0.4, 0.25, 0.35])
    purpose = rng.choice(["Personal", "Auto", "Home", "Education", "Business"], n)
    logit = -2.3 + 2.2 * util - 0.006 * (score - 650) + 0.8 * (loan / income) - 0.03 * years + 0.4 * (emp == "Self-employed")
    defaulted = (rng.random(n) < _sig(logit)).astype(int)
    return pd.DataFrame(
        {
            "application_date": _dates(rng, n),
            "age": rng.integers(21, 70, n),
            "income": _mask_nan(rng, pd.Series(income), 0.03),
            "loan_amount": loan,
            "credit_score": _mask_nan(rng, pd.Series(score), 0.02),
            "employment_years": years,
            "open_accounts": rng.poisson(4, n),
            "credit_utilization": util,
            "employment_type": emp,
            "home_ownership": home,
            "loan_purpose": purpose,
            "defaulted": defaulted,
        }
    )


def _synthetic_churn() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 4000
    tenure = rng.integers(1, 61, n)
    spend = rng.lognormal(6.0, 0.6, n).round(0)
    orders = rng.poisson(3, n)
    tickets = rng.poisson(1.2, n)
    basket = rng.lognormal(6.3, 0.4, n).round(0)
    device = rng.choice(["Mobile", "Desktop", "Tablet"], n, p=[0.65, 0.3, 0.05])
    plan = rng.choice(["Basic", "Plus", "Premium"], n, p=[0.5, 0.35, 0.15])
    country = rng.choice(["IN", "SG", "AE", "US", "UK"], n, p=[0.55, 0.12, 0.1, 0.13, 0.1])
    logit = -0.9 - 0.04 * tenure + 0.35 * tickets - 0.12 * orders + 0.6 * (plan == "Basic") - 0.4 * (plan == "Premium")
    churned = (rng.random(n) < _sig(logit)).astype(int)
    return pd.DataFrame(
        {
            "signup_date": _dates(rng, n, "2022-01-01", 1000),
            "tenure_months": tenure,
            "monthly_spend": _mask_nan(rng, pd.Series(spend), 0.02),
            "orders_last_90d": orders,
            "support_tickets": tickets,
            "avg_basket_value": basket,
            "device": device,
            "plan": plan,
            "country": country,
            "churned": churned,
        }
    )


def _synthetic_housing() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 3000
    area = np.clip(rng.normal(1250, 380, n), 350, 4000).round(0)
    beds = np.clip((area / 450 + rng.normal(0, 0.5, n)).round(), 1, 6).astype(int)
    baths = np.clip(beds - rng.integers(0, 2, n), 1, 5)
    age = rng.integers(0, 40, n)
    dist = np.clip(rng.gamma(2.5, 3.0, n), 0.5, 40).round(1)
    hood = rng.choice(["Central", "North", "South", "East", "West"], n, p=[0.15, 0.25, 0.25, 0.2, 0.15])
    mult = pd.Series(hood).map({"Central": 1.35, "North": 1.0, "South": 0.95, "East": 0.85, "West": 1.1}).to_numpy()
    parking = rng.integers(0, 2, n)
    price = 0.05 * area * mult - 0.35 * age - 1.4 * dist + 5 * beds + 4 * parking + rng.normal(0, 7, n)
    return pd.DataFrame(
        {
            "listing_date": _dates(rng, n, "2024-01-01", 600),
            "area_sqft": area,
            "bedrooms": beds,
            "bathrooms": baths,
            "age_years": age,
            "distance_center_km": dist,
            "neighborhood": hood,
            "has_parking": parking,
            "price_lakhs": price.round(2),
        }
    )


def _synthetic_students() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 2500
    study = np.clip(rng.normal(3.5, 1.5, n), 0, 10).round(1)
    attend = np.clip(rng.normal(82, 12, n), 30, 100).round(1)
    prev = np.clip(rng.normal(62, 15, n), 10, 100).round(1)
    sleep = np.clip(rng.normal(6.8, 1.2, n), 3, 10).round(1)
    tutoring = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    school = rng.choice(["Public", "Private", "Charter"], n, p=[0.5, 0.35, 0.15])
    score = 18 + 0.38 * prev + 2.6 * study + 0.14 * attend + 0.8 * sleep + 3 * (tutoring == "Yes") + rng.normal(0, 4, n)
    return pd.DataFrame(
        {
            "study_hours": study,
            "attendance_pct": _mask_nan(rng, pd.Series(attend), 0.02),
            "previous_score": prev,
            "sleep_hours": sleep,
            "tutoring": tutoring,
            "school_type": school,
            "final_score": np.clip(score, 0, 100).round(1),
        }
    )


def _synthetic_machines() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n = 3000
    temp = rng.normal(75, 9, n).round(1)
    vib = np.clip(rng.gamma(2.0, 1.2, n), 0.1, 15).round(2)
    pressure = rng.normal(30, 4, n).round(1)
    rpm = rng.normal(1500, 180, n).round(0)
    hours = rng.integers(10, 9000, n)
    mtype = rng.choice(["Press", "Lathe", "Mill", "Pump"], n)
    shift = rng.choice(["Day", "Evening", "Night"], n, p=[0.45, 0.35, 0.2])
    logit = -5.3 + 0.045 * (temp - 75) + 0.55 * vib + 0.0002 * hours + 0.5 * (shift == "Night")
    fail = (rng.random(n) < _sig(logit)).astype(int)
    return pd.DataFrame(
        {
            "timestamp": _dates(rng, n, "2024-01-01", 365),
            "temperature_c": temp,
            "vibration_mm_s": vib,
            "pressure_psi": _mask_nan(rng, pd.Series(pressure), 0.02),
            "rpm": rpm,
            "runtime_hours": hours,
            "machine_type": mtype,
            "shift": shift,
            "failure": fail,
        }
    )


def _sk(loader: str, rename: dict | None = None, target_map: dict | None = None):
    def _load() -> pd.DataFrame:
        from sklearn import datasets as skd

        bunch = getattr(skd, loader)(as_frame=True)
        df = bunch.frame.copy()
        if target_map:
            df["target"] = df["target"].map(target_map)
        return df.rename(columns=rename or {})

    return _load


@dataclass
class CatalogItem:
    id: str
    name: str
    domains: list[str]
    tasks: list[str]
    description: str
    default_target: str
    loader: Callable[[], pd.DataFrame]
    license: str = "Generated for demos"
    synthetic: bool = True
    source: str = "builtin"
    url: str | None = None
    rows: int | None = None  # declared shape, so listing a generated dataset never has to build it
    cols: int | None = None


BUILTIN: list[CatalogItem] = [
    CatalogItem("builtin:credit-risk", "Loan default risk (synthetic)", ["Finance"], ["classification"],
                "Applicant profile, loan size and credit signals with a default flag. Has missing values and a date column.",
                "defaulted", _synthetic_credit),
    CatalogItem("builtin:churn", "Customer churn (synthetic)", ["E-commerce"], ["classification"],
                "Subscription customers with spend, orders and support tickets; predict who churns.",
                "churned", _synthetic_churn),
    CatalogItem("builtin:housing", "House prices (synthetic)", ["Real estate"], ["regression"],
                "Listings with area, rooms, age and neighbourhood; predict price in lakhs.",
                "price_lakhs", _synthetic_housing),
    CatalogItem("builtin:students", "Student final score (synthetic)", ["Education"], ["regression"],
                "Study habits, attendance and previous score; predict the final score.",
                "final_score", _synthetic_students),
    CatalogItem("builtin:machines", "Machine failure (synthetic)", ["Manufacturing"], ["classification"],
                "Sensor readings and runtime from factory machines; predict failures.",
                "failure", _synthetic_machines),
    CatalogItem("builtin:breast-cancer", "Breast cancer (Wisconsin)", ["Healthcare"], ["classification"],
                "569 tumour samples with 30 measurements; benign vs malignant.",
                "diagnosis", _sk("load_breast_cancer", {"target": "diagnosis"}),
                license="CC BY 4.0 (UCI)", synthetic=False),
    CatalogItem("builtin:diabetes", "Diabetes progression", ["Healthcare"], ["regression"],
                "442 patients with 10 baseline variables; predict disease progression after one year.",
                "progression", _sk("load_diabetes", {"target": "progression"}),
                license="Public (scikit-learn)", synthetic=False),
    CatalogItem("builtin:wine", "Wine cultivars", ["Manufacturing", "Other"], ["classification"],
                "178 wines from 3 cultivars with 13 chemical measurements.",
                "cultivar", _sk("load_wine", {"target": "cultivar"}), license="CC BY 4.0 (UCI)", synthetic=False),
    CatalogItem("builtin:iris", "Iris flowers", ["Education", "Other"], ["classification"],
                "The classic 150-row flower measurements dataset, 3 species.",
                "species", _sk("load_iris", {"target": "species"}, {0: "setosa", 1: "versicolor", 2: "virginica"}),
                license="Public domain", synthetic=False),
]
# --------------------------------------------------------------------------- large generated datasets
# 60,000 rows x 34 columns, generated on demand from a fixed seed (nothing is stored). They follow the same
# size rule as the Kaggle / Hugging Face / UCI results: more than 50,000 rows and at least 30 columns.
BIG_ROWS = 60_000


@dataclass(frozen=True)
class _BigSpec:
    item_id: str
    name: str
    domain: str
    kind: str  # classification | regression
    description: str
    target: str
    date_col: str
    numeric: tuple[str, ...]
    cats: dict
    seed: int
    positive_rate: float = 0.12


_BIG: list[_BigSpec] = [
    _BigSpec("builtin:big-fraud", "Card transaction fraud (synthetic, large)", "Finance", "classification",
             "60,000 card transactions with 34 behavioural, device and risk columns; flag fraudulent ones.", "is_fraud", "txn_date",
             ("txn_amount", "account_age_days", "balance_before", "balance_after", "txn_count_24h", "txn_count_7d", "avg_txn_30d",
              "std_txn_30d", "distinct_merchants_7d", "distinct_countries_30d", "failed_logins_24h", "device_changes_30d",
              "ip_risk_score", "merchant_risk_score", "hour_of_day", "days_since_last_txn", "credit_limit", "utilization_ratio",
              "num_cards", "chargebacks_12m", "geo_distance_km", "velocity_score", "session_length_s", "page_views", "pin_retries",
              "kyc_score", "income_est", "tenure_months"),
             {"channel": ["POS", "Online", "ATM", "Mobile"], "merchant_category": ["Grocery", "Travel", "Electronics", "Fuel", "Dining", "Other"],
              "card_type": ["Debit", "Credit", "Prepaid"], "region": ["North", "South", "East", "West"]}, 101, 0.08),
    _BigSpec("builtin:big-readmission", "Hospital readmission (synthetic, large)", "Healthcare", "classification",
             "60,000 hospital stays with vitals, lab results and history; predict readmission within 30 days.", "readmitted_30d", "admission_date",
             ("age", "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "glucose", "hba1c", "cholesterol", "ldl", "hdl", "triglycerides",
              "creatinine", "egfr", "hemoglobin", "wbc_count", "platelets", "sodium", "potassium", "num_prior_admissions",
              "length_of_stay_days", "num_medications", "num_procedures", "num_diagnoses", "er_visits_12m", "smoking_pack_years",
              "alcohol_units_week", "exercise_hours_week", "comorbidity_index"),
             {"gender": ["F", "M"], "admission_type": ["Emergency", "Elective", "Urgent"], "insurance": ["Public", "Private", "None"],
              "discharge_to": ["Home", "Rehab", "Nursing", "Other"]}, 102, 0.16),
    _BigSpec("builtin:big-clv", "Customer lifetime value (synthetic, large)", "E-commerce", "regression",
             "60,000 shoppers with browsing, ordering and loyalty signals; predict lifetime value.", "lifetime_value", "signup_date",
             ("sessions_30d", "pages_per_session", "avg_session_minutes", "cart_adds_30d", "cart_abandon_rate", "orders_12m",
              "avg_order_value", "items_per_order", "return_rate", "discount_usage_rate", "days_since_last_order", "tenure_days",
              "email_open_rate", "email_click_rate", "push_opt_in_score", "wishlist_items", "reviews_written", "avg_rating_given",
              "support_tickets_12m", "nps_score", "loyalty_points", "referral_count", "mobile_share", "weekend_share", "night_share",
              "category_breadth", "brand_loyalty_score", "price_sensitivity"),
             {"acquisition_channel": ["Organic", "Paid", "Referral", "Social"], "device": ["Mobile", "Desktop", "Tablet"],
              "city_tier": ["Tier 1", "Tier 2", "Tier 3", "Rural"], "payment_pref": ["Card", "UPI", "COD", "Wallet"]}, 103),
    _BigSpec("builtin:big-property", "Property valuation (synthetic, large)", "Real estate", "regression",
             "60,000 listings with size, location quality and neighbourhood signals; predict price in lakhs.", "price_lakhs", "listing_date",
             ("area_sqft", "plot_area_sqft", "bedrooms", "bathrooms", "floors", "age_years", "distance_center_km", "distance_metro_km",
              "distance_school_km", "distance_hospital_km", "crime_index", "walk_score", "noise_db", "air_quality_index",
              "green_cover_pct", "parking_spots", "balcony_area", "ceiling_height", "renovation_years_ago", "maintenance_fee",
              "property_tax", "hoa_fee", "nearby_listings", "days_on_market", "neighborhood_income", "school_rating",
              "flood_risk_score", "view_score"),
             {"neighborhood": ["Central", "North", "South", "East", "West"], "property_type": ["Apartment", "Villa", "Townhouse", "Studio"],
              "facing": ["North", "South", "East", "West"], "furnishing": ["Unfurnished", "Semi", "Furnished"]}, 104),
    _BigSpec("builtin:big-students", "Student outcomes (synthetic, large)", "Education", "regression",
             "60,000 student records with study habits, engagement and context; predict the final score.", "final_score", "term_start",
             ("attendance_pct", "study_hours_week", "previous_gpa", "assignment_avg", "quiz_avg", "midterm_score", "lab_score",
              "participation_score", "library_visits", "online_logins", "forum_posts", "video_watch_hours", "tutoring_hours",
              "sleep_hours", "commute_minutes", "part_time_work_hours", "extracurricular_hours", "stress_score", "motivation_score",
              "family_support_score", "peer_study_sessions", "late_submissions", "absences", "course_load_credits", "teacher_rating",
              "class_size", "homework_completion", "reading_hours"),
             {"school_type": ["Public", "Private", "Charter"], "major": ["STEM", "Business", "Arts", "Health"],
              "scholarship": ["None", "Partial", "Full"], "housing": ["Home", "Hostel", "Rented"]}, 105),
    _BigSpec("builtin:big-machines", "Machine failure sensors (synthetic, large)", "Manufacturing", "classification",
             "60,000 machine readings from 28 sensors and context; predict the next failure.", "failure", "timestamp",
             ("temperature_c", "vibration_mm_s", "pressure_psi", "rpm", "torque_nm", "current_a", "voltage_v", "power_kw", "oil_temp_c",
              "oil_viscosity", "bearing_temp_c", "humidity_pct", "ambient_temp_c", "noise_db", "runtime_hours", "hours_since_service",
              "load_pct", "cycle_count", "tool_wear_min", "feed_rate", "spindle_speed", "coolant_flow", "coolant_temp_c", "air_pressure",
              "particle_count", "vibration_x", "vibration_y", "vibration_z"),
             {"machine_type": ["Press", "Lathe", "Mill", "Pump"], "shift": ["Day", "Evening", "Night"], "plant": ["A", "B", "C", "D"],
              "operator_grade": ["Junior", "Mid", "Senior"]}, 106, 0.07),
]


def _make_big(spec: _BigSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    n, k = BIG_ROWS, len(spec.numeric)
    z = rng.standard_normal((n, k))
    for b in range(0, k - 3, 4):  # blocks of four correlated features, as real sensor / behaviour data tends to be
        z[:, b + 1] = 0.55 * z[:, b] + 0.84 * z[:, b + 1]
        z[:, b + 3] = 0.45 * z[:, b + 2] + 0.89 * z[:, b + 3]
    cols: dict[str, np.ndarray] = {}
    for i, name in enumerate(spec.numeric):
        loc, sc = rng.uniform(8, 120), rng.uniform(0.08, 0.35)
        if any(w in name for w in ("count", "visits", "orders", "tickets", "num_", "posts", "admissions", "referral", "items", "cards", "absences", "late")):
            v = rng.poisson(np.clip(loc / 12, 0.5, 9), n).astype(float) + np.where(z[:, i] > 1.2, 1, 0)
        elif i % 5 == 2:
            v = np.exp(np.log(loc) + sc * 2 * z[:, i])
        else:
            v = loc + loc * sc * z[:, i]
        cols[name] = np.round(np.clip(v, 0, None), 2)
    cat_effect = np.zeros(n)
    for cname, levels in spec.cats.items():
        idx = rng.integers(0, len(levels), n)
        cat_effect += rng.normal(0, 0.25, len(levels))[idx]
        cols[cname] = np.array(levels, dtype=object)[idx]
    w = np.array([1.0, -0.8, 0.65, 0.5, -0.4, 0.3, 0.22, -0.15])
    score = z[:, :8] @ w + 0.5 * z[:, 0] * z[:, 1] + 0.35 * z[:, 2] ** 2 + cat_effect + rng.normal(0, 0.9, n)
    if spec.kind == "classification":
        thr = np.quantile(score, 1 - spec.positive_rate)
        y = (score > thr).astype(int)
    else:
        y = np.round(np.clip(55 + 8 * score, 1, 100 if spec.item_id.endswith('students') else None), 2)
    df = pd.DataFrame({spec.date_col: _dates(rng, n, "2023-01-01", 900), **cols, spec.target: y})
    for name in list(spec.numeric)[5:9]:  # a little missing data to make preparation worth doing
        df[name] = _mask_nan(rng, df[name], 0.02)
    return df


def _big_item(sp: _BigSpec) -> CatalogItem:
    return CatalogItem(sp.item_id, sp.name, [sp.domain], [sp.kind], sp.description, sp.target, lambda sp=sp: _make_big(sp),
                       rows=BIG_ROWS, cols=1 + len(sp.numeric) + len(sp.cats) + 1)


BUILTIN.extend(_big_item(sp) for sp in _BIG)
_BUILTIN_BY_ID = {b.id: b for b in BUILTIN}


@lru_cache(maxsize=3)
def _builtin_frame(item_id: str) -> pd.DataFrame:
    return _BUILTIN_BY_ID[item_id].loader()


def load_builtin(item_id: str) -> tuple[pd.DataFrame, CatalogItem]:
    item = _BUILTIN_BY_ID.get(item_id)
    if not item:
        raise DatasetError("Unknown dataset", 404)
    return _builtin_frame(item_id).copy(), item


@lru_cache(maxsize=None)
def _shape_of(item_id: str) -> tuple[int, int, float]:
    it = _BUILTIN_BY_ID[item_id]
    if it.rows and it.cols:  # large generated sets: ~8 bytes per numeric cell, no need to build them to list them
        return it.rows, it.cols, round(it.rows * it.cols * 8 / 1e6, 1)
    df = _builtin_frame(item_id)
    return int(len(df)), int(df.shape[1]), round(df.memory_usage(deep=True).sum() / 1e6, 2)


def _card(item: CatalogItem, match: bool) -> dict:
    rows, cols, mb = _shape_of(item.id)
    return {
        "id": item.id, "name": item.name, "source": item.source, "description": item.description,
        "rows": rows, "cols": cols, "size_mb": mb,
        "domains": item.domains, "tasks": item.tasks, "license": item.license, "synthetic": item.synthetic,
        "url": item.url, "default_target": item.default_target, "match": match,
    }


# --------------------------------------------------------------------------- live providers
SOURCES = ("recommended", "kaggle", "huggingface", "uci")

# Search keywords per domain, because "Finance classification" is a poor full-text query.
_DOMAIN_WORDS = {
    "finance": ["credit", "fraud", "loan", "bank"],
    "healthcare": ["patient", "disease", "hospital", "health"],
    "e-commerce": ["customer", "retail", "sales", "churn"],
    "real estate": ["house", "property", "housing", "rent"],
    "education": ["student", "school", "exam", "university"],
    "manufacturing": ["sensor", "machine", "failure", "industrial"],
}
_UCI_AREAS = {
    "finance": {"business"}, "e-commerce": {"business"}, "real estate": {"business"},
    "healthcare": {"health and medicine", "biology"}, "education": {"social science"},
    "manufacturing": {"engineering", "physics and chemistry"},
}


def size_rule() -> dict:
    s = get_settings()
    return {"min_rows": s.min_dataset_rows, "min_cols": s.min_dataset_cols}


def meets_size(rows: int | None, cols: int | None) -> bool:
    r = size_rule()
    return (rows or 0) >= r["min_rows"] and (cols or 0) >= r["min_cols"]


def _field(d: dict, name: str):
    """Kaggle returns both `name` and `nameNullable` depending on the endpoint version."""
    v = d.get(name)
    return v if v is not None else d.get(name + "Nullable")


def kaggle_enabled() -> bool:
    s = get_settings()
    return bool(s.kaggle_api_token or (s.kaggle_username and s.kaggle_key))


def _kaggle_auth() -> tuple[dict, tuple[str, str] | None]:
    """New-style KGAT_ tokens use a Bearer header; legacy accounts use username + key (basic auth)."""
    s = get_settings()
    if s.kaggle_api_token:
        return {"Authorization": f"Bearer {s.kaggle_api_token}"}, None
    return {}, (s.kaggle_username, s.kaggle_key)


def kaggle_search(query: str, limit: int = 20, min_bytes: int | None = None) -> list[dict]:
    """CSV datasets big enough to plausibly hold 50k rows x 30 columns. Kaggle's list API reports file size but not
    rows/columns, so size filters the search and the real shape is verified after download."""
    if not kaggle_enabled():
        return []
    r = get_settings()
    rule = size_rule()
    floor = min_bytes if min_bytes is not None else max(5_000_000, int(rule["min_rows"] * rule["min_cols"] * 4))
    headers, basic = _kaggle_auth()
    resp = httpx.get(
        "https://www.kaggle.com/api/v1/datasets/list",
        params={"search": query, "fileType": "csv", "sortBy": "votes", "minSize": floor,
                "maxSize": r.max_external_mb * 1024 * 1024 * 3, "page": 1},
        headers=headers, auth=basic, timeout=12,
    )
    if resp.status_code in (401, 403):
        raise PermissionError("Kaggle rejected the credentials")
    resp.raise_for_status()
    out = []
    for d in resp.json()[:limit]:
        ref = _field(d, "ref")
        if not ref:
            continue
        size = _field(d, "totalBytes")
        out.append({
            "id": f"kaggle:{ref}", "name": _field(d, "title") or ref, "source": "kaggle",
            "description": (_field(d, "subtitle") or "")[:200], "rows": None, "cols": None,
            "size_mb": round(size / 1e6, 1) if size else None, "domains": [], "tasks": [],
            "license": _field(d, "licenseName") or "See Kaggle page", "synthetic": False,
            "url": _field(d, "url") or f"https://www.kaggle.com/datasets/{ref}", "default_target": None, "match": True,
            "votes": d.get("voteCount") or 0, "downloads": d.get("downloadCount") or 0, "shape_checked": False,
        })
    return out


_search_cache: dict[tuple, tuple[float, list[dict]]] = {}
_SEARCH_TTL = 600  # seconds; identical searches within 10 minutes are served instantly


def _cached_search(key: tuple, fn: Callable[[], list[dict]]) -> list[dict]:
    hit = _search_cache.get(key)
    if hit and time.time() - hit[0] < _SEARCH_TTL:
        return hit[1]
    res = fn()
    if res:  # never cache an empty or failed search
        _search_cache[key] = (time.time(), res)
    return res


def kaggle_search_domain(domain: str, query: str = "") -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    words = [query] if query else (_DOMAIN_WORDS.get((domain or "").lower()) or [domain or "data"])[:3]

    def run() -> list[dict]:
        def one(w: str) -> list[dict]:
            try:
                return kaggle_search(w)
            except PermissionError:
                raise
            except Exception:  # noqa: BLE001 - one slow keyword must not sink the others
                return []

        with ThreadPoolExecutor(max_workers=len(words)) as pool:  # keywords searched side by side, not one after another
            results = list(pool.map(one, words))
        seen: dict[str, dict] = {}
        for res in results:
            for it in res:
                seen.setdefault(it["id"], it)
        return sorted(seen.values(), key=lambda i: -(i.get("downloads") or 0))[:30]

    return _cached_search(("kaggle", tuple(words)), run)


def _hf_parquet_urls(did: str) -> list[str]:
    """Train-split parquet shards from Hugging Face's auto-converted copy, best split first."""
    try:
        r = httpx.get(f"https://huggingface.co/api/datasets/{did}/parquet", timeout=10, headers=_hf_headers())
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    for cfg in (data.values() if isinstance(data, dict) else []):
        if not isinstance(cfg, dict):
            continue
        splits = sorted(cfg.items(), key=lambda kv: (0 if kv[0] == "train" else 1, kv[0]))
        for _, urls in splits:
            out += [u for u in urls if isinstance(u, str)]
        if out:
            break
    return out


_hf_cache: dict[str, tuple[float, tuple[int | None, int | None]]] = {}
_hf_sizes: dict[str, int] = {}  # download size in bytes, filled in by _hf_shape


def _hf_shape(did: str) -> tuple[int | None, int | None]:
    """(rows, columns) from the Hugging Face datasets server; (None, None) if there is no viewer for it."""
    hit = _hf_cache.get(did)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    shape: tuple[int | None, int | None] = (None, None)
    try:
        info = httpx.get("https://datasets-server.huggingface.co/info", params={"dataset": did}, timeout=6, headers=_hf_headers()).json()
        cfgs = info.get("dataset_info") or {}
        if cfgs:
            c = next(iter(cfgs.values()))
            rows = sum(int(sp.get("num_examples", 0)) for sp in (c.get("splits") or {}).values())
            shape = (rows or None, len(c.get("features") or {}) or None)
            if c.get("download_size"):
                _hf_sizes[did] = int(c["download_size"])
    except Exception:  # noqa: BLE001
        pass
    _hf_cache[did] = (time.time(), shape)
    return shape


def hf_search(query: str, limit: int = 30, task: str = "") -> list[dict]:
    return _cached_search(("hf", query, task, limit), lambda: _hf_search(query, limit, task))


def _hf_search(query: str, limit: int = 30, task: str = "") -> list[dict]:
    base = {"search": query, "limit": 40, "sort": "downloads", "direction": -1}
    items: list[dict] = []
    if task in ("classification", "regression"):  # prefer tabular datasets for the chosen task
        r = httpx.get("https://huggingface.co/api/datasets", params={**base, "filter": f"task_categories:tabular-{task}"}, timeout=8, headers=_hf_headers())
        r.raise_for_status()
        items = r.json()
    if len(items) < 10:
        r = httpx.get("https://huggingface.co/api/datasets", params=base, timeout=8, headers=_hf_headers())
        r.raise_for_status()
        seen = {d.get("id") for d in items}
        items += [d for d in r.json() if d.get("id") not in seen]
    items = [d for d in items if d.get("id")][:40]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=16) as pool:
        shapes = list(pool.map(lambda d: _hf_shape(d["id"]), items))
    out = []
    for d, (rows, cols) in zip(items, shapes):
        if not meets_size(rows, cols):
            continue
        did = d["id"]
        out.append({
            "id": f"hf:{did}", "name": did, "source": "huggingface",
            "description": (d.get("description") or ", ".join((d.get("tags") or [])[:5]))[:200], "rows": rows, "cols": cols,
            "size_mb": round(_hf_sizes[did] / 1e6, 1) if did in _hf_sizes else None, "domains": [], "tasks": [],
            "license": next((t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")), "See dataset card"),
            "synthetic": False, "url": f"https://huggingface.co/datasets/{did}", "default_target": None, "match": True,
            "downloads": d.get("downloads") or 0, "shape_checked": True,
        })
        if len(out) >= limit:
            break
    return out


_uci_cache: dict = {"at": 0.0, "items": []}
_uci_lock = threading.Lock()


def _uci_index(block: bool = True) -> list[dict]:
    """Every UCI dataset that has a CSV, with its real shape (cached for a day; building it takes a few seconds)."""
    if _uci_cache["items"] and time.time() - _uci_cache["at"] < 86400:
        return _uci_cache["items"]
    if not block:  # a search never waits ~20 s for the one-time build: start it in the background and say so
        if not _uci_lock.locked():
            threading.Thread(target=_uci_index, daemon=True).start()
        raise CatalogWarming("UCI catalogue is loading")
    _uci_lock.acquire()
    try:
        if _uci_cache["items"] and time.time() - _uci_cache["at"] < 86400:
            return _uci_cache["items"]
        lst = httpx.get("https://archive.ics.uci.edu/api/datasets/list", params={"filter": "python", "take": 400, "skip": 0}, timeout=15)
        lst.raise_for_status()
        ids = [d["id"] for d in lst.json().get("data", []) if d.get("id") is not None]
        from concurrent.futures import ThreadPoolExecutor

        def one(i: int) -> dict | None:
            try:
                d = httpx.get("https://archive.ics.uci.edu/api/dataset", params={"id": i}, timeout=10).json().get("data") or {}
                if not d.get("data_url"):
                    return None
                targets = d.get("target_col") or []
                return {"uci_id": i, "name": d.get("name", f"UCI {i}"), "abstract": (d.get("abstract") or "")[:300],
                        "area": d.get("area") or "", "tasks": d.get("tasks") or [], "rows": int(d.get("num_instances") or 0),
                        "cols": int(d.get("num_features") or 0) + len(targets), "target": targets[0] if targets else None,
                        "data_url": d["data_url"], "page": d.get("repository_url") or f"https://archive.ics.uci.edu/dataset/{i}"}
            except Exception:  # noqa: BLE001
                return None

        with ThreadPoolExecutor(max_workers=24) as pool:
            items = [x for x in pool.map(one, ids) if x]
        if items:
            _uci_cache.update(at=time.time(), items=items)
        return items
    finally:
        _uci_lock.release()


def uci_search(domain: str = "", task: str = "", query: str = "", limit: int = 30) -> list[dict]:
    items = _uci_index(block=False)
    q = (query or "").lower().split()
    areas = _UCI_AREAS.get((domain or "").lower(), set())
    out = []
    for it in items:
        if not meets_size(it["rows"], it["cols"]):
            continue
        if task and it["tasks"] and task.capitalize() not in it["tasks"]:
            continue
        hay = f"{it['name']} {it['abstract']} {it['area']}".lower()
        hits = sum(w in hay for w in q)
        if q and hits == 0:
            continue
        out.append((0 if it["area"].lower() in areas else 1, -hits, -it["rows"], {
            "id": f"uci:{it['uci_id']}", "name": it["name"], "source": "uci", "description": it["abstract"][:200],
            "rows": it["rows"], "cols": it["cols"], "size_mb": None, "domains": [], "tasks": [t.lower() for t in it["tasks"]],
            "license": "CC BY 4.0 (UCI)", "synthetic": False, "url": it["page"], "default_target": it["target"], "match": it["area"].lower() in areas,
            "shape_checked": True,
        }))
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in out[:limit]]


def suggest(domain: str, task: str, query: str = "", live: bool = True, source: str = "recommended",
            include_small: bool = False) -> dict:
    """One source at a time. Every source follows the same size rule (rows and columns minimums) except the user's
    own uploads. `include_small` adds the small practice datasets to the recommended list."""
    source = source if source in SOURCES else "recommended"
    domain_l = (domain or "").lower()
    warnings: list[str] = []
    rule = size_rule()
    if source == "recommended":
        ranked = []
        for item in BUILTIN:
            if task and task not in item.tasks:
                continue
            rows, cols, _ = _shape_of(item.id)
            big = meets_size(rows, cols)
            if not big and not include_small:
                continue
            match = any(d.lower() == domain_l for d in item.domains)
            ranked.append((0 if match else 1, 0 if big else 1, item.name, _card(item, match)))
        ranked.sort(key=lambda r: (r[0], r[1], r[2]))
        return {"items": [r[3] for r in ranked], "warnings": warnings, "source": source, "rule": rule}
    items: list[dict] = []
    if not live:
        return {"items": [], "warnings": warnings, "source": source, "rule": rule}
    label = {"kaggle": "Kaggle", "huggingface": "Hugging Face", "uci": "UCI"}[source]
    try:
        if source == "kaggle":
            if not kaggle_enabled():
                warnings.append("Kaggle search is off: the server has no Kaggle API token yet.")
            else:
                items = kaggle_search_domain(domain, query)
        elif source == "huggingface":
            items = hf_search(query or f"{domain} {task}".strip(), task=task)
        else:
            items = uci_search(domain, task, query)
        if not items and not warnings:
            hint = (" UCI has very few datasets this large (about 8 in total, none in finance): clear the search box to see them, "
                    "or try Hugging Face / Kaggle.") if source == "uci" else ""
            warnings.append(f"{label} had no datasets with at least {rule['min_rows']:,} rows and {rule['min_cols']} columns for this search. "
                            "Try other keywords, another source, or upload your own file." + hint)
    except CatalogWarming:
        warnings.append("The UCI catalogue is still loading (the first load after a restart takes about 20 seconds). "
                        "Press Search again in a moment.")
    except PermissionError:
        warnings.append(f"{label} rejected the server's credentials. Check the token.")
    except Exception as exc:  # noqa: BLE001 - network providers are best-effort
        warnings.append(f"{label} search failed ({type(exc).__name__}). Try again in a moment.")
    return {"items": items, "warnings": warnings, "source": source, "rule": rule}


def _download_zip_csv(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        best = None
        for info in zf.infolist():
            if info.filename.lower().endswith(".csv") and 0 < info.file_size <= 300 * 1024 * 1024:
                if best is None or info.file_size > best.file_size:  # the biggest CSV is the main table
                    best = info
        if best is not None:
            return zf.read(best)
    raise DatasetError("No CSV file found inside the downloaded archive.")


def _enforce_size(df: pd.DataFrame, where: str) -> None:
    r = size_rule()
    if len(df) < r["min_rows"] or df.shape[1] < r["min_cols"]:
        raise DatasetError(
            f"This {where} dataset has {len(df):,} rows and {df.shape[1]} columns. Search results must have at least "
            f"{r['min_rows']:,} rows and {r['min_cols']} columns. Pick another one, or use the Upload tab for your own data.", 422)


def load_external(item_id: str) -> tuple[pd.DataFrame, str, str | None]:
    """Returns (frame, display name, default target). Cached on disk, and reads only the part of a remote file it
    needs (see fastload.py), falling back to a full download when a host has no Range support."""
    from . import fastload as FL

    hit = FL.cache_get(item_id)
    if hit:
        return hit
    df, name, target = _load_external_uncached(item_id)
    FL.cache_put(item_id, df, name, target)
    return df, name, target


def _load_external_uncached(item_id: str) -> tuple[pd.DataFrame, str, str | None]:
    from . import fastload as FL

    s = get_settings()
    max_bytes = s.max_external_mb * 1024 * 1024
    if item_id.startswith("kaggle:"):
        ref = item_id.split(":", 1)[1]
        if not kaggle_enabled() or ref.count("/") != 1:
            raise DatasetError("Kaggle downloads are not configured.")
        headers, basic = _kaggle_auth()
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
        try:
            raw_csv = FL.zip_csv_range(url, headers, basic)
        except Exception:  # noqa: BLE001 - any failure of the fast path falls back to the plain download
            raw_csv = _download_zip_csv(fetch_bytes(url, max_bytes, auth=basic, headers=headers))
        df = parse_csv(raw_csv)
        _enforce_size(df, "Kaggle")
        return df, ref, None
    if item_id.startswith("hf:"):
        did = item_id.split(":", 1)[1]
        if did.count("/") > 1 or not re.fullmatch(r"[\w.\-]+(/[\w.\-]+)?", did):
            raise DatasetError("Unknown Hugging Face dataset.", 404)
        with ThreadPoolExecutor(2) as pool:  # the two metadata lookups are independent
            meta_f = pool.submit(lambda: httpx.get(f"https://huggingface.co/api/datasets/{did}", timeout=8, headers=_hf_headers()).json())
            urls_f = pool.submit(_hf_parquet_urls, did)
            meta, urls = meta_f.result(), urls_f.result()
        files = [f["rfilename"] for f in meta.get("siblings", []) if f["rfilename"].lower().endswith(".csv")]
        if files:
            files.sort(key=lambda f: (0 if "train" in f.lower() else 1, len(f)))
            df = parse_csv(FL.csv_head(f"https://huggingface.co/datasets/{did}/resolve/main/{files[0]}", max_bytes))
        else:  # most tabular datasets are parquet; Hugging Face also publishes an auto-converted copy of every dataset
            if not urls:
                raise DatasetError("This Hugging Face dataset has no CSV or Parquet file the server can read (JSON and "
                                   "image/text-only datasets are not supported). Pick another one.", 422)
            try:
                df = FL.parquet_range(urls[0], headers=_hf_headers())
            except Exception:  # noqa: BLE001 - fast path failed; fall back to the plain download
                try:
                    df = parse_parquet(fetch_bytes(urls[0], max_bytes, headers=_hf_headers()))
                except DatasetError as exc:
                    if exc.status == 413:
                        raise DatasetError(f"This Hugging Face dataset is bigger than the {s.max_external_mb} MB download limit. "
                                           "Pick a smaller one (the card shows the size), or raise MAX_EXTERNAL_MB.", 413) from exc
                    raise
        _enforce_size(df, "Hugging Face")
        return df, did, None
    if item_id.startswith("uci:"):
        uid = item_id.split(":", 1)[1]
        if not uid.isdigit():
            raise DatasetError("Unknown UCI dataset.", 404)
        meta = httpx.get("https://archive.ics.uci.edu/api/dataset", params={"id": int(uid)}, timeout=10).json().get("data") or {}
        url = meta.get("data_url")
        if not url:
            raise DatasetError("UCI has no CSV for this dataset.", 404)
        try:
            raw = FL.zip_csv_range(url, None, None)  # UCI usually serves a ZIP; read just the CSV inside
        except Exception:  # noqa: BLE001
            raw = fetch_bytes(url, max_bytes)
            if raw[:2] == b"PK":
                raw = _download_zip_csv(raw)
        df = parse_csv(raw)
        _enforce_size(df, "UCI")
        targets = meta.get("target_col") or []
        return df, meta.get("name") or f"UCI {uid}", (targets[0] if targets and targets[0] in df.columns else None)
    raise DatasetError("Unknown dataset source", 404)


# --------------------------------------------------------------------------- URL safety
_ALLOWED_HOSTS = {
    "drive.google.com", "docs.google.com", "raw.githubusercontent.com", "gist.githubusercontent.com", "github.com",
    "huggingface.co", "www.kaggle.com", "archive.ics.uci.edu", "storage.googleapis.com",
}

def _hf_headers() -> dict:
    """Optional Hugging Face token (HF_TOKEN) for higher rate limits; public datasets work without it."""
    t = get_settings().hf_token
    return {"Authorization": f"Bearer {t}"} if t else {}


_ALLOWED_SUFFIXES = (".huggingface.co", ".hf.co", ".googleusercontent.com", ".githubusercontent.com", ".kaggle.com", ".kaggleusercontent.com", ".googleapis.com", ".uci.edu")


def normalize_url(url: str) -> str:
    u = urlparse(url.strip())
    host = (u.hostname or "").lower()
    if host == "drive.google.com":
        parts = [p for p in u.path.split("/") if p]
        if "d" in parts and parts.index("d") + 1 < len(parts):
            return f"https://drive.google.com/uc?export=download&id={parts[parts.index('d') + 1]}"
        qid = parse_qs(u.query).get("id")
        if qid:
            return f"https://drive.google.com/uc?export=download&id={qid[0]}"
    if host == "github.com" and "/blob/" in u.path:
        return "https://raw.githubusercontent.com" + u.path.replace("/blob/", "/", 1)
    return url.strip()


def validate_public_url(url: str, resolver=socket.getaddrinfo) -> str:
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not host:
        raise DatasetError("Only https links are allowed.")
    if u.username or u.password:
        raise DatasetError("Links with credentials are not allowed.")
    if host not in _ALLOWED_HOSTS and not host.endswith(_ALLOWED_SUFFIXES):
        raise DatasetError(
            "That host is not allowed. Use a Google Drive, GitHub, Hugging Face, Kaggle or UCI link, or upload the CSV."
        )
    try:
        infos = resolver(host, u.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise DatasetError("Could not resolve that host.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise DatasetError("That address is not allowed.")
    return url


def fetch_bytes(url: str, max_bytes: int, auth: tuple[str, str] | None = None, headers: dict | None = None) -> bytes:
    """Download with redirect re-validation, a hard size cap and a timeout."""
    current = normalize_url(url)
    with httpx.Client(follow_redirects=False, timeout=20) as client:
        for _ in range(5):
            validate_public_url(current)
            host = (urlparse(current).hostname or "").lower()
            kaggle = host == "www.kaggle.com"  # credentials never follow a redirect to another host
            with client.stream("GET", current, auth=auth if kaggle else None, headers=headers if kaggle else None) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    loc = r.headers.get("location")
                    if not loc:
                        raise DatasetError("Bad redirect from source.")
                    current = httpx.URL(current).join(loc).__str__()
                    continue
                if r.status_code >= 400:
                    raise DatasetError(f"The source returned HTTP {r.status_code}.", 502)
                chunks, total = [], 0
                for chunk in r.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise DatasetError(f"File is larger than {max_bytes // (1024 * 1024)} MB.", 413)
                    chunks.append(chunk)
                return b"".join(chunks)
    raise DatasetError("Too many redirects.")


# --------------------------------------------------------------------------- parsing & profiling
def parse_parquet(data: bytes) -> pd.DataFrame:
    """Parquet (Hugging Face's usual format). Reads at most MAX_LOAD_ROWS rows and about MAX_CELLS cells so a wide
    table cannot exhaust the server's memory."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise DatasetError("Parquet support is not installed on the server (pip install pyarrow).", 501) from exc
    try:
        pf = pq.ParquetFile(io.BytesIO(data))
        ncols = max(1, len(pf.schema_arrow.names))
        cap = max(1000, min(MAX_LOAD_ROWS, MAX_CELLS // ncols))
        frames, n = [], 0
        for batch in pf.iter_batches(batch_size=20_000):
            frames.append(batch.to_pandas())
            n += len(frames[-1])
            if n >= cap:
                break
        df = pd.concat(frames, ignore_index=True).head(cap) if frames else pd.DataFrame()
    except DatasetError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read this file as Parquet: {str(exc)[:120]}") from exc
    return _clean_frame(df)


def parse_csv(data: bytes) -> pd.DataFrame:
    if not data.strip():
        raise DatasetError("The file is empty.")
    text = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    assert text is not None
    sample = text[:4096]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, nrows=MAX_LOAD_ROWS)
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not read this file as CSV: {str(exc)[:120]}") from exc
    return _clean_frame(df)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] < 2 or len(df) < 20:
        raise DatasetError("Need at least 2 columns and 20 rows.")
    if df.shape[1] > MAX_COLUMNS:
        raise DatasetError(f"Too many columns (max {MAX_COLUMNS}).")
    cols, seen = [], set()
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"col_{i}"
        base, k = name, 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen.add(name)
        cols.append(name)
    df.columns = cols
    for c in df.columns:
        s = df[c]
        if (pd.api.types.is_string_dtype(s) or s.dtype == object) and any(k in c.lower() for k in ("date", "time")):
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() >= 0.9:
                df[c] = parsed
    return df


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    nun = int(s.nunique(dropna=True))
    n = max(len(s), 1)
    avg_len = float(s.dropna().astype(str).str.len().mean()) if nun else 0.0
    if avg_len > 40 and nun > 0.5 * n:
        return "text"
    return "categorical"


def profile(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": c, "kind": column_kind(s), "dtype": str(s.dtype),
            "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True)),
        })
    return {"rows": int(len(df)), "cols": int(df.shape[1]), "columns": cols}


# --------------------------------------------------------------------------- in-memory dataset store
@dataclass
class StoredDataset:
    id: str
    uid: str
    name: str
    source: str
    df: pd.DataFrame
    created: float = field(default_factory=time.time)
    default_target: str | None = None
    ref: str | None = None


class DatasetStore:
    MAX_PER_USER = 3

    def __init__(self) -> None:
        self._items: dict[str, StoredDataset] = {}
        self._lock = threading.Lock()

    def sweep(self) -> None:
        ttl = get_settings().dataset_ttl_seconds
        now = time.time()
        with self._lock:
            for k in [k for k, v in self._items.items() if now - v.created > ttl]:
                del self._items[k]

    def put(self, uid: str, name: str, df: pd.DataFrame, source: str, default_target: str | None = None,
            ref: str | None = None) -> StoredDataset:
        self.sweep()
        with self._lock:
            mine = sorted((v for v in self._items.values() if v.uid == uid), key=lambda v: v.created)
            while len(mine) >= self.MAX_PER_USER:
                del self._items[mine.pop(0).id]
            ds = StoredDataset(id=uuid.uuid4().hex[:16], uid=uid, name=name, source=source, df=df, default_target=default_target,
                               ref=ref)
            self._items[ds.id] = ds
            return ds

    def get(self, ds_id: str, uid: str) -> StoredDataset:
        self.sweep()
        with self._lock:
            ds = self._items.get(ds_id)
        if not ds or ds.uid != uid:
            raise DatasetError("Dataset expired or not found. Please select it again.", 404)
        return ds

    def keep(self, ds_id: str) -> None:
        """Restart the in-memory clock; the dataset is wiped DATASET_TTL_SECONDS after its last use."""
        with self._lock:
            if ds_id in self._items:
                self._items[ds_id].created = time.time()

    def drop(self, ds_id: str) -> None:
        with self._lock:
            self._items.pop(ds_id, None)


datasets = DatasetStore()
