"""Exploratory data analysis: chart-ready JSON plus rule-based written inference."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .datasets import column_kind


def _n(x):
    """numpy/pandas scalar -> JSON-safe python value (NaN/inf -> None)."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _hist(s: pd.Series, bins: int = 10) -> list[dict]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return []
    counts, edges = np.histogram(s, bins=bins)
    return [{"label": f"{_n(edges[i])}–{_n(edges[i + 1])}", "start": _n(edges[i]), "count": int(counts[i])}
            for i in range(len(counts))]


def build_eda(df: pd.DataFrame, target: str, task: str, columns: list[str] | None = None) -> dict:
    kinds = {c: column_kind(df[c]) for c in df.columns}
    feats = [c for c in df.columns if c != target]
    if columns:  # wide dataset: the user (or the automatic picker) chose which features to explore
        keep = set(columns)
        feats = [c for c in feats if c in keep]
    num = [c for c in feats if kinds[c] == "numeric"]
    cat = [c for c in feats if kinds[c] in ("categorical", "boolean")]
    n_rows = len(df)

    miss = df.isna().sum()
    missing = [{"column": c, "missing": int(miss[c]), "pct": round(100 * miss[c] / max(n_rows, 1), 2)}
               for c in miss.sort_values(ascending=False).index if miss[c] > 0][:15]

    dup = int(df.duplicated().sum())
    overview = {
        "rows": n_rows, "cols": int(df.shape[1]), "duplicates": dup,
        "missing_cells": int(miss.sum()), "missing_pct": round(100 * miss.sum() / max(df.size, 1), 2),
        "numeric_cols": len(num), "categorical_cols": len(cat),
        "datetime_cols": sum(1 for c in feats if kinds[c] == "datetime"),
        **({"explored_cols": len(feats) + 1} if columns else {}),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }

    summary, skewed, outlier_cols = [], [], []
    for c in num[:20]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        skew = _n(s.skew())
        summary.append({"column": c, "mean": _n(s.mean()), "std": _n(s.std()), "min": _n(s.min()), "p25": _n(q1),
                        "median": _n(s.median()), "p75": _n(q3), "max": _n(s.max()), "skew": skew, "outliers": outliers})
        if skew is not None and abs(skew) > 1:
            skewed.append(c)
        if outliers > 0.05 * len(s):
            outlier_cols.append(c)

    hist_cols = [c for c in num if df[c].nunique() > 6][:8]
    histograms = [{"column": c, "bins": _hist(df[c])} for c in hist_cols]

    # target
    tgt: dict = {"column": target, "task": task}
    imbalance = None
    tser = df[target]
    if task == "classification":
        vc = tser.value_counts(dropna=True).head(20)
        total = int(vc.sum()) or 1
        tgt["classes"] = [{"label": str(k), "count": int(v), "pct": round(100 * v / total, 2)} for k, v in vc.items()]
        if len(vc) >= 2 and vc.min() > 0:
            imbalance = round(float(vc.max() / vc.min()), 2)
        tgt["imbalance_ratio"] = imbalance
    else:
        ts = pd.to_numeric(tser, errors="coerce").dropna()
        tgt["bins"] = _hist(ts)
        tgt["mean"], tgt["std"], tgt["skew"] = _n(ts.mean()), _n(ts.std()), _n(ts.skew())

    # correlation
    tnum = pd.to_numeric(tser, errors="coerce")
    corr_cols = list(num)
    target_corr: dict[str, float] = {}
    if tnum.notna().mean() > 0.9 and (task == "regression" or tnum.nunique() <= 20):
        for c in num:
            r = pd.to_numeric(df[c], errors="coerce").corr(tnum)
            if r is not None and not math.isnan(r):
                target_corr[c] = float(r)
        corr_cols = sorted(num, key=lambda c: -abs(target_corr.get(c, 0)))
    corr_cols = corr_cols[:10]
    correlation = {"columns": [], "matrix": []}
    pairs = []
    if len(corr_cols) >= 2:
        cm = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        correlation = {"columns": corr_cols, "matrix": [[_n(v) for v in row] for row in cm.to_numpy()]}
        for i, a in enumerate(corr_cols):
            for b in corr_cols[i + 1:]:
                v = cm.loc[a, b]
                if not math.isnan(v) and abs(v) >= 0.85:
                    pairs.append((a, b, round(float(v), 3)))

    categorical = []
    for c in cat[:5]:
        vc = df[c].astype(str).value_counts().head(8)
        categorical.append({"column": c, "unique": int(df[c].nunique()),
                            "values": [{"label": k, "count": int(v)} for k, v in vc.items()]})

    # --- written inference
    inf: list[dict] = []
    add = lambda level, text: inf.append({"level": level, "text": text})  # noqa: E731
    add("info", f"The dataset has {n_rows:,} rows and {df.shape[1]} columns: {len(num)} numeric, {len(cat)} categorical"
                f"{', ' + str(overview['datetime_cols']) + ' date/time' if overview['datetime_cols'] else ''}.")
    heavy = [m for m in missing if m["pct"] > 30]
    if heavy:
        add("warn", "Columns with more than 30% missing values: " + ", ".join(m["column"] for m in heavy[:5])
            + ". Consider dropping them; imputing this much invented data can mislead the model.")
    elif missing:
        add("info", f"{len(missing)} column(s) have missing values (highest: {missing[0]['column']} at {missing[0]['pct']}%). "
                    "Imputation in the preprocessing step handles this without leaking test data.")
    else:
        add("good", "No missing values.")
    if dup:
        add("warn", f"{dup} duplicate row(s) found. Duplicates can leak between train and test and inflate scores.")
    if skewed:
        add("info", f"{len(skewed)} numeric column(s) are strongly skewed (e.g. {', '.join(skewed[:3])}). "
                    "Tree models ignore this; linear, KNN and SVM models benefit from scaling or a log transform.")
    if outlier_cols:
        add("info", "Notable outliers (over 5% of values) in: " + ", ".join(outlier_cols[:4]) + ". RobustScaler is a safe choice.")
    if task == "classification" and imbalance:
        if imbalance >= 3:
            add("warn", f"Class imbalance: the largest class is {imbalance}x the smallest. Judge models by F1 / ROC-AUC, not accuracy alone.")
        else:
            add("good", f"Classes are reasonably balanced (largest/smallest = {imbalance}x).")
    if task == "regression" and tgt.get("skew") is not None and abs(tgt["skew"]) > 1:
        add("info", f"The target is skewed (skew {tgt['skew']}). A log transform of the target may improve linear models.")
    if target_corr:
        top = sorted(target_corr.items(), key=lambda kv: -abs(kv[1]))[:3]
        strong = [(k, v) for k, v in top if abs(v) >= 0.3]
        if strong:
            add("good", "Most related to the target: " + ", ".join(f"{k} (r={v:.2f})" for k, v in strong) + ".")
        else:
            add("info", "No numeric feature has a strong linear relationship with the target; non-linear models may do better.")
    if pairs:
        add("warn", "Highly correlated feature pairs (|r| ≥ 0.85): " + ", ".join(f"{a}/{b}" for a, b, _ in pairs[:3])
            + ". This can hurt linear models; PCA or Ridge helps.")
    consts = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    if consts:
        add("warn", "Constant columns carry no information and will be dropped: " + ", ".join(consts[:5]) + ".")
    if df.shape[1] > max(n_rows / 10, 1):
        add("warn", "Many columns compared with rows; consider feature selection to reduce overfitting.")

    return {
        "overview": overview, "missing": missing, "numeric_summary": summary, "histograms": histograms,
        "target": tgt, "correlation": correlation, "top_pairs": [{"a": a, "b": b, "r": r} for a, b, r in pairs],
        "target_correlation": [{"column": k, "r": round(v, 3)} for k, v in
                               sorted(target_corr.items(), key=lambda kv: -abs(kv[1]))[:10]],
        "categorical": categorical, "inference": inf,
        "columns": [{"name": c, "kind": kinds[c], "role": "target" if c == target else "feature",
                     "missing": int(miss[c]), "unique": int(df[c].nunique(dropna=True)),
                     "explored": c == target or c in feats} for c in df.columns],
    }


# --------------------------------------------------------------------------- wide datasets
WIDE_THRESHOLD = 15  # more feature columns than this -> ask which ones to explore


def recommend_eda_columns(df: pd.DataFrame, target: str, task: str, k: int = 12) -> dict:
    """Statistically rank feature columns so wide datasets get a focused EDA.

    Score = strength of association with the target (|Pearson r| for numeric targets, correlation ratio for
    categorical ones), after dropping ID-like, constant and mostly-missing columns. Purely rule-based and cheap;
    the user can always change the selection.
    """
    n = max(len(df), 1)
    tser = df[target]
    tnum = pd.to_numeric(tser, errors="coerce")
    target_numeric = bool(tnum.notna().mean() > 0.9 and (task == "regression" or tnum.nunique() <= 20))
    tcodes = pd.factorize(tser.astype(str))[0] if not target_numeric or task == "classification" else None
    cols, skipped = [], {}
    for c in df.columns:
        if c == target:
            continue
        kind = column_kind(df[c])
        nun = int(df[c].nunique(dropna=True))
        miss = float(df[c].isna().mean())
        if kind == "datetime":
            skipped[c] = "date/time column (not charted)"
            continue
        if nun <= 1:
            skipped[c] = "constant"
            continue
        if miss > 0.5:
            skipped[c] = f"{round(100 * miss)}% missing"
            continue
        if kind in ("categorical", "text") and nun > 0.9 * n:
            skipped[c] = "looks like an ID"
            continue
        score = 0.0
        try:
            if kind == "numeric":
                x = pd.to_numeric(df[c], errors="coerce")
                if task == "regression" and target_numeric:
                    r = x.corr(tnum)
                    score = 0.0 if r is None or math.isnan(r) else abs(float(r))
                else:  # correlation ratio (eta) of the numeric feature against target classes
                    g = pd.DataFrame({"x": x, "t": tcodes}).dropna()
                    if len(g) > 2 and g["x"].var() > 0:
                        means = g.groupby("t")["x"].transform("mean")
                        score = math.sqrt(max(0.0, 1 - ((g["x"] - means) ** 2).sum() / ((g["x"] - g["x"].mean()) ** 2).sum()))
            else:
                if nun > 50:
                    score = 0.0
                elif task == "regression" and target_numeric:
                    g = pd.DataFrame({"x": df[c].astype(str), "t": tnum}).dropna()
                    if len(g) > 2 and g["t"].var() > 0:
                        means = g.groupby("x")["t"].transform("mean")
                        score = math.sqrt(max(0.0, 1 - ((g["t"] - means) ** 2).sum() / ((g["t"] - g["t"].mean()) ** 2).sum()))
                else:
                    ct = pd.crosstab(df[c].astype(str), tser.astype(str))
                    tot = ct.to_numpy().sum()
                    if tot and ct.shape[0] > 1 and ct.shape[1] > 1:
                        exp = ct.sum(axis=1).to_numpy()[:, None] * ct.sum(axis=0).to_numpy()[None, :] / tot
                        chi2 = float((((ct.to_numpy() - exp) ** 2) / exp).sum())
                        score = math.sqrt(chi2 / (tot * (min(ct.shape) - 1)))  # Cramer's V
        except Exception:  # noqa: BLE001 - a column we cannot score simply ranks last
            score = 0.0
        cols.append({"column": c, "kind": kind, "score": round(float(score), 3), "missing_pct": round(100 * miss, 1)})
    cols.sort(key=lambda r: -r["score"])
    recommended = [r["column"] for r in cols[:k]]
    for r in cols:
        r["recommended"] = r["column"] in recommended
    return {"columns": cols, "recommended": recommended, "skipped": skipped,
            "wide": len(df.columns) - 1 > WIDE_THRESHOLD, "threshold": WIDE_THRESHOLD}
