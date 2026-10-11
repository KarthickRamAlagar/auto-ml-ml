"""Data-balancing and feature-selection methods.

The user can pick several methods of each kind. When more than one is picked we try each on a quick baseline
model (3-fold CV on a sample of the TRAINING rows only) and keep the best one, so the extra methods are a real
comparison, not decoration. The first 3 methods per run are free; each further one costs INR 2 (see pricing.py).
"""
from __future__ import annotations

import warnings
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest, VarianceThreshold, f_classif, f_regression,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

EVAL_ROWS = 4000

SELECTION_INFO = {
    "kbest": ("SelectKBest (ANOVA / F-test)", "Keeps the K features with the strongest statistical link to the target."),
    "variance": ("Variance threshold", "Removes features that barely change; quick and safe."),
    "mutual_info": ("Mutual information", "Like K-best but also catches non-linear relationships."),
    "rfe": ("Recursive feature elimination", "Repeatedly drops the weakest feature of a linear model."),
    "lasso": ("L1 (Lasso) selection", "Keeps features a sparse linear model finds useful."),
    "tree": ("Tree importance", "Keeps features a Random Forest uses most."),
}
BALANCING_INFO = {
    "class_weight": ("Class weights", "Makes rare classes count more during training; no rows are added or removed."),
    "oversample": ("Random oversampling", "Duplicates minority-class rows until classes are even."),
    "smote": ("SMOTE", "Creates synthetic minority rows between real neighbours."),
    "undersample": ("Random undersampling", "Drops majority-class rows until classes are even."),
}


# --------------------------------------------------------------------------- feature selection
def make_selector(name: str, task: str, k: int, n_feat: int):
    """Return (estimator, description, code)."""
    k = max(1, min(k, n_feat))
    clf = task == "classification"
    if name == "kbest":
        fn = "f_classif" if clf else "f_regression"
        return (SelectKBest(f_classif if clf else f_regression, k=k), f"SelectKBest keeps {k} of {n_feat} features",
                f"from sklearn.feature_selection import SelectKBest, {fn}\nsteps.append(('select', SelectKBest({fn}, k={k})))")
    if name == "variance":
        return (VarianceThreshold(threshold=0.01), f"VarianceThreshold(0.01) over {n_feat} features",
                "from sklearn.feature_selection import VarianceThreshold\n"
                "steps.append(('select', VarianceThreshold(threshold=0.01)))")
    if name == "mutual_info":
        fn = "mutual_info_classif" if clf else "mutual_info_regression"
        base = mutual_info_classif if clf else mutual_info_regression
        return (SelectKBest(partial(base, random_state=42), k=k), f"Mutual information keeps {k} of {n_feat} features",
                f"from functools import partial\nfrom sklearn.feature_selection import SelectKBest, {fn}\n"
                f"steps.append(('select', SelectKBest(partial({fn}, random_state=42), k={k})))")
    if name == "rfe":
        est = LogisticRegression(max_iter=300) if clf else Ridge()
        e_code = "LogisticRegression(max_iter=300)" if clf else "Ridge()"
        imp = "from sklearn.linear_model import LogisticRegression" if clf else "from sklearn.linear_model import Ridge"
        return (RFE(est, n_features_to_select=k, step=0.2), f"RFE keeps {k} of {n_feat} features",
                f"from sklearn.feature_selection import RFE\n{imp}\n"
                f"steps.append(('select', RFE({e_code}, n_features_to_select={k}, step=0.2)))")
    if name == "lasso":
        if clf:
            est = LogisticRegression(penalty="l1", solver="liblinear", C=0.5)
            e_code = "LogisticRegression(penalty='l1', solver='liblinear', C=0.5)"
            imp = "from sklearn.linear_model import LogisticRegression"
        else:
            est, e_code, imp = Lasso(alpha=0.01, max_iter=5000), "Lasso(alpha=0.01, max_iter=5000)", "from sklearn.linear_model import Lasso"
        return (SelectFromModel(est, threshold=1e-5), "L1 selection keeps features with non-zero weights",
                f"from sklearn.feature_selection import SelectFromModel\n{imp}\n"
                f"steps.append(('select', SelectFromModel({e_code}, threshold=1e-5)))")
    if name == "tree":
        cls = RandomForestClassifier if clf else RandomForestRegressor
        cn = cls.__name__
        est = cls(n_estimators=80, max_depth=8, random_state=42, n_jobs=1)
        return (SelectFromModel(est, threshold="median"), "Tree importance keeps the top half of features",
                f"from sklearn.feature_selection import SelectFromModel\nfrom sklearn.ensemble import {cn}\n"
                f"steps.append(('select', SelectFromModel({cn}(n_estimators=80, max_depth=8, random_state=42), threshold='median')))")
    raise ValueError(f"unknown selection method {name}")


# --------------------------------------------------------------------------- balancing
def make_sampler(name: str, ytr: np.ndarray):
    """Return (sampler | None, description, code). class_weight has no sampler (handled by the model)."""
    if name == "class_weight":
        return None, "class_weight='balanced' on models that support it", "# models that support it get class_weight='balanced'"
    counts = np.bincount(ytr)
    counts = counts[counts > 0]
    if name == "oversample":
        return (RandomOverSampler(random_state=42), "Random oversampling of minority classes",
                "from imblearn.over_sampling import RandomOverSampler\nsteps.append(('sampler', RandomOverSampler(random_state=42)))")
    if name == "undersample":
        return (RandomUnderSampler(random_state=42), "Random undersampling of majority classes",
                "from imblearn.under_sampling import RandomUnderSampler\nsteps.append(('sampler', RandomUnderSampler(random_state=42)))")
    if name == "smote":
        k = int(min(5, counts.min() - 1))
        if k < 1:
            return (RandomOverSampler(random_state=42), "SMOTE needs at least 2 rows per class; used random oversampling instead",
                    "from imblearn.over_sampling import RandomOverSampler\nsteps.append(('sampler', RandomOverSampler(random_state=42)))")
        return (SMOTE(random_state=42, k_neighbors=k), f"SMOTE synthetic rows (k_neighbors={k})",
                f"from imblearn.over_sampling import SMOTE\nsteps.append(('sampler', SMOTE(random_state=42, k_neighbors={k})))")
    raise ValueError(f"unknown balancing method {name}")


# --------------------------------------------------------------------------- compare and pick
def _cv(task: str, y: np.ndarray, folds: int = 3):
    if task == "classification":
        c = np.bincount(y)
        c = c[c > 0]
        if c.min() >= folds:
            return StratifiedKFold(folds, shuffle=True, random_state=42)
    return KFold(folds, shuffle=True, random_state=42)


def _baseline(task: str, class_weight: bool = False):
    if task == "classification":
        return LogisticRegression(max_iter=300, class_weight="balanced" if class_weight else None)
    return Ridge()


def _sample(X: pd.DataFrame, y: np.ndarray, n: int = EVAL_ROWS):
    if len(X) <= n:
        return X, y
    idx = np.sort(np.random.default_rng(7).choice(len(X), n, replace=False))
    return X.iloc[idx], y[idx]


def _score(task: str, steps: list, sampler, class_weight: bool, X, y) -> float:
    pipe = ImbPipeline([(n, clone(e)) for n, e in steps] + ([("sampler", clone(sampler))] if sampler is not None else [])
                       + [("model", _baseline(task, class_weight))])
    scoring = "f1_macro" if task == "classification" else "r2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(pipe, X, y, cv=_cv(task, y), scoring=scoring, error_score="raise")))


def compare(task: str, candidates: list[str], build: Callable[[str], tuple[list, object, bool]], Xtr, ytr, log) -> dict:
    """Score every candidate on a baseline model. `build(name)` returns (steps, sampler, class_weight).
    A candidate that fails is reported and skipped. Returns {tried: [{method, score|error}], chosen}."""
    Xs, ys = _sample(Xtr, ytr)
    tried: list[dict] = []
    for name in candidates:
        try:
            steps, sampler, cw = build(name)
            sc = _score(task, steps, sampler, cw, Xs, ys)
            tried.append({"method": name, "score": round(sc, 4)})
            log(f"  {name}: baseline {'F1 (macro)' if task == 'classification' else 'R²'} = {sc:.4f}")
        except Exception as exc:  # noqa: BLE001 - one failing candidate must not stop the run
            tried.append({"method": name, "error": str(exc)[:120]})
            log(f"  {name}: could not be evaluated ({str(exc)[:80]})")
    ok = [t for t in tried if "score" in t]
    chosen = max(ok, key=lambda t: t["score"])["method"] if ok else None
    return {"tried": tried, "chosen": chosen, "metric": "F1 (macro)" if task == "classification" else "R²"}


# --------------------------------------------------------------------------- advice (before the run)
def advise(df: pd.DataFrame, target: str, task: str) -> dict:
    """Rule-based recommendations shown after EDA. The user can accept them or choose their own."""
    y = df[target].dropna()
    feats = [c for c in df.columns if c != target]
    n_feat = len(feats)
    out: dict = {"features": n_feat, "rows": int(len(df))}
    bal: dict = {"required": False, "recommended": [], "reason": "", "ratio": None, "distribution": []}
    if task == "classification":
        vc = y.astype(str).value_counts()
        if len(vc) >= 2:
            ratio = float(vc.iloc[0] / max(vc.iloc[-1], 1))
            bal["ratio"] = round(ratio, 2)
            bal["distribution"] = [{"label": str(k), "count": int(v)} for k, v in vc.head(12).items()]
            minority = int(vc.iloc[-1])
            if ratio >= 3:
                bal["required"] = True
                rec = ["class_weight"]
                rec.append("smote" if minority >= 6 and len(df) <= 50000 else "undersample")
                bal["recommended"] = rec
                bal["reason"] = (f"The largest class is {ratio:.1f}x the smallest, so models may ignore the rare class. "
                                 "Balancing is recommended.")
            elif ratio >= 1.5:
                bal["recommended"] = ["class_weight"]
                bal["reason"] = f"Mild imbalance ({ratio:.1f}x). Class weights are enough; resampling is optional."
            else:
                bal["reason"] = f"Classes are well balanced ({ratio:.1f}x). Balancing is not required."
    else:
        bal["reason"] = "Balancing applies to classification only."
    out["balancing"] = bal
    sel: dict = {"required": n_feat > 15, "recommended": [], "reason": ""}
    if n_feat > 40:
        sel["recommended"], sel["reason"] = ["lasso", "tree", "mutual_info"], f"{n_feat} features is a lot; selecting the useful ones speeds training and reduces noise."
    elif n_feat > 15:
        sel["recommended"], sel["reason"] = ["tree", "kbest"], f"{n_feat} features; trimming the weakest may help."
    elif n_feat > 8:
        sel["recommended"], sel["reason"] = ["variance"], "A modest number of features; selection is optional."
    else:
        sel["reason"] = f"Only {n_feat} features, so feature selection is not required."
    out["selection"] = sel
    out["info"] = {"balancing": {k: {"name": v[0], "blurb": v[1]} for k, v in BALANCING_INFO.items()},
                   "selection": {k: {"name": v[0], "blurb": v[1]} for k, v in SELECTION_INFO.items()}}
    return out
