"""Model advisor: understands a typed model request and recommends settings for each chosen model.

Recommendations are computed on a small sample of the user's data (the sample is never stored), so each value
comes with a measured reason, for example "depth 6 scored 0.81 on held-out rows vs 0.78 unpruned".
An LLM, when configured, only rewrites the one-paragraph summary; it never picks numbers.
"""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd

from .datasets import column_kind
from .models_registry import CLF, MODELS, REG

# ---- which parameters the advisor asks the user about, per model (everything else keeps a sensible default)
ASK: dict[str, list[str]] = {
    "logreg": ["C", "class_weight"],
    "linreg": [],
    "ridge": ["alpha"],
    "lasso": ["alpha"],
    "elastic": ["alpha", "l1_ratio"],
    "knn": ["n_neighbors", "weights"],
    "svm": ["C", "kernel", "gamma"],
    "tree": ["pruning", "criterion", "max_depth", "min_samples_leaf", "ccp_alpha"],
    "rf": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
    "extra": ["n_estimators", "max_depth", "min_samples_leaf", "max_features"],
    "gb": ["n_estimators", "learning_rate", "max_depth", "subsample"],
    "ada": ["n_estimators", "learning_rate"],
    "nb": ["var_smoothing"],
    "mlp": ["hidden_layers", "activation", "learning_rate_init", "alpha", "max_iter", "early_stopping"],
}

# ---- typed requests: "random forest and svm", "xgboost?", "ann", "decision tree"
ALIASES: dict[str, list[str]] = {
    "logreg": ["logistic regression", "logistic", "logreg", "log reg"],
    "linreg": ["linear regression", "linreg", "ols", "least squares"],
    "ridge": ["ridge"],
    "lasso": ["lasso"],
    "elastic": ["elastic net", "elasticnet", "elastic"],
    "knn": ["k nearest", "k-nearest", "knn", "nearest neighbour", "nearest neighbor"],
    "svm": ["support vector", "svm", "svc", "svr"],
    "tree": ["decision tree", "decision-tree", "cart", "dtree", "tree"],
    "rf": ["random forest", "random-forest", "randomforest", "forest", "rf"],
    "extra": ["extra trees", "extra-trees", "extratrees", "extremely randomized"],
    "gb": ["gradient boosting", "gradient boost", "gbm", "gbdt", "gb", "boosting"],
    "ada": ["adaboost", "ada boost", "ada"],
    "nb": ["naive bayes", "naive-bayes", "gaussian nb", "nb", "bayes"],
    "mlp": ["neural network", "neural net", "mlp", "ann", "deep learning", "perceptron", "nn"],
}
NOT_OFFERED = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "lstm": "LSTM", "cnn": "CNN",
               "transformer": "Transformers"}


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9+\- ]+", " ", text.lower()) + " "


def parse_request(text: str, task: str) -> dict:
    """Map free text to model ids with plain rules. Longest alias wins, so 'extra trees' is not read as 'tree'."""
    t = _norm(text)
    hits: dict[str, int] = {}
    claimed = t
    pairs = sorted(((a, mid) for mid, al in ALIASES.items() for a in al), key=lambda p: -len(p[0]))
    for alias, mid in pairs:
        pat = f" {alias} "
        pos = claimed.find(pat)
        if pos < 0 and len(alias) > 3:
            pos = claimed.find(alias)
        if pos >= 0:
            hits.setdefault(mid, pos)
            claimed = claimed.replace(alias, " " * len(alias))  # keep positions stable for the next alias
    found = sorted(hits, key=hits.get)  # order of appearance in the user's text
    unavailable = [MODELS[m].name for m in found if task not in MODELS[m].classes]
    ok = [m for m in found if task in MODELS[m].classes]
    wanted_but_missing = [label for key, label in NOT_OFFERED.items() if key in t]
    return {"model_ids": ok, "not_for_task": unavailable, "not_offered": wanted_but_missing}


# ---- quick numeric matrix from the raw frame (sample only, never stored)
def _quick_xy(df: pd.DataFrame, target: str, task: str, max_rows: int = 8000):
    d = df.sample(min(len(df), max_rows), random_state=7) if len(df) > max_rows else df
    y = d[target]
    ok = y.notna()
    d, y = d[ok], y[ok]
    feats = []
    for c in d.columns:
        if c == target:
            continue
        kind = column_kind(d[c])
        if kind == "numeric" or kind == "boolean":
            feats.append(d[c].astype(float).fillna(d[c].astype(float).median()).rename(c))
        elif kind == "categorical" and d[c].nunique() <= 15:
            feats.append(pd.get_dummies(d[c].astype(str), prefix=c, dtype=float))
    if not feats:
        return None, None
    X = pd.concat(feats, axis=1).fillna(0.0)
    if task == CLF:
        y = pd.factorize(y.astype(str))[0]
    else:
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna().to_numpy()
        X, y = X[keep], y[keep].to_numpy()
    return X.to_numpy(dtype=float), np.asarray(y)


def _shape_facts(df: pd.DataFrame, target: str, task: str) -> dict:
    n, c = df.shape
    facts = {"rows": int(n), "cols": int(c - 1), "task": task}
    if task == CLF:
        vc = df[target].value_counts()
        facts["classes"] = int(len(vc))
        facts["imbalance"] = round(float(vc.max() / max(vc.min(), 1)), 2) if len(vc) > 1 else 1.0
    return facts


def _best_simple(scores: list[tuple], tol: float = 0.005):
    """scores: [(value, score, complexity_rank)]. Pick the simplest option within `tol` of the best score."""
    best = max(s[1] for s in scores)
    near = [s for s in scores if s[1] >= best - tol]
    return min(near, key=lambda s: s[2]), best


def _score(task: str, model, Xtr, ytr, Xva, yva) -> float:
    model.fit(Xtr, ytr)
    return float(model.score(Xva, yva))


def _recommend_tree(X, y, task: str, facts: dict) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cls = DecisionTreeClassifier if task == CLF else DecisionTreeRegressor
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y if task == CLF and len(set(y)) > 1 else None)
    full = cls(random_state=42).fit(Xtr, ytr)
    full_score = float(full.score(Xva, yva))
    depths = [3, 4, 5, 6, 8, 10, 12, None]
    d_scores = [(d, _score(task, cls(max_depth=d, random_state=42), Xtr, ytr, Xva, yva), i) for i, d in enumerate(depths)]
    (depth, d_best, _), _b = _best_simple(d_scores)
    leaves = [1, 5, 10, 20, 50]
    l_scores = [(v, _score(task, cls(min_samples_leaf=v, random_state=42), Xtr, ytr, Xva, yva), len(leaves) - i) for i, v in enumerate(leaves)]
    (leaf, l_best, _), _b2 = _best_simple(l_scores)
    path = full.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas
    alphas = np.unique(np.quantile(path, np.linspace(0, 0.6, 14))) if len(path) > 14 else path
    a_scores = [(float(a), _score(task, cls(ccp_alpha=float(a), random_state=42), Xtr, ytr, Xva, yva), -float(a)) for a in alphas]
    (alpha, a_best, _), _b3 = _best_simple(a_scores)
    alpha = float(min(max(alpha, 0.0), 0.1))
    metric = "accuracy" if task == CLF else "R²"
    pre_best = max(d_best, l_best)
    if max(pre_best, a_best) <= full_score + 0.003:
        pruning = "none"
    elif a_best >= pre_best + 0.003:
        pruning = "post"
    else:
        pruning = "pre"
    return {
        "pruning": (pruning, f"An unpruned tree scored {metric} {full_score:.3f} on held-out rows. Best pre-pruning "
                              f"{max(d_best, l_best):.3f}, best post-pruning {a_best:.3f}."),
        "max_depth": (depth, f"Depth {depth if depth else 'unlimited'} scored {d_best:.3f} on held-out rows; shallower trees were within 0.5% so the simplest was chosen."),
        "min_samples_leaf": (int(leaf), f"At least {leaf} rows per leaf scored {l_best:.3f} and keeps leaves from fitting noise."),
        "ccp_alpha": (round(alpha, 5), f"Cost-complexity alpha {alpha:.4f} scored {a_best:.3f} along the pruning path; it removes the weakest branches."),
        "criterion": ("gini" if task == CLF else "squared_error", "The default criterion is a safe choice; the others rarely change accuracy much."),
    }


def _recommend_linear(kind: str, X, y, task: str) -> dict:
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    if kind == "ridge":
        m = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(Xs, y)
        return {"alpha": (round(float(m.alpha_), 4), f"Cross-validation over 11 values picked alpha {m.alpha_:.3g}.")}
    if kind == "lasso":
        m = LassoCV(cv=3, n_alphas=30, max_iter=3000).fit(Xs, y)
        return {"alpha": (round(float(max(m.alpha_, 0.0005)), 5), f"Cross-validation picked alpha {m.alpha_:.3g}; larger values drop more features.")}
    m = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=3, n_alphas=20, max_iter=3000).fit(Xs, y)
    return {"alpha": (round(float(max(m.alpha_, 0.0005)), 5), f"Cross-validation picked alpha {m.alpha_:.3g}."),
            "l1_ratio": (float(m.l1_ratio_), f"L1 ratio {m.l1_ratio_} scored best in cross-validation.")}


def _recommend_knn(X, y, task: str, facts: dict) -> dict:
    n = facts["rows"]
    k = int(max(3, min(31, round(np.sqrt(min(n, 10000)) / 2))))
    k += 1 - k % 2  # odd K avoids ties
    return {"n_neighbors": (k, f"About half of √n ({n:,} rows) and odd, which balances noise against detail."),
            "weights": ("distance", "Closer rows count more, which helps when neighbours differ in distance.")}


def build_plan(df: pd.DataFrame, target: str, task: str, model_ids: list[str]) -> dict:
    facts = _shape_facts(df, target, task)
    X, y = _quick_xy(df, target, task)
    n, nf = facts["rows"], facts["cols"]
    imbalance = facts.get("imbalance", 1.0)
    plans: dict[str, dict] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mid in model_ids:
            m = MODELS.get(mid)
            if not m or task not in m.classes:
                continue
            rec: dict[str, tuple] = {}
            try:
                if mid == "tree" and X is not None:
                    rec = _recommend_tree(X, y, task, facts)
                elif mid in ("ridge", "lasso", "elastic") and X is not None:
                    rec = _recommend_linear(mid, X, y, task)
                elif mid == "knn":
                    rec = _recommend_knn(X, y, task, facts)
            except Exception:  # noqa: BLE001 - fall back to the rules below
                rec = {}
            big = n > 30000
            rules: dict[str, tuple] = {
                "C": (1.0, "C = 1 is the standard starting point; the Auto-tune option searches 0.1 to 10."),
                "class_weight": (("balanced", f"Classes are imbalanced ({imbalance:g}:1), so rare classes get more weight.") if imbalance >= 3
                                 else ("none", "Classes are reasonably balanced, so no re-weighting is needed.")),
                "n_estimators": ((200, "More trees reduce variance; 200 is a good balance of accuracy and time.") if n > 5000 and not big and mid != "gb"
                                 else (100, "100 trees is plenty" + (" and keeps training fast on a large dataset." if big else " for a dataset this size."))),
                "min_samples_leaf": ((5, f"With {n:,} rows, leaves of at least 5 rows reduce noise.") if n > 20000
                                     else (2, "A small leaf size keeps detail but stops single-row leaves.") if n > 3000
                                     else (1, "A small dataset needs fine leaves.")),
                "max_features": ("sqrt", "√(features) per split is the usual choice for decorrelated trees."),
                "learning_rate": ((0.1, "0.1 is the standard speed/accuracy trade-off.") if mid != "ada" else (0.5, "A gentler rate is steadier for AdaBoost.")),
                "subsample": ((0.8, "Each tree sees 80% of rows: faster and less prone to overfit on large data.") if big
                              else (1.0, "Use all rows; the dataset is small enough.")),
                "kernel": ("rbf", "RBF handles non-linear boundaries and is the usual first choice."),
                "gamma": ("scale", "'scale' adapts gamma to the data's variance."),
                "var_smoothing": (1e-9, "The default adds just enough smoothing to avoid zero variances."),
                "hidden_layers": (("50", f"{nf} features is small; one layer of 50 units is enough.") if nf < 12
                                  else ("100,50", f"{nf} features suits two layers (100 then 50).") if nf < 40
                                  else ("128,64", f"{nf} features suits a wider network (128 then 64).")),
                "activation": ("relu", "ReLU trains fastest and works well on scaled data."),
                "learning_rate_init": (0.001, "0.001 is the standard Adam learning rate."),
                "alpha": ((0.001, "A stronger L2 penalty helps a small dataset avoid overfitting.") if n < 2000 else (0.0001, "The default L2 penalty is enough with this much data.")),
                "max_iter": ((500, "A small dataset needs more passes to converge.") if n < 2000 else (300, "300 epochs is usually enough with early stopping.")),
                "early_stopping": (("yes", "Large dataset: stop when validation stalls to save time and avoid overfitting.") if n > 10000
                                   else ("no", "Small dataset: keep every row for training.")),
                "max_depth": ((3, "Shallow trees are the standard for boosting.") if mid in ("gb",)
                              else (None if not big else 20, "Unlimited depth is fine for forests; capped at 20 on very large data for speed.")),
            }
            if mid in ("ridge", "lasso", "elastic"):
                rules["alpha"] = (1.0 if mid == "ridge" else 0.1, "A moderate penalty is a safe starting point.")
            params = []
            for spec in m.params_for(task):
                asked = spec.name in ASK.get(mid, [])
                pub = spec.public()
                val, why = rec.get(spec.name) or rules.get(spec.name) or (spec.default, "Library default.")
                if mid == "ada" and spec.name == "n_estimators":
                    val, why = 100, "100 weak learners is a common choice."
                pub.update(recommended=val, reason=why, ask=asked)
                params.append(pub)
            plans[mid] = {"id": mid, "name": m.name, "params": params, "measured": bool(rec)}
    return {"facts": facts, "plans": plans}


def summarise(facts: dict, plans: dict, router=None) -> dict:
    """One paragraph for the form header. Rule text always works; an LLM may rewrite it."""
    names = ", ".join(p["name"] for p in plans.values())
    text = (f"I checked your data ({facts['rows']:,} rows, {facts['cols']} features"
            + (f", class imbalance {facts['imbalance']:g}:1" if facts.get("imbalance") else "")
            + f") and prepared recommended settings for {names}. Change any value, or keep my picks.")
    if router is None:
        return {"text": text, "source": "rules"}
    try:
        payload = {"facts": facts, "models": [p["name"] for p in plans.values()]}
        out, _meta = router.complete([
            {"role": "system", "content": "You advise students on machine-learning settings. In at most 60 words, say what the data's size "
                                          "and balance imply for the chosen models. Use ONLY the facts given. No invented numbers."},
            {"role": "user", "content": "Facts (JSON): " + json.dumps(payload)}], max_tokens=140)
        return {"text": out[:500], "source": "llm"}
    except Exception:  # noqa: BLE001
        return {"text": text, "source": "rules"}
