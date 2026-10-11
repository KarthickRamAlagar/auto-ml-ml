"""Model registry: which algorithms exist, their tunable parameters, grids and domain-based suggestions.
The same kwargs dict is used both to build the estimator and to generate the exported code, so the code
a user sees is exactly what ran.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

CLF, REG = "classification", "regression"
_CRITERIA = {CLF: ("gini", "entropy", "log_loss"), REG: ("squared_error", "friedman_mse", "absolute_error")}


@dataclass
class ParamSpec:
    name: str
    label: str
    type: str  # int | float | choice
    default: Any
    min: float | None = None
    max: float | None = None
    options: list[Any] = field(default_factory=list)
    help: str = ""

    def public(self) -> dict:
        return {"name": self.name, "label": self.label, "type": self.type, "default": self.default,
                "min": self.min, "max": self.max, "options": self.options, "help": self.help}


@dataclass
class ModelDef:
    id: str
    name: str
    blurb: str
    classes: dict[str, tuple[str, str]]  # task -> (module, class name)
    base: dict[str, dict]  # task -> fixed kwargs
    params: list[ParamSpec]
    grid: dict[str, dict]  # task -> param grid (sklearn kwarg names)
    scale_sensitive: bool = False
    slow: bool = False

    def tasks(self) -> list[str]:
        return list(self.classes)

    def kwargs(self, task: str, user: dict | None = None) -> dict:
        kw = dict(self.base[task])
        user = dict(user or {})
        pruning = user.pop("pruning", None)  # decision tree: none | pre | post | both
        for k, v in user.items():
            if k == "hidden_units":
                kw["hidden_layer_sizes"] = (int(v),)
            elif k == "hidden_layers":
                kw["hidden_layer_sizes"] = tuple(int(x) for x in str(v).split(",") if x.strip())
            elif k == "early_stopping":
                kw[k] = v in (True, "yes", "true")
            elif k == "class_weight":
                kw[k] = None if v in (None, "none") else v
            elif k == "max_features":
                kw[k] = None if v == "all" else v
            elif k == "criterion":
                if v in _CRITERIA[task]:
                    kw[k] = v
            else:
                kw[k] = v
        if self.id == "tree" and pruning is not None:
            pre_keys, post_keys = ("max_depth", "min_samples_leaf"), ("ccp_alpha",)
            drop = {"none": pre_keys + post_keys, "pre": post_keys, "post": pre_keys, "both": ()}.get(pruning, ())
            for k in drop:
                kw.pop(k, None)
        return kw

    def params_for(self, task: str) -> list[ParamSpec]:
        """Parameter list for one task (e.g. split criteria differ between classification and regression)."""
        out = []
        for p in self.params:
            if p.name == "criterion":
                p = ParamSpec(p.name, p.label, p.type, _CRITERIA[task][0], options=list(_CRITERIA[task]), help=p.help)
            if p.name == "class_weight" and task != CLF:
                continue
            out.append(p)
        return out

    def make(self, task: str, user: dict | None = None):
        mod, cls = self.classes[task]
        return getattr(importlib.import_module(mod), cls)(**self.kwargs(task, user))

    def class_path(self, task: str) -> tuple[str, str]:
        return self.classes[task]


def _p(name, label, typ, default, **kw):
    return ParamSpec(name, label, typ, default, **kw)


_DEPTH = _p("max_depth", "Max depth (blank = unlimited)", "int", None, min=1, max=30)
_LEAF = _p("min_samples_leaf", "Min samples per leaf", "int", 1, min=1, max=200,
           help="Larger values stop the tree from isolating tiny groups, which reduces overfitting.")
_MAXF = _p("max_features", "Features tried per split", "choice", "sqrt", options=["sqrt", "log2", "all"],
           help="Fewer features per split makes trees more different from each other.")
_CW = _p("class_weight", "Class weights", "choice", "none", options=["none", "balanced"],
         help="'balanced' gives rare classes more weight; useful when the classes are imbalanced.")
_TREES = _p("n_estimators", "Number of trees", "int", 100, min=10, max=500)
_SK = "sklearn"

MODELS: dict[str, ModelDef] = {}


def _add(m: ModelDef) -> None:
    MODELS[m.id] = m


_add(ModelDef("logreg", "Logistic Regression", "Fast, interpretable baseline for classification.",
              {CLF: (f"{_SK}.linear_model", "LogisticRegression")}, {CLF: {"max_iter": 1000}},
              [_p("C", "Inverse regularisation C", "float", 1.0, min=0.01, max=100), _CW],
              {CLF: {"C": [0.1, 1.0, 10.0]}}, scale_sensitive=True))
_add(ModelDef("linreg", "Linear Regression", "Ordinary least squares baseline.",
              {REG: (f"{_SK}.linear_model", "LinearRegression")}, {REG: {}}, [], {REG: {}}, scale_sensitive=True))
_add(ModelDef("ridge", "Ridge Regression", "Linear model with L2 regularisation; stable with correlated features.",
              {REG: (f"{_SK}.linear_model", "Ridge")}, {REG: {}},
              [_p("alpha", "Alpha", "float", 1.0, min=0.001, max=100)], {REG: {"alpha": [0.1, 1.0, 10.0]}}, scale_sensitive=True))
_add(ModelDef("lasso", "Lasso Regression", "L1 regularisation; can zero out weak features.",
              {REG: (f"{_SK}.linear_model", "Lasso")}, {REG: {"max_iter": 5000}},
              [_p("alpha", "Alpha", "float", 0.1, min=0.0005, max=10)], {REG: {"alpha": [0.001, 0.01, 0.1, 1.0]}}, scale_sensitive=True))
_add(ModelDef("elastic", "Elastic Net", "Mix of L1 and L2 regularisation.",
              {REG: (f"{_SK}.linear_model", "ElasticNet")}, {REG: {"max_iter": 5000}},
              [_p("alpha", "Alpha", "float", 0.1, min=0.0005, max=10), _p("l1_ratio", "L1 ratio", "float", 0.5, min=0.05, max=0.95)],
              {REG: {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}}, scale_sensitive=True))
_add(ModelDef("knn", "K-Nearest Neighbours", "Predicts from the K closest rows; K is the key choice.",
              {CLF: (f"{_SK}.neighbors", "KNeighborsClassifier"), REG: (f"{_SK}.neighbors", "KNeighborsRegressor")},
              {CLF: {}, REG: {}},
              [_p("n_neighbors", "K (neighbours)", "int", 5, min=1, max=50),
               _p("weights", "Weights", "choice", "uniform", options=["uniform", "distance"])],
              {CLF: {"n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"]},
               REG: {"n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21], "weights": ["uniform", "distance"]}}, scale_sensitive=True))
_SVM_PARAMS = [
    _p("C", "C (margin softness)", "float", 1.0, min=0.01, max=100),
    _p("kernel", "Kernel", "choice", "rbf", options=["rbf", "linear", "poly"]),
    _p("gamma", "Gamma", "choice", "scale", options=["scale", "auto"]),
]
_add(ModelDef("svm", "Support Vector Machine", "Maximum-margin model; C, kernel and gamma matter.",
              {CLF: (f"{_SK}.svm", "SVC"), REG: (f"{_SK}.svm", "SVR")}, {CLF: {}, REG: {}}, _SVM_PARAMS,
              {CLF: {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
               REG: {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}},
              scale_sensitive=True, slow=True))
_add(ModelDef("tree", "Decision Tree", "Readable rules; prone to overfit without a depth limit.",
              {CLF: (f"{_SK}.tree", "DecisionTreeClassifier"), REG: (f"{_SK}.tree", "DecisionTreeRegressor")},
              {CLF: {"random_state": 42}, REG: {"random_state": 42}},
              [_p("pruning", "Pruning strategy", "choice", "none", options=["none", "pre", "post", "both"],
                  help="Pre-pruning stops growth early (max depth, min samples per leaf). Post-pruning grows the full tree, "
                       "then cuts weak branches (cost-complexity alpha)."),
               _p("criterion", "Split criterion", "choice", "gini", options=list(_CRITERIA[CLF] + _CRITERIA[REG])),
               _DEPTH, _LEAF,
               _p("ccp_alpha", "Post-pruning strength (ccp_alpha)", "float", 0.0, min=0.0, max=0.1,
                  help="0 keeps the full tree. Larger values prune more branches.")],
              {CLF: {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 10], "ccp_alpha": [0.0, 0.001]},
               REG: {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 10], "ccp_alpha": [0.0, 0.001]}}))
_add(ModelDef("rf", "Random Forest", "Many trees voting; strong default for tabular data.",
              {CLF: (f"{_SK}.ensemble", "RandomForestClassifier"), REG: (f"{_SK}.ensemble", "RandomForestRegressor")},
              {CLF: {"random_state": 42, "n_jobs": 1}, REG: {"random_state": 42, "n_jobs": 1}},
              [_TREES, _DEPTH, _LEAF, _MAXF],
              {CLF: {"n_estimators": [100, 200], "max_depth": [None, 8]},
               REG: {"n_estimators": [100, 200], "max_depth": [None, 8]}}))
_add(ModelDef("extra", "Extra Trees", "Like Random Forest with more randomness; often faster.",
              {CLF: (f"{_SK}.ensemble", "ExtraTreesClassifier"), REG: (f"{_SK}.ensemble", "ExtraTreesRegressor")},
              {CLF: {"random_state": 42, "n_jobs": 1}, REG: {"random_state": 42, "n_jobs": 1}},
              [_TREES, _DEPTH, _LEAF, _MAXF],
              {CLF: {"n_estimators": [100, 200], "max_depth": [None, 8]},
               REG: {"n_estimators": [100, 200], "max_depth": [None, 8]}}))
_add(ModelDef("gb", "Gradient Boosting", "Trees built one after another to fix earlier errors.",
              {CLF: (f"{_SK}.ensemble", "GradientBoostingClassifier"), REG: (f"{_SK}.ensemble", "GradientBoostingRegressor")},
              {CLF: {"random_state": 42}, REG: {"random_state": 42}},
              [_TREES, _p("learning_rate", "Learning rate", "float", 0.1, min=0.01, max=1.0),
               _p("max_depth", "Max depth", "int", 3, min=1, max=8),
               _p("subsample", "Row sample per tree", "float", 1.0, min=0.3, max=1.0,
                  help="Below 1.0 each tree sees a random share of rows, which speeds training and can reduce overfitting.")],
              {CLF: {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
               REG: {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}}, slow=True))
_add(ModelDef("ada", "AdaBoost", "Boosting that re-weights hard examples.",
              {CLF: (f"{_SK}.ensemble", "AdaBoostClassifier"), REG: (f"{_SK}.ensemble", "AdaBoostRegressor")},
              {CLF: {"random_state": 42}, REG: {"random_state": 42}},
              [_p("n_estimators", "Number of estimators", "int", 50, min=10, max=500),
               _p("learning_rate", "Learning rate", "float", 1.0, min=0.01, max=2.0)],
              {CLF: {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
               REG: {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]}}))
_add(ModelDef("nb", "Gaussian Naive Bayes", "Very fast probabilistic baseline.",
              {CLF: (f"{_SK}.naive_bayes", "GaussianNB")}, {CLF: {}},
              [_p("var_smoothing", "Variance smoothing", "float", 1e-9, min=1e-12, max=1e-3)],
              {CLF: {"var_smoothing": [1e-9, 1e-8, 1e-7]}}))
_add(ModelDef("mlp", "Neural Network (MLP)", "Small feed-forward network.",
              {CLF: (f"{_SK}.neural_network", "MLPClassifier"), REG: (f"{_SK}.neural_network", "MLPRegressor")},
              {CLF: {"max_iter": 300, "random_state": 42}, REG: {"max_iter": 300, "random_state": 42}},
              [_p("hidden_layers", "Hidden layers (units per layer)", "choice", "100",
                  options=["50", "100", "100,50", "128,64", "128,64,32", "256,128,64"],
                  help="Each number is one layer. More and wider layers can learn more, but overfit and train slower."),
               _p("activation", "Activation function", "choice", "relu", options=["relu", "tanh", "logistic"]),
               _p("learning_rate_init", "Learning rate", "float", 0.001, min=0.0001, max=0.1),
               _p("alpha", "L2 penalty (alpha)", "float", 1e-4, min=1e-6, max=0.1),
               _p("max_iter", "Training epochs (max)", "int", 300, min=50, max=1000),
               _p("early_stopping", "Stop early when validation stalls", "choice", "no", options=["yes", "no"],
                  help="Holds out 10% of the training rows and stops when the score stops improving.")],
              {CLF: {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "alpha": [1e-4, 1e-3]},
               REG: {"hidden_layer_sizes": [(50,), (100,), (100, 50)], "alpha": [1e-4, 1e-3]}}, scale_sensitive=True, slow=True))

# Suggested order per domain (most useful first).
_RANKS = {
    CLF: {
        "finance": ["logreg", "rf", "gb", "tree", "svm"],
        "healthcare": ["logreg", "rf", "svm", "knn", "gb"],
        "e-commerce": ["gb", "rf", "logreg", "knn", "nb"],
        "education": ["logreg", "tree", "knn", "rf", "nb"],
        "manufacturing": ["rf", "gb", "svm", "tree", "knn"],
        "other": ["logreg", "rf", "knn", "svm", "tree"],
    },
    REG: {
        "finance": ["ridge", "gb", "rf", "linreg", "lasso"],
        "healthcare": ["ridge", "rf", "gb", "linreg", "svm"],
        "e-commerce": ["gb", "rf", "ridge", "lasso", "knn"],
        "real estate": ["gb", "rf", "ridge", "linreg", "extra"],
        "education": ["linreg", "ridge", "rf", "gb", "knn"],
        "manufacturing": ["rf", "gb", "ridge", "svm", "extra"],
        "other": ["linreg", "ridge", "rf", "gb", "knn"],
    },
}

_WHY = {
    "logreg": "Interpretable baseline; a good first model for most tabular classification.",
    "linreg": "Simple baseline that shows how much is explained by straight-line effects.",
    "ridge": "Handles correlated features well and is hard to overfit.",
    "lasso": "Drops weak features automatically.",
    "rf": "Strong on mixed, non-linear tabular data with little tuning.",
    "gb": "Often the most accurate tree model on structured data.",
    "svm": "Works well on smaller, cleaner datasets after scaling.",
    "knn": "Easy to understand; a good place to practise choosing K.",
    "tree": "Gives human-readable rules.",
    "nb": "Very fast and a useful sanity check.",
    "extra": "Fast tree ensemble with different randomness from Random Forest.",
}


def suggest_models(task: str, domain: str) -> list[dict]:
    d = (domain or "other").lower()
    order = _RANKS[task].get(d) or _RANKS[task]["other"]
    available = [m for m in MODELS.values() if task in m.classes]
    ids = [i for i in order if i in MODELS and task in MODELS[i].classes] + [m.id for m in available if m.id not in order]
    out = []
    for rank, mid in enumerate(ids):
        m = MODELS[mid]
        out.append({
            "id": m.id, "name": m.name, "blurb": m.blurb, "recommended": rank < 5,
            "why": _WHY.get(mid, m.blurb) if rank < 5 else "",
            "params": [p.public() for p in m.params_for(task)], "has_grid": bool(m.grid.get(task)),
            "slow": m.slow,
        })
    return out


def clean_params(model_id: str, params: dict | None) -> dict:
    """Whitelist + bound user-supplied params. Unknown keys are dropped (never passed to sklearn)."""
    m = MODELS[model_id]
    out: dict = {}
    for spec in m.params:
        if not params or spec.name not in params:
            continue
        v = params[spec.name]
        if v in (None, ""):
            if spec.name == "max_depth":
                out[spec.name] = None
            continue
        try:
            if spec.type == "int":
                out[spec.name] = int(max(spec.min, min(spec.max, int(float(v)))))
            elif spec.type == "float":
                out[spec.name] = float(max(spec.min, min(spec.max, float(v))))
            elif spec.type == "choice" and str(v) in [str(o) for o in spec.options]:
                out[spec.name] = str(v) if not isinstance(v, (int, float)) or isinstance(v, bool) else v
        except (TypeError, ValueError):
            continue
    return out
