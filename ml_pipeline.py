"""
ml_pipeline.py - data loading, preprocessing pipeline, model training & selection
for the Auto-MPG fuel-efficiency predictor.

End-to-end pipeline (everything lives INSIDE one sklearn Pipeline, so raw user input
can be passed straight to .predict()):

  raw row -> feature engineering (brand, origin name, power/weight, disp/cyl)
          -> ColumnTransformer
               numeric     : median imputation -> StandardScaler
               categorical : constant/most-frequent imputation -> OneHotEncoder
          -> regressor
"""
from __future__ import annotations

import hashlib
import io
import re
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model" / "mpg_model.joblib"

NUM_RAW = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
FEATURES = NUM_RAW + ["origin", "car_name"]          # what the user supplies
NUM_ALL = NUM_RAW + ["power_to_weight", "disp_per_cyl"]  # numeric after engineering
CAT_ALL = ["origin_name", "brand"]
ORIGIN_MAP = {1: "usa", 2: "europe", 3: "japan"}
ORIGIN_LABEL = {1: "🇺🇸 USA", 2: "🇪🇺 Europe", 3: "🇯🇵 Japan"}

BRAND_FIX = {
    "chevroelt": "chevrolet", "chevy": "chevrolet", "vw": "volkswagen",
    "vokswagen": "volkswagen", "toyouta": "toyota", "maxda": "mazda",
    "mercedes-benz": "mercedes",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names / types of any Auto-MPG style file."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"name": "car_name", "carname": "car_name", "car": "car_name",
                            "year": "model_year", "modelyear": "model_year"})
    missing = [c for c in ["mpg"] + NUM_RAW + ["origin"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
    for c in ["mpg"] + NUM_RAW:                      # '?' etc. -> NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["origin"].dtype == object or str(df["origin"].dtype).startswith("str"):
        df["origin"] = df["origin"].astype(str).str.strip().str.lower().map(
            {"usa": 1, "europe": 2, "japan": 3, "1": 1, "2": 2, "3": 3})
    df["origin"] = pd.to_numeric(df["origin"], errors="coerce")
    if df["model_year"].dropna().gt(1900).any():     # 1970 -> 70
        df["model_year"] = df["model_year"] - 1900
    if "car_name" not in df.columns:
        df["car_name"] = np.nan
    df = df[["mpg"] + NUM_RAW + ["origin", "car_name"]]
    return df.dropna(subset=["mpg"]).reset_index(drop=True)


def _read_any(path_or_buf) -> pd.DataFrame:
    """Read CSV or the original whitespace separated UCI auto-mpg.data."""
    try:
        if hasattr(path_or_buf, "seek"):
            path_or_buf.seek(0)
        raw = pd.read_csv(path_or_buf)
        if "mpg" in [str(c).strip().lower() for c in raw.columns]:
            return raw
    except Exception:
        pass
    if hasattr(path_or_buf, "seek"):
        path_or_buf.seek(0)
    return pd.read_csv(path_or_buf, sep=r"\s+", header=None, na_values="?", quotechar='"',
                       names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                              "acceleration", "model_year", "origin", "car_name"])


def make_demo_data(n: int = 392, seed: int = 7) -> pd.DataFrame:
    """Synthetic stand-in (ONLY used if the real dataset cannot be found)."""
    rng = np.random.default_rng(seed)
    origin = rng.choice([1, 2, 3], n, p=[0.62, 0.18, 0.20])
    cyl = np.where(origin == 1, rng.choice([4, 6, 8], n, p=[.25, .25, .5]),
                   rng.choice([3, 4, 5, 6], n, p=[.03, .88, .04, .05]))
    disp = cyl * rng.normal(48, 8, n).clip(25, 70)
    hp = (disp * rng.normal(0.42, 0.06, n)).clip(45, 230)
    wt = (900 + disp * 5.5 + rng.normal(0, 250, n)).clip(1600, 5100)
    acc = (22 - hp / 18 + rng.normal(0, 1.4, n)).clip(8, 24.8)
    year = rng.integers(70, 83, n)
    mpg = (50 - wt * 0.0055 - hp * 0.02 + (year - 70) * 0.55 + (origin > 1) * 1.5
           + rng.normal(0, 1.8, n)).clip(9, 46)
    brands = {1: ["chevrolet", "ford", "buick", "plymouth", "dodge", "amc"],
              2: ["volkswagen", "bmw", "fiat", "peugeot", "audi"],
              3: ["toyota", "honda", "datsun", "mazda", "subaru"]}
    names = [f"{rng.choice(brands[o])} model {i}" for i, o in enumerate(origin)]
    df = pd.DataFrame(dict(mpg=mpg.round(1), cylinders=cyl, displacement=disp.round(0),
                           horsepower=hp.round(0), weight=wt.round(0),
                           acceleration=acc.round(1), model_year=year, origin=origin,
                           car_name=names))
    df.loc[rng.choice(n, 6, replace=False), "horsepower"] = np.nan
    return df


def get_data(uploaded=None):
    """Return (df, source_label, is_demo). Order: upload -> local file -> online -> demo."""
    if uploaded is not None:
        return standardize(_read_any(uploaded)), "uploaded file", False
    for p in [DATA_DIR / "auto-mpg.csv", DATA_DIR / "auto-mpg.data", BASE_DIR / "auto-mpg.csv",
              BASE_DIR / "auto-mpg.data", DATA_DIR / "mpg.csv"]:
        if p.exists():
            return standardize(_read_any(p)), f"local file ({p.name})", False
    try:
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
        with urllib.request.urlopen(url, timeout=8) as r:
            return standardize(pd.read_csv(io.BytesIO(r.read()))), "online copy of Auto-MPG", False
    except Exception:
        pass
    return standardize(make_demo_data()), "SYNTHETIC DEMO DATA", True


# --------------------------------------------------------------------------- #
# Feature engineering + preprocessing
# --------------------------------------------------------------------------- #
def clean_brand(name) -> str:
    if not isinstance(name, str) or not name.strip():
        return "other"
    first = re.split(r"\s+", name.strip().lower())[0]
    return BRAND_FIX.get(first, first)


def add_features(X):
    X = pd.DataFrame(X).copy()
    for c in NUM_RAW + ["origin"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["origin_name"] = X["origin"].map(ORIGIN_MAP).fillna("unknown")
    if "car_name" not in X.columns:
        X["car_name"] = np.nan
    X["brand"] = X["car_name"].map(clean_brand)
    X["power_to_weight"] = X["horsepower"] / X["weight"] * 1000
    X["disp_per_cyl"] = X["displacement"] / X["cylinders"]
    return X.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="other")),
                    ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=3,
                                             sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_ALL), ("cat", cat, CAT_ALL)],
                             remainder="drop")


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=2.0),
        "Support Vector Regressor": SVR(kernel="rbf", C=20, epsilon=0.2, gamma="scale"),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=1,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=42),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline([("features", FunctionTransformer(add_features, validate=False)),
                     ("prep", build_preprocessor()),
                     ("model", model)])


# --------------------------------------------------------------------------- #
# Training / selection
# --------------------------------------------------------------------------- #
def data_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()


def train_all(df: pd.DataFrame, source: str = "", progress=None) -> dict:
    """Train >=3 algorithms, compare with CV + hold-out, pick the best, refit on all data."""
    progress = progress or (lambda f, t: None)
    X, y = df[FEATURES], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(5, shuffle=True, random_state=42)
    models = get_models()
    rows, fitted_tr = [], {}
    for i, (name, est) in enumerate(models.items()):
        progress(i / (len(models) + 2), f"Training {name} …")
        pipe = make_pipeline(est)
        t0 = time.time()
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv,
                                   scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append(dict(Model=name,
                         **{"R² (test)": r2_score(y_te, pred),
                            "RMSE (test)": float(np.sqrt(mean_squared_error(y_te, pred))),
                            "MAE (test)": mean_absolute_error(y_te, pred),
                            "CV RMSE (5-fold)": cv_rmse,
                            "Train time (s)": time.time() - t0}))
        fitted_tr[name] = pipe
    results = pd.DataFrame(rows).sort_values("CV RMSE (5-fold)").reset_index(drop=True)
    best_name = results.loc[0, "Model"]

    progress(len(models) / (len(models) + 2), "Explaining the winner …")
    best_tr = fitted_tr[best_name]
    perm = permutation_importance(best_tr, X_te, y_te, n_repeats=10, random_state=42,
                                  scoring="neg_root_mean_squared_error")
    importance = (pd.DataFrame({"Feature": FEATURES, "Importance": perm.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))
    test_pred = pd.DataFrame({"actual": y_te.values, "predicted": best_tr.predict(X_te)})

    progress((len(models) + 1) / (len(models) + 2), "Refitting every model on all data …")
    all_pipes = {n: make_pipeline(get_models()[n]).fit(X, y) for n in models}
    progress(1.0, "Done!")
    return dict(best_name=best_name, best_pipeline=all_pipes[best_name], all_pipelines=all_pipes,
                results=results, importance=importance, test_pred=test_pred,
                cv_rmse=float(results.loc[0, "CV RMSE (5-fold)"]),
                n_rows=len(df), source=source, data_hash=data_hash(df),
                sklearn_version=sklearn.__version__)


def load_or_train(df, source="", save=True, progress=None) -> dict:
    """Reuse the saved model if it matches the data & sklearn version, else retrain."""
    if MODEL_PATH.exists():
        try:
            b = joblib.load(MODEL_PATH)
            if b.get("data_hash") == data_hash(df) and b.get("sklearn_version") == sklearn.__version__:
                return b
        except Exception:
            pass
    bundle = train_all(df, source, progress)
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
    return bundle


def predict_row(pipe, row: dict) -> float:
    clean = {k: (np.nan if row.get(k) in (None, "") else row.get(k)) for k in FEATURES}
    return float(pipe.predict(pd.DataFrame([clean], columns=FEATURES))[0])
