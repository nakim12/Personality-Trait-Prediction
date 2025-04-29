import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ── cli ───────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",   required=True, type=Path,  help="Merged label+feature CSV")
p.add_argument("--model", required=True, type=Path,  help="Where to save .pkl")
args = p.parse_args()


df = pd.read_csv(args.csv)

FEATURE_COLS = [
    "fwh_ratio",
    "mouth_angle_deg",
    "brow_raise_norm",
    "inner_brow_raise_norm",
    "brow_furrow_norm",
    "kind_proj",
    "empath_proj",
]
missing = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    raise SystemExit(f"CSV missing columns: {missing}")

X = df[FEATURE_COLS]
y = df["compassion_pct"]

# ── scaler and xgboost regressor ──────────────────────
model = make_pipeline(
    StandardScaler(),
    XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
    ),
)
model.fit(X, y)


args.model.parent.mkdir(exist_ok=True)
joblib.dump(model, args.model)
print(f"Saved trained model: {args.model}")
