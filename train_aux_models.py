from pathlib import Path
import pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "model_artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# ── Testosterone regression ──────────────────────────
t_df = pd.read_csv(ROOT / "fWHR to average Testosterone - Sample2_forEHB.csv")
reg_T = make_pipeline(StandardScaler(), LinearRegression())
reg_T.fit(t_df[["fWHR"]], t_df["av_T"])
joblib.dump(reg_T, ARTIFACTS / "regressor_testosterone.joblib")

# ── Rank classification ───────────────────────────
r_csv = ROOT / (
    "fWHR to lifetime reprodcutive success and military rank. - "
    "Loehr+OHara+Finnish+soldier+data.csv"
)
r_df = pd.read_csv(r_csv)
r_df["fWHR"] = pd.to_numeric(r_df["fWHR"], errors="coerce")
r_df = r_df.dropna(subset=["RANKWINT", "fWHR"])

print(f"Training rank model on {len(r_df)} rows after cleaning")

X_R = r_df[["fWHR"]].values
y_R = r_df["RANKWINT"].astype(int).values

clf_R = make_pipeline(
    StandardScaler(),
    LogisticRegression(multi_class="multinomial", max_iter=200, n_jobs=-1),
)
clf_R.fit(X_R, y_R)
joblib.dump(clf_R, ARTIFACTS / "classifier_rank.joblib")

print("\nModels saved in:", ARTIFACTS)
