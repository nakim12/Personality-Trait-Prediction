import pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent
df = pd.read_csv(
    ROOT / "fWHR to lifetime reprodcutive success and military rank. - Loehr+OHara+Finnish+soldier+data.csv"
).dropna(subset=["RANKWINT", "fWHR"])
X = df[["fWHR"]].values
y = df["RANKWINT"].astype(int).values

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class="multinomial",
        max_iter=200,
        class_weight="balanced" 
    )
)
model.fit(X, y)
joblib.dump(model, ROOT / "model_artifacts" / "classifier_rank.joblib")
print("Saved balanced rank model")
