from __future__ import annotations
import math, sys
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import joblib
import streamlit as st

try:
    import mediapipe as mp
except ModuleNotFoundError:
    sys.exit("MediaPipe missing → pip install mediapipe==0.10.21")


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model_artifacts"
REG_COMPASSION   = joblib.load(MODEL_DIR / "regressor_xgb.pkl")           
REG_TESTOSTERONE = joblib.load(MODEL_DIR / "regressor_testosterone.joblib")
CLS_RANK         = joblib.load(MODEL_DIR / "classifier_rank.joblib")

RANK_LABELS = {1: "Low intelligence", 2: "Medium intelligence", 3: "High intelligence"}
COMP_THRESH = {"Semikind": (0, 25),
               "Kind":     (25, 60),
               "Empathetic": (60, 101)}
TEST_THRESH = {"Low":    (0, 250),
               "Normal": (250, 750),
               "High":   (750, 10_000)}       

# ── features ────────────────────────────────────────────
@dataclass
class LandmarkFeatures:
    fwh_ratio: float
    mouth_angle_deg: float
    brow_raise_norm: float
    inner_brow_raise_norm: float = 0.0
    brow_furrow_norm: float      = 0.0
    kind_proj: float             = 0.0
    empath_proj: float           = 0.0
    def as_array(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), np.float32)

# ── mediapipe ───────────────────────────────────────────────
mp_face = mp.solutions.face_mesh
FACE = mp_face.FaceMesh(static_image_mode=True,
                        refine_landmarks=True,
                        max_num_faces=1,
                        min_detection_confidence=0.5)

LEFT_ZY, RIGHT_ZY = 234, 454
NATION, UPPER_LIP = 168, 13
LEFT_MOUTH, RIGHT_MOUTH = 61, 291
LEFT_BROW, LEFT_EYE_LID = 105, 159

def extract_landmark_features(img: np.ndarray) -> LandmarkFeatures | None:
    res = FACE.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm  = res.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    fwh_ratio = np.linalg.norm(
        np.array([lm[RIGHT_ZY].x*w, lm[RIGHT_ZY].y*h]) -
        np.array([lm[LEFT_ZY ].x*w, lm[LEFT_ZY ].y*h])
    ) / max(
        1e-3,
        np.linalg.norm(
            np.array([lm[NATION].x*w, lm[NATION].y*h]) -
            np.array([lm[UPPER_LIP].x*w, lm[UPPER_LIP].y*h])
        )
    )
    mouth_vec = (
        np.array([lm[RIGHT_MOUTH].x*w, lm[RIGHT_MOUTH].y*h]) -
        np.array([lm[LEFT_MOUTH ].x*w, lm[LEFT_MOUTH ].y*h])
    )
    mouth_angle_deg = math.degrees(math.atan2(mouth_vec[1], mouth_vec[0])) if mouth_vec.any() else 0
    brow_raise_norm = (
        (lm[LEFT_EYE_LID].y - lm[LEFT_BROW].y) * h
    ) / (lm[UPPER_LIP].y*h - lm[NATION].y*h + 1e-3)
    return LandmarkFeatures(fwh_ratio, mouth_angle_deg, brow_raise_norm)


def predict_traits(feat: LandmarkFeatures) -> dict[str, float]:
    comp = float(REG_COMPASSION.predict(feat.as_array().reshape(1, -1))[0])
    test = float(REG_TESTOSTERONE.predict([[feat.fwh_ratio]])[0])
    rank_class = int(CLS_RANK.predict([[feat.fwh_ratio]])[0])
    rank_conf  = float(CLS_RANK.predict_proba([[feat.fwh_ratio]])[0].max())
    return dict(compassion=comp, testosterone=test,
                rank_class=rank_class, rank_conf=rank_conf)

# ── preference ───────────────────────────────────────
def within(val: float, bounds: tuple[int, int]) -> bool:
    lo, hi = bounds
    return lo <= val < hi

def matches_prefs(res: dict, prefs: dict) -> bool:
    
    if prefs["comp"] is not None:
        if not within(res["compassion"], COMP_THRESH[prefs["comp"]]):
            return False
   
    if prefs["test"] is not None:
        if not within(res["testosterone"], TEST_THRESH[prefs["test"]]):
            return False
    
    if prefs["intel"] is not None:
        wanted_rank = {"Low":1, "Medium":2, "High":3}[prefs["intel"]]
        if res["rank_class"] != wanted_rank:
            return False
    return True

# ── streamlit ────────────────────────────────────────────────
st.set_page_config(page_title="Face-Forward", page_icon="")
st.title("How Compatible and Sexy is Your man?")


with st.sidebar:
    st.header("Your ideal match")
    comp_pref = st.radio("Compassion level",
                         ("No preference", "Semikind", "Kind", "Empathetic"))
    test_pref = st.radio("Testosterone level",
                         ("No preference",
                          "Low (<250 ng/dL)",
                          "Normal (250-750 ng/dL)",
                          "High (>750 ng/dL)"))
    intel_pref = st.radio("Intelligence level",
                          ("No preference", "Low", "Medium", "High"))
prefs = {
    "comp": comp_pref if comp_pref != "No preference" else None,
    "test": test_pref.split()[0] if "No preference" not in test_pref else None,
    "intel": intel_pref if intel_pref != "No preference" else None,
}

uploaded = st.file_uploader("Upload a clear, front-facing photo", ["jpg","jpeg","png"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    cols = st.columns([1,1.2])
    with cols[0]:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image")
    feats = extract_landmark_features(img)
    if feats is None:
        st.error("No face detected.")
        st.stop()
    res = predict_traits(feats)
    is_match = matches_prefs(res, prefs)
    with cols[1]:
        st.header("Predictions")
        st.metric("Compassion",    f"{res['compassion']:.1f} %")
        st.metric("Testosterone",  f"{res['testosterone']:.0f} ng/dL")
        rank_lbl  = RANK_LABELS[res['rank_class']]
        conf_pc   = res['rank_conf'] * 100
        fwh_ratio = feats.fwh_ratio
        st.metric("Intelligence", rank_lbl)
        st.caption(f"Confidence {conf_pc:.0f}% • fWHR {fwh_ratio:.2f}")
        st.markdown("---")
        st.subheader("Match verdict")
        if is_match:
            st.success("You and him are a perfect match!")
        else:
            st.warning("You guys are not a match :(")
        st.caption("Ranges: Low < 250 ng/dL, Normal 250-750 ng/dL, High > 750 ng/dL")
else:
    st.info("⬆️  Upload an image to begin.")
