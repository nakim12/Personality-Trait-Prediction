
import cv2, numpy as np, streamlit as st, mediapipe as mp, pathlib, joblib, math

ROOT  = pathlib.Path(__file__).resolve().parent.parent.parent   # 3 × .. to project root
reg   = joblib.load(ROOT/'model_artifacts'/'regressor_xgb.pkl')
cls_pkl = ROOT/'model_artifacts'/'classifier_xgb.pkl'
cls   = joblib.load(cls_pkl)["model"] if cls_pkl.exists() else None
cls_names = joblib.load(cls_pkl)["classes"] if cls_pkl.exists() else None

mpm   = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
L,R,U,C,ML,MR,BL,BR = 127,356,13,152,61,291,65,295

def quick(c):
    w  = np.linalg.norm(c[R,:2]-c[L,:2]);  h = np.linalg.norm(c[U,:2]-c[C,:2])
    fwh= w/h
    dx,dy = c[MR,:2]-c[ML,:2]; ang = math.degrees(math.atan2(dy,dx))
    br = (c[ML,1]-c[BL,1])/w
    fur= np.linalg.norm(c[BR,:2]-c[BL,:2])/w
    return np.array([[fwh,ang,br,br,fur,0,0]], np.float32)

st.sidebar.title('Preferences')
min_c = st.sidebar.slider('Minimum compassion %',0,100,50)
pref  = st.sidebar.radio('Preferred type',
         ['no preference'] + (cls_names if cls_names else ['neutral']))

file = st.file_uploader('Upload a face', type=['jpg','jpeg','png'])
if file:
    img = cv2.imdecode(np.frombuffer(file.read(),np.uint8), cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    res = mpm.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if res.multi_face_landmarks:
        h,w = img.shape[:2]
        c = np.array([(l.x*w,l.y*h,l.z*w) for l in res.multi_face_landmarks[0].landmark])
        feats = quick(c)
        comp = float(reg.predict(feats)[0])
        st.metric('Predicted Compassion %', f'{comp:0.1f}%')
        label = 'neutral'
        if cls:
            label = cls_names[int(cls.predict(feats)[0])]
            st.write('Type:', label)
        ok = (comp>=min_c) and (pref=='no preference' or pref==label)
        st.success('💘 Compatible') if ok else st.error('Not a match')
    else:
        st.error('No face detected')
