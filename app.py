import streamlit as st
import pandas as pd
import cv2
import tempfile
import os
import joblib
from utils import extract_angles_from_video, get_deepest_ohs_angles, save_deepest_frame_with_angles

st.set_page_config(page_title="Análisis OHS", layout="centered")
st.title("📊 Análisis de Sentadilla Overhead (OHS)")

@st.cache_resource
def load_model():
    return joblib.load("modelo_ohs_rf_en.pkl")

clf = load_model()

video_file = st.file_uploader("Sube tu vídeo de sentadilla (.mp4)", type=["mp4"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name

    st.video(video_path)
    st.info("Procesando vídeo...")

    try:
        df_all = extract_angles_from_video(video_path, show_preview=False)
        # Guardar imagen con textos y clasificacion
        output_img_path = "resultado_ohs.png"
        tabla_eval = save_deepest_frame_with_angles(video_path, df_all, clf, output_img_path)

        # Leer imagen generada con OpenCV (como array) y mostrarla
        img = cv2.imread(output_img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mostrar imagen en Streamlit
        st.image(img_rgb, caption="Resultado del análisis", use_container_width=True)

        # Mostrar tabla
        st.markdown("### 📋 Evaluación por ángulos")
        st.dataframe(tabla_eval.style.format({"Valor (deg)": "{:.1f}"}))


    except Exception as e:
        st.error(f"❌ Error al procesar el vídeo: {str(e)}")