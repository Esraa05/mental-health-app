import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime


st.set_page_config(
    page_title="Prediksi Kesehatan Mental AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Memuat model dan semua transformator data dari folder model_assets."""
    model = load_model('model_assets/mental_health_model.h5')
    scaler = joblib.load('model_assets/scaler.joblib')
    label_encoder = joblib.load('model_assets/label_encoder.joblib')
    model_columns = joblib.load('model_assets/model_columns.joblib')
    return model, scaler, label_encoder, model_columns

try:
    model, scaler, label_encoder, model_columns = load_assets()
except Exception as e:
    st.error(f"⚠️ Gagal memuat aset model: {e}")
    st.info("Pastikan folder 'model_assets' berisi file: .h5, scaler.joblib, label_encoder.joblib, dan model_columns.joblib")
    st.stop()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3034/3034873.png", width=100)
    st.title("Informasi Proyek")
    st.markdown("""
    Aplikasi ini menggunakan **Deep Neural Network** untuk memprediksi status mental berdasarkan dataset aktivitas media sosial.
    """)
    
    with st.expander("Debug: Kolom Model"):
        st.write("Daftar fitur yang dipelajari AI:")
        st.write(model_columns)
        
    st.divider()
    st.write("Dataset: Mental Health Social Media")

st.title("Klasifikasi Kesehatan Mental Pengguna Media Sosial")

with st.form("input_form"):
    st.subheader("Isi Data Diri")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        person_name = st.text_input("Person Name", "John Doe")
    with c2:
        age = st.number_input("Age", 10, 100, 25)
    with c3:
        date_input = st.date_input("Date", datetime.now())
    with c4:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Aktivitas Digital")
        platform = st.selectbox("Platform", ["Instagram", "Twitter", "TikTok", "YouTube", "Snapchat", "Facebook", "WhatsApp"])
        screen_time = st.number_input("Daily Screen Time (Min)", 0, 1440, 300)
        sm_time = st.number_input("Social Media Time (Min)", 0, 1440, 120)
        neg_int = st.number_input("Negative Interactions Count", 0, 500, 2)
        pos_int = st.number_input("Positive Interactions Count", 0, 500, 10)
        
    with col2:
        st.subheader("Gaya Hidup")
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        phys_act = st.number_input("Physical Activity (Min)", 0, 300, 30)
        
        st.subheader("Level Psikologis (1-10)")
        mood = st.slider("Mood Level", 1, 10, 5)
        anxiety = st.slider("Anxiety Level", 1, 10, 5)
        stress = st.slider("Stress Level", 1, 10, 5)

    submit_button = st.form_submit_button("JALANKAN ANALISIS (PREDIKSI MENTAL STATE)")

if submit_button:
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    mapping = {
        'age': age,
        'daily_screen_time_min': screen_time,
        'social_media_time_min': sm_time,
        'negative_interactions_count': neg_int,
        'positive_interactions_count': pos_int,
        'sleep_hours': sleep_hours,
        'physical_activity_min': phys_act,
        'anxiety_level': anxiety,
        'stress_level': stress,
        'mood_level': mood
    }
    
    for col, val in mapping.items():
        if col in input_df.columns:
            input_df[col] = val

    gender_col = f"gender_{gender}"
    platform_col = f"platform_{platform}"

    for col in input_df.columns:
        if col.lower() == gender_col.lower():
            input_df[col] = 1
        if col.lower() == platform_col.lower():
            input_df[col] = 1

    scaled_input = scaler.transform(input_df)

    with st.spinner(f"AI sedang menganalisis data {person_name}..."):
        prediction_proba = model.predict(scaled_input)
        prediction_idx = np.argmax(prediction_proba)
        final_label = label_encoder.inverse_transform([prediction_idx])[0]
        confidence = np.max(prediction_proba)

    st.divider()
    
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        st.subheader(f"Analisis untuk: {person_name}")
        st.write(f"**Tanggal Pemeriksaan:** {date_input}")

        if final_label == "Normal" or final_label == "Healthy":
            st.success(f"Mental State: **{final_label}**")
            st.write(f"Halo {person_name}, AI menilai kondisi mental Anda dalam rentang stabil. Tetap jaga pola interaksi positif Anda.")
        else:
            st.warning(f"Mental State: **{final_label}**")
            st.write(f"Halo {person_name}, model mendeteksi adanya pola kecenderungan **{final_label}**.")
            st.write("Saran: Cobalah kurangi waktu layar dan tingkatkan kualitas istirahat Anda.")
        
        st.metric("AI Confidence Score", f"{confidence*100:.2f}%")

    with res_col2:
        st.subheader("Distribusi Probabilitas")
        prob_df = pd.DataFrame({
            'Kategori': label_encoder.classes_,
            'Skor': prediction_proba[0]
        }).sort_values('Skor', ascending=True)
        
        st.bar_chart(data=prob_df, x='Kategori', y='Skor', use_container_width=True)

st.markdown("---")
st.caption(f"Aplikasi ini disinkronkan dengan kolom dataset Mental Health Social Media. Analisis dilakukan pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")