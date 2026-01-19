### Klasifikasi Kesehatan Mental Pengguna Media Sosial

Aplikasi berbasis web ini menggunakan teknologi Deep Learning untuk memprediksi kondisi kesehatan mental seseorang berdasarkan pola aktivitas mereka di media sosial dan gaya hidup sehari-hari.

### Deskripsi Proyek

Proyek ini bertujuan untuk memberikan deteksi dini atau analisis awal terhadap kondisi mental pengguna (seperti Stressed, Anxious, atau Normal) dengan memanfaatkan arsitektur Deep Neural Network (DNN). Model dilatih menggunakan dataset yang mencakup berbagai indikator perilaku digital dan psikologis.

### Fitur Utama

- **Prediksi Berbasis AI** : Menggunakan model saraf tiruan (Neural Network) dengan akurasi tinggi.
- **Input Data Lengkap** : Mendukung berbagai parameter mulai dari durasi penggunaan layar, jam tidur, hingga level stres subjektif.
- **Dukungan Multi-Platform** : Menganalisis dampak dari berbagai platform seperti Instagram, Twitter, TikTok, YouTube, Snapchat, Facebook, dan WhatsApp.
- **Analisis Probabilitas** : Menampilkan grafik distribusi probabilitas untuk menunjukkan tingkat keyakinan AI pada setiap kategori status mental.
- **Antarmuka Interaktif** : Dibuat menggunakan Streamlit untuk pengalaman pengguna yang intuitif.

### Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.11
- **Framework Web**: Streamlit
- **Deep Learning**: TensorFlow & Keras
- **Pengolahan Data**: Pandas & NumPy
- **Preprocessing**: Scikit-learn (StandardScaler & LabelEncoder)
- **Penyimpanan Model**: Joblib

📂 Struktur Folder
mental-health-app/
│
├── app.py                # File utama aplikasi Streamlit
├── requirements.txt      # Daftar dependensi library Python
├── README.md             # Dokumentasi proyek
└── model_assets/         # Folder aset model dari Google Colab
    ├── mental_health_model.h5
    ├── scaler.joblib
    ├── label_encoder.joblib
    └── model_columns.joblib


### Instalasi dan Cara Menjalankan

1. Buka Terminal/CMD dan arahkan ke folder proyek.
Instal Library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

2. Jalankan Aplikasi:
```bash
streamlit run app.py
```

### Dataset

Dataset yang digunakan mencakup kolom-kolom berikut:
1. age, gender, platform
2. daily_screen_time_min, social_media_time_min
3. negative_interactions_count, positive_interactions_count
4. sleep_hours, physical_activity_min
5. anxiety_level, stress_level, mood_level

⚠️ Disclaimer
Aplikasi ini dibuat untuk tujuan edukasi dan analisis data riset. Hasil prediksi AI ini bukan merupakan diagnosa medis profesional. Silakan konsultasikan dengan ahli kesehatan mental (psikolog/psikiater) untuk hasil yang akurat.