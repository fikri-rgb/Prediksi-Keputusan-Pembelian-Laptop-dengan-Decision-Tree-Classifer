import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# =========================
# LOAD MODEL (TIDAK DIUBAH)
# =========================
model = joblib.load("model.pkl")

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Prediksi Pembelian Laptop",
    page_icon="💻",
    layout="centered"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("💻 Menu Aplikasi")
menu = st.sidebar.radio(
    "Navigasi",
    ["📊 Dashboard", "🔍 Prediksi"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Project Kecerdasan Buatan\n\n"
    "Algoritma: **Decision Tree Classifier**"
)

# =========================
# DASHBOARD
# =========================
if menu == "📊 Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Dashboard Project</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("📌 Deskripsi Project")
    st.write(
        """
        Aplikasi ini bertujuan untuk **memprediksi keputusan pembelian laptop**
        berdasarkan karakteristik pengguna menggunakan algoritma  
        **Decision Tree Classifier**.
        """
    )

    st.subheader("📂 Fitur yang Digunakan")
    st.markdown("""
    - 👤 Umur  
    - 💰 Pendapatan  
    - 🎓 Status Mahasiswa  
    - 💳 Rating Kredit  
    """)

    st.subheader("⚙️ Teknologi")
    st.markdown("""
    - Python  
    - Streamlit  
    - Scikit-Learn  
    - Decision Tree Classifier  
    """)

    st.info("Gunakan menu **Prediksi** untuk melakukan klasifikasi.")

# =========================
# MENU PREDIKSI
# =========================
if menu == "🔍 Prediksi":
    st.markdown("<h1 style='text-align:center;'>🔍 Prediksi Pembelian Laptop</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # =========================
    # INPUT DATA
    # =========================
    st.subheader("📝 Masukkan Data Pengguna")

    col1, col2 = st.columns(2)

    with col1:
        umur = st.selectbox("👤 Umur", ["Muda", "Paruh Baya", "Tua"])
        pendapatan = st.selectbox("💰 Pendapatan", ["Rendah", "Sedang", "Tinggi"])

    with col2:
        mahasiswa = st.selectbox("🎓 Status Mahasiswa", ["Ya", "Tidak"])
        kredit = st.selectbox("💳 Rating Kredit", ["Baik", "Buruk"])

    # =========================
    # MAPPING (TIDAK DIUBAH)
    # =========================
    umur_map = {"Muda": 0, "Paruh Baya": 1, "Tua": 2}
    pendapatan_map = {"Rendah": 0, "Sedang": 1, "Tinggi": 2}
    mahasiswa_map = {"Tidak": 0, "Ya": 1}
    kredit_map = {"Buruk": 0, "Baik": 1}

    st.markdown("---")

    # =========================
    # PREDIKSI
    # =========================
    if st.button("🔍 Prediksi Keputusan", use_container_width=True):
        data = np.array([[ 
            umur_map[umur],
            pendapatan_map[pendapatan],
            mahasiswa_map[mahasiswa],
            kredit_map[kredit]
        ]])

        hasil = model.predict(data)
        prob = model.predict_proba(data)

        # =========================
        # HASIL & PROBABILITAS
        # =========================
        col_hasil, col_prob = st.columns(2)

        with col_hasil:
            st.subheader("📊 Hasil Prediksi")
            if hasil[0] == 1:
                st.success("✅ **AKAN MEMBELI LAPTOP**")
            else:
                st.error("❌ **TIDAK MEMBELI LAPTOP**")

        with col_prob:
            st.subheader("📈 Probabilitas")
            prob_df = pd.DataFrame({
                "Keputusan": ["Tidak Membeli", "Membeli Laptop"],
                "Probabilitas (%)": [
                    round(prob[0][0] * 100, 2),
                    round(prob[0][1] * 100, 2)
                ]
            })
            st.table(prob_df)

        # =========================
        # ALASAN PREDIKSI
        # =========================
        st.markdown("### 🧠 Alasan Prediksi")

        alasan = []
        if mahasiswa == "Ya":
            alasan.append("Status mahasiswa meningkatkan kebutuhan laptop")
        if pendapatan == "Tinggi":
            alasan.append("Pendapatan tinggi meningkatkan kemampuan membeli")
        if kredit == "Baik":
            alasan.append("Rating kredit baik menunjukkan kelayakan finansial")
        if umur == "Muda":
            alasan.append("Usia muda cenderung membutuhkan perangkat teknologi")

        if not alasan:
            alasan.append("Kombinasi fitur menunjukkan kecenderungan tidak membeli laptop")

        for a in alasan:
            st.write("•", a)

        # =========================
        # DECISION TREE (MUNCUL DI HASIL)
        # =========================
        st.markdown("### 🌳 Pohon Keputusan (Decision Tree)")
        st.write(
            "Visualisasi berikut menunjukkan aturan IF–THEN yang digunakan "
            "oleh model dalam menentukan hasil prediksi."
        )

        feature_names = ["Umur", "Pendapatan", "Mahasiswa", "Rating Kredit"]
        class_names = ["Tidak Membeli", "Membeli"]

        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(
            model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=ax
        )

        st.pyplot(fig)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:12px;'>"
    "Project Kecerdasan Buatan • Decision Tree Classifier</p>",
    unsafe_allow_html=True
)
