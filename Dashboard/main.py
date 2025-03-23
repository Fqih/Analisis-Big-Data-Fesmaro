import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# Sidebar - Pilih Mode
st.sidebar.title("🔍 Navigasi")
mode = st.sidebar.selectbox("Pilih Halaman", ["Dashboard Visualisasi", "Prediksi Sentimen"])

# ===================== DASHBOARD VISUALISASI =====================
if mode == "Dashboard Visualisasi":
    st.title("📊 Dashboard Analisis Ulasan Produk Amazon")

    # Load dataset
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/Fqih/Analisis-Big-Data-Fesmaro/refs/heads/main/Dataset/Dataset.csv",
            encoding="utf-8", on_bad_lines='skip', engine='python')
        if df.shape[1] == 3:
            df.columns = ['Kelas', 'Judul', 'Ulasan']
        else:
            st.error("❌ Format dataset error: bukan 3 kolom.")
            df = pd.DataFrame(columns=['Kelas', 'Judul', 'Ulasan'])
    except Exception as e:
        st.error(f"❌ Gagal membaca dataset: {e}")
        df = pd.DataFrame(columns=['Kelas', 'Judul', 'Ulasan'])

    df = df.dropna().reset_index(drop=True)
    df["Sentimen"] = df["Kelas"].map({1: "Negatif", 2: "Positif"})

    # === Informasi Dataset ===
    st.markdown("## 📌 Informasi Dataset")
    total_data_formatted = f"{len(df):,}".replace(",", " ")
    st.markdown(f"**Total Data:** `{total_data_formatted}` ulasan")

    kelas_counts = df["Kelas"].value_counts().sort_index()
    st.markdown("### 🔢 Jumlah Data per Kelas:")
    st.markdown(f"- **Negatif (1):** `{kelas_counts.get(1, 0):,}` ulasan".replace(",", " "))
    st.markdown(f"- **Positif (2):** `{kelas_counts.get(2, 0):,}` ulasan".replace(",", " "))

    # Sidebar filter
    st.sidebar.header("⚙️ Filter Data")
    if not df.empty:
        selected_kelas = st.sidebar.multiselect("Pilih Kelas", df["Kelas"].unique(), default=df["Kelas"].unique())
        keyword = st.sidebar.text_input("🔍 Cari Kata Kunci")
        review_length = st.sidebar.slider("📏 Panjang Ulasan (kata)", 1, 500, (1, 500))

        filtered_df = df[df["Kelas"].isin(selected_kelas)]
        if keyword:
            filtered_df = filtered_df[filtered_df["Ulasan"].str.contains(keyword, case=False, na=False)]
        filtered_df = filtered_df[filtered_df["Ulasan"].apply(lambda x: review_length[0] <= len(x.split()) <= review_length[1])]

        st.write("### 🔍 Data Setelah Filter")
        st.dataframe(filtered_df)
    else:
        filtered_df = df

    # Visualisasi Pie Chart
    if not filtered_df.empty:
        st.subheader("📊 Distribusi Kelas Sentimen")
        pie_counts = filtered_df["Kelas"].value_counts().sort_index()
        labels = ['Negatif', 'Positif']
        sizes = [pie_counts.get(1, 0), pie_counts.get(2, 0)]
        colors = ['red', 'green']
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.warning("⚠️ Tidak ada data untuk ditampilkan.")

    # Word Cloud
    st.subheader("☁️ Word Cloud Ulasan")
    if not filtered_df.empty:
        text = " ".join(filtered_df["Ulasan"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array())
    else:
        st.warning("⚠️ Tidak ada data untuk Word Cloud.")

    # Bigram Visualisasi
    st.subheader("🔗 Bigram Paling Umum")
    if not filtered_df.empty:
        for sentiment in ["Positif", "Negatif"]:
            sentiment_text = " ".join(filtered_df[filtered_df["Sentimen"] == sentiment]["Ulasan"])
            words = sentiment_text.split()

            bigrams = zip(words, words[1:])
            bigram_freq = Counter(bigrams).most_common(20)
            bigram_words = [' '.join(bigram) for bigram, _ in bigram_freq]
            bigram_counts = [count for _, count in bigram_freq]

            st.markdown(f"#### Top 20 Bigram ({sentiment})")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.bar(bigram_words, bigram_counts, color='blue' if sentiment == "Positif" else 'orange')
            plt.xticks(rotation=45)
            st.pyplot(fig3)
    else:
        st.warning("⚠️ Tidak ada data untuk visualisasi bigram.")

    # Contoh Ulasan
    st.subheader("📜 Contoh Ulasan")
    if not filtered_df.empty:
        st.dataframe(filtered_df[["Kelas", "Sentimen", "Judul", "Ulasan"]].sample(min(50, len(filtered_df))))
    else:
        st.warning("⚠️ Tidak ada data ulasan.")

    # Kata Umum
    st.subheader("🔍 Kata Paling Umum")
    if not filtered_df.empty:
        for sentiment in ["Positif", "Negatif"]:
            sentiment_text = " ".join(filtered_df[filtered_df["Sentimen"] == sentiment]["Ulasan"])
            words = sentiment_text.split()
            common_words = Counter(words).most_common(20)
            word_df = pd.DataFrame(common_words, columns=["Kata", "Frekuensi"])

            st.markdown(f"#### Top 20 Kata ({sentiment})")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.bar(word_df["Kata"], word_df["Frekuensi"], color='green' if sentiment == "Positif" else 'red')
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    else:
        st.warning("⚠️ Tidak ada data untuk analisis kata umum.")

# ===================== PREDIKSI SENTIMEN =====================
elif mode == "Prediksi Sentimen":
    st.title("🧠 Prediksi Sentimen Ulasan Produk Amazon")

    judul = st.text_input("📝 Masukkan Judul Ulasan Produk Amazon:")
    ulasan = st.text_area("🗒️ Masukkan Isi Ulasan Produk Amazon:")

    if st.button("🔍 Prediksi"):
        if judul and ulasan:
            st.markdown("#### 🔮 Prediksi Sentimen:")
            st.info("Model belum termuat. Silakan integrasikan model untuk melakukan prediksi.")
        else:
            st.warning("⚠️ Silakan masukkan **judul** dan **ulasan** terlebih dahulu.")
