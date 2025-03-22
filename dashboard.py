import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter
import re

# Download stopwords untuk pemrosesan teks
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset dengan penanganan error
try:
    df = pd.read_csv("https://raw.githubusercontent.com/Fqih/Analisis-Big-Data-Fesmaro/refs/heads/main/Dataset/Dataset.csv", encoding="utf-8", on_bad_lines='skip', engine='python')
    if df.shape[1] == 3:
        df.columns = ['Kelas', 'Judul', 'Ulasan']
    else:
        st.error("Dataset tidak memiliki jumlah kolom yang sesuai.")
        df = pd.DataFrame(columns=['Kelas', 'Judul', 'Ulasan'])
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    df = pd.DataFrame(columns=['Kelas', 'Judul', 'Ulasan'])

# Hapus nilai kosong dan reset index
df = df.dropna().reset_index(drop=True)

# Fungsi untuk analisis sentimen
def get_sentiment(text):
    analysis = TextBlob(text)
    return "Positif" if analysis.sentiment.polarity > 0 else "Negatif"

df["Sentimen"] = df["Ulasan"].apply(get_sentiment)

# Dashboard Title
st.title("📊 Dashboard Analisis Ulasan Produk Amazon")

# Informasi Dataset
st.markdown("""
## 📌 Informasi Dataset

Dataset ini merupakan sampel dari dataset asli yang berisi lebih dari **34 juta** ulasan produk Amazon yang telah diklasifikasikan menjadi dua kategori sentimen: **positif** dan **negatif**. Ulasan dalam dataset ini berasal dari berbagai produk yang dikumpulkan selama **18 tahun**, sehingga dapat digunakan untuk **analisis sentimen berbasis Machine Learning**.

### Struktur Dataset:
- **Judul Ulasan**: Ringkasan singkat dari ulasan pengguna.
- **Isi Ulasan**: Teks lengkap ulasan yang diberikan pengguna.
- **Sentimen**:
  - **Kelas 1** (Negatif: rating 1 dan 2) 
  - **Kelas 2** (Positif: rating 4 dan 5) 
""")

# Sidebar
st.sidebar.header("⚙️ Filter Data")
if not df.empty:
    selected_kelas = st.sidebar.multiselect("Pilih Kelas", df["Kelas"].unique(), default=df["Kelas"].unique())
    keyword = st.sidebar.text_input("🔍 Cari Kata Kunci dalam Ulasan")
    review_length = st.sidebar.slider("📏 Panjang Ulasan (Jumlah Kata)", min_value=1, max_value=500, value=(1, 500))
    filtered_df = df[df["Kelas"].isin(selected_kelas)]
    if keyword:
        filtered_df = filtered_df[filtered_df["Ulasan"].str.contains(keyword, case=False, na=False)]
    filtered_df = filtered_df[filtered_df["Ulasan"].apply(lambda x: review_length[0] <= len(x.split()) <= review_length[1])]
    st.write("### 🔍 Data Setelah Filter")
    st.dataframe(filtered_df)
else:
    filtered_df = df

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistik", "☁️ Word Cloud", "📜 Ulasan", "🔍 Kata Umum"])

with tab1:
    st.subheader("📊 Distribusi Kelas Sentimen")
    if not filtered_df.empty:
        kelas_count = filtered_df["Kelas"].value_counts().reset_index()
        kelas_count.columns = ["Kelas", "Jumlah"]
        kelas_count["Kelas"] = kelas_count["Kelas"].astype(str).replace({"1": "Sentimen Negatif", "2": "Sentimen Positif"})
        fig = px.pie(kelas_count, names='Kelas', values='Jumlah', title='Distribusi Kelas Sentimen')
        st.plotly_chart(fig)
        
        st.subheader("📊 Distribusi Panjang Karakter dalam Ulasan")
        filtered_df["Panjang Karakter"] = filtered_df["Ulasan"].apply(len)
        for sentiment in ["Positif", "Negatif"]:
            sentiment_df = filtered_df[filtered_df["Sentimen"] == sentiment]
            fig = px.histogram(sentiment_df, x="Panjang Karakter", nbins=30, title=f"Distribusi Panjang Karakter dalam Ulasan ({sentiment})")
            st.plotly_chart(fig)
    else:
        st.warning("Tidak ada data yang tersedia untuk ditampilkan.")

with tab2:
    st.subheader("☁️ Word Cloud dari Ulasan")
    if not filtered_df.empty:
        text = " ".join(filtered_df["Ulasan"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wordcloud.to_array())
    else:
        st.warning("Tidak ada data yang tersedia untuk Word Cloud.")

with tab3:
    st.subheader("📜 Contoh Ulasan")
    if not filtered_df.empty:
        st.dataframe(filtered_df[["Kelas", "Sentimen", "Judul", "Ulasan"]].dropna().sample(min(50, len(filtered_df))))
    else:
        st.warning("Tidak ada data ulasan yang tersedia.")

with tab4:
    st.subheader("🔍 Kata-Kata Paling Umum dalam Ulasan")
    if not filtered_df.empty:
        for sentiment in ["Positif", "Negatif"]:
            sentiment_text = " ".join(filtered_df[filtered_df["Sentimen"] == sentiment]["Ulasan"].dropna())
            words = re.findall(r'\b\w+\b', sentiment_text.lower())
            words = [word for word in words if word not in stop_words]
            common_words = Counter(words).most_common(20)
            word_df = pd.DataFrame(common_words, columns=["Kata", "Frekuensi"])
            st.subheader(f"📊 Kata-Kata Paling Umum ({sentiment})")
            fig = px.bar(word_df, x='Kata', y='Frekuensi', title=f'Top 20 Kata Umum ({sentiment})')
            st.plotly_chart(fig)
    else:
        st.warning("Tidak ada data yang tersedia untuk analisis kata umum.")
