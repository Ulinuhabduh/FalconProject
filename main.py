import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import time

# Konfigurasi halaman untuk lebar penuh
st.set_page_config(page_title="Dashboard Penerbangan Qatar", layout="wide")

# Load model hanya sekali saat aplikasi pertama kali dijalankan
@st.cache_resource
def load_model():
    model = joblib.load("Trained_Review_Classifier.pkl") 
    return model

model = load_model()

# Inisialisasi session state untuk menyimpan hasil prediksi jika belum ada
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# Sidebar untuk memilih halaman
page = st.sidebar.selectbox("Pilih Halaman", ["Analisis Data", "Prediksi Sentimen dan Topik"])

# Jika halaman "Analisis Data" dipilih
if page == "Analisis Data":
    st.markdown(
        """
        <style>
        .metric-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 10px;
            border-radius: 10px;
            color: black;
            font-weight: bold;
            text-align: center;
            font-size: 16px;
            margin-top: -10px;
            height: 100px; /* Atur tinggi yang konsisten */
        }
        .blue-bg { background-color: #4A90E2; }
        .green-bg { background-color: #7ED321; }
        .yellow-bg { background-color: #F8E71C; }
        .red-bg { background-color: #FF6B6B; }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        
        .title-center {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load data
    file_path = "New_qatar.xlsx"
    data = pd.read_excel(file_path)

    # Konversi kolom tanggal menjadi tipe datetime
    data['Date Published'] = pd.to_datetime(data['Date Published'], errors='coerce')
    data['Month Published'] = data['Date Published'].dt.month

    # Judul Utama
    # st.title("Dashboard Analisis Ulasan Penerbangan")
    st.markdown("<h1 class='title-center'>Dashboard Analisis Ulasan Penerbangan</h1>", unsafe_allow_html=True)

    # Sidebar untuk Logo dan Filter
    st.sidebar.image("qatar logo.png", use_container_width=True)
    st.sidebar.header("Filter Data")
    selected_month = st.sidebar.selectbox("Pilih Bulan", options=["All"] + list(data['Date Published'].dt.month_name().unique()))
    selected_year = st.sidebar.selectbox("Pilih Tahun", options=["All"] + list(data['Date Published'].dt.year.unique()))

    # Filter data berdasarkan bulan dan tahun yang dipilih
    if selected_month != "All" and selected_year != "All":
        filtered_data = data[(data['Date Published'].dt.month_name() == selected_month) &
                            (data['Date Published'].dt.year == selected_year)]
    elif selected_month == "All" and selected_year != "All":
        filtered_data = data[data['Date Published'].dt.year == selected_year]
    elif selected_month != "All" and selected_year == "All":
        filtered_data = data[data['Date Published'].dt.month_name() == selected_month]
    else:
        filtered_data = data 

    # Bagian untuk menampilkan metrik dengan warna background dan posisi tengah
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_reviews = filtered_data.shape[0]
        st.markdown(f"""
            <div class="metric-container blue-bg">
                <div>Total Ulasan</div>
                <div class="metric-value">{total_reviews}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        average_rating = filtered_data['Rating'].mean()
        st.markdown(f"""
            <div class="metric-container green-bg">
                <div>Rata-rata Rating</div>
                <div class="metric-value">{average_rating:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        positive_reviews = (filtered_data['Sentiment'] == 'positive').sum()
        st.markdown(f"""
            <div class="metric-container yellow-bg">
                <div>Ulasan Positif</div>
                <div class="metric-value">{positive_reviews}</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        negative_reviews = (filtered_data['Sentiment'] == 'negative').sum()
        st.markdown(f"""
            <div class="metric-container red-bg">
                <div>Ulasan Negatif</div>
                <div class="metric-value">{negative_reviews}</div>
            </div>
        """, unsafe_allow_html=True)


    st.markdown("<hr>", unsafe_allow_html=True)

    # Row untuk Tipe Traveller dan Kelas Kursi
    col5, col6, col7 = st.columns(3)

    with col5:
        st.subheader("Distribusi Rating")
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.histplot(filtered_data['Rating'].dropna(), bins=10, kde=True, ax=ax)
        ax.set_title("Distribusi Rating")
        ax.set_xlabel("Rating")
        st.pyplot(fig)

    with col6:
        st.subheader("Distribusi Sentimen")
        sentiment_counts = filtered_data['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        sentiment_counts.plot(kind='bar', color=['skyblue', 'salmon'], ax=ax)
        ax.set_title("Sentimen Ulasan")
        ax.set_xlabel("Sentimen")
        ax.set_ylabel("Jumlah")

        # Mengatur orientasi teks sumbu-x menjadi horizontal
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        st.pyplot(fig)

    with col7:
        st.subheader("Tipe Traveller")
        traveller_counts = filtered_data['Type Of Traveller'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        traveller_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
        ax.set_ylabel('')
        ax.set_title("Distribusi Tipe Traveller")
        st.pyplot(fig)

    col8, col9, col10 = st.columns(3)  # Sesuaikan jumlah kolom jika tidak menampilkan `col10`

    with col8:
        st.subheader("Kelas Kursi")
        seat_counts = filtered_data['Seat Type'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        seat_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, ax=ax)
        ax.set_ylabel('')
        ax.set_title("Distribusi Kelas Kursi")
        st.pyplot(fig)

    with col10:
        st.subheader("Top 10 Negara Berdasarkan Jumlah Ulasan")
        country_counts = filtered_data['Country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax, palette="viridis")
        ax.set_title("Top 10 Negara Berdasarkan Ulasan")
        ax.set_xlabel("Jumlah Ulasan")
        ax.set_ylabel("Negara")
        st.pyplot(fig)

    # Tampilkan `col10` hanya jika bulan adalah "All"
    if selected_month == "All":
        with col9:
            st.subheader("Tren Rating Bulanan")
            
            # Jika "All" tahun dipilih, ambil rata-rata per bulan dari semua tahun
            if selected_year == "All":
                monthly_ratings = filtered_data.groupby('Month Published')['Rating'].mean().reset_index()
                monthly_ratings['Month Published'] = pd.to_datetime(monthly_ratings['Month Published'], format='%m').dt.strftime('%b')
            else:
                # Jika tahun tertentu dipilih, ambil data per bulan di tahun itu
                selected_data_year = filtered_data[filtered_data['Date Published'].dt.year == selected_year]
                monthly_ratings = selected_data_year.set_index('Date Published').resample('M')['Rating'].mean().dropna().reset_index()
                monthly_ratings['Month Published'] = monthly_ratings['Date Published'].dt.strftime('%b')

            # Plotting dengan garis tren
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.lineplot(data=monthly_ratings, x='Month Published', y='Rating', marker='o', color="steelblue", linewidth=2.5, ax=ax)
            
            # Menambahkan garis tren menggunakan regresi linear
            z = np.polyfit(range(len(monthly_ratings)), monthly_ratings['Rating'], 1)
            p = np.poly1d(z)
            ax.plot(monthly_ratings['Month Published'], p(range(len(monthly_ratings))), linestyle="--", color="red", linewidth=2)

            # Atur judul dan label sumbu
            title_suffix = f"untuk Tahun {int(selected_year)}" if selected_year != "All" else "seluruh tahun"
            ax.set_title(f"Rata-rata Rating Bulanan {title_suffix}")
            ax.set_xlabel("Bulan")
            ax.set_ylabel("Rata-rata Rating")
            
            # Menampilkan nilai rata-rata rating di atas setiap titik
            for x, y in zip(monthly_ratings['Month Published'], monthly_ratings['Rating']):
                ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=10, color="darkblue")
            
            st.pyplot(fig)

    # Menambahkan garis pemisah sebelum footer
    st.markdown("---")

if page == "Prediksi Sentimen dan Topik":
    st.markdown("<h1 class='title-center'>Prediksi Sentimen dan Topik</h1>", unsafe_allow_html=True)
    
    # Pilih metode input: manual atau file
    input_method = st.radio("Pilih metode input:", ("Manual", "File"))

    if input_method == "Manual":
        # Input manual untuk ulasan pengguna
        review_text = st.text_area("Masukkan ulasan penerbangan untuk diprediksi:")

        if st.button("Prediksi"):
            if review_text:
                # Mengukur waktu prediksi
                start_time = time.time()
                sentiment_prediction, topic_prediction = model.predict([review_text])[0]
                end_time = time.time()
                prediction_time = end_time - start_time

                # Tambahkan hasil prediksi ke session state
                st.session_state["predictions"].append({
                    "Review": review_text,
                    "Topic": sentiment_prediction,
                    "Sentiment": topic_prediction,
                    "Prediction Time (s)": round(prediction_time, 4)
                })

            else:
                st.warning("Silakan masukkan teks ulasan terlebih dahulu.")

        # Tampilkan DataFrame hasil prediksi jika ada
        if st.session_state["predictions"]:
            prediction_df = pd.DataFrame(st.session_state["predictions"])
            st.table(prediction_df)

    elif input_method == "File":
        # Input file dan pilih kolom review
        uploaded_file = st.file_uploader("Unggah file CSV berisi ulasan:", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            review_column = st.selectbox("Pilih kolom yang berisi ulasan:", options=data.columns)

            if st.button("Prediksi dari File"):
                # Hapus nilai NaN dari kolom yang dipilih
                review_data = data[review_column].dropna()

                # Proses prediksi untuk setiap review di kolom yang dipilih
                predictions = []
                start_time = time.time()
                
                for review in review_data:
                    sentiment_prediction, topic_prediction = model.predict([review])[0]
                    predictions.append({
                        "Review": review,
                        "Sentiment": topic_prediction,
                        "Topic": sentiment_prediction
                    })
                
                end_time = time.time()
                prediction_time = end_time - start_time

                # Menyimpan hasil prediksi ke DataFrame dan ke session_state
                prediction_df = pd.DataFrame(predictions)
                st.session_state["predictions"] = prediction_df.to_dict('records')  # Menyimpan di session state

                # Tampilkan hasil prediksi
                st.table(prediction_df.head())

                                    # Filter data berdasarkan sentimen negatif dan positif
                negative_topics = prediction_df[prediction_df['Sentiment'] == 'negative']['Topic'].value_counts()
                positive_topics = prediction_df[prediction_df['Sentiment'] == 'positive']['Topic'].value_counts()

                # Membagi layout menjadi dua kolom untuk visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart untuk topik dengan sentimen negatif
                    if not negative_topics.empty:
                        fig, ax = plt.subplots()
                        ax.pie(negative_topics, labels=negative_topics.index, autopct='%1.1f%%', startangle=140)
                        ax.set_title("Topik Review Negatif")
                        plt.tight_layout()  # Menambah padding agar pie chart terlihat jelas
                        st.pyplot(fig)
                    else:
                        st.write("Tidak ada topik dengan sentimen negatif.")

                with col2:
                    # Pie chart untuk topik dengan sentimen positif
                    if not positive_topics.empty:
                        fig, ax = plt.subplots()
                        ax.pie(positive_topics, labels=positive_topics.index, autopct='%1.1f%%', startangle=140)
                        ax.set_title("Topik Review Positif")
                        plt.tight_layout()  # Menambah padding agar pie chart terlihat jelas
                        st.pyplot(fig)
                    else:
                        st.write("Tidak ada topik dengan sentimen positif.")