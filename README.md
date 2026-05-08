# Wine Quality Classification — UTS

**Nama:** Hayu Dhiya Rahmadani  
**NIM:** 2304020005

---

## 📋 Deskripsi

Soal ini bertujuan untuk membangun model klasifikasi kualitas anggur (*wine quality*) menggunakan berbagai algoritma *machine learning*. Dataset terdiri dari data kimiawi anggur beserta label kualitasnya (skala 0–10). Model yang dibangun akan mempelajari hubungan antara karakteristik kimia anggur dengan skor kualitasnya, lalu digunakan untuk memprediksi kualitas anggur pada data yang belum memiliki label.

Proyek ini mencakup seluruh alur kerja *machine learning* secara end-to-end: mulai dari eksplorasi data, pembersihan, pemilihan fitur, pelatihan beberapa model, perbandingan performa, hingga prediksi akhir dan penyimpanan hasil.

---

## 📁 Struktur File

```
├── wine_classification_Hayu_Dhiya_Rahmadani__005_.ipynb   # Notebook utama
├── data_training.csv                                      # Dataset pelatihan (dengan label)
├── data_testing.csv                                       # Dataset pengujian (tanpa label)
├── hasil_prediksi.csv                                     # Output prediksi model
└── README.md                                              # Dokumentasi proyek
```

---

## Library yang Digunakan

| Library | Kegunaan |
|---|---|
| `pandas` | Manipulasi dan analisis data tabular |
| `numpy` | Operasi numerik dan array |
| `matplotlib` | Visualisasi data (grafik, plot) |
| `seaborn` | Visualisasi statistik (heatmap, boxplot) |
| `s  tatsmodels` | Analisis statistik tambahan |
| `scikit-learn` | Algoritma machine learning, evaluasi, dan tuning |

---

## 📊 Dataset

Dataset yang digunakan adalah **Wine Quality Dataset** yang terdiri dari dua file:

- `data_training.csv` → digunakan untuk melatih dan mengevaluasi model (memiliki label `quality`)
- `data_testing.csv` → digunakan untuk prediksi akhir (tidak memiliki label `quality`)

### Variabel / Fitur dalam Dataset

| No. | Variabel | Deskripsi |
|---|---|---|
| 1 | `fixed acidity` | Kadar asam tetap (non-volatile) dalam anggur |
| 2 | `volatile acidity` | Kadar asam volatil; nilai tinggi dapat menyebabkan rasa seperti cuka |
| 3 | `citric acid` | Asam sitrat yang menambah kesegaran dan kompleksitas rasa |
| 4 | `residual sugar` | Kadar gula yang tersisa setelah proses fermentasi selesai |
| 5 | `chlorides` | Kadar garam dalam anggur |
| 6 | `free sulfur dioxide` | SO₂ bebas yang berperan mencegah pertumbuhan mikroba dan oksidasi |
| 7 | `total sulfur dioxide` | Total SO₂ dalam anggur (bebas + terikat) |
| 8 | `density` | Kerapatan cairan anggur |
| 9 | `pH` | Tingkat keasaman anggur (skala 0–14) |
| 10 | `sulphates` | Aditif anggur yang berkontribusi pada kadar SO₂ |
| 11 | `alcohol` | Persentase kadar alkohol dalam anggur |
| 12 | `type` | Jenis anggur (red / white) |
| 13 | `quality` | **(Target)** Skor kualitas anggur, skala 0–10 |

---

## 🔄 Alur Analisis (Pipeline)

```
Load Dataset → EDA → Pembersihan Data → Persiapan Fitur → Split Data
     ↓
Pelatihan 4 Model: [Logistic Regression | KNN | Decision Tree | Random Forest]
     ↓
Evaluasi & Perbandingan Akurasi
     ↓
Pemilihan Model Terbaik (Random Forest)
     ↓
Retrain dengan Seluruh Data → Prediksi Data Testing → Simpan CSV
```

---

## 📝 Penjelasan Setiap Langkah Analisis

### Langkah 1 — Import Library

Langkah pertama adalah mengimpor seluruh library yang diperlukan. `pandas` dan `numpy` digunakan untuk manipulasi data, `matplotlib` dan `seaborn` untuk visualisasi, serta berbagai modul dari `scikit-learn` untuk membangun dan mengevaluasi model machine learning. `warnings.filterwarnings('ignore')` ditambahkan untuk menyembunyikan peringatan yang tidak relevan agar output lebih bersih.

---

### Langkah 2 — Load Dataset

Dataset dimuat dari dua file CSV menggunakan `pd.read_csv()`. Langkah ini juga mencetak bentuk (*shape*) dari masing-masing dataset untuk memastikan data berhasil dimuat dengan jumlah baris dan kolom yang benar. Pemisahan antara data training dan data testing sejak awal penting dilakukan agar tidak terjadi *data leakage* (informasi dari data testing bocor ke proses pelatihan model).

---

### Langkah 3 — Eksplorasi Data (EDA)

EDA (*Exploratory Data Analysis*) adalah tahap paling krusial sebelum membangun model. Tujuannya adalah memahami karakteristik data secara mendalam sebelum membuat keputusan teknis apapun.

#### 3.1 Informasi dan Statistik Deskriptif
`df_train.info()` digunakan untuk melihat tipe data setiap kolom dan memastikan tidak ada kolom yang memiliki tipe data yang tidak sesuai. `df_train.describe()` memberikan ringkasan statistik seperti nilai minimum, maksimum, rata-rata, dan standar deviasi. Dari sini terlihat bahwa skala antar fitur sangat bervariasi — misalnya `total sulfur dioxide` bisa mencapai ratusan, sementara `citric acid` berada di bawah 1. Perbedaan skala ini penting dipertimbangkan karena model seperti KNN sangat sensitif terhadapnya.

#### 3.2 Distribusi Target (Quality)
Visualisasi distribusi kelas `quality` dilakukan dengan bar chart. Hasilnya menunjukkan bahwa sebagian besar data terkonsentrasi pada kualitas 5 dan 6, sementara kualitas ekstrem (3 atau 8-9) sangat sedikit. Kondisi ini disebut **class imbalance** (ketidakseimbangan kelas), yang dapat menyebabkan model lebih pintar memprediksi kelas mayoritas dibanding kelas minoritas. Hal ini harus dipertimbangkan saat menginterpretasikan metrik evaluasi.

#### 3.3 Distribusi Setiap Fitur Kimia (Histogram)
Histogram dibuat untuk setiap fitur kimia dengan garis penanda rata-rata (merah) dan median (oranye). Ketika rata-rata lebih besar dari median, ini menandakan distribusi *right-skewed* (condong ke kanan), yang mengindikasikan adanya **outlier** dengan nilai sangat tinggi. Fitur seperti `residual sugar`, `chlorides`, dan `total sulfur dioxide` menunjukkan pola ini. Outlier dapat mempengaruhi performa model tertentu (terutama yang berbasis jarak seperti KNN) dan kadang perlu ditangani.

#### 3.4 Boxplot Alcohol per Kelas Quality
Boxplot ini menampilkan hubungan antara kadar alkohol dan kelas kualitas. Tren yang terlihat sangat jelas: **semakin tinggi kualitas anggur, semakin tinggi median kadar alkoholnya**. Ini adalah temuan yang sangat penting karena mengkonfirmasi bahwa `alcohol` adalah fitur dengan daya prediktif yang kuat. Boxplot juga membantu mengidentifikasi adanya outlier dalam setiap kelompok kualitas.

#### 3.5 Boxplot Semua Fitur per Kelas Quality
Visualisasi ini memperluas analisis ke semua fitur kimia. Fitur yang memiliki perbedaan distribusi antar kelas yang jelas (misalnya box-nya tidak banyak tumpang tindih) merupakan fitur yang paling informatif untuk model klasifikasi. Dari analisis ini ditemukan bahwa:
- **`alcohol`** → distribusi meningkat jelas seiring naiknya kualitas
- **`volatile acidity`** → distribusi menurun seiring naiknya kualitas
- **`sulphates`** dan **`citric acid`** → menunjukkan tren naik yang konsisten
- **`density`** dan **`pH`** → distribusi antar kelas relatif mirip, artinya pengaruhnya lebih kecil

#### 3.6 Tren Rata-rata Fitur Utama per Kelas Quality (Line Chart)
Grafik tren ini mempertegas temuan dari boxplot dengan menampilkan rata-rata fitur utama di setiap kelas quality:
- **Alcohol** → tren naik konsisten: anggur berkualitas tinggi cenderung memiliki alkohol lebih tinggi
- **Volatile acidity** → tren turun konsisten: kadar asam volatil tinggi merusak cita rasa anggur
- **Sulphates** → tren naik: berfungsi sebagai pengawet yang menjaga kualitas
- **Citric acid** → tren naik: menambah kompleksitas dan kesegaran rasa

#### 3.7 Heatmap Korelasi
Heatmap menampilkan koefisien korelasi Pearson antar semua fitur numerik. Nilai mendekati +1 menunjukkan korelasi positif kuat, mendekati -1 menunjukkan korelasi negatif kuat, dan mendekati 0 berarti tidak ada korelasi linear. Temuan utama:
- `alcohol` memiliki korelasi **positif** yang cukup kuat terhadap `quality` (anggur lebih beralkohol → kualitas cenderung lebih tinggi)
- `volatile acidity` memiliki korelasi **negatif** terhadap `quality` (asam volatil tinggi → kualitas turun)
- `density` dan `alcohol` berkorelasi negatif kuat satu sama lain (ini masuk akal karena alkohol lebih ringan dari air)
- Heatmap juga membantu mendeteksi **multikolinearitas** — ketika dua fitur berkorelasi sangat tinggi, salah satunya mungkin tidak memberikan informasi tambahan yang berarti

---

### Langkah 4 — Pembersihan Data

Sebelum melatih model, data harus dibersihkan dari dua masalah umum:

1. **Missing values (nilai yang hilang)**: Diperiksa dengan `df_train.isnull().sum()`. Jika ada, baris tersebut dihapus menggunakan `dropna()`. Data dengan nilai yang hilang dapat menyebabkan error saat pelatihan model atau menghasilkan prediksi yang tidak akurat.

2. **Duplikat**: Baris yang identik dengan baris lain diperiksa dengan `duplicated()` dan dihapus menggunakan `drop_duplicates()`. Data duplikat dapat menyebabkan model "menghafal" sampel tertentu (overfitting) dan memberikan gambaran performa yang terlalu optimis.

Setelah pembersihan, shape dataset dicek kembali untuk memastikan data sudah bersih dan siap digunakan.

---

### Langkah 5 — Persiapan Fitur dan Target

Dataset dipisahkan menjadi:
- **`x`** (variabel independen / fitur): semua kolom kecuali `quality` dan `Id`
- **`y`** (variabel dependen / target): kolom `quality`

Kolom `Id` dipisahkan dari data testing agar bisa digunakan kembali saat menyimpan hasil prediksi. Pemisahan ini penting karena `Id` adalah pengenal unik dan bukan fitur kimia, sehingga tidak boleh dimasukkan sebagai input model.

---

### Langkah 6 — Split Data Training dan Validasi

Data training dibagi menjadi dua subset menggunakan `train_test_split()`:
- **70% untuk training** → digunakan untuk melatih model
- **30% untuk validasi** → digunakan untuk mengevaluasi performa model secara "jujur"

Parameter `random_state=1` memastikan pembagian selalu sama setiap kali kode dijalankan (reproducible). Tanpa pembagian ini, kita tidak bisa mengetahui seberapa baik model akan bekerja pada data yang belum pernah dilihat sebelumnya. Mengevaluasi model pada data yang sama dengan data training akan memberikan estimasi performa yang terlalu optimis.

---

### Langkah 7 — Model 1: Logistic Regression

**Cara kerja:** Logistic Regression menghitung probabilitas suatu data masuk ke kelas tertentu menggunakan fungsi sigmoid. Meskipun namanya mengandung kata "regression", model ini digunakan untuk klasifikasi.

**Kelebihan:** Sederhana, cepat, mudah diinterpretasikan, tidak memerlukan tuning yang kompleks.

**Kekurangan:** Hanya mampu menangkap hubungan *linear* antara fitur dan target. Untuk data kimiawi anggur yang memiliki pola hubungan kompleks dan non-linear, performa Logistic Regression cenderung terbatas.

**Peran dalam proyek:** Digunakan sebagai **baseline model** — tolok ukur awal untuk menilai apakah model yang lebih kompleks benar-benar memberikan peningkatan yang signifikan.

**Evaluasi:** Confusion matrix menampilkan tabel yang menunjukkan jumlah prediksi benar dan salah untuk setiap kelas. Classification report menampilkan precision, recall, dan F1-score per kelas — metrik yang lebih informatif dibanding akurasi saja, terutama pada dataset dengan class imbalance.

---

### Langkah 8 — Model 2: K-Nearest Neighbors (KNN)

**Cara kerja:** KNN mengklasifikasikan data baru berdasarkan mayoritas kelas dari K tetangga terdekatnya di ruang fitur. Jika K=7, model akan mencari 7 data training yang paling mirip dengan data baru tersebut, lalu mengambil keputusan berdasarkan suara terbanyak.

**Pemilihan nilai K yang optimal:** Percobaan dilakukan dengan berbagai nilai K (1, 3, 5, 7, ..., 21) dan akurasi train/test dicatat untuk setiap nilai. Grafik yang dihasilkan menunjukkan fenomena penting:
- K kecil → model terlalu sensitif terhadap noise → **overfitting** (akurasi training tinggi, test rendah)
- K besar → model terlalu sederhana → **underfitting** (akurasi training dan test sama-sama rendah)
- K optimal → titik keseimbangan di mana akurasi test tertinggi

**Catatan penting:** KNN sensitif terhadap perbedaan skala fitur karena menggunakan jarak Euclidean. Idealnya, normalisasi (seperti StandardScaler) perlu diterapkan sebelum KNN. Ini merupakan poin penting untuk pengembangan lebih lanjut.

---

### Langkah 9 — Model 3: Decision Tree

**Cara kerja:** Decision Tree membangun aturan klasifikasi berbentuk pohon keputusan dengan cara membagi data secara rekursif berdasarkan fitur yang paling efektif memisahkan kelas. Di setiap node, model mencari *split* yang memaksimalkan kemurnian kelas (diukur dengan Gini impurity atau Information Gain / Entropy).

**Kelebihan:** Mudah divisualisasikan dan diinterpretasikan, tidak memerlukan normalisasi fitur.

**Kekurangan:** Sangat rentan terhadap **overfitting** tanpa pembatasan kedalaman, karena pohon tanpa batas akan "menghafalkan" data training.

**Visualisasi pohon:** Pohon ditampilkan dengan kedalaman maksimal 3 level agar bisa dibaca. Node paling atas (root) adalah fitur yang paling informatif dalam memisahkan kelas. Warna node yang lebih gelap menunjukkan dominasi kelas yang lebih kuat (lebih "murni").

**Tuning dengan GridSearchCV:** Untuk mengatasi overfitting, dilakukan pencarian hyperparameter terbaik secara sistematis:
- `max_depth` → membatasi kedalaman pohon agar tidak terlalu kompleks
- `criterion` → memilih antara Gini impurity atau Entropy sebagai ukuran kemurnian

`StratifiedKFold` dengan 3 lipatan (folds) digunakan untuk cross-validation, memastikan setiap fold memiliki proporsi kelas yang sama dengan dataset asli.

---

### Langkah 10 — Model 4: Random Forest

**Cara kerja:** Random Forest adalah algoritma **ensemble** yang membangun ratusan Decision Tree secara paralel. Setiap pohon dilatih pada subset data yang dipilih secara acak (*bootstrap sampling*) dan hanya menggunakan subset fitur yang acak di setiap pembelahan. Prediksi akhir ditentukan melalui **voting mayoritas** dari semua pohon.

**Mengapa Random Forest lebih baik dari Decision Tree tunggal?**
1. **Mengurangi overfitting**: Dengan menggabungkan banyak pohon yang "berbeda-beda", kesalahan individual saling meniadakan
2. **Lebih stabil**: Tidak sensitif terhadap satu perubahan kecil pada data
3. **Robust terhadap outlier**: Pengaruh outlier "dilunakkan" karena ada banyak pohon
4. **Tidak perlu normalisasi**: Tidak sensitif terhadap skala fitur

Parameter `n_estimators=200` berarti dibangun 200 pohon keputusan. Semakin banyak pohon, semakin stabil prediksi, namun dengan biaya komputasi yang lebih tinggi.

**Feature Importance:** Salah satu keunggulan Random Forest adalah kemampuannya menghasilkan skor kepentingan fitur (*feature importance*). Skor ini mengukur seberapa sering dan efektif setiap fitur digunakan dalam pembelahan di seluruh pohon. Fitur dengan skor tinggi berarti paling berpengaruh dalam menentukan kualitas anggur. Visualisasi ini ditampilkan dalam dua format: horizontal bar chart dan pie chart untuk menunjukkan proporsi kontribusi setiap fitur.

---

### Langkah 11 — Perbandingan Semua Model

Setelah semua model dievaluasi, akurasi masing-masing dibandingkan dalam satu tabel dan grafik batang horizontal. Model dengan bar berwarna **emas** menunjukkan model dengan akurasi tertinggi. Perbandingan ini memungkinkan pemilihan model final secara objektif berdasarkan data, bukan asumsi.

---

### Langkah 12 — Kesimpulan Pemilihan Model

**Random Forest** dipilih sebagai model final berdasarkan bukti empiris dari perbandingan akurasi. Alasan teknis pemilihannya:

1. **Ensemble method** yang menggabungkan ratusan pohon → lebih akurat dan stabil
2. **Tidak perlu normalisasi fitur** → praktis dan tidak memerlukan preprocessing tambahan
3. **Tahan terhadap overfitting** → menggunakan teknik *bagging* dan *random feature selection*
4. **Mampu menangkap pola non-linear** → cocok untuk hubungan kompleks dalam data kimia anggur
5. **Menghasilkan feature importance** → memberikan interpretabilitas tambahan meski bersifat ensemble

---

### Langkah 13 — Prediksi Data Testing

Sebelum memprediksi data testing, model **dilatih ulang menggunakan seluruh data training** (100%, bukan hanya 70%). Ini adalah praktik standar dalam machine learning: setelah model terbaik dipilih melalui evaluasi pada validation set, model tersebut dilatih ulang dengan semua data yang tersedia untuk memaksimalkan informasi yang bisa dipelajari.

Hasilnya adalah prediksi nilai `quality` untuk setiap baris di data testing. Distribusi prediksi kemudian divisualisasikan untuk memastikan hasilnya masuk akal dan konsisten dengan distribusi di data training.

---

### Langkah 14 — Simpan Hasil Prediksi

Hasil prediksi disimpan dalam file `hasil_prediksi.csv` dengan dua kolom: `Id` dan `quality`. Format ini sesuai dengan kebutuhan submission. Kolom `Id` digunakan untuk mengidentifikasi setiap baris, sedangkan `quality` berisi prediksi model.

---

### Langkah 15 — Ringkasan Akhir

| Tahap | Keterangan |
|---|---|
| Dataset | Wine Quality (training + testing) |
| Fitur | 12 variabel kimiawi anggur |
| Target | `quality` (skala 0–10) |
| Pembersihan | Hapus missing values dan duplikat |
| Model yang diuji | Logistic Regression, KNN, Decision Tree, Random Forest |
| Model terpilih | **Random Forest** (akurasi tertinggi) |
| Output | `hasil_prediksi.csv` (Id + quality) |

---

## 📈 Hasil dan Performa Model

| Model | Keterangan |
|---|---|
| Logistic Regression | Baseline — performa terbatas karena hanya menangkap pola linear |
| KNN (K=7) | Performa moderat; sensitif terhadap skala fitur |
| Decision Tree (Default) | Rentan overfitting tanpa pembatasan kedalaman |
| Decision Tree (Tuned) | Meningkat setelah GridSearchCV menemukan parameter optimal |
| **Random Forest** | **Akurasi tertinggi** — dipilih sebagai model final |

> Nilai akurasi aktual masing-masing model dapat dilihat langsung di output notebook.

---

## 💡 Insight Utama dari Analisis

1. **`alcohol`** adalah fitur paling berpengaruh terhadap kualitas anggur — anggur dengan kadar alkohol lebih tinggi cenderung berkualitas lebih baik.

2. **`volatile acidity`** memiliki pengaruh negatif yang kuat — kadar asam volatil yang tinggi menurunkan kualitas rasa anggur karena memberikan rasa seperti cuka.

3. **`sulphates`** dan **`citric acid`** berkorelasi positif dengan kualitas — sulphates berfungsi sebagai pengawet alami dan citric acid menambah kompleksitas rasa.

4. **Class imbalance** perlu diperhatikan — kelas kualitas 5 dan 6 mendominasi dataset, sehingga model cenderung lebih baik memprediksi kelas-kelas ini dibanding kelas ekstrem (3 atau 9).

5. **Random Forest** terbukti paling unggul karena mampu menangkap hubungan kompleks dan non-linear antara fitur kimia dengan kualitas anggur.

---

## ▶️ Cara Menjalankan

1. Clone repository ini atau download semua file
2. Pastikan `data_training.csv` dan `data_testing.csv` berada di direktori yang sama dengan notebook
3. Buka `wine_classification_Hayu_Dhiya_Rahmadani__005_.ipynb` di Google Colab atau Jupyter Notebook
4. Jalankan semua sel secara berurutan (Runtime → Run All)
5. Hasil prediksi akan tersimpan otomatis sebagai `hasil_prediksi.csv`

### Prasyarat
```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```
