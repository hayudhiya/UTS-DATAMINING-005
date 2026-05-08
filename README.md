# UTS Data Mining — Klasifikasi Kualitas Anggur (Wine Quality Classification)

**Nama**: Hayu Dhiya Rahmadani  
**NIM**: 2304020005  
**Mata Kuliah**: Data Mining

---

## Deskripsi Proyek

Repository ini berisi hasil pengerjaan UTS Data Mining dengan topik klasifikasi kualitas anggur menggunakan dataset Wine Quality. Tujuan utama adalah membangun model machine learning yang mampu memprediksi nilai kualitas anggur (`quality`, skala 0–10) berdasarkan fitur-fitur kimiawi yang tersedia, kemudian menerapkan model terbaik untuk memprediksi kualitas pada dataset testing yang tidak memiliki label.

---

## Pendahuluan

Anggur merupakan salah satu minuman fermentasi yang kualitasnya sangat dipengaruhi oleh komposisi kimiawi yang terkandung di dalamnya. Penilaian kualitas anggur secara konvensional dilakukan oleh para ahli melalui uji organoleptik — penilaian berdasarkan rasa, aroma, dan penampilan — yang bersifat subjektif dan memerlukan keahlian khusus.

Pendekatan berbasis machine learning menawarkan cara yang lebih objektif dan efisien. Dengan memanfaatkan data kimiawi seperti kadar alkohol, keasaman volatil, dan kandungan sulfat, sebuah model klasifikasi dapat dilatih untuk mengenali pola yang membedakan anggur berkualitas tinggi dari yang berkualitas rendah.

Proyek ini merupakan bagian dari UTS mata kuliah Data Mining. Lima algoritma klasifikasi diuji dan dibandingkan — Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, dan Support Vector Machine — untuk menemukan model dengan performa terbaik dalam memprediksi kualitas anggur berdasarkan fitur kimiawi yang tersedia.

---

## Struktur Repository

```
├── wine_classification_Hayu_Dhiya_R.ipynb   # Notebook analisis lengkap dengan interpretasi
├── data_training.csv                         # Dataset untuk melatih model (berlabel)
├── data_testing.csv                          # Dataset untuk prediksi akhir (tanpa label)
├── hasil_prediksi.csv                        # File hasil prediksi (Id + quality)
└── README.md                                 # Dokumentasi proyek ini
```

---

## Dataset

Dataset yang digunakan adalah Wine Quality dataset yang memuat data anggur beserta fitur-fitur kimiawinya.

| Dataset | Keterangan |
|---|---|
| `data_training.csv` | Memiliki label `quality` (target), digunakan untuk melatih dan mengevaluasi model |
| `data_testing.csv` | Tidak memiliki label `quality`, digunakan untuk prediksi akhir |

Fitur-fitur yang digunakan sebagai input model:

| Fitur | Deskripsi |
|---|---|
| `fixed acidity` | Kadar asam tetap dalam anggur |
| `volatile acidity` | Kadar asam volatil; nilai tinggi dapat menyebabkan rasa cuka |
| `citric acid` | Asam sitrat yang menambah kesegaran rasa |
| `residual sugar` | Kadar gula yang tersisa setelah fermentasi |
| `chlorides` | Kadar garam dalam anggur |
| `free sulfur dioxide` | SO₂ bebas yang mencegah pertumbuhan mikroba |
| `total sulfur dioxide` | Total SO₂ (bebas + terikat) |
| `density` | Kerapatan anggur |
| `pH` | Tingkat keasaman anggur |
| `sulphates` | Aditif yang berkontribusi pada SO₂ |
| `alcohol` | Kadar alkohol dalam anggur |
| `type` | Jenis anggur (red/white) |
| **`quality`** | **(Target)** Kualitas anggur, skala 0–10 |

Distribusi kelas pada data training menunjukkan adanya **class imbalance** — kelas 5 dan 6 mendominasi secara signifikan, sementara kelas dengan kualitas sangat rendah (3) atau sangat tinggi (8–9) sangat sedikit. Kondisi ini perlu diperhatikan dalam evaluasi model agar tidak bias terhadap kelas mayoritas.

---

## Alur Pengerjaan

### Langkah 1 — Persiapan dan Pemuatan Data

Dataset training dan testing dimuat menggunakan `pandas`. Dataset training memuat fitur-fitur kimiawi beserta kolom target `quality`, sedangkan dataset testing tidak memiliki kolom target. Kolom `Id` pada data testing dipisahkan dan disimpan terpisah untuk digunakan kembali saat menyimpan hasil prediksi akhir.

### Langkah 2 — Eksplorasi Data (EDA)

Beberapa temuan utama dari tahap eksplorasi:

- **Distribusi target**: Mayoritas data bernilai kualitas 5 dan 6. Kelas dengan kualitas sangat rendah (3) atau sangat tinggi (8–9) sangat sedikit, mencerminkan class imbalance yang nyata.
- **Korelasi fitur**: Fitur `alcohol` memiliki korelasi positif yang cukup kuat terhadap `quality` — anggur dengan kadar alkohol lebih tinggi cenderung berkualitas lebih baik. Sebaliknya, `volatile acidity` berkorelasi negatif, artinya keasaman volatil yang tinggi berkaitan dengan kualitas yang lebih rendah.
- **Skala fitur**: Nilai antar fitur bervariasi sangat besar. `total sulfur dioxide` bisa mencapai ratusan, sementara `chlorides` bernilai di bawah 1. Hal ini perlu diperhatikan pada model yang sensitif terhadap skala fitur.

### Langkah 3 — Pembersihan Data

Dilakukan pemeriksaan terhadap:
- **Missing values**: Diperiksa per kolom menggunakan `isnull().sum()`. Baris yang mengandung nilai hilang dihapus.
- **Duplikasi**: Diperiksa menggunakan `duplicated()`. Baris duplikat dihapus agar model tidak belajar dari data yang berulang.

Setelah proses pembersihan, dataset dipastikan bersih dan siap digunakan untuk pelatihan model.

### Langkah 4 — Persiapan Fitur dan Target

Variabel fitur (`x`) dipisahkan dari variabel target (`y` = `quality`) dengan menghapus kolom `quality` dan `Id` dari data training. Data training kemudian dibagi menjadi **70% untuk training** dan **30% untuk validasi** menggunakan `train_test_split` dengan `random_state=1` agar hasil dapat direproduksi.

### Langkah 5 — Pelatihan dan Evaluasi Model

Lima model dilatih dan dievaluasi menggunakan data validasi. Berikut ringkasan pendekatan masing-masing model:

**Model 1 — Logistic Regression**  
Digunakan sebagai **baseline**. Model linear yang bekerja dengan menghitung probabilitas suatu data masuk ke kelas tertentu. Sederhana dan mudah diinterpretasikan, namun kemungkinan besar belum mampu menangkap pola kompleks dan non-linear dalam data kimiawi anggur.

**Model 2 — K-Nearest Neighbors (KNN)**  
Mengklasifikasikan data baru berdasarkan mayoritas kelas dari K tetangga terdekatnya. Eksperimen dilakukan pada berbagai nilai K (1, 3, 5, ..., 21) untuk menemukan nilai yang optimal. Grafik perbandingan train accuracy vs test accuracy digunakan untuk mendiagnosis overfitting atau underfitting.

**Model 3 — Decision Tree**  
Membagi data secara rekursif berdasarkan fitur yang paling efektif memisahkan kelas. Pohon divisualisasikan (3 level pertama) untuk memperlihatkan logika pembelahan. Tuning dilakukan dengan **GridSearchCV** untuk mencari kombinasi terbaik dari `max_depth` dan `criterion` (gini/entropy) menggunakan 3-Fold Stratified Cross-Validation.

**Model 4 — Random Forest**  
Metode **ensemble** yang membangun 200 Decision Tree secara paralel dan menggabungkan hasilnya via voting mayoritas. Lebih stabil dan tahan terhadap overfitting dibandingkan Decision Tree tunggal. Feature importance divisualisasikan untuk mengetahui fitur mana yang paling berpengaruh dalam prediksi.

**Model 5 — Support Vector Machine (SVM)**  
Mencari **hyperplane** terbaik yang memisahkan kelas di ruang fitur berdimensi tinggi. Digunakan kernel linear dengan parameter `C=1` dan `gamma=0.1`. Efektif pada dataset berdimensi tinggi, namun membutuhkan waktu komputasi lebih lama.

### Langkah 6 — Perbandingan Model

Seluruh model dibandingkan akurasinya secara visual menggunakan horizontal bar chart. Model dengan akurasi tertinggi ditandai dengan warna emas. Perbandingan ini menjadi dasar pemilihan model final.

### Langkah 7 — Prediksi Data Testing

**Random Forest** dipilih sebagai model final karena menghasilkan akurasi tertinggi pada data validasi. Alasan pemilihan:

1. Merupakan metode **ensemble** yang lebih stabil dan akurat dibandingkan model lain
2. Mampu menangani **fitur dengan skala bervariasi** tanpa memerlukan normalisasi eksplisit
3. Lebih **tahan terhadap overfitting** dibandingkan Decision Tree tunggal karena menggunakan subset fitur secara acak
4. Mampu menangkap **hubungan non-linear** yang tidak bisa ditangkap oleh Logistic Regression

Sebelum prediksi, model dilatih ulang menggunakan **seluruh** data training (100%) untuk memaksimalkan informasi yang dipelajari — praktik umum setelah model final dipilih.

Distribusi hasil prediksi pada data testing menunjukkan pola yang konsisten dengan data training, di mana kualitas 5 dan 6 mendominasi. Ini mengindikasikan bahwa model berhasil mempelajari distribusi kelas dengan baik dan tidak mengalami bias ekstrem ke salah satu kelas.

### Langkah 8 — Penyimpanan Hasil

Hasil prediksi disimpan dalam file `hasil_prediksi.csv` yang hanya memuat dua kolom: `Id` dan `quality`, sesuai format yang diminta.

---

## Hasil Ringkasan

| Aspek | Hasil |
|---|---|
| Model yang diuji | Logistic Regression, KNN, Decision Tree, Random Forest, SVM |
| Model terpilih | **Random Forest** (akurasi tertinggi) |
| Jumlah estimator | 200 |
| Tuning | GridSearchCV pada Decision Tree (max_depth, criterion) |
| Fitur terpenting | `alcohol`, `volatile acidity` |
| Output prediksi | `hasil_prediksi.csv` (Id + quality) |

### Performa Model

Model Random Forest yang dibangun menunjukkan akurasi tertinggi di antara semua model yang diuji. Hasil ini perlu diinterpretasikan dalam konteks permasalahannya — klasifikasi multi-kelas dengan distribusi yang sangat tidak merata. Kelas mayoritas (quality 5 dan 6) menyumbang sebagian besar total data, sementara kelas minoritas (quality 3 dan 8–9) hanya memiliki sangat sedikit sampel.

Sebagai perbandingan, jika model hanya menebak kelas yang paling sering muncul (quality 5) untuk semua prediksi, akurasi yang diperoleh hanya setara dengan proporsi kelas tersebut. Dengan demikian, akurasi Random Forest menunjukkan bahwa model berhasil mempelajari pola yang bermakna dari data, jauh di atas tebakan naif tersebut.

**Feature Importance — Random Forest**

Fitur `alcohol` dan `volatile acidity` secara konsisten muncul sebagai fitur paling berpengaruh. Temuan ini konsisten dengan hasil EDA di mana keduanya memiliki korelasi tertinggi terhadap `quality`, serta dengan literatur enologi yang menyatakan bahwa kadar alkohol dan keasaman merupakan faktor kunci penentu kualitas anggur.

---

## Kesimpulan

Berdasarkan seluruh tahapan yang telah dilakukan, dapat ditarik beberapa kesimpulan:

1. Dataset Wine Quality dalam kondisi bersih setelah melalui tahap pembersihan. Namun terdapat **ketidakseimbangan kelas** yang cukup signifikan, di mana kelas 5 dan 6 mendominasi dan kelas di ujung ekstrem sangat sedikit.

2. Lima model klasifikasi diuji dan dibandingkan. **Random Forest** menghasilkan akurasi tertinggi karena kemampuannya menangkap pola non-linear dan ketahanannya terhadap overfitting melalui mekanisme ensemble.

3. Dari analisis feature importance, kadar alkohol (`alcohol`) terbukti menjadi faktor paling berpengaruh dalam menentukan kualitas anggur, diikuti oleh keasaman volatil (`volatile acidity`). Hasil ini konsisten dengan temuan pada tahap EDA dan literatur di bidang enologi.

4. Model berhasil menghasilkan prediksi untuk seluruh data testing dengan distribusi yang konsisten terhadap pola distribusi data training, yaitu didominasi oleh kelas kualitas 5 dan 6.

5. Pendekatan machine learning terbukti mampu menangkap pola dari data kimiawi anggur untuk melakukan klasifikasi kualitas secara otomatis, meski masih terdapat ruang untuk peningkatan, terutama dalam menangani kelas-kelas minoritas.

---

## Referensi

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553.
- Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5–32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- Dataset Wine Quality: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## Cara Menjalankan Notebook

1. Buka Google Colab di [colab.research.google.com](https://colab.research.google.com)
2. Upload file `wine_classification_Hayu_Dhiya_R.ipynb`
3. Jalankan cell pertama untuk mengimpor library yang dibutuhkan
4. Upload `data_training.csv` dan `data_testing.csv` ke direktori yang sesuai
5. Jalankan seluruh cell secara berurutan dari atas ke bawah
6. File `hasil_prediksi.csv` akan otomatis terunduh setelah cell terakhir dijalankan

---

## Dependensi

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn

Semua library di atas tersedia secara default di Google Colab dan tidak memerlukan instalasi tambahan.
