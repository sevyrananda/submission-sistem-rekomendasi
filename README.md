# Laporan Proyek Machine Learning
Project ini merupakan submission dicoding pada kelas Machine learning Terapan. Dalam project ini, akan membuat sistem rekomendasi film KDrama menggunakan model berbasis *content based filtering* dan model *KNN* yang dapat menentukan rekomendasi judul KDrama teratas.
## 1. Project Overview
Industri hiburan Korea Selatan, khususnya KDrama, mengalami pertumbuhan pesat dan mendapat perhatian luas dari penonton global. Dengan banyaknya pilihan drama yang tersedia, pengguna sering mengalami kesulitan dalam menemukan KDrama yang sesuai dengan preferensi mereka.

Proyek ini bertujuan untuk membangun sistem rekomendasi KDrama berbasis Content-Based Filtering dan K-Nearest Neighbor (KNN), untuk membantu pengguna menemukan tontonan yang relevan berdasarkan kesamaan genre dan rating.

Model yang dikembangkan menggunakan dataset berisi 250 KDrama populer dan dievaluasi menggunakan metrik evaluasi sistem rekomendasi.

![image](https://github.com/user-attachments/assets/6841377f-4718-4d1a-8163-04fa8ae7923a)

## 2. Business Understanding
Sistem rekomendasi KDrama ini dapat digunakan untuk membantu pengguna menemukan KDrama yang sesuai dengan selera mereka dengan lebih mudah dan efisien. Dengan memberikan rekomendasi judul KDrama teratas sesuai preferensi, menjadikan penonton KDrama tidak bingung saat memilih tontonan KDrama yang akan di tonton. 

### Problem Statements
- Bagaimana membangun sistem rekomendasi KDrama berdasarkan genre untuk membantu pengguna menemukan tontonan serupa?
- Bagaimana memanfaatkan data rating untuk merekomendasikan KDrama yang relevan namun belum ditonton pengguna?
- Bagaimana mengimplementasikan dan mengoptimalkan model berbasis Cosine Similarity dan K-Nearest Neighbor?
- Bagaimana mengevaluasi performa sistem rekomendasi menggunakan metrik evaluasi yang sesuai?

### Goals
Untuk menjawab permasalahan yang ada, dikembangkanlah sistem rekomendasi dengan tujuan :
- Menghasilkan rekomendasi Top-N KDrama berdasarkan genre pengguna.
- Memberikan rekomendasi KDrama relevan yang belum pernah ditonton pengguna berdasarkan preferensi.
- Membangun dan membandingkan dua model rekomendasi: Content-Based Filtering (Cosine Similarity) dan KNN berbasis rating dan genre.
- Mengevaluasi performa sistem menggunakan Precision@K, RMSE, dan metrik klasterisasi.

### Solution Approach
Dalam proses analisis data, dilakukan Exploratory Data Analysis (EDA) serta visualisasi data untuk memperoleh pemahaman yang lebih baik terhadap dataset. Untuk menghasilkan model prediksi yang optimal, dilakukan beberapa tahap data cleaning seperti menghapus nilai yang hilang (missing values), memeriksa keberadaan data duplikat, menghapus karakter alfanumerik yang tidak diperlukan, serta menghilangkan tautan (URL) dari data. Selain itu, dilakukan proses one-hot encoding untuk mengonversi data kategorikal menjadi format numerik. Guna menilai kinerja model yang dibangun, digunakan beberapa metrik evaluasi seperti Precision, Calinski-Harabasz Score, dan Davies-Bouldin Score.

## 3. Data Understanding
### EDA - Deskripsi Variabel

Dataset : [Kaggle](https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset)                      

## Kolom-kolom penting :
- Name : Judul KDrama
- Genre : Genre drama
- Rating : Skor penilaian pengguna
- Original Network, Production Companies, dan fitur terkait lainnya

**Berikut informasi pada dataset** :
 - Datasets berupa file csv (Comma-Seperated Values).
 - Dataset berupa 1 buah file CSV yaitu: 
    * kdarama.csv  
 - Dataset memiliki 250 data review dengan 17 fitur.
 - Dataset memilik 14 fitur object, 2 fitur int64, dan 1 fitur float64.
 - Tidak ada data yang duplikat.
 - Terdapat *Missing value* pada fitur Content Rating sebanyak 5, Director sebanyak 1, Screenwriter sebanyak 1, dan Production companies sebanyak 2.

### Kolom datasets anime memiliki informasi berikut:

*    Name : Judul KDrama
*    Aired Date : Tanggal dari saat hingga KDrama pertama kali ditayangkan.
*    Year of release : Tahun rilis KDrama
*    Original Network : Penayangan KDrama
*    Aired On : Hari-hari dalam seminggu saat KDrama ditayangkan.
*    Number of Episodes : Jumlah episode yang ada
*    Duration : Lama penayangan (dalam jam dan menit)
*    Content Rating : Batasan usia
*    Rating : Skor atau penilaian yang diberikan penonton pada KDrama
*    Sypnosis : Sipnosis dari KDrama
*    Genre : Jenis KDrama (Life, Drama, Romantic, dan lainnya)
*    Tags : Tema
*    Director : Sutradara
*    Screenwriter : Pnulis
*    Cast : Pemeran
*    Production companies : Perusahaan yang memproduksi KDrama
*    Rank : Peringkat KDrama

## 4. Data Preparation
Langkah-langkah persiapan data:

- **Menghapus Missing Values**: Semua entri dengan nilai kosong dihapus menggunakan `data = data.dropna()`.
- **Pemilihan Fitur**: Menggunakan fitur `Genre` untuk proses rekomendasi.
- **TF-IDF Vectorization**: Mengubah teks pada kolom `Genre` menjadi representasi numerik menggunakan TF-IDF.

 
## 5. Modeling 
- Menggunakan **Cosine Similarity** antar fitur drama.
- Fungsi rekomendasi mencari drama paling mirip berdasarkan drama pilihan pengguna.
- 2 Skema rekomendasi:
  - Berdasarkan **Genre saja**
  - Berdasarkan **Genre + Rating**
    
### Content-Based Filtering (Cosine Similarity)

**Metode**:
- Menghitung kesamaan antar drama menggunakan Cosine Similarity berdasarkan TF-IDF dari genre.

**Rumus**:

### Cosine similarity
Cosine Similarity dituliskan dalam rumus: 

$$Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)$$ 

dimana: 
- (A·B)menyatakan produk titik dari vektor A dan B.
- ||A|| mewakili norma Euclidean (magnitudo) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitudo) dari vektor B.

untuk menguji model  mencoba seperti ini 
1. Rekomendasi berdasarkan genre saja
Jika input: `Memory`, maka output:

| Rekomendasi                       | Genre                                      |
|-----------------------------------|--------------------------------------------|
| Extraordinary Attorney Woo        | Law, Romance, Life, Drama                  |
| Oh My Venus	                      | Comedy, Law, Romance, Life                 |
| Juvenile Justice		                | Law, Drama                                 |
| Touch Your Heart	                 | Comedy, Law, Romance, Drama                |
| Confession                        | Thriller, Mystery, Law, Drama              |

2. Rekomendasi berdasarkan genre + rating
Input yang digunakan `Memory`, maka output:

|Name	| Genre |	Rating |
|-----|-------|--------|
| Extraordinary Attorney Woo	Law| Romance| Life | Drama	9.0 |
| Defendant	Thriller | Mystery | Law | Drama	8.7 |
| Big Mouth	Thriller | Mystery | Law, Drama	8.6 |
| Juvenile Justice	Law | Drama	8.6 |
| Stranger 2	Thriller | Mystery | Law | Drama	8.6 |
| Because This Is My First Life	Comedy | Romance | Life | Drama	8.5 |
| My Unfamiliar Family	Life | Drama	8.4 |
| Oh My Venus	Comedy | Law | Romance | Life	8.3 |
| Touch Your Heart	Comedy | Law | Romance | Drama	8.3 |
| When the Weather Is Fine	Romance | Life | Drama | Melodrama	8.3 |

📌 Kesimpulan:

- Skema pertama cukup baik untuk menemukan drama serupa secara konten.
- Skema kedua menambahkan filter kualitas (rating) yang lebih membantu pengguna dalam memilih tontonan terbaik dari genre yang mirip.


## 6. Evaluation
### 1. Metrik Evaluasi
Sistem rekomendasi dievaluasi menggunakan:

- Precision: Mengukur proporsi rekomendasi yang relevan terhadap total rekomendasi.
- Recall: Mengukur sejauh mana sistem berhasil menemukan item relevan.
- F1-Score: Harmonik rata-rata Precision dan Recall.

---

### 2. Hasil Evaluasi

| Skema | Precision |
|:------|:----------|
| Genre saja | 1.00 |
| Genre + Rating | 1.00 |

**Insight**:
- Semua rekomendasi pada kedua skema relevan terhadap genre yang dicari.
- Menambahkan filter rating tidak menurunkan kualitas, bahkan membantu menyaring tontonan lebih berkualitas.

### 3. Average Precision, Recall, F1-Score

| Precision | Recall | F1-Score |
|:------|:----------|:----------|
| 1.00 | 0.07 | 0.13 |

**Insight**:

- Precision tinggi → sistem sangat tepat dalam memberikan rekomendasi yang sesuai.
- Recall rendah → sistem hanya merekomendasikan sebagian kecil dari semua drama yang sesuai genre → sistem terlalu selektif.
- F1-score rendah → sistem sangat hati-hati, tapi mengorbankan cakupan.

## 🔎 Additional Analysis (Opsional)

Sebagai tambahan, dilakukan analisis clustering terhadap drama menggunakan KMeans.

- **Calinski-Harabasz Score**: 13.28 → menunjukkan kualitas klaster cukup baik.
- **Davies-Bouldin Score**: 3.10 → menunjukkan masih ada sedikit overlap antar klaster.

> *Catatan:*  
> Analisis clustering ini **bukan** digunakan untuk mengevaluasi performa sistem rekomendasi, melainkan hanya untuk memahami struktur data.


### Hubungan dengan Business Understanding

- Sistem dapat membantu pengguna memilih drama sesuai preferensi genre dan rating.
- Model mampu memberikan rekomendasi relevan, meningkatkan engagement dan kepuasan pengguna.
- Evaluasi metrik clustering menunjukkan performa yang cukup baik, namun masih terdapat ruang penyempurnaan, khususnya dalam mengurangi overlap antar klaster rekomendasi.

  
## 7. Kesimpulan
- Sistem berhasil menjawab seluruh pertanyaan pada Business Understanding.
- Evaluasi menunjukkan bahwa model sangat akurat, namun bisa lebih baik dalam menjangkau lebih banyak tontonan.
- Sistem dapat dikembangkan lebih lanjut dengan kombinasi metode Content-Based dan Collaborative Filtering, serta mempertimbangkan data pengguna dan ulasan.
