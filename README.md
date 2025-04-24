# Laporan Proyek Machine Learning
Project ini merupakan submission dicoding pada kelas Machine learning Terapan. Dalam project ini, saya membuat sistem rekomendasi film KDrama menggunakan model berbasis *content based filtering* dan model *KNN* yang dapat menentukan rekomendasi judul KDrama teratas.
## Project Overview

![image](https://github.com/user-attachments/assets/6841377f-4718-4d1a-8163-04fa8ae7923a)

## Business Understanding
Sistem rekomendasi KDrama ini dapat digunakan untuk membantu pengguna menemukan KDrama yang sesuai dengan selera mereka dengan lebih mudah dan efisien. Dengan memberikan rekomendasi judul KDrama teratas sesuai preferensi, menjadikan penonton KDrama tidak bingung saat memilih tontonan KDrama yang akan di tonton. 

### Problem Statements
- Bagaimana cara membuat sistem rekomendasi KDrama yang merekomendasikan pengguna berdasarkan genre yang ada?
- Dari data rating yang pengguna berikan terhadap KDrama, bagaimana perusahaan channel penayangan dapat merekomendasikan anime yang belum pernah ditonton pengguna?
- Bagimana cara membuat model sistem rekomendasi Cosine Similarity dan K-Nearest Neighbor?
- Bagaimana cara mengukur nilai perfoma model pada sistem rekomendasi yang telah dibangun?

### Goals
Untuk menjawab permasalahan yang ada, dikembangkanlah sistem rekomendasi dengan tujuan :
- Menghasilkan rekomendasi KDrama sebanyak Top-N, dimana N tersebut digunakan top 5/10 Rekomendasi kepada pengguna berdasarkan genre.
- Menghasilkan rekomendasi KDrama yang telah sesuai dengan preferensi pengguna dan juga belum pernah dilihat oleh pengguna.
- Membuat model sistem rekomendasi menggunakan Cosine Similarity dan K-Nearest Neighbor berdasarkan fitur yang telah dipilih dari dataset
- Mengukur perfoma model pada sistem rekomendasi dengan menggunakan metrik evaluasi

### Solution Approach
Dalam proses analisis data, dilakukan Exploratory Data Analysis (EDA) serta visualisasi data untuk memperoleh pemahaman yang lebih baik terhadap dataset. Untuk menghasilkan model prediksi yang optimal, dilakukan beberapa tahap data cleaning seperti menghapus nilai yang hilang (missing values), memeriksa keberadaan data duplikat, menghapus karakter alfanumerik yang tidak diperlukan, serta menghilangkan tautan (URL) dari data. Selain itu, dilakukan proses one-hot encoding untuk mengonversi data kategorikal menjadi format numerik. Guna menilai kinerja model yang dibangun, digunakan beberapa metrik evaluasi seperti Precision, Calinski-Harabasz Score, dan Davies-Bouldin Score.

## Data Understanding
### EDA - Deskripsi Variabel

Dataset : [Kaggle](https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset)                      

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

## Data Preparation

Langkah-langkah :
- Menghapus missing value secara keseluruhan menggunakan `data = data.dropna()`
- Menghapus kolom yang tidak dibutuhkan seperti  'Aired Date','Year of release', 'Aired On', 'Number of Episodes', 'Duration', 'Content Rating', 'Synopsis', 'Tags',
    'Director',
    'Screenwriter',
    'Cast',
    'Rank'
- Fitur yang digunakan:
  - *Content-based Filtering*: `Name`, `Genre`
 
## Modeling 
Pada project ini saya menggunakan Model Cosine Similarity dan K-Nearest Neighbor. Kedua algoritma ini akan mempelajari kesamaan antar data dalam fitur yang ada.

### Cosine similarity
Cosine Similarity dituliskan dalam rumus: 

$$Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||)$$ 

dimana: 
- (A·B)menyatakan produk titik dari vektor A dan B.
- ||A|| mewakili norma Euclidean (magnitudo) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitudo) dari vektor B.

untuk menguji model saya mencoba seperti ini 
Jika input: `Dr. Romantic`, maka output:

| Rekomendasi                       | Genre                                      |
|-----------------------------------|--------------------------------------------|
| D-Day	                            | Romance, Drama, Medical                    |
| Good Doctor	                      | Romance, Life, Drama, Medical              |
| If You Wish Upon Me	              | Romance, Life, Drama, Medical              |
| God's Quiz: Reboot	               | Mystery, Medical                           |
| Dr. Romantic 2                    | Romance, Drama, Medical, Melodrama         |

### K-Nearest Neighbor
K-Nearest Neighbor dituliskan dalam rumus:

 $$Euclidean Distance (P, Q) = sqrt(∑(Pi - Qi)^2)$$

dimana:
- Pi, mewakili fitur ke-i dari titik data P.
- Qi, mewakili fitur ke-i dari titik data Q (titik data dari kumpulan data D).
- ∑ adalah simbol penjumlahan pada semua fitur titik data.

- Input: data genre & rating
- Output: Top-N rekomendasi
- Parameter penting: jumlah tetangga (`k`), metrik jarak
  
berikut merupakan hasil pengujian model _K-Nearest Neighbor_ dengan _metrik Euclidean Distance_: 

apabila pengguna menyukai drama :_**"Tomorrow"**_

Berikut ini adalah aplikasi yang juga mungkin akan disukai :
| KDrama Name                                  | Similarity Score |
|----------------------------------------------|------------------|
| Our Beloved Summer                           | 98.57%           |
| One Spring Night                            	| 98.5%            |
| Rookie Historian Goo Hae Ryung	              | 98.46%           |
| Happiness                                    | 98.0%            |
| SKY Castle	                                  | 98.0%            |

## Evaluation
- **Precision**: Seberapa relevan rekomendasi Top-k
- **Calinski-Harabasz Score**: Evaluasi kualitas klaster,
  Skor ini digunakan untuk mengevaluasi kualitas hasil clustering. Semakin tinggi nilainya, semakin baik performa clustering karena :
  - Variasi antar klaster (jarak antara klaster) besar.
  - Variasi dalam klaster (jarak antar data dalam satu klaster) kecil.
Nilai 13.28 menunjukkan bahwa terdapat pemisahan klaster yang cukup baik, meskipun skor ini belum terlalu tinggi. Bisa jadi masih ada ruang untuk meningkatkan pemisahan antar klaster.
- **Davies-Bouldin Score**: Evaluasi klaster berdasarkan kesamaan dalam dan antar klaster,
  Skor ini juga digunakan untuk mengevaluasi hasil clustering, namun dengan logika yang berbeda. Semakin rendah nilainya, semakin baik, karena menunjukkan:
  - Klaster saling berjauhan satu sama lain.
  - Data dalam satu klaster lebih kompak.
Nilai 3.10 tergolong kurang baik karena idealnya nilai ini mendekati 0. Ini mengindikasikan bahwa beberapa klaster mungkin masih terlalu tumpang tindih atau data dalam satu klaster belum cukup homogen.

## 7. Kesimpulan
- Sistem berhasil merekomendasikan KDrama serupa berdasarkan genre dan rating.
- Cosine Similarity efektif untuk pendekatan content-based.
- KNN menambah variasi hasil rekomendasi.
- Sistem dapat dikembangkan dengan memasukkan feedback pengguna atau metode hybrid filtering.
