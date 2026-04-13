# Tugas Data Mining — Clustering Kelayakan Pendidikan SD Indonesia

Proyek clustering untuk menganalisis kelayakan pendidikan SD di Indonesia berdasarkan data provinsi, menggunakan algoritma **KMeans** dan **Agglomerative Clustering**.

## Struktur Proyek

```
tugas_data_mining/
├── main.py                        # Entry point utama
├── modules/
│   ├── utils.py                   # Konfigurasi output, fungsi helper
│   ├── preprocessing.py           # Loading data, feature engineering, scaling, PCA
│   ├── clustering.py              # KMeans, Agglomerative, labeling, evaluasi
│   ├── visualisasi.py             # Semua plotting
│   └── laporan.py                 # Ekspor laporan.txt dan hasil_clustering.csv
├── kelayakan-pendidikan-indonesia.csv
└── hasil datamining/              # Output dihasilkan otomatis
```

## Cara Penggunaan

Pastikan file `kelayakan-pendidikan-indonesia.csv` berada di folder yang sama dengan `main.py`, lalu jalankan:

```bash
python main.py
```

## Output

Semua hasil disimpan di folder `hasil datamining/`:

| File | Keterangan |
|---|---|
| `hasil_clustering.csv` | Hasil cluster per provinsi |
| `laporan.txt` | Laporan lengkap evaluasi model |
| `elbow_method.png` | Grafik elbow untuk menentukan k optimal |
| `scatter_kmeans.png` | Scatter plot KMeans (PCA 2D) |
| `scatter_agglomerative.png` | Scatter plot Agglomerative (PCA 2D) |
| `heatmap_kmeans.png` | Heatmap fitur turunan KMeans |
| `heatmap_agglomerative.png` | Heatmap fitur turunan Agglomerative |
| `bar_distribusi.png` | Distribusi provinsi per cluster |
| `dendrogram.png` | Dendrogram Agglomerative (Ward Linkage) |

## Fitur Turunan

| Fitur | Keterangan |
|---|---|
| `rasio_putus_sekolah` | % siswa putus sekolah |
| `rasio_guru_berkualifikasi` | % guru/kepala sekolah ≥S1 |
| `rasio_kelas_rusak` | % ruang kelas rusak (ringan/sedang/berat) |

## Label Cluster

- **Provinsi Maju** — rasio positif tinggi, rasio negatif rendah
- **Provinsi Berkembang** — kondisi menengah
- **Provinsi Perlu Perhatian** — rasio negatif tinggi, rasio positif rendah

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
tabulate
```
