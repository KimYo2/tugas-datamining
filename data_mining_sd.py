import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from tabulate import tabulate
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hasil dataminibf')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(nama_file):
    return os.path.join(OUTPUT_DIR, nama_file)

# ==========================================
# FUNGSI HELPER
# ==========================================

def cetak_tabel(data, judul, jumlah=5):
    print(f"\n{'='*60}")
    print(f"  {judul}")
    print(f"{'='*60}")
    if isinstance(data, pd.DataFrame):
        print(tabulate(data.head(jumlah), headers='keys', tablefmt='grid', floatfmt='.4f'))
    else:
        print(tabulate(data, headers='keys', tablefmt='grid', floatfmt='.4f'))

def simpan_plot(nama_file):
    plt.savefig(nama_file, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# ==========================================
# LANGKAH 1: DATA LOADING & INSPEKSI AWAL
# ==========================================

df = pd.read_csv('kelayakan-pendidikan-indonesia.csv')

cetak_tabel(df, "5 Baris Pertama Dataset", jumlah=5)

print("\n--- INFO DATASET ---")
df.info()

print("\n--- STATISTIK DESKRIPTIF ---")
print(tabulate(df.describe(), headers='keys', tablefmt='grid', floatfmt='.2f'))

print("\n--- MISSING VALUE PER KOLOM ---")
missing = df.isnull().sum()
print(tabulate([[col, val] for col, val in missing.items()],
               headers=['Kolom', 'Missing Value'], tablefmt='grid'))

df = df[df['Provinsi'] != 'Luar Negeri'].reset_index(drop=True)

labels = df['Provinsi'].reset_index(drop=True)
X = df.drop(columns=['Provinsi']).copy()

print(f"\n1. Dataset berhasil dimuat. {len(df)} provinsi, {df.shape[1]} kolom.")

# ==========================================
# LANGKAH 2: FEATURE ENGINEERING
# ==========================================

X['rasio_putus_sekolah'] = (X['Putus Sekolah'] / X['Siswa']) * 100

guru_total = X['Kepala Sekolah dan Guru(<S1)'] + X['Kepala Sekolah dan Guru(>S1)']
X['rasio_guru_berkualifikasi'] = (X['Kepala Sekolah dan Guru(>S1)'] / guru_total) * 100

kelas_rusak = X['Ruang kelas(rusak ringan)'] + X['Ruang kelas(rusak sedang)'] + X['Ruang kelas(rusak berat)']
kelas_total = X['Ruang kelas(baik)'] + kelas_rusak
X['rasio_kelas_rusak'] = (kelas_rusak / kelas_total) * 100

print("2. Feature engineering selesai. 3 fitur baru ditambahkan.")

cetak_tabel(
    X[['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']].assign(Provinsi=labels.values).set_index('Provinsi'),
    "Fitur Turunan (5 Baris Pertama)",
    jumlah=5
)

# ==========================================
# LANGKAH 3: HANDLING MISSING VALUE & SCALING
# ==========================================

fitur_numerik = X.select_dtypes(include=[np.number]).columns.tolist()
for col in fitur_numerik:
    X[col] = X[col].fillna(X[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[fitur_numerik])
n_fitur = X_scaled.shape[1]

print(f"3. Scaling selesai. Total fitur: {n_fitur}")

# ==========================================
# LANGKAH 4: REDUKSI DIMENSI DENGAN PCA
# ==========================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pc1_var = pca.explained_variance_ratio_[0] * 100
pc2_var = pca.explained_variance_ratio_[1] * 100
total_var = pc1_var + pc2_var

print(f"4. PCA selesai. Variansi dijelaskan: {total_var:.2f}%")
print(f"   PCA: PC1={pc1_var:.2f}%, PC2={pc2_var:.2f}%, Total={total_var:.2f}%")

# ==========================================
# LANGKAH 5: MENENTUKAN K OPTIMAL (ELBOW METHOD)
# ==========================================

inertia_list = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_list, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Jumlah Cluster (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method - Menentukan K Optimal', fontsize=14)
plt.xticks(k_range)
plt.grid(True, alpha=0.3)
for i, (k, inertia) in enumerate(zip(k_range, inertia_list)):
    plt.annotate(f'{inertia:.0f}', (k, inertia), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=8)
simpan_plot(out('elbow_method.png'))

diffs = np.diff(inertia_list)
diffs2 = np.diff(diffs)
k_optimal = int(np.argmax(np.abs(diffs2)) + 2)
if k_optimal < 2:
    k_optimal = 3

print(f"5. K optimal yang disarankan: {k_optimal}")

# ==========================================
# LANGKAH 6: TRAINING MODEL KMEANS
# ==========================================

kmeans = KMeans(n_clusters=k_optimal, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.predict(X_scaled)
df['Cluster_KMeans'] = kmeans_labels

print("6. Model KMeans berhasil dilatih.")

# ==========================================
# LANGKAH 7: TRAINING MODEL AGGLOMERATIVE CLUSTERING
# ==========================================

agg = AgglomerativeClustering(n_clusters=k_optimal, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)
df['Cluster_Agglomerative'] = agg_labels

print("7. Model Agglomerative berhasil dilatih.")

# ==========================================
# LANGKAH 8: INTERPRETASI & LABELING CLUSTER
# ==========================================

def beri_label_cluster(df_input, cluster_col):
    rasio_cols = ['rasio_putus_sekolah', 'rasio_kelas_rusak']
    kualifikasi_col = 'rasio_guru_berkualifikasi'
    cluster_means = df_input.groupby(cluster_col)[rasio_cols + [kualifikasi_col]].mean()

    skor = {}
    for c in cluster_means.index:
        skor_negatif = float(cluster_means.at[c, 'rasio_putus_sekolah'] + cluster_means.at[c, 'rasio_kelas_rusak']) / 2
        skor_positif = float(cluster_means.at[c, kualifikasi_col])
        skor[c] = skor_positif - skor_negatif

    sorted_clusters = sorted(skor.keys(), key=lambda c_key: skor[c_key], reverse=True)

    label_map = {}
    n = len(sorted_clusters)
    for i, c in enumerate(sorted_clusters):
        if i == 0:
            label_map[c] = "Provinsi Maju"
        elif i == n - 1:
            label_map[c] = "Provinsi Perlu Perhatian"
        else:
            label_map[c] = "Provinsi Berkembang"

    return label_map

fitur_cols_eng = fitur_numerik
df_with_features = df.copy()
for col in ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']:
    df_with_features[col] = X[col].values

label_map_kmeans = beri_label_cluster(df_with_features, 'Cluster_KMeans')
label_map_agg = beri_label_cluster(df_with_features, 'Cluster_Agglomerative')

df['Label_KMeans'] = df['Cluster_KMeans'].map(label_map_kmeans)
df['Label_Agglomerative'] = df['Cluster_Agglomerative'].map(label_map_agg)

rasio_fitur = ['rasio_putus_sekolah', 'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']

print("\n--- INTERPRETASI CLUSTER KMEANS ---")
km_stats = df_with_features.groupby('Cluster_KMeans')[rasio_fitur].mean().round(4)
km_stats.index = [f"Cluster {i} ({label_map_kmeans[i]})" for i in km_stats.index]
print(tabulate(km_stats, headers='keys', tablefmt='grid', floatfmt='.4f'))

print("\n--- DAFTAR PROVINSI (KMEANS) ---")
km_list = df[['Provinsi', 'Cluster_KMeans', 'Label_KMeans']].copy()
km_list.columns = ['Provinsi', 'Cluster', 'Label']
print(tabulate(km_list, headers='keys', tablefmt='grid', showindex=False))

print("\n--- INTERPRETASI CLUSTER AGGLOMERATIVE ---")
agg_stats = df_with_features.groupby('Cluster_Agglomerative')[rasio_fitur].mean().round(4)
agg_stats.index = [f"Cluster {i} ({label_map_agg[i]})" for i in agg_stats.index]
print(tabulate(agg_stats, headers='keys', tablefmt='grid', floatfmt='.4f'))

print("\n--- DAFTAR PROVINSI (AGGLOMERATIVE) ---")
agg_list = df[['Provinsi', 'Cluster_Agglomerative', 'Label_Agglomerative']].copy()
agg_list.columns = ['Provinsi', 'Cluster', 'Label']
print(tabulate(agg_list, headers='keys', tablefmt='grid', showindex=False))

# ==========================================
# LANGKAH 9: EVALUASI & PERBANDINGAN MODEL
# ==========================================

sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
inertia_kmeans = kmeans.inertia_

sil_agg = silhouette_score(X_scaled, agg_labels)
db_agg = davies_bouldin_score(X_scaled, agg_labels)

print(f"\n9. Silhouette KMeans: {sil_kmeans:.4f} | Silhouette Agglomerative: {sil_agg:.4f}")

tabel_eval = [
    ['Silhouette Score', f'{sil_kmeans:.4f}', f'{sil_agg:.4f}'],
    ['Davies-Bouldin Score', f'{db_kmeans:.4f}', f'{db_agg:.4f}'],
    ['Inertia', f'{inertia_kmeans:.2f}', 'N/A'],
]
print("\n--- TABEL PERBANDINGAN MODEL ---")
print(tabulate(tabel_eval,
               headers=['Metrik', 'KMeans', 'Agglomerative'],
               tablefmt='grid'))

sil_winner = 'KMeans' if sil_kmeans >= sil_agg else 'Agglomerative'
db_winner = 'KMeans' if db_kmeans <= db_agg else 'Agglomerative'

if sil_winner == db_winner:
    best_model = sil_winner
    alasan = (f"memiliki Silhouette Score lebih tinggi ({max(sil_kmeans, sil_agg):.4f}) "
              f"dan Davies-Bouldin Score lebih rendah ({min(db_kmeans, db_agg):.4f})")
elif sil_winner == 'KMeans':
    best_model = 'KMeans'
    alasan = (f"memiliki Silhouette Score lebih tinggi ({sil_kmeans:.4f} vs {sil_agg:.4f}), "
              f"meskipun Davies-Bouldin Score sedikit lebih tinggi")
else:
    best_model = 'Agglomerative'
    alasan = (f"memiliki Silhouette Score lebih tinggi ({sil_agg:.4f} vs {sil_kmeans:.4f}), "
              f"meskipun tidak memiliki inertia")

kesimpulan = f"Kesimpulan: Model terbaik adalah {best_model} karena {alasan}."
print(f"\n{kesimpulan}")
print(f"9. Model terbaik: {best_model}")

# ==========================================
# LANGKAH 10: SIMPAN HASIL EVALUASI KE LAPORAN.TXT
# ==========================================

fitur_digunakan = fitur_numerik
distribusi_kmeans = {}
for c in sorted(df['Cluster_KMeans'].unique()):
    label = label_map_kmeans[c]
    provinsi_list = df[df['Cluster_KMeans'] == c]['Provinsi'].tolist()
    distribusi_kmeans[c] = (label, provinsi_list)

distribusi_agg = {}
for c in sorted(df['Cluster_Agglomerative'].unique()):
    label = label_map_agg[c]
    provinsi_list = df[df['Cluster_Agglomerative'] == c]['Provinsi'].tolist()
    distribusi_agg[c] = (label, provinsi_list)

with open(out('laporan.txt'), 'w', encoding='utf-8') as f:
    f.write("================================================\n")
    f.write("LAPORAN HASIL CLUSTERING\n")
    f.write("Dataset: Pendidikan SD Indonesia 2023-2024\n")
    f.write(f"Tanggal: {datetime.now().strftime('%d %B %Y %H:%M:%S')}\n")
    f.write("================================================\n\n")

    f.write("[INFORMASI DATASET]\n")
    f.write(f"- Total provinsi: {len(df)}\n")
    f.write(f"- Total fitur: {n_fitur}\n")
    f.write(f"- Fitur yang digunakan: {', '.join(fitur_digunakan)}\n")
    f.write("- Fitur turunan: rasio_putus_sekolah, rasio_guru_berkualifikasi, rasio_kelas_rusak\n\n")

    f.write("[PCA]\n")
    f.write(f"- PC1 explained variance: {pc1_var:.2f}%\n")
    f.write(f"- PC2 explained variance: {pc2_var:.2f}%\n")
    f.write(f"- Total variance explained: {total_var:.2f}%\n\n")

    f.write("[PARAMETER MODEL]\n")
    f.write(f"- Jumlah Cluster (k): {k_optimal}\n")
    f.write("- KMeans: init=k-means++, max_iter=300, n_init=10, random_state=42\n")
    f.write("- Agglomerative: linkage=ward\n\n")

    f.write("[HASIL EVALUASI]\n")
    f.write(f"- KMeans Silhouette Score    : {sil_kmeans:.4f}\n")
    f.write(f"- KMeans Davies-Bouldin Score: {db_kmeans:.4f}\n")
    f.write(f"- KMeans Inertia             : {inertia_kmeans:.2f}\n")
    f.write(f"- Agglomerative Silhouette   : {sil_agg:.4f}\n")
    f.write(f"- Agglomerative Davies-Bouldin: {db_agg:.4f}\n\n")

    f.write("[KESIMPULAN]\n")
    f.write(f"Model terbaik: {best_model}\n")
    f.write(f"Alasan: {best_model} {alasan}.\n\n")

    f.write("[DISTRIBUSI PROVINSI PER CLUSTER — KMEANS]\n")
    for c, (label, provinsi_list) in distribusi_kmeans.items():
        f.write(f"Cluster {c} ({label}): {', '.join(provinsi_list)}\n")

    f.write("\n[DISTRIBUSI PROVINSI PER CLUSTER — AGGLOMERATIVE]\n")
    for c, (label, provinsi_list) in distribusi_agg.items():
        f.write(f"Cluster {c} ({label}): {', '.join(provinsi_list)}\n")

    f.write("\n================================================\n")

print("10. Laporan berhasil disimpan ke laporan.txt")

# ==========================================
# LANGKAH 11: VISUALISASI
# ==========================================

colors_palette = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']

# 1. elbow_method.png sudah disimpan di Langkah 5

# 2. scatter_kmeans.png
fig, ax = plt.subplots(figsize=(14, 10))
unique_clusters = sorted(df['Cluster_KMeans'].unique())
for c in unique_clusters:
    mask = kmeans_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors_palette[c % len(colors_palette)],
               label=f"Cluster {c} ({label_map_kmeans[c]})",
               s=100, alpha=0.8, edgecolors='white', linewidth=0.5)

centroids_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           marker='*', s=400, c='black', zorder=5, label='Centroid')

for i, prov in enumerate(labels.values):
    ax.annotate(prov.replace('Prov. ', ''), (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6.5, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
ax.set_title('Hasil Clustering KMeans (PCA 2D)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
simpan_plot(out('scatter_kmeans.png'))

# 3. scatter_agglomerative.png
fig, ax = plt.subplots(figsize=(14, 10))
unique_clusters_agg = sorted(df['Cluster_Agglomerative'].unique())
for c in unique_clusters_agg:
    mask = agg_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors_palette[c % len(colors_palette)],
               label=f"Cluster {c} ({label_map_agg[c]})",
               s=100, alpha=0.8, edgecolors='white', linewidth=0.5)

for c in unique_clusters_agg:
    mask = agg_labels == c
    cx = X_pca[mask, 0].mean()
    cy = X_pca[mask, 1].mean()
    ax.scatter(cx, cy, marker='*', s=400, c='black', zorder=5)

ax.scatter([], [], marker='*', s=400, c='black', label='Centroid (rata-rata)')

for i, prov in enumerate(labels.values):
    ax.annotate(prov.replace('Prov. ', ''), (X_pca[i, 0], X_pca[i, 1]),
                fontsize=6.5, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points')

ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=11)
ax.set_title('Hasil Agglomerative Clustering (PCA 2D)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
simpan_plot(out('scatter_agglomerative.png'))

# 4. heatmap_kmeans.png
hm_km = df_with_features.groupby('Cluster_KMeans')[rasio_fitur].mean()
hm_km.index = [f"Cluster {i}\n({label_map_kmeans[i]})" for i in hm_km.index]
hm_km.columns = ['Rasio Putus\nSekolah (%)', 'Rasio Guru\nBerkualifikasi (%)', 'Rasio Kelas\nRusak (%)']

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(hm_km, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Nilai'})
ax.set_title('Heatmap Rata-rata Fitur Turunan per Cluster (KMeans)', fontsize=13, fontweight='bold')
ax.set_xlabel('Fitur', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
plt.tight_layout()
simpan_plot(out('heatmap_kmeans.png'))

# 5. heatmap_agglomerative.png
hm_agg = df_with_features.groupby('Cluster_Agglomerative')[rasio_fitur].mean()
hm_agg.index = [f"Cluster {i}\n({label_map_agg[i]})" for i in hm_agg.index]
hm_agg.columns = ['Rasio Putus\nSekolah (%)', 'Rasio Guru\nBerkualifikasi (%)', 'Rasio Kelas\nRusak (%)']

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(hm_agg, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Nilai'})
ax.set_title('Heatmap Rata-rata Fitur Turunan per Cluster (Agglomerative)', fontsize=13, fontweight='bold')
ax.set_xlabel('Fitur', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
plt.tight_layout()
simpan_plot(out('heatmap_agglomerative.png'))

# 6. bar_distribusi.png
km_counts = df['Cluster_KMeans'].value_counts().sort_index()
agg_counts = df['Cluster_Agglomerative'].value_counts().sort_index()

all_clusters = sorted(set(km_counts.index) | set(agg_counts.index))
km_vals = [km_counts.get(c, 0) for c in all_clusters]
agg_vals = [agg_counts.get(c, 0) for c in all_clusters]

x = np.arange(len(all_clusters))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, km_vals, width, label='KMeans',
               color='#3498db', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x + width/2, agg_vals, width, label='Agglomerative',
               color='#e74c3c', edgecolor='white', linewidth=0.5)

for bar in bars1:
    ax.annotate(f'{int(bar.get_height())}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.annotate(f'{int(bar.get_height())}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=10)

ax.set_xlabel('Nomor Cluster', fontsize=11)
ax.set_ylabel('Jumlah Provinsi', fontsize=11)
ax.set_title('Distribusi Provinsi per Cluster: KMeans vs Agglomerative', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Cluster {c}' for c in all_clusters])
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
simpan_plot(out('bar_distribusi.png'))

# 7. dendrogram.png
Z = linkage(X_scaled, method='ward')
fig, ax = plt.subplots(figsize=(18, 8))
dendrogram(Z, labels=labels.values, leaf_rotation=90, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]), ax=ax)
ax.set_title('Dendrogram Agglomerative Clustering (Ward Linkage)', fontsize=14, fontweight='bold')
ax.set_xlabel('Provinsi', fontsize=11)
ax.set_ylabel('Jarak (Distance)', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
simpan_plot(out('dendrogram.png'))

print("11. Semua visualisasi berhasil disimpan.")

# ==========================================
# SIMPAN HASIL CLUSTERING KE CSV
# ==========================================

hasil = df[['Provinsi', 'Cluster_KMeans', 'Label_KMeans',
            'Cluster_Agglomerative', 'Label_Agglomerative']].copy()
hasil.to_csv(out('hasil_clustering.csv'), index=False, encoding='utf-8-sig')

print("\n--- SELESAI ---")
print(f"File output:")
print("  - data_mining_sd.py")
print("  - hasil_clustering.csv")
print("  - laporan.txt")
print("  - elbow_method.png")
print("  - scatter_kmeans.png")
print("  - scatter_agglomerative.png")
print("  - heatmap_kmeans.png")
print("  - heatmap_agglomerative.png")
print("  - bar_distribusi.png")
print("  - dendrogram.png")
