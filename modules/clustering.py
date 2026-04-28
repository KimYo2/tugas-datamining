import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
def cari_k_optimal(k_range, inertia_list, output_dir):
    """Cari k optimal menggunakan KneeLocator, fallback ke k=3 jika kneed tidak tersedia."""
    # Cek ketersediaan library kneed
    try:
        from kneed import KneeLocator
        kneed_available = True
    except ImportError:
        kneed_available = False
    
    if kneed_available:
        kn = KneeLocator(k_range, inertia_list, curve='convex', direction='decreasing')
        k_optimal = kn.knee
        if k_optimal is None:
            k_optimal = 3
    else:
        k_optimal = 3
        print("⚠️  Library 'kneed' tidak tersedia, menggunakan fallback k=3")
    
    # Generate elbow plot dengan titik elbow merah
    plt.figure(figsize=(10,6))
    plt.plot(k_range, inertia_list, marker='o')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method untuk Menentukan k Optimal')
    plt.axvline(x=k_optimal, color='red', linestyle='--', linewidth=2, label=f'Elbow Point (k={k_optimal})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'elbow_method.png'))
    plt.close()
    
    return k_optimal
def beri_label_cluster(df_clustered, cluster_centers):
    """Beri label cluster berdasarkan ranking dan override nilai absolut."""
    # Hitung skor cluster untuk ranking (semakin tinggi semakin baik)
    skor = (df_clustered.groupby('cluster')['rasio_guru_berkualifikasi'].mean() -
            df_clustered.groupby('cluster')['rasio_putus_sekolah'].mean() -
            df_clustered.groupby('cluster')['rasio_kelas_rusak'].mean())
    skor_sorted = skor.sort_values(ascending=False)
    
    # Label awal berdasarkan ranking
    label_awal = {}
    label_list = ["Provinsi Maju", "Provinsi Berkembang", "Provinsi Perlu Perhatian"]
    for i, cluster in enumerate(skor_sorted.index):
        label_awal[cluster] = label_list[i] if i < len(label_list) else "Provinsi Berkembang"
    
    # Override label berdasarkan nilai absolut
    label_final = {}
    for cluster in df_clustered['cluster'].unique():
        avg_rasio_guru = df_clustered[df_clustered['cluster']==cluster]['rasio_guru_berkualifikasi'].mean()
        avg_rasio_putus = df_clustered[df_clustered['cluster']==cluster]['rasio_putus_sekolah'].mean()
        
        # Override: putus sekolah >5% → paksa Perlu Perhatian
        if avg_rasio_putus > 5:
            label_final[cluster] = "Provinsi Perlu Perhatian"
        else:
            # Override: guru berkualifikasi <50% → turunkan dari Maju ke Berkembang
            if avg_rasio_guru < 50 and label_awal[cluster] == "Provinsi Maju":
                label_final[cluster] = "Provinsi Berkembang"
            else:
                label_final[cluster] = label_awal[cluster]
    
    # Tambahkan kolom label ke dataframe
    df_clustered['label_cluster'] = df_clustered['cluster'].map(label_final)
    return df_clustered, label_final
def evaluasi_clustering(X_pca, labels, model_name):
    """Evaluasi hasil clustering menggunakan Silhouette dan Davies-Bouldin."""
    sil = silhouette_score(X_pca, labels)
    db = davies_bouldin_score(X_pca, labels)
    return {
        'Model': model_name,
        'Silhouette Score': sil,
        'Davies-Bouldin Score': db
    }
def clustering_kmeans(X_scaled, k):
    """Lakukan KMeans clustering."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans
def clustering_agglomerative(X_scaled, k):
    """Lakukan Agglomerative Clustering."""
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)
    return labels, agg