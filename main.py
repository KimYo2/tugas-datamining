import os
import time
from modules.preprocessing import (muat_data, bersihkan_data, feature_engineering,
                                   pilih_fitur, scaling_data, reduksi_pca)
from modules.clustering import (cari_k_optimal, beri_label_cluster, evaluasi_clustering,
                                clustering_kmeans, clustering_agglomerative)
from modules.visualisasi import buat_semua_visualisasi
from modules.laporan import buat_laporan
from modules.utils import buat_folder_output
def main():
    start_time = time.perf_counter()
    
    # Konfigurasi
    FILE_CSV = 'kelayakan-pendidikan-indonesia.csv'
    OUTPUT_DIR = 'hasil datamining'
    
    # Buat folder output jika belum ada
    buat_folder_output(OUTPUT_DIR)
    
    # 1. Preprocessing
    print("=== MEMULAI PREPROCESSING DATA ===")
    df = muat_data(FILE_CSV)
    df = bersihkan_data(df)
    df = feature_engineering(df)
    X = pilih_fitur(df)
    X_scaled, scaler = scaling_data(X)
    X_pca, pca = reduksi_pca(X_scaled)
    
    # 2. Cari k optimal
    print("\n=== MENCARI K OPTIMAL ===")
    k_range = range(2, 11)
    inertia_list = []
    for k in k_range:
        kmeans = clustering_kmeans(X_scaled, k)[1]
        inertia_list.append(kmeans.inertia_)
    k_optimal = cari_k_optimal(k_range, inertia_list, OUTPUT_DIR)
    print(f"k optimal yang dipilih: {k_optimal}")
    
    # 3. Clustering KMeans
    print("\n=== CLUSTERING KMEANS ===")
    labels_kmeans, model_kmeans = clustering_kmeans(X_scaled, k_optimal)
    df['cluster'] = labels_kmeans
    df_kmeans, label_cluster_kmeans = beri_label_cluster(df.copy(), model_kmeans.cluster_centers_)
    eval_kmeans = evaluasi_clustering(X_pca, labels_kmeans, 'KMeans')
    print(f"KMeans - Silhouette: {eval_kmeans['Silhouette Score']:.3f}, Davies-Bouldin: {eval_kmeans['Davies-Bouldin Score']:.3f}")
    
    # 4. Clustering Agglomerative
    print("\n=== CLUSTERING AGGLOMERATIVE ===")
    labels_agg, model_agg = clustering_agglomerative(X_scaled, k_optimal)
    df['cluster'] = labels_agg
    df_agg, label_cluster_agg = beri_label_cluster(df.copy(), None)
    eval_agg = evaluasi_clustering(X_pca, labels_agg, 'Agglomerative')
    print(f"Agglomerative - Silhouette: {eval_agg['Silhouette Score']:.3f}, Davies-Bouldin: {eval_agg['Davies-Bouldin Score']:.3f}")
    
    # 5. Evaluasi dan Visualisasi
    print("\n=== MEMBUAT VISUALISASI ===")
    evaluasi = {
        'KMeans': eval_kmeans,
        'Agglomerative': eval_agg
    }
    buat_semua_visualisasi(X_pca, labels_kmeans, labels_agg, 
                           label_cluster_kmeans, label_cluster_agg, 
                           evaluasi, OUTPUT_DIR)
    
    # 6. Laporan
    print("\n=== MEMBUAT LAPORAN ===")
    buat_laporan(df_kmeans, df_agg, evaluasi, OUTPUT_DIR)
    
    end_time = time.perf_counter()
    print(f"\nSELESAI: Total waktu eksekusi: {end_time - start_time:.2f} detik")
if __name__ == "__main__":
    main()