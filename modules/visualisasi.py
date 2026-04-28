import os
import matplotlib.pyplot as plt
import numpy as np
def buat_chart_evaluasi(evaluasi, output_dir):
    """Buat bar chart horizontal perbandingan evaluasi model."""
    metrics = ['Silhouette Score', 'Davies-Bouldin Score']
    kmeans_vals = [evaluasi['KMeans']['Silhouette Score'], evaluasi['KMeans']['Davies-Bouldin Score']]
    agg_vals = [evaluasi['Agglomerative']['Silhouette Score'], evaluasi['Agglomerative']['Davies-Bouldin Score']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(metrics))
    bar_height = 0.35
    
    # Plot bars untuk KMeans dan Agglomerative
    bars1 = ax.barh(y_pos - bar_height/2, kmeans_vals, bar_height, label='KMeans', color='#1f77b4')
    bars2 = ax.barh(y_pos + bar_height/2, agg_vals, bar_height, label='Agglomerative', color='#ff7f0e')
    
    # Tambahkan label nilai di ujung bar
    ax.bar_label(bars1, fmt='%.3f', padding=3)
    ax.bar_label(bars2, fmt='%.3f', padding=3)
    
    # Konfigurasi plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Nilai Score')
    ax.set_title('Perbandingan Evaluasi Model Clustering')
    ax.legend()
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Keterangan higher/lower is better
    ax.text(0.02, 0.98, 'Silhouette: Higher is Better\nDavies-Bouldin: Lower is Better', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluasi_model.png'))
    plt.close()
def visualisasi_cluster_2d(X_pca, labels, label_cluster, output_dir, model_name):
    """Visualisasi cluster 2D hasil PCA."""
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title(f'Visualisasi Cluster 2D - {model_name}')
    
    # Legenda
    handles = []
    for label in np.unique(labels):
        cluster_label = label_cluster.get(label, f'Cluster {label}')
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(label/np.max(labels)), markersize=10, label=cluster_label))
    plt.legend(handles=handles)
    
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'cluster_2d_{model_name.lower()}.png'))
    plt.close()
def buat_semua_visualisasi(X_pca, labels_kmeans, labels_agg, label_cluster_kmeans, label_cluster_agg, evaluasi, output_dir):
    """Buat semua visualisasi yang diperlukan."""
    visualisasi_cluster_2d(X_pca, labels_kmeans, label_cluster_kmeans, output_dir, 'KMeans')
    visualisasi_cluster_2d(X_pca, labels_agg, label_cluster_agg, output_dir, 'Agglomerative')
    buat_chart_evaluasi(evaluasi, output_dir)