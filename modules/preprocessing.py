import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
def muat_data(file_path):
    """Muat dataset CSV dan lakukan validasi keberadaan file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File CSV tidak ditemukan di: {file_path}. Pastikan path file benar.")
    return pd.read_csv(file_path)
def bersihkan_data(df):
    """Bersihkan data dari missing values dan duplikat."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df
def feature_engineering(df):
    """Lakukan feature engineering dengan guard division by zero."""
    # Hitung total guru dan kelas untuk denominator
    df['guru_total'] = df['Guru Bersertifikat'] + df['Guru Tidak Bersertifikat']
    df['kelas_total'] = df['Kelas Layak'] + df['Kelas Rusak']
    
    # Rasio putus sekolah (guard X['Siswa'] != 0)
    df['rasio_putus_sekolah'] = np.where(
        df['Siswa'] != 0,
        (df['Siswa Putus Sekolah'] / df['Siswa']) * 100,
        0
    )
    
    # Rasio guru berkualifikasi (guard guru_total != 0)
    df['rasio_guru_berkualifikasi'] = np.where(
        df['guru_total'] != 0,
        (df['Guru Bersertifikat'] / df['guru_total']) * 100,
        0
    )
    
    # Rasio kelas rusak (guard kelas_total != 0)
    df['rasio_kelas_rusak'] = np.where(
        df['kelas_total'] != 0,
        (df['Kelas Rusak'] / df['kelas_total']) * 100,
        0
    )
    
    # Hapus kolom temporary
    df = df.drop(['guru_total', 'kelas_total'], axis=1)
    return df
def pilih_fitur(df):
    """Pilih fitur yang relevan untuk clustering."""
    fitur = ['Siswa', 'Guru Bersertifikat', 'rasio_putus_sekolah', 
             'rasio_guru_berkualifikasi', 'rasio_kelas_rusak']
    return df[fitur]
def scaling_data(X):
    """Scaling fitur menggunakan StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
def reduksi_pca(X_scaled, n_components=2):
    """Reduksi dimensi menggunakan PCA dengan peringatan variansi."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    total_var = sum(pca.explained_variance_ratio_) * 100
    if total_var < 70.0:
        print(f"⚠️  WARNING: PCA hanya menjelaskan {total_var:.1f}% variansi. Hasil visualisasi 2D mungkin tidak representatif.")
    return X_pca, pca