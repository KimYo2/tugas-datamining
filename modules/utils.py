import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hasil datamining')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(nama_file):
    return os.path.join(OUTPUT_DIR, nama_file)

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
