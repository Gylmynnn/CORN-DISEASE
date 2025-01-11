import os
import h5py
import cv2
import numpy as np
import tqdm

# Path ke dataset
letak_dataset = 'dataset/'
hasil_model_h5 = 'model/corn_leaf_disease_model.h5'

# Ukuran gambar yang akan diubah
ukuran_gambar = (150, 150)

# Daftar kelas (label) untuk dataset
daftar_kelas = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Array untuk menyimpan data gambar dan label
data_gambar = []
data_kelas = []

for kelas, nama_kelas in enumerate(daftar_kelas):
    # jalur ke folder kelas
    letak_kelas = os.path.join(letak_dataset, nama_kelas)


    if not os.path.exists(letak_kelas):
        print(f"Folder {letak_kelas} tidak ditemukan")
        continue

    for file_gambar in tqdm.tqdm(os.listdir(letak_kelas), desc=f"Memproses kelas {nama_kelas}"):
        # Path ke gambar
        letak_gambar = os.path.join(letak_kelas, file_gambar)
        # Baca gambar
        try :
            img_array = cv2.imread(letak_gambar)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_array = cv2.resize(img_array, ukuran_gambar)
            data_gambar.append(img_array)
            data_kelas.append(kelas)
        except Exception as e:
            print(f"Error membaca {letak_gambar}: {e}")

# Konversi ke numpy array
data_gambar = np.array(data_gambar, dtype=np.float32)
normalisasi_data_gambar = data_gambar / 255.0
data_kelas = np.array(data_kelas, dtype=int)

# Simpan ke file h5
with h5py.File(hasil_model_h5, 'w') as f:

    # Simpan data ke file h5
    f.create_dataset('gambar', data=normalisasi_data_gambar)
    f.create_dataset('kelas', data=data_kelas)

    # Cek isi file h5
    semua_gambar = f['gambar'][:]
    semua_kelas = f['kelas'][:]
    print(f"Shape images: {semua_gambar.shape}")
    print(f"Shape labels: {semua_kelas.shape}")

print(f"Dataset berhasil disimpan ke {hasil_model_h5}")
