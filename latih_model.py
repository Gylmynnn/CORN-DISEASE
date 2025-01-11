import h5py
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Path ke file h5
letak_file_h5 = 'model/corn_leaf_disease_model.h5'


# Baca file h5
# x = gambar, y = kelas
with h5py.File(letak_file_h5, 'r') as f:
    x = f['gambar'][:]
    y = f['kelas'][:]

#encode kelas
nomer_kelas = len(set(y))
y = to_categorical(y, nomer_kelas)

# Bagi data menjadi data latih dan data uji
x_latih, x_uji, y_latih, y_uji = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Shape x_latih: {x_latih.shape}, y_latih: {y_latih.shape}")
print(f"Shape x_uji: {x_uji.shape}, y_uji: {y_uji.shape}")


#---------- Membuat model CNN ----------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(nomer_kelas, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


#---------- Melatih model ----------
from tensorflow.keras.callbacks import EarlyStopping

# Inisialisasi EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Latih model
model.fit(x_latih, y_latih, validation_data=(x_uji, y_uji), epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluasi model
loss, accuracy = model.evaluate(x_uji, y_uji)
print(f"Loss: {loss}")
print(f"Akurasi: {accuracy}")

# Simpan model
model.save('model/cnn/corn_leaf_disease_model.h5')
