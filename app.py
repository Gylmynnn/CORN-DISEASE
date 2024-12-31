from flask import Flask, render_template, request
from keras._tf_keras.keras.models import load_model
import numpy as np
from PIL import Image


# Membuat aplikasi Flask
app = Flask(__name__)


# Muat model yang sudah dilatih
model = load_model('model/corn_disease_model.h5')
# Daftar kelas untuk prediksi
class_labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Fungsi untuk memproses gambar
def preprocess_image(img):
    img = img.resize((150, 150))  # Ubah ukuran gambar sesuai input model
    img_array = np.array(img)  # Ubah gambar ke array NumPy
    img_array = img_array / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array


# Rute untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk menangani unggahan gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')
    try:
        img = Image.open(file.stream)
        img_array = preprocess_image(img)
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Convert to percentage

        # Render the result back on the same page
        return render_template('index.html', prediction=predicted_class, confidence=confidence, imageData=img)

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
