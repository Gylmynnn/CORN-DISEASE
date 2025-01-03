from flask import Flask, render_template, request, url_for
# from keras._tf_keras.keras.models import load_model
# from matplotlib import pyplot as plt
# from tensorflow.python.keras.models import load_model
# from tensorflow.keras.models import load_model, save_model
from tensorflow.python.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Membuat aplikasi Flask
app = Flask(__name__)


# Atur folder untuk file statis
app.config['UPLOAD_FOLDER'] = 'static/images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Muat model yang sudah dilatih
model = load_model('model/corn_disease_model.h5')

# Daftar kelas untuk prediksi
class_labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Fungsi untuk memproses gambar
def preprocess_image(img, save_intermediate=False, filename=""):
    # Resize gambar
    img_resized = img.resize((150, 150))
    if save_intermediate:
        resized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_resized.jpg")
        img_resized.save(resized_path)

    # Normalisasi gambar
    img_array = np.array(img_resized) / 255.0
    if save_intermediate:
        normalized_image = Image.fromarray((img_array * 255).astype('uint8'))
        normalized_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_normalized.jpg")
        normalized_image.save(normalized_path)

    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)

    # Return gambar yang diproses dan jalur file
    return img_array


# Rute untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Rute untuk menangani unggahan gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Bukan bertipe file')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='Tidak ada gambar yang dipilih')

    try:
        img = Image.open(file.stream)

        # Proses gambar dan simpan hasil tiap tahap
        filename = os.path.splitext(file.filename)[0]
        img_array = preprocess_image(img, save_intermediate=True, filename=filename)

        # Prediksi menggunakan model
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Convert to percentage


        # # Buat diagram batang
        # plt.figure(figsize=(8, 6))
        # plt.bar(class_labels, predictions * 100, color='skyblue')
        # plt.title('Hasil Prediksi')
        # plt.xlabel('Kelas')
        # plt.ylabel('Probabilitas (%)')
        # plt.xticks(rotation=45)

        # # Simpan diagram ke dalam folder statis
        # chart_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_chart.png")
        # plt.savefig(chart_path)
        # plt.close()

        # # URL untuk diagram
        # chart_url = url_for('static', filename=f'images/{filename}_chart.png')

        # Kirim hasil prediksi dan URL gambar ke template
        resized_url = url_for('static', filename=f'images/{filename}_resized.jpg')
        normalized_url = url_for('static', filename=f'images/{filename}_normalized.jpg')

        return render_template(
            'index.html',
            prediction=predicted_class,
            confidence=confidence,
            resized_url=resized_url,
            normalized_url=normalized_url,
            # chart_url=chart_url
        )

    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')


# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
