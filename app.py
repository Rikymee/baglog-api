import os
import io
import gdown
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Mengizinkan akses CORS dari domain frontend Hostinger
CORS(app)

MODEL_PATH = 'model_baglog_cnn.h5'
GDRIVE_FILE_ID = '1_PYaxjs5w89yeOTEiqlfhIPtz1AuwkeG'

# Unduh model dari Google Drive jika belum ada di server Railway
if not os.path.exists(MODEL_PATH):
    print("Mengunduh model dari Google Drive...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    gdown.download(url, MODEL_PATH, quiet=False)

# Load Model Keras/TensorFlow
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

# Nama kelas sesuai urutan dataset saat training
CLASS_NAMES = ['Mucor', 'Penicillium', 'Sehat', 'Trichoderma']

def preprocess_image(image_bytes):
    # Buka gambar dan pastikan format RGB (3 channel)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize gambar ke ukuran 224x224 piksel sesuai input training CNN
    img = img.resize((224, 224))
    
    # Ubah menjadi array numpy
    img_array = np.array(img)
    
    # Tambahkan dimensi batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "success",
        "message": "API Klasifikasi Baglog Jamur Tiram Siap Digunakan!"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum siap atau gagal dimuat di server'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'File gambar tidak valid'}), 400

    try:
        image_bytes = file.read()
        processed_img = preprocess_image(image_bytes)
        
        # Prediksi menggunakan model CNN
        predictions = model.predict(processed_img)
        
        # Mengambil indeks dan probabilitas tertinggi
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(predictions[0]) * 100)
        
        return jsonify({
            'class': predicted_class,
            'probability': round(confidence, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500

if __name__ == '__main__':
    # Railway akan menyediakan port melalui environment variable PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
