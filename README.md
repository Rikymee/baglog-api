# Baglog Jamur Tiram Classifier API

Backend REST API berbasis **Flask** dan **TensorFlow** untuk melakukan klasifikasi penyakit pada baglog jamur tiram menggunakan model Convolutional Neural Network (CNN).

## 📌 Fitur API
- Endpoint `/predict` untuk memproses unggahan citra baglog.
- Mendukung 4 kelas identifikasi: `Mucor`, `Penicillium`, `Sehat`, dan `Trichoderma`.
- Mengembalikan hasil respon berformat JSON berupa nama kelas dan nilai akurasi/probabilitas (`%`).
- Terintegrasi dengan CORS agar dapat diakses dari frontend web Hostinger.

## 📁 Struktur Repositori
```text
.
├── Procfile                # Konfigurasi deployment Railway
├── README.md               # Dokumentasi proyek
├── app.py                  # Server Flask & logika inferensi model
├── model_baglog_cnn.h5     # Model CNN hasil pelatihan
└── requirements.txt        # Pustaka dependency Python
