# Student Academic Success - MLflow Project

**Oleh:** Ahmad Ngiliyun (ahmadngiliyun00)

Proyek ini dibuat untuk memenuhi **Proyek Akhir: Membangun Sistem Machine Learning** di kursus Dicoding. Proyek ini mendemonstrasikan end-to-end Machine Learning pipeline tracking menggunakan MLflow.

## Deskripsi

Proyek ini bertujuan untuk memprediksi *Student Academic Success* (Keberhasilan Akademik Mahasiswa) menggunakan algoritma klasifikasi `Logistic Regression`.
Kinerja model dievaluasi dengan menggunakan metrik *Accuracy* dan *F1-Macro Score*, di mana setiap *run* model akan direkam oleh MLflow.

Sistem ini menyimpan artifak (*artifacts*) berikut di dalam *run* MLflow:

- Model klasifikasi (berbasis Scikit-Learn Logistic Regression)
- Confusion Matrix dalam bentuk gambar (`confusion_matrix.png`)
- Classification Report dalam bentuk file teks (`classification_report.txt`)

## Struktur Direktori

- `MLProject/`: Tempat MLflow Project didefinisikan secara deklaratif di `MLproject`, dan dependencies tersimpan di `conda.yaml`.
- `MLProject/modelling.py`: Script Python untuk memuat *preprocessed dataset*, melatih model, mengevaluasi *test accuracy*, dan melakukan pencatatan eksperimen via `mlflow`.
- `MLProject/namadataset_preprocessing/`: Memuat matriks Numpy (contohnya `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`) sebagai dataset siap-pakai yang dipakai oleh skrip `modelling.py`.
- `LICENSE`: Lisensi MIT file dengan informasi hak cipta.

## Cara Menjalankan (How to Run)

1. Pastikan Anda sudah menginstall Python, dan dependensi (seperti `numpy`, `scikit-learn`, `mlflow`, `matplotlib`).
2. Masuk ke direktori `MLProject`:

   ```bash
   cd MLProject
   ```

3. Eksekusi program menggunakan interpretator Python:

   ```bash
   python modelling.py
   ```

   Atau Anda dapat menjalankannya langsung melalui *MLflow cli command*:

   ```bash
   mlflow run .
   ```

4. Eksperimen log dapat Anda amati dengan mengaktifkan ui MLflow:

   ```bash
   mlflow ui
   ```

   Lalu buka [http://127.0.0.1:5000](http://127.0.0.1:5000) pada browser komputer Anda.

## Lisensi

Proyek ini terlisensi di bawah Lisensi MIT. Baca dokumen [LICENSE](LICENSE) terkait detail legal penggunaan, duplikasi, dan distribusi proyek ini.
