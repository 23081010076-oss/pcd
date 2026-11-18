# Progres pertama CLAHE pada rontgen lutut

Buka [notebook](penelitian_oa_clahe_full_code.ipynb), pilih kernel `.venv`, lalu jalankan **Run All** dari folder proyek.

1. Lihat jumlah PNG per grade KL dan contoh asli dari folder `train`. [Dataset Kaggle](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) ? [Arsip lokal](aset_penelitian/dataset/dataset.zip)
2. Bandingkan B0 tanpa CLAHE dengan clip limit 1, 2, 3, dan 5 pada grid tetap 8 x 8. [Panel hasil](aset_penelitian/hasil/progres_pertama.png)

Jika perlu menyiapkan lingkungan Python baru:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install ipykernel matplotlib opencv-python-headless
```

Jika folder data belum tersedia, ekstrak arsip ke `aset_penelitian/dataset/raw/`.

[Acuan Ahmed dan Mstafa (2022)](https://doi.org/10.3390/diagnostics12122939) ? [Dokumentasi CLAHE OpenCV](https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html)
