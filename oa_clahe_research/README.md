# Proyek OA lutut: pengaruh clip limit CLAHE

Folder proyek baru ini mengikuti slide proposal: EfficientNet-B0 ImageNet, seluruh parameter dilatih, dropout 0,3, lima keluaran KL. Word masih menyebut pilihan EfficientNet-B0/ResNet50; implementasi menetapkan EfficientNet-B0 sesuai slide dan permintaan pengguna. File asli tidak diubah. Teks sumber tersimpan di `references/`.

## Lingkungan dan instalasi (PowerShell)

Jalankan seluruh perintah dari folder ini. Python yang diverifikasi: 3.14, PyTorch 2.9.0 CPU dan torchvision 0.24.0. GPU CUDA tidak tersedia pada sesi ini. Lingkungan lokal `.venv` memakai paket sistem yang sudah ada; `environment-lock.txt` mencatat versi aktual. Untuk lingkungan baru yang terisolasi:

```powershell
cd D:\pcd\oa_clahe_research
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Untuk GPU, pasang pasangan torch/torchvision sesuai driver CUDA dan dokumentasi resmi PyTorch terlebih dahulu. Reproduksibilitas lintas versi/perangkat tidak dijamin identik secara bit.

## Data dan audit

Sumber: [Knee Osteoarthritis Dataset with Severity Grading, Shashwat Tiwari](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity). Unduhan publik Kaggle disimpan sebagai `data/dataset.zip`, ekstraksi di `data/raw`, URL dan checksum SHA-256 di `data/download.json`. Downloader memvalidasi ZIP serta jalur ekstraksi. Jika akses Kaggle berubah menjadi wajib autentikasi, unduh arsip dari halaman sumber dan letakkan di lokasi yang sama. Ikuti ketentuan penggunaan pada sumber untuk pemakaian/redistribusi.

```powershell
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe audit_data.py
.\.venv\Scripts\python.exe verify.py
```

Audit membaca seluruh citra untuk integritas data, termasuk calon subset uji, tetapi tidak melakukan evaluasi model pada uji. Audit menghasilkan inventaris ukuran/mode/label/hash, daftar duplikat dan berkas rusak, serta manifest pembagian tetap. Duplikasi diperiksa dengan hash piksel grayscale sebelum resize; salinan identik disisakan satu, semua citra dengan piksel sama tetapi label berbeda dikarantina dari manifest. Berkas asli tetap utuh. Near-duplicate dan kebenaran diagnosis label belum diverifikasi.

Arsip memiliki folder bawaan train/val/test/auto_test. Folder auto_test dikecualikan karena bersifat tambahan dan memiliki ID yang juga muncul pada data utama. Penelitian memakai 8.260 citra train/val/test lalu membangun ulang pembagian dari citra unik. Sepuluh stratified group folds dialokasikan 7/1/2 untuk mendekati 70/10/20; rasio aktual dapat berbeda karena kelompok pasien harus utuh. Split seed 42 tetap untuk semua skenario dan seed pelatihan.

Prefix tujuh digit pada nama file diperlakukan sebagai kandidat ID pasien OAI. Ini heuristik konservatif, belum dikonfirmasi melalui metadata pasien resmi. Semua prefix yang sama dikelompokkan dalam satu subset; jika tersedia pemetaan terverifikasi, gunakan `audit_data.py --patient-map patients.csv` dengan kolom `path,patient_id` (path relatif terhadap data/raw). Jika tidak ada ID, pemisahan pasien tidak dijamin. Jangan mengubah manifest sesudah eksperimen dimulai; perubahan audit memerlukan pengulangan seluruh run.

## Preprocessing dan pelatihan

Urutan: grayscale -> resize bilinear 224x224 -> baseline tanpa CLAHE atau CLAHE grid 8x8 -> augmentasi latih -> salin tiga kanal -> skala 0-1 dan normalisasi ImageNet. Augmentasi hanya rotasi uniform -5 sampai +5 derajat (bilinear, isi tepi hitam) dan flip horizontal p=0,5. Validasi/uji tanpa augmentasi. Softmax diperlukan bila menampilkan probabilitas; loss cross-entropy menerima logits agar stabil.

`config.json` menyimpan semua pengaturan eksperimen utama: B0 tanpa CLAHE, C1/C2/C3/C5 dengan clip limit 1/2/3/5; seed 42/123/2026; Adam lr 0,0001, weight decay 0,00001; batch 32; maksimum 150 epoch; patience 12 tanpa peningkatan macro-F1. Bobot kelas = N_latih/(5 x jumlah_latih_kelas), dihitung hanya dari data latih. Checkpoint dipilih berdasarkan macro-F1 validasi tertinggi, loss validasi lebih rendah memecahkan seri. Loss validasi memakai bobot kelas latih yang sama.

## Pilot baseline

```powershell
.\.venv\Scripts\python.exe train.py --scenario B0 --seed 42 --pilot
```

Pilot memakai citra asli, 8 per kelas latih (40 total), 4 per kelas validasi (20 total), batch 4, satu epoch, bobot ImageNet resmi. Ini menguji alur forward/backward, optimizer, validasi dan simpan/muat checkpoint, bukan estimasi kinerja penelitian. Hasil tersimpan di `runs/pilot/B0_seed42/`, terpisah dari eksperimen utama. Opsi `--random-weights` hanya untuk pilot jika bobot asli tidak tersedia, dan ditandai dalam metadata. Run yang sudah mempunyai checkpoint tidak ditimpa.

## Lima belas pelatihan utama

```powershell
# Hanya cetak/verifikasi 15 perintah dan simpan rencananya:
.\.venv\Scripts\python.exe experiments.py
# Jalankan semuanya berurutan (dapat lama di CPU):
.\.venv\Scripts\python.exe experiments.py --execute
# Contoh satu pelatihan:
.\.venv\Scripts\python.exe train.py --scenario C2 --seed 123
# Setelah 15 run selesai:
.\.venv\Scripts\python.exe summarize.py
```

Setiap run menyimpan `settings.json`, `history.csv`, `best.pt`, `best_metrics.json` dan `verification.json`. Semua skenario pada seed sama memakai RNG, inisialisasi classifier, urutan data dan pengaturan yang sama. `reports/experiment_plan.json` memuat 15 perintah. Rekap CSV tidak memuat angka perkiraan. `summarize.py` menolak seleksi jika belum lengkap atau konfigurasi/manifest berbeda. Rerata dan simpangan baku sampel (ddof=1) dihitung dari tiga seed. CLAHE terpilih ditentukan dari rerata macro-F1 validasi, lalu rerata loss; kode skenario memecahkan seri yang persis sama. Baseline tetap dilaporkan meskipun mengungguli semua CLAHE. Seleksi tidak menyatakan bahwa CLAHE pasti lebih baik.

## Evaluasi akhir

```powershell
.\.venv\Scripts\python.exe evaluate_final.py
```

Hanya setelah seleksi selesai, evaluasi keenam checkpoint (baseline dan CLAHE terpilih x 3 seed) pada subset uji yang sama dalam satu tahap final. Seleksi dan hash checkpoint dibekukan sebelum citra uji dibuka. Keluaran di `results/final/`: accuracy/macro-F1 per seed dan rerata/SD, precision/recall/F1/support setiap kelas, confusion matrix aktual sebagai baris/prediksi sebagai kolom, dan prediksi per citra. Skrip menolak menimpa folder final; jika terputus, periksa artefak sebelum memindahkannya dan mengulang. Jangan memakai hasil uji untuk memilih konfigurasi atau epoch.

Format CSV kosong tersedia di `templates/`. Angka pilot tidak dimasukkan ke tabel eksperimen utama. Lihat `reports/STATUS.md` untuk hasil yang benar-benar diperoleh dan keterbatasannya.
