# Hasil pelaksanaan

Proyek dibuat di D:/pcd/oa_clahe_research. Workspace awal kosong; kedua file sumber dibaca tanpa perubahan. Rancangan slide menjadi acuan; EfficientNet-B0 menggantikan alternatif ResNet50 pada Word. Ekstraksi teks dan checksum sumber tersedia di aset_penelitian/referensi/.

## Unduhan dan audit aktual

Arsip Kaggle berhasil diunduh: 213.819.666 byte. SHA-256 tercatat di aset_penelitian/dataset/download.json. Audit membaca 9.786 PNG grayscale 224x224: 0 rusak, 0 duplikat piksel identik, 0 konflik label untuk piksel identik, 0 citra tanpa label folder. Konflik label klinis tidak dapat dinilai hanya dari nama folder.

Folder utama train/val/test berjumlah 8.260 citra. Folder auto_test berisi 1.526 citra tambahan, dikecualikan karena memiliki 811 kandidat ID pasien yang juga muncul di data utama; tidak diasumsikan independen. Tidak ada berkas asli yang dihapus.

Distribusi utama KL0=3.253, KL1=1.495, KL2=2.175, KL3=1.086, KL4=251. Kelas tidak seimbang; weighted cross-entropy memakai jumlah kelas latih saja.

Pembagian tetap setelah grouping ID: latih 5.782, validasi 826, uji 1.652 (70/10/20 persen tepat pada jumlah citra). Ada 4.130 prefix ID unik pada 8.260 citra utama. Prefix tujuh digit adalah heuristik nama berkas, belum identitas pasien yang dikonfirmasi. Pemeriksaan tidak menemukan overlap ID tersedia atau piksel identik antar subset. Near-duplicate belum diuji; validitas diagnosis label memerlukan anotasi/ahli.

## Pilot aktual

B0 seed42, satu epoch, 40 citra latih dan 20 validasi, batch4, EfficientNet-B0 bobot resmi IMAGENET1K_V1, CPU. Seluruh parameter dilatih; bukan data sintetis atau bobot acak.

- Training loss: 1.61398396
- Validation weighted loss: 1.63290145
- Validation macro-F1: 0.12272727
- Validation accuracy: 0.1500
- Durasi bagian pelatihan/validasi dan checkpoint: 3.54 detik (tidak mencakup audit/instalasi/unduhan).

Angka ini hanya bukti pilot sangat kecil; tidak digunakan untuk menyimpulkan manfaat CLAHE, menyeleksi konfigurasi, atau mengisi tabel eksperimen utama. Checkpoint reload dan evaluasi ulang memberi macro-F1 yang sama. Data uji belum dievaluasi.

## Verifikasi aktual

Kelima skenario menghasilkan tensor 3x224x224 finite; validasi deterministik; augmentasi berulang dengan seed sama konsisten; inisialisasi classifier konsisten; dropout0,3, lima output, semua parameter trainable. Pemeriksaan metrik dan overlap subset lulus. CLI dan kompilasi kode lulus. Lima belas perintah dicetak dan tersimpan dalam experiment_plan.json, tanpa menjalankan eksperimen utama. Seleksi ditolak saat 15 run belum lengkap; evaluasi final ditolak saat belum ada seleksi beku.

## Pekerjaan lanjutan dan keterbatasan

Tidak ada hambatan dataset/bobot untuk pilot. Eksperimen lengkap belum dijalankan sesuai lingkup permintaan, sehingga belum ada rerata/SD tiga seed, CLAHE terpilih, atau metrik uji. GPU CUDA tidak tersedia; 15 pelatihan sampai maksimum150 epoch akan lebih berat daripada pilot. Identitas pasien resmi dan near-duplicate masih belum terverifikasi. Hasil kelak terbatas pada dataset/protokol ini.

Gunakan [notebook kode lengkap](../../penelitian_oa_clahe_full_code.ipynb) untuk menjalankan audit, demonstrasi, pilot, pelatihan utama, dan evaluasi. Laporan ini merekam pelaksanaan sebelum alur dipindahkan ke notebook.
