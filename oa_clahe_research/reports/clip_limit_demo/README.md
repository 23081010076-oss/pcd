# Demonstrasi clip limit dan pilot singkat

Gambar comparison.png memakai satu citra dari setiap KL grade pada subset latih. Setiap baris berasal dari citra yang sama; kolom menunjukkan baseline, clip1, clip2, clip3, dan clip5. Semua gambar memakai rentang tampilan 0-255 yang sama agar perbandingan tidak dipengaruhi normalisasi otomatis tampilan. Gambar disimpan sebelum normalisasi ImageNet dan tanpa augmentasi untuk memperlihatkan pengaruh CLAHE saja.

Bandingkan detail dan perubahan tekstur dalam satu baris. Clip limit yang lebih tinggi mengizinkan pembatasan kontras yang lebih longgar, tetapi perubahan tiap citra dapat berbeda. Detail yang tampak lebih tegas belum membuktikan ketepatan diagnosis atau peningkatan kinerja CNN. image_statistics.csv mencatat perubahan intensitas; simpangan baku intensitas bukan metrik keberhasilan klasifikasi.

Seluruh pilot memakai seed42, ImageNet, satu epoch, 40 citra latih dan 20 validasi yang sama, batch4. Angka pada pilot_comparison.csv adalah hasil aktual pilot, bukan hasil eksperimen utama. Sampel sangat kecil, validasi seimbang (4 per kelas), dan satu seed belum mewakili distribusi validasi lengkap ataupun variasi tiga seed. Jangan memilih konfigurasi terbaik dari demonstrasi ini. Data uji tidak dievaluasi. Checkpoint semua pilot berhasil dimuat ulang.

Gunakan [notebook kode lengkap](../../penelitian_oa_clahe_full_code.ipynb) untuk menjalankan audit, demonstrasi, pilot, pelatihan utama, dan evaluasi. Laporan ini merekam pelaksanaan sebelum alur dipindahkan ke notebook.
