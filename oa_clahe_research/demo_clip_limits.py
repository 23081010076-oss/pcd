"""Compare real training images and short CLAHE pilots, without accessing test images."""
import json
from pathlib import Path
import subprocess
import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from core import load_config, file_hash


def main():
    cfg=load_config()
    out=Path('reports/clip_limit_demo');out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv('data/manifest.csv')
    train=frame[frame.split=='train']
    samples=train.groupby('label',group_keys=False).sample(n=1,random_state=42).sort_values('label')
    samples[['path','label','split']].to_csv(out/'image_sources.csv',index=False)
    fig,axes=plt.subplots(5,5,figsize=(12,13))
    stats=[]
    for row,(_,record) in enumerate(samples.iterrows()):
        with Image.open(Path('data/raw')/record.path) as im:
            original=np.array(im.convert('L').resize((cfg['image_size'],cfg['image_size']),Image.Resampling.BILINEAR))
        for col,(name,clip) in enumerate(cfg['scenarios'].items()):
            pixels=original if clip is None else cv2.createCLAHE(clip,tuple(cfg['tile_grid'])).apply(original)
            assert pixels.shape==(224,224) and pixels.dtype==np.uint8
            Image.fromarray(pixels).save(out/f'KL{record.label}_{name}.png')
            ax=axes[row,col]
            ax.imshow(pixels,cmap='gray',vmin=0,vmax=255)
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title('B0: tanpa CLAHE' if clip is None else f'{name}: clip {clip:g}',fontsize=12)
            if col==0: ax.set_ylabel(f'KL {record.label}',fontsize=13)
            stats.append(dict(label=int(record.label),scenario=name,clip_limit=clip,path=record.path,
                mean_intensity=float(pixels.mean()),intensity_std=float(pixels.std()),
                mean_absolute_change=float(np.abs(pixels.astype(float)-original).mean())))
    fig.suptitle('Citra latih asli: B0 dan empat clip limit CLAHE\nGrid tetap 8x8; skala tampilan sama 0-255; tanpa augmentasi',fontsize=15)
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    pd.DataFrame(stats).to_csv(out/'image_statistics.csv',index=False)
    rows=[]
    for scenario in cfg['scenarios']:
        run=Path(f'runs/pilot/{scenario}_seed42')
        if not (run/'best.pt').exists():
            subprocess.run([sys.executable,'train.py','--scenario',scenario,'--seed','42','--pilot'],check=True)
        metadata=json.loads((run/'best_metrics.json').read_text())
        verification=json.loads((run/'verification.json').read_text())
        assert metadata['pilot'] and metadata['weights']=='IMAGENET1K_V1'
        assert metadata['config_sha256']==file_hash('config.json') and metadata['manifest_sha256']==file_hash('data/manifest.csv')
        assert all(file_hash(f)==h for f,h in metadata['source_sha256'].items())
        assert verification['checkpoint_reload'] and not verification['test_evaluated']
        rows.append({k:metadata[k] for k in ['scenario','seed','epoch','train_samples','val_samples','train_loss','val_loss','val_accuracy','val_macro_f1']})
    results=pd.DataFrame(rows);results.to_csv(out/'pilot_comparison.csv',index=False)
    text='''# Demonstrasi clip limit dan pilot singkat

Gambar comparison.png memakai satu citra dari setiap KL grade pada subset latih. Setiap baris berasal dari citra yang sama; kolom menunjukkan baseline, clip1, clip2, clip3, dan clip5. Semua gambar memakai rentang tampilan 0-255 yang sama agar perbandingan tidak dipengaruhi normalisasi otomatis tampilan. Gambar disimpan sebelum normalisasi ImageNet dan tanpa augmentasi untuk memperlihatkan pengaruh CLAHE saja.

Bandingkan detail dan perubahan tekstur dalam satu baris. Clip limit yang lebih tinggi mengizinkan pembatasan kontras yang lebih longgar, tetapi perubahan tiap citra dapat berbeda. Detail yang tampak lebih tegas belum membuktikan ketepatan diagnosis atau peningkatan kinerja CNN. image_statistics.csv mencatat perubahan intensitas; simpangan baku intensitas bukan metrik keberhasilan klasifikasi.

Seluruh pilot memakai seed42, ImageNet, satu epoch, 40 citra latih dan 20 validasi yang sama, batch4. Angka pada pilot_comparison.csv adalah hasil aktual pilot, bukan hasil eksperimen utama. Sampel sangat kecil, validasi seimbang (4 per kelas), dan satu seed belum mewakili distribusi validasi lengkap ataupun variasi tiga seed. Jangan memilih konfigurasi terbaik dari demonstrasi ini. Data uji tidak dievaluasi. Checkpoint semua pilot berhasil dimuat ulang.

Ulangi pembuatan demonstrasi dengan:

```powershell
.\\.venv\\Scripts\\python.exe demo_clip_limits.py
```

Pilot lengkap yang sudah ada dipakai kembali setelah hash konfigurasi/data/kode diperiksa; checkpoint tidak ditimpa. Untuk penelitian utama, jalankan experiments.py --execute, kemudian summarize.py dan evaluate_final.py sesuai README.
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    print(results.to_string(index=False))
    print('Gambar:',out/'comparison.png')

if __name__=='__main__':main()
