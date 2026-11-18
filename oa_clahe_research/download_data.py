"""Unduh arsip publik Kaggle; tidak membutuhkan kredensial jika akses publik tersedia."""
import hashlib
import json
from pathlib import Path
import zipfile
import requests

URL = 'https://www.kaggle.com/api/v1/datasets/download/shashwatwork/knee-osteoarthritis-dataset-with-severity'

def main():
    data = Path('data'); data.mkdir(exist_ok=True)
    archive = data / 'dataset.zip'
    if not archive.exists():
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with archive.with_suffix('.part').open('wb') as f:
                for chunk in r.iter_content(1024 * 1024): f.write(chunk)
        archive.with_suffix('.part').replace(archive)
    with zipfile.ZipFile(archive) as z:
        target = (data / 'raw').resolve()
        for entry in z.infolist():
            if not (target / entry.filename).resolve().is_relative_to(target):
                raise ValueError('Unsafe archive path')
        z.extractall(target)
    metadata = {'url': URL, 'bytes': archive.stat().st_size,
                'sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()}
    (data / 'download.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__': main()
