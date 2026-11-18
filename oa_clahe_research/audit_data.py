"""Audit penuh dan pembagian tetap berdasarkan kelompok pasien/duplikat."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='data/raw')
    ap.add_argument('--patient-map', help='CSV path,patient_id relatif terhadap root; opsional')
    args = ap.parse_args(); root = Path(args.root).resolve()
    out = Path('data'); out.mkdir(exist_ok=True)
    cfg = json.loads(Path('config.json').read_text())
    mapping = pd.read_csv(args.patient_map).set_index('path').patient_id.to_dict() if args.patient_map else {}
    records, errors, ignored = [], [], []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}: continue
        labels = [int(x) for x in p.relative_to(root).parts[:-1] if x in {'0','1','2','3','4'}]
        if len(set(labels)) != 1:
            ignored.append(str(p)); continue
        try:
            with Image.open(p) as im: im.verify()
            with Image.open(p) as im:
                im.load(); w,h=im.size; mode=im.mode
                pixels = np.asarray(im.convert('L'))
                digest=hashlib.sha256(str((w,h)).encode()+pixels.tobytes()).hexdigest()
            rel=p.relative_to(root).as_posix()
            # OAI filenames start with seven-digit participant ID; heuristic, explicitly recorded.
            match=re.match(r'^(\d{7})(?:[LR]|[_\-])', p.stem)
            pid=str(mapping.get(rel, match.group(1) if match else ''))
            records.append(dict(path=rel,label=labels[0],width=w,height=h,mode=mode,
                sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),pixel_hash=digest,
                patient_id=pid,patient_source='mapping' if rel in mapping else 'filename_heuristic' if pid else 'unavailable',
                original_split=p.relative_to(root).parts[0]))
        except Exception as e: errors.append({'path':str(p),'error':str(e)})
    if not records: raise RuntimeError('Tidak ada citra berlabel terbaca')
    df=pd.DataFrame(records); df.to_csv(out/'inventory.csv',index=False)
    duplicate=df[df.duplicated('pixel_hash',keep=False)]; duplicate.to_csv(out/'duplicates.csv',index=False)
    conflicts=df.groupby('pixel_hash').label.nunique(); bad=set(conflicts[conflicts>1].index)
    # Quarantine every conflicting pixel group; retain one representative of exact duplicates.
    eligible=df[df.original_split.isin(['train','val','test'])].copy()
    if eligible.empty: eligible=df.copy()
    clean=eligible[~eligible.pixel_hash.isin(bad)].drop_duplicates('pixel_hash').copy().reset_index(drop=True)
    parent=list(range(len(clean)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen={}
    for i,r in clean.iterrows():
        for key in [('pixel',r.pixel_hash)]+([('patient',r.patient_id)] if r.patient_id else []):
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    clean['group']=[str(find(i)) for i in range(len(clean))]
    clean['split']='train'
    folds=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=cfg['split_seed'])
    for k,(_,idx) in enumerate(folds.split(clean,clean.label,clean.group)):
        clean.loc[idx,'split']='test' if k<2 else 'val' if k==2 else 'train'
    for col in ['group','pixel_hash']:
        assert clean.groupby(col).split.nunique().max()==1
    known=clean[clean.patient_id!='']; assert known.empty or known.groupby('patient_id').split.nunique().max()==1
    assert all(set(clean[clean.split==s].label)==set(range(5)) for s in ['train','val','test'])
    clean.to_csv(out/'manifest.csv',index=False)
    report={'root':str(root),'readable':len(df),'corrupt':errors,'unlabeled':ignored,
      'class_counts':{str(k):int(v) for k,v in df.label.value_counts().sort_index().items()},
      'sizes':df.groupby(['width','height']).size().to_dict().__str__(),'modes':df['mode'].value_counts().to_dict(),
      'duplicate_extra_images':len(df)-df.pixel_hash.nunique(),'conflicting_hash_groups':len(bad),
      'excluded_conflict_images':int(df.pixel_hash.isin(bad).sum()),'retained':len(clean),
      'patient_id_sources':df.patient_source.value_counts().to_dict(),
      'patient_id_note':'ID 7 digit dari nama file adalah heuristik OAI, belum diverifikasi dengan metadata resmi. Semua ID sama dikelompokkan. Jika tidak tersedia, pemisahan pasien tidak dijamin.',
      'split_counts':{},'original_split_counts':df.original_split.value_counts().to_dict(),
      'split_method':'10 stratified group folds: 7 train, 1 val, 2 test; rasio mendekati 70/10/20; split_seed tetap',
      'leakage_checks':'Tidak ada overlap group, ID pasien yang tersedia, atau piksel identik antar subset. Near-duplicate belum diuji.'}
    available=clean[clean.patient_id!='']
    report.update(auxiliary_images_excluded=int((~df.original_split.isin(['train','val','test'])).sum()),primary_images=len(eligible),primary_class_counts=eligible.label.value_counts().sort_index().to_dict(),valid_images=len(df),corrupt_images=len(errors),ignored_images=len(ignored),
        duplicate_extra_images=len(df)-len(df.drop_duplicates('pixel_hash')),
        conflicting_pixel_groups=len(bad),retained_images=len(clean),
        patients_available=int((clean.patient_id!='').sum()),unique_patient_ids=int(available.patient_id.nunique()),
        dimensions=df.groupby(['width','height']).size().to_dict().__str__(),
        modes=df['mode'].value_counts().to_dict(),class_counts=df.label.value_counts().sort_index().to_dict(),
        patient_id_note='Seven digit prefix is a filename heuristic, not independently verified clinical metadata.',
        split_counts=clean.split.value_counts().to_dict(),
        split_class_counts=clean.groupby(['split','label']).size().unstack(fill_value=0).to_dict(orient='index'))
    pd.DataFrame(errors).to_csv(out/'corrupt.csv',index=False)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
