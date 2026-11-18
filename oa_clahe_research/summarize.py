"""Validation-only selection. Refuse incomplete/mixed main experiments."""
import json
from pathlib import Path
import pandas as pd
from core import load_config,file_hash

def main():
    c=load_config();rows=[];missing=[]
    for scenario in c['scenarios']:
        for seed in c['seeds']:
            p=Path(f'runs/main/{scenario}_seed{seed}/best_metrics.json')
            if not p.exists() or not (p.parent/'verification.json').exists(): missing.append(str(p));continue
            d=json.loads(p.read_text())
            assert not d['pilot'] and d['weights']=='IMAGENET1K_V1'
            assert d['scenario']==scenario and d['seed']==seed
            assert all(file_hash(name)==digest for name,digest in d['source_sha256'].items())
            assert d['config_sha256']==file_hash('config.json') and d['manifest_sha256']==file_hash('data/manifest.csv')
            assert all(file_hash(name)==digest for name,digest in d['source_sha256'].items())
            rows.append({'scenario':scenario,'seed':seed,'epoch':d['epoch'],'val_macro_f1':d['val_macro_f1'] if 'val_macro_f1' in d else d['macro_f1'], 'val_loss':d['val_loss'],'val_accuracy':d['val_accuracy']})
    out=Path('results');out.mkdir(exist_ok=True)
    pd.DataFrame(rows,columns=['scenario','seed','epoch','val_macro_f1','val_loss','val_accuracy']).to_csv(out/'per_seed.csv',index=False)
    if missing: raise SystemExit(f'Belum lengkap: {len(missing)} run. Seleksi tidak dilakukan.')
    df=pd.DataFrame(rows)
    summary=df.groupby('scenario').agg(n=('seed','count'),macro_f1_mean=('val_macro_f1','mean'),macro_f1_std=('val_macro_f1','std'),loss_mean=('val_loss','mean'),loss_std=('val_loss','std'),accuracy_mean=('val_accuracy','mean'),accuracy_std=('val_accuracy','std')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    winner=summary[summary.scenario!='B0'].sort_values(['macro_f1_mean','loss_mean','scenario'],ascending=[False,True,True]).iloc[0].scenario
    checkpoints={f'{s}_{seed}':file_hash(f'runs/main/{s}_seed{seed}/best.pt') for s in ['B0',winner] for seed in c['seeds']}
    frozen={'selected_clahe':winner,'selection_rule':'mean validation macro-F1 descending; mean weighted validation loss ascending; scenario code for exact ties', 'config_sha256':file_hash('config.json'),'manifest_sha256':file_hash('data/manifest.csv'),'checkpoints':checkpoints}
    (out/'selection.json').write_text(json.dumps(frozen,indent=2));print(json.dumps(frozen,indent=2))
if __name__=='__main__':main()
