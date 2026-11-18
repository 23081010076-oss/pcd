"""Final held-out evaluation after all 15 validation runs and frozen selection."""
import json
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from core import *

def main():
    c=load_config(); selection=json.loads(Path('results/selection.json').read_text())
    assert selection['config_sha256']==file_hash('config.json') and selection['manifest_sha256']==file_hash('data/manifest.csv')
    output=Path('results/final')
    if output.exists(): raise FileExistsError('Evaluasi final sudah dimulai/selesai. Periksa hasil yang tersedia.')
    # Verify all checkpoints before opening test images.
    for key,digest in selection['checkpoints'].items():
        scenario,seed=key.split('_'); assert file_hash(f'runs/main/{scenario}_seed{seed}/best.pt')==digest
    output.mkdir();(output/'frozen_selection.json').write_text(json.dumps(selection,indent=2))
    frame=pd.read_csv('data/manifest.csv');test=frame[frame.split=='test'];rows=[];perclass=[]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(4)
    for scenario in ['B0',selection['selected_clahe']]:
        for seed in c['seeds']:
            ck=torch.load(f'runs/main/{scenario}_seed{seed}/best.pt',map_location=device,weights_only=False)
            assert not ck['provenance']['pilot'] and ck['provenance']['weights']=='IMAGENET1K_V1'
            model=model_for(c,False).to(device);model.load_state_dict(ck['model'])
            loader=DataLoader(Knees(test,ck['provenance']['root'],c,scenario),batch_size=c['batch_size'],num_workers=0)
            result,ys,ps=evaluate(model,loader,nn.CrossEntropyLoss(),device)
            rows.append(dict(scenario=scenario,seed=seed,accuracy=result['accuracy'],macro_f1=result['macro_f1']))
            for k in range(5): perclass.append(dict(scenario=scenario,seed=seed,grade=k,**result['per_class'][str(k)]))
            pd.DataFrame(result['confusion_matrix'],index=range(5),columns=range(5)).to_csv(output/f'{scenario}_{seed}_confusion.csv',index_label='actual/predicted')
            pd.DataFrame({'path':test.path,'label':ys,'prediction':ps}).to_csv(output/f'{scenario}_{seed}_predictions.csv',index=False)
            (output/f'{scenario}_{seed}_metrics.json').write_text(json.dumps(result,indent=2))
    df=pd.DataFrame(rows);df.to_csv(output/'per_seed.csv',index=False)
    df.groupby('scenario')[['accuracy','macro_f1']].agg(['mean','std']).to_csv(output/'summary.csv')
    pd.DataFrame(perclass).to_csv(output/'per_class.csv',index=False)
    (output/'COMPLETE').write_text('Final evaluation complete')
if __name__=='__main__':main()
