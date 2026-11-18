import argparse, json, os, platform, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from core import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenario',choices=['B0','C1','C2','C3','C5'],default='B0')
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--root',default='data/raw');ap.add_argument('--random-weights',action='store_true',help='Hanya pilot; bukan hasil penelitian')
    a=ap.parse_args();c=load_config()
    if a.seed not in c['seeds']: raise ValueError('Seed di luar konfigurasi')
    if a.random_weights and not a.pilot: raise ValueError('Eksperimen utama wajib ImageNet')
    run=Path('runs')/('pilot' if a.pilot else 'main')/f'{a.scenario}_seed{a.seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'best.pt').exists(): raise FileExistsError(f'{run}: run sudah ada; simpan/ubah nama sebelum menjalankan ulang')
    frame=pd.read_csv('data/manifest.csv');train=frame[frame.split=='train'];val=frame[frame.split=='val']
    if a.pilot:
        train=train.groupby('label',group_keys=False).sample(n=c['pilot']['train_per_class'],random_state=a.seed)
        val=val.groupby('label',group_keys=False).sample(n=c['pilot']['val_per_class'],random_state=a.seed)
    counts=train.label.value_counts().reindex(range(5),fill_value=0)
    if (counts==0).any(): raise ValueError('Ada kelas latih kosong')
    weights=torch.tensor(len(train)/(5*counts.values),dtype=torch.float32)
    seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.hub.set_dir(str(Path('weights').resolve()))
    model=model_for(c,not a.random_weights).to(device)
    criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    gen=torch.Generator().manual_seed(a.seed)
    batch=c['pilot']['batch_size'] if a.pilot else c['batch_size']
    tl=DataLoader(Knees(train,a.root,c,a.scenario,True),batch_size=batch,shuffle=True,generator=gen,num_workers=0)
    vl=DataLoader(Knees(val,a.root,c,a.scenario),batch_size=batch,num_workers=0)
    provenance=dict(config=c,scenario=a.scenario,seed=a.seed,pilot=a.pilot,
        weights='random' if a.random_weights else 'IMAGENET1K_V1',manifest_sha256=file_hash('data/manifest.csv'),
        config_sha256=file_hash('config.json'),root=str(Path(a.root).resolve()),
        torch=torch.__version__,device=str(device),python=platform.python_version(),
        class_weights=weights.tolist(),train_samples=len(train),val_samples=len(val),
        source_sha256={name:file_hash(name) for name in ['core.py','train.py']})
    (run/'settings.json').write_text(json.dumps(provenance,indent=2))
    best=(-1.,float('-inf'));wait=0;history=[];start=time.time()
    for epoch in range(1,(c['pilot']['epochs'] if a.pilot else c['max_epochs'])+1):
        model.train();total=0.;denom=0.
        for x,y in tl:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
            logits=model(x);loss=criterion(logits,y);loss.backward();opt.step()
            total+=float(loss.detach())*float(criterion.weight[y].sum());denom+=float(criterion.weight[y].sum())
        result,_,_=evaluate(model,vl,criterion,device)
        row=dict(epoch=epoch,train_loss=total/denom,val_loss=result['loss'],val_macro_f1=result['macro_f1'],val_accuracy=result['accuracy'])
        history.append(row);pd.DataFrame(history).to_csv(run/'history.csv',index=False);print(row,flush=True)
        score=(result['macro_f1'],-result['loss'])
        # Patience counts epochs without strictly improved macro-F1; loss only resolves checkpoint ties.
        wait=0 if score[0]>best[0] else wait+1
        if score>best:
            best=score
            torch.save({'model':model.state_dict(),'provenance':provenance,'epoch':epoch,'validation':result},run/'best.pt')
            (run/'best_metrics.json').write_text(json.dumps(dict(row,**provenance),indent=2))
        if wait>=c['patience']: break
    ck=torch.load(run/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    again,_,_=evaluate(model,vl,criterion,device)
    assert abs(again['macro_f1']-ck['validation']['macro_f1'])<1e-12
    (run/'verification.json').write_text(json.dumps({'checkpoint_reload':True,'elapsed_seconds':time.time()-start,'test_evaluated':False},indent=2))

if __name__=='__main__':main()
