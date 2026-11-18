"""Checks on real train/validation inputs; never evaluates held-out test data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import *

def main():
    c=load_config();df=pd.read_csv('data/manifest.csv');sample=df[df.split=='val'].iloc[:2]
    shape={}
    for s in c['scenarios']:
        ds=Knees(sample,'data/raw',c,s)
        x,y=ds[0];x2,_=ds[0]
        assert x.shape==(3,224,224) and torch.isfinite(x).all() and torch.equal(x,x2)
        shape[s]=list(x.shape)
        ds=Knees(sample,'data/raw',c,s,True)
        seed_all(42);a,_=ds[0];seed_all(42);b,_=ds[0];assert torch.equal(a,b)
    seed_all(42);m=model_for(c,False);head=m.classifier[1].weight.detach().clone()
    seed_all(42);m2=model_for(c,False);assert torch.equal(head,m2.classifier[1].weight)
    assert m.classifier[0].p==0.3 and m.classifier[1].out_features==5
    assert all(p.requires_grad for p in m.parameters())
    assert metrics(list(range(5)),list(range(5)))['macro_f1']==1.
    for col in ['group','pixel_hash']:
        assert df.groupby(col).split.nunique().max()==1
    known=df[df.patient_id.notna()];assert known.empty or known.groupby('patient_id').split.nunique().max()==1
    result={'passed':True,'scenario_tensor_shapes':shape,'validation_deterministic':True,
        'augmentation_reproducible':True,'classifier_initialization_reproducible':True,
        'all_parameters_trainable':True,'group_and_pixel_overlap':False,'test_evaluated':False}
    Path('reports/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
