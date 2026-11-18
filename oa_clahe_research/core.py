"""Shared preprocessing, EfficientNet-B0 and metrics. No test access in training."""
import hashlib, json, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import functional as TF, InterpolationMode
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def load_config(): return json.loads(Path('config.json').read_text())
def file_hash(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(load_config()['threads'])

class Knees(Dataset):
    def __init__(self, frame, root, cfg, scenario, train=False):
        self.frame=frame.reset_index(drop=True);self.root=Path(root);self.cfg=cfg
        self.clip=cfg['scenarios'][scenario];self.train=train
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]; c=self.cfg
        with Image.open(self.root/row.path) as im:
            im=im.convert('L').resize((c['image_size'],c['image_size']),Image.Resampling.BILINEAR)
        if self.clip is not None:
            im=Image.fromarray(cv2.createCLAHE(self.clip,tuple(c['tile_grid'])).apply(np.array(im)))
        if self.train:
            angle=float(torch.empty(1).uniform_(-c['rotation'],c['rotation']).item())
            im=TF.rotate(im,angle,interpolation=InterpolationMode.BILINEAR)
            if torch.rand(1).item()<c['horizontal_flip']: im=TF.hflip(im)
        x=TF.to_tensor(im).repeat(3,1,1)
        return TF.normalize(x,c['mean'],c['std']),int(row.label)

def model_for(cfg,pretrained=True):
    weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model=efficientnet_b0(weights=weights,dropout=cfg['dropout'])
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,5)
    return model

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0)),
            'per_class':classification_report(y,p,labels=list(range(5)),output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,p,labels=list(range(5))).tolist()}

@torch.no_grad()
def evaluate(model,loader,criterion,device):
    model.eval(); ys=[];ps=[];total=0.; denom=0.
    for x,y in loader:
        x,y=x.to(device),y.to(device); logits=model(x)
        # Exact weighted CE denominator across batches, independent of batch composition.
        total+=float(nn.functional.cross_entropy(logits,y,weight=criterion.weight,reduction='sum'))
        denom+=float(criterion.weight[y].sum()) if criterion.weight is not None else len(y)
        ys.extend(y.cpu().tolist());ps.extend(logits.argmax(1).cpu().tolist())
    result=metrics(ys,ps); result['loss']=total/denom
    return result,ys,ps
