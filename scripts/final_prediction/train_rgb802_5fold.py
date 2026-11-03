#!/usr/bin/env python3
"""Paper-aligned ResTrans21 40-band input; group-safe checkpoint selection + full-fold refit.

Data layout:
  PROJECT/data/matched_sections_40band.h5
  PROJECT/data/matched_sections_manifest.csv
  PROJECT/data/groundtruth_sections_n.csv

HDF5 datasets: genuine_mjolnir [N,15,100,40] required;
sony_bgr_only and sony_bgr_plus_802 optional.
Run: python run_restrans21_historical_40band.py --data-dir ./data --output-dir ./results

Historical section-level fertilizer-stratified StratifiedKFold (five folds, shuffled,
random_state=3; first fold only). Validation selects target-specific checkpoints
using historical reversed R2; standard R2 is also reported. No independent test.

This adaptation trains and selects checkpoints on its fixed reconstructed HSI source.
Input normalization uses the maximum from the selected representation's training
sections; the genuine-HSI historical baseline used a full-dataset input maximum.
Target normalization retains the historical full-dataset min-max procedure.
Dropout remains 0.0. Model architecture and original torch.bmm attention are unchanged,
including the 13 defined but never called BatchNorm modules.
"""
import argparse
import json
import random
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from copy import deepcopy
from vit_pytorch import SimpleViT
import torch.nn.functional as F

TARGET_COLUMNS = ["dm_gpermsq", "n-content_perc", "n-uptake_gpermsq"]
TARGET_NAMES = ["DM", "NC", "NU"]
SOURCES = {"genuine": "genuine_mjolnir", "rgb_only": "sony_bgr_only", "rgb_plus_802": "sony_bgr_plus_802"}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_dir):
    h5_path=data_dir/'matched_sections_40band.h5'
    manifest_path=data_dir/'matched_sections_manifest.csv'
    labels_path=data_dir/'groundtruth_sections_n.csv'
    if not labels_path.is_file():
        labels_path=Path('/trinity/home/jizh/RGB2msi/groundtruth/groundtruth_sections_n.csv')
    for p in (h5_path,manifest_path,labels_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")
    with h5py.File(h5_path,'r') as h5:
        if SOURCES['genuine'] not in h5:
            raise KeyError(f"Missing required HDF5 dataset: {SOURCES['genuine']}; found {list(h5.keys())}")
        xs = {}
        for name, key in SOURCES.items():
            if key in h5:
                xs[name] = np.asarray(h5[key], dtype=np.float32)
                print(f"Using source: {name} (HDF5 key: {key})", flush=True)
            else:
                print(f"Optional source unavailable: {name} (HDF5 key: {key}); skipping", flush=True)
    manifest=pd.read_csv(manifest_path)
    labels=pd.read_csv(labels_path)
    missing=[c for c in TARGET_COLUMNS if c not in labels]
    if missing: raise ValueError(f"Ground-truth columns missing: {missing}")
    expected_cols=['label_id_1','label_id_2','label_id_3','parent_plot']
    missing=[c for c in expected_cols if c not in manifest]
    if missing: raise ValueError(f"Manifest columns missing: {missing}")
    ids=list(labels.columns[:3]) # same mapping as original 40-band training code
    lookup={}
    for _,row in labels.iterrows():
        key=tuple(int(float(row[c])) for c in ids)
        if key in lookup: raise ValueError(f"Duplicate ground-truth ID: {key}")
        lookup[key]=np.asarray([row[c] for c in TARGET_COLUMNS],dtype=np.float32)
    keep=[]; yy=[]
    for _,row in manifest.iterrows():
        key=tuple(int(row[c]) for c in ['label_id_1','label_id_2','label_id_3'])
        val=lookup.get(key)
        ok=val is not None and np.all(np.isfinite(val))
        keep.append(ok)
        if ok: yy.append(val)
    if not yy: raise ValueError('No matching, finite trait labels found')
    keep=np.asarray(keep,dtype=bool)
    if len(keep)!=len(xs['genuine']):
        raise ValueError(f"Manifest/H5 length mismatch: {len(keep)} vs {len(xs['genuine'])}")
    xs={k:v[keep] for k,v in xs.items()}
    manifest=manifest.loc[keep].reset_index(drop=True)
    y=np.stack(yy).astype(np.float32)
    for name,x in xs.items():
        if x.shape!=(len(y),15,100,40):
            raise ValueError(f"{name}: expected {(len(y),15,100,40)}, got {x.shape}")
        if not np.isfinite(x).all(): raise ValueError(f"Nonfinite data in {name}")
    groups=manifest['parent_plot'].astype(str).to_numpy()
    if len(set(groups)) < 5: raise ValueError('Need at least 5 parent plots')
    print(f"Loaded {len(y)} sections and {len(set(groups))} parent plots; 40 bands; available sources={list(xs)}")
    return xs,y,groups,manifest

class InputDataset(Dataset):
    def __init__(self, x,y,indices,x_mean,x_std,y_min,y_range):
        self.x=x;self.y=y;self.indices=np.asarray(indices,dtype=np.int64)
        self.x_mean=x_mean;self.x_std=x_std;self.y_min=y_min;self.y_range=y_range
    def __len__(self):return len(self.indices)
    def __getitem__(self,ix):
        i=int(self.indices[ix])
        xx=(self.x[i]-self.x_mean)/self.x_std
        xx=np.moveaxis(xx,-1,0).copy() # [40,15,100]
        yy=(self.y[i]-self.y_min)/self.y_range
        return torch.from_numpy(xx),torch.from_numpy(yy.astype(np.float32)),i

def get_loader(x,y,ix,xm,xs,ym,yr,batch,shuffle):
    return DataLoader(InputDataset(x,y,ix,xm,xs,ym,yr),batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=torch.cuda.is_available())

class SelfAttention(nn.Module):
    """
    Canonical implementation of multi-head self attention.
    """

    def __init__(self, emb, heads=1, mask=False, kqnorm=False):
        """
        :param emb:
        :param heads:
        :param mask:
        :param kqnorm:
        """
        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues  = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm
        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        # assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values  = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        # assert dot.size() == (b*h, t, t)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)
##
##
class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    # self.norm0 = nn.LayerNorm(k)

    self.attention1 = SelfAttention(k, heads=heads,kqnorm=True)
    self.norm1 = nn.LayerNorm(k)
    
    # self.attention2 = SelfAttention(k, heads=heads,kqnorm=True)
    # self.norm2 = nn.LayerNorm(k)
    # #
    # self.attention3 = SelfAttention(k, heads=heads,kqnorm=True)
    # self.norm3 = nn.LayerNorm(k)
    

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))
    self.norm_ff = nn.LayerNorm(k)

    # self.attention2 = SelfAttention(k, kqnorm=False)

    # self.norm2 = nn.LayerNorm(k)
    
    self.dropout = nn.Dropout(0.0)
    
  def forward(self, x):
    # x = self.norm0(x)
    #
    attended_1 = self.attention1(x)
    x = self.norm1(attended_1 + x)
    #
    # attended_2 = self.attention2(x)
    # x = self.norm2(attended_2 + x)
    # #
    #
    fedforward = self.ff(x)
    x = self.norm_ff(fedforward + x)
    x = self.dropout(x)
    return x


from vit_pytorch import SimpleViT

###########
# from vit_pytorch import ViT
# from SimpleViT_n_final import SimpleViT_n
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, channel):
        super(Net, self).__init__()
        ###
        self.vit = SimpleViT(
            image_size = (15, 100),
            patch_size = (3, 10),
            num_classes = 200*4,
            dim = 200*4,
            depth = 6,
            heads = 6,
            mlp_dim = 200*4,
            channels = channel
        )

        
        #
        self.conv9 = nn.Linear(100*8,200, bias = False)
        self.bn9 = nn.BatchNorm1d(200)
        self.relu9 = nn.ReLU()
        self.pool9 = nn.AvgPool1d(2)
        self.transformer9 = TransformerBlock(100,1) #12
        #
        self.conv10 = nn.Linear(100*9,200, bias = False)
        self.bn10 = nn.BatchNorm1d(200)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.AvgPool1d(2)
        self.transformer10 = TransformerBlock(100,1) #12
        #
        self.conv11 = nn.Linear(100*10,200, bias = False)
        self.bn11 = nn.BatchNorm1d(200)
        self.relu11 = nn.ReLU()
        self.pool11 = nn.AvgPool1d(2)
        self.transformer11 = TransformerBlock(100,1) #12

        self.conv12 = nn.Linear(100*11,200, bias = False)
        self.bn12 = nn.BatchNorm1d(200)
        self.relu12 = nn.ReLU()
        self.pool12 = nn.AvgPool1d(2)
        self.transformer12 = TransformerBlock(100,1) #12
        #
        self.conv13 = nn.Linear(100*12,200, bias = False)
        self.bn13 = nn.BatchNorm1d(200)
        self.relu13 = nn.ReLU()
        self.pool13 = nn.AvgPool1d(2)
        self.transformer13 = TransformerBlock(100,1) #12
        # #
        self.conv14 = nn.Linear(100*13,200, bias = False)
        self.bn14 = nn.BatchNorm1d(200)
        self.relu14 = nn.ReLU()
        self.pool14 = nn.AvgPool1d(2)
        self.transformer14 = TransformerBlock(100,1) #12
        #
        self.conv15 = nn.Linear(100*6,400, bias = False)
        self.bn15 = nn.BatchNorm1d(400)
        self.relu15 = nn.ReLU()
        self.pool15 = nn.AvgPool1d(2)
        self.transformer15 = TransformerBlock(200,1) #12
        
        self.dropout = nn.Dropout(0.0)
        self.dropout_f = nn.Dropout(0.0)
        self.dropout_dm = nn.Dropout(0.0)
        self.dropout_nc = nn.Dropout(0.0)
        self.dropout_nu = nn.Dropout(0.0)


        ##  heads for different variables !!!
        # self.conv21_0 = nn.Linear(200,10, bias = False)
        # self.conv21 = nn.Linear(200, 3, bias = False) # classes = 4
        ###
        # self.conv22_0 = nn.Linear(200,10, bias = False)
        self.conv22 = nn.Linear(100,1, bias = False)
        ###
        self.conv23_0 = nn.Linear(100*2,200, bias = False)
        self.bn23_0 = nn.BatchNorm1d(200)
        self.relu23_0 = nn.ReLU()
        self.pool23_0 = nn.AvgPool1d(2)
        self.transformer23_0 = TransformerBlock(100,1)
        #
        self.conv23_1 = nn.Linear(100*3,200, bias = False)
        self.bn23_1 = nn.BatchNorm1d(200)
        self.relu23_1 = nn.ReLU()
        self.pool23_1 = nn.AvgPool1d(2)
        self.transformer23_1 = TransformerBlock(100,1)
        #
        self.conv23_2 = nn.Linear(100*4,200, bias = False)
        self.bn23_2 = nn.BatchNorm1d(200)
        self.relu23_2 = nn.ReLU()
        self.pool23_2 = nn.AvgPool1d(2)
        self.transformer23_2 = TransformerBlock(100,1)
        #
        self.conv23_3 = nn.Linear(100*5,200, bias = False)
        self.bn23_3 = nn.BatchNorm1d(200)
        self.relu23_3 = nn.ReLU()
        self.pool23_3 = nn.AvgPool1d(2)
        self.transformer23_3 = TransformerBlock(100,1)
        #
        self.conv23_4 = nn.Linear(100*6,200, bias = False)
        self.bn23_4 = nn.BatchNorm1d(200)
        self.relu23_4 = nn.ReLU()
        self.pool23_4 = nn.AvgPool1d(2)
        self.transformer23_4 = TransformerBlock(100,1)
        #
        self.conv23_5 = nn.Linear(100*7,200, bias = False)
        self.bn23_5 = nn.BatchNorm1d(200)
        self.relu23_5 = nn.ReLU()
        self.pool23_5 = nn.AvgPool1d(2)
        self.transformer23_5 = TransformerBlock(100,1)
        #
        self.conv23 = nn.Linear(100, 1, bias = False) # soil type = 3
        #
        self.conv24 = nn.Linear(100, 1, bias = False) # soil type = 3
        #
        self.relu = nn.ReLU()
        #
    def forward(self, x):
        # print(x.size())
        layer_1 = self.vit(x)
        #
        layer_9 = self.relu9(self.conv9(layer_1))
        layer_9 = self.pool9(layer_9) # 100
        layer_9 = self.transformer9(layer_9.unsqueeze(1)).squeeze(1)
        layer_9 = self.dropout(layer_9)
        cat_9 = torch.cat((layer_1, layer_9), 1) # 400
        # cat_9 = self.pool9(cat_9) # 200
        #
        layer_10 = self.relu10(self.conv10(cat_9))
        layer_10 = self.pool10(layer_10) # 200
        layer_10 = self.transformer10(layer_10.unsqueeze(1)).squeeze(1)
        layer_10 = self.dropout(layer_10)
        #
        cat_10 = torch.cat((cat_9, layer_10), dim = 1)
        # cat_10 = self.pool10(cat_10)
        #
        layer_11 = self.relu11(self.conv11(cat_10))
        layer_11 = self.pool11(layer_11) # 200
        layer_11 = self.transformer11(layer_11.unsqueeze(1)).squeeze(1)
        layer_11 = self.dropout(layer_11)
        #
        cat_11 = torch.cat((cat_10, layer_11), dim = 1)
        # cat_11 = self.pool11(cat_11)
        #
        layer_12 = self.relu12(self.conv12(cat_11))
        layer_12 = self.pool12(layer_12) # 200
        layer_12 = self.transformer12(layer_12.unsqueeze(1)).squeeze(1)
        layer_12 = self.dropout(layer_12)
        #
        cat_12 = torch.cat((cat_11, layer_12), dim = 1)
        # cat_12 = self.pool12(cat_12)
        #
        layer_13 = self.relu13(self.conv13(cat_12))
        layer_13 = self.pool13(layer_13) # 200
        layer_13 = self.transformer13(layer_13.unsqueeze(1)).squeeze(1)
        layer_13 = self.dropout(layer_13)
        # #
        cat_13 = torch.cat((cat_12, layer_13), dim = 1)
        # #
        layer_14 = self.relu14(self.conv14(cat_13))
        layer_14 = self.pool14(layer_14) # 200
        layer_14 = self.transformer14(layer_14.unsqueeze(1)).squeeze(1)
        layer_14 = self.dropout(layer_14)
        
        # cat_14 = torch.cat((cat_13, layer_14), dim = 1)
        cat_1_x = torch.cat((layer_9, layer_10, layer_11, layer_12, layer_13, layer_14), dim = 1)
        
        layer_15 = self.relu15(self.conv15(cat_1_x))
        layer_15 = self.pool15(layer_15)
        layer_15 = self.transformer15(layer_15.unsqueeze(1)).squeeze(1)
        layer_15 = self.dropout(layer_15)
        #
        ### the rest are outputs
        # layer_21_0 = self.conv21_0(layer_15)
        # layer_21 = self.conv21(layer_11)
        # # layer_22_0 = self.conv22_0(layer_15)
        # layer_22 = self.conv22(layer_11)
        ##
        layer_23_0 = self.relu23_0(self.conv23_0(layer_15))
        layer_23_0 = self.dropout_f(layer_23_0)
        layer_23_0 = self.pool23_0(layer_23_0)
        layer_23_0 = self.transformer23_0(layer_23_0.unsqueeze(1)).squeeze(1)
        cat_23_0 = torch.cat((layer_15, layer_23_0), dim = 1)
        #
        layer_23_1 = self.relu23_1(self.conv23_1(cat_23_0))
        layer_23_1 = self.dropout_f(layer_23_1)
        layer_23_1 = self.pool23_1(layer_23_1)
        layer_23_1 = self.transformer23_1(layer_23_1.unsqueeze(1)).squeeze(1)
        cat_23_1 = torch.cat((cat_23_0, layer_23_1), dim = 1)
        #
        layer_23_2 = self.relu23_2(self.conv23_2(cat_23_1))
        layer_23_2 = self.dropout_f(layer_23_2)
        layer_23_2 = self.pool23_2(layer_23_2)
        layer_23_2 = self.transformer23_2(layer_23_2.unsqueeze(1)).squeeze(1)
        cat_23_2 = torch.cat((cat_23_1, layer_23_2), dim = 1)
        #
        layer_23_3 = self.relu23_3(self.conv23_3(cat_23_2))
        layer_23_3 = self.dropout_f(layer_23_3)
        layer_23_3 = self.pool23_3(layer_23_3)
        layer_23_3 = self.transformer23_3(layer_23_3.unsqueeze(1)).squeeze(1)
        cat_23_3 = torch.cat((cat_23_2, layer_23_3), dim = 1)
        #
        layer_23_4 = self.relu23_4(self.conv23_4(cat_23_3))
        layer_23_4 = self.dropout_f(layer_23_4)
        layer_23_4 = self.pool23_4(layer_23_4)
        layer_23_4 = self.transformer23_4(layer_23_4.unsqueeze(1)).squeeze(1)
        # cat_23_4 = torch.cat((cat_23_3, layer_23_4), dim = 1)
        #
        cat_23_x = torch.cat((layer_15, layer_23_0, layer_23_1, layer_23_2, layer_23_3, layer_23_4), dim = 1)
        layer_23_5 = self.relu23_5(self.conv23_5(cat_23_x))
        layer_23_5 = self.dropout_f(layer_23_5)
        layer_23_5 = self.pool23_5(layer_23_5)
        layer_23_5 = self.transformer23_5(layer_23_5.unsqueeze(1)).squeeze(1)

        ####     
        layer_22 = self.conv22(self.dropout_dm(layer_23_5))
        layer_23 = self.conv23(self.dropout_nc(layer_23_5))
        layer_24 = self.conv24(self.dropout_nu(layer_23_5))

        #
        return layer_22,layer_23,layer_24

###

def evaluate(model,loader,ymin,yrange,device):
    model.eval();preds=[];trues=[];indices=[]
    with torch.no_grad():
        for xb,yb,ib in loader:
            outputs=model(xb.to(device))
            pred=torch.cat(outputs,dim=1).cpu().numpy()
            preds.append(pred*yrange+ymin)
            trues.append(yb.numpy()*yrange+ymin)
            indices.append(ib.numpy())
    return np.concatenate(preds),np.concatenate(trues),np.concatenate(indices)

def metrics(y_true,y_pred):
    return {name:{'R2':float(r2_score(y_true[:,i],y_pred[:,i])),
                  'RMSE':float(np.sqrt(mean_squared_error(y_true[:,i],y_pred[:,i]))),
                  'MAE':float(mean_absolute_error(y_true[:,i],y_pred[:,i]))}
            for i,name in enumerate(TARGET_NAMES)}



def get_fertilizer(data_dir,manifest):
    path=data_dir/'groundtruth_sections_n.csv'
    if not path.exists(): path=Path('/trinity/home/jizh/RGB2msi/groundtruth/groundtruth_sections_n.csv')
    labels=pd.read_csv(path)
    if 'fertilizer' not in labels.columns: raise ValueError('Missing fertilizer column')
    ids=list(labels.columns[:3])
    lookup={}
    for _,row in labels.iterrows():
        key=tuple(int(float(row[c])) for c in ids)
        if np.isfinite(float(row['fertilizer'])):lookup[key]=int(float(row['fertilizer']))
    result=[]
    for _,row in manifest.iterrows():
        key=tuple(int(row[c]) for c in ('label_id_1','label_id_2','label_id_3'))
        if key not in lookup:raise ValueError(f'Missing fertilizer for {key}')
        result.append(lookup[key])
    return np.asarray(result,dtype=int)


# Adaptation source. Architecture above is IDENTICAL to the corrected original.
TRAINING_SOURCE='rgb_plus_802'
def run(args):
    seed_all(args.seed)
    data_dir=Path(args.data_dir).expanduser().resolve()
    out=Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    xs,Y,groups,manifest=load_data(data_dir)
    genuine=xs['genuine']
    adapted=xs[TRAINING_SOURCE]
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type!='cuda' and not args.cpu:
        raise RuntimeError('CUDA unavailable; submit this script as a GPU Slurm job')
    fold=1
    fertilizer=get_fertilizer(data_dir,manifest)
    outer_train,test=next(iter(StratifiedKFold(n_splits=5,shuffle=True,random_state=3).split(genuine,fertilizer)))
    audit=pd.DataFrame({'sample_index':np.arange(len(Y)),'parent_plot':groups,'fertilizer':fertilizer,
                        'partition':np.where(np.isin(np.arange(len(Y)),outer_train),'train','validation')})
    audit.to_csv(out/'split_audit.csv',index=False)
    overlap=set(groups[outer_train]) & set(groups[test])
    print('Historical section-level fertilizer split. Shared plots across train/validation:',len(overlap),flush=True)
    print(pd.crosstab(audit.partition,audit.fertilizer).to_string(),flush=True)
    seed_all(args.seed)
    # Historical full-dataset input scaling: use ALL labeled sections of selected reconstructed source.
    xm=np.zeros(adapted.shape[-1],dtype=np.float32)
    scalar=float(np.max(adapted))
    if not np.isfinite(scalar) or scalar<=0: raise RuntimeError('Invalid training image maximum')
    xstd=np.full(adapted.shape[-1],scalar,dtype=np.float32)
    ymin=Y.min(axis=0).astype(np.float32)
    yrange=np.maximum(Y.max(axis=0)-ymin,1e-8).astype(np.float32)
    model=Net(40).to(device)
    criterion=nn.MSELoss()
    train_loader=get_loader(adapted,Y,outer_train,xm,xstd,ymin,yrange,len(outer_train),False)
    eval_train_loader=get_loader(adapted,Y,outer_train,xm,xstd,ymin,yrange,args.eval_batch_size,False)
    test_loaders={source:get_loader(x,Y,test,xm,xstd,ymin,yrange,args.eval_batch_size,False) for source,x in xs.items()}
    config={'strategy':'historical script one-fold training and validation replication; validation guides checkpoints',
            'fold':1,'split':'StratifiedKFold fertilizer 5 folds shuffle=True random_state=3; first fold only','train_sections':len(outer_train),
            'test_sections':len(test),'max_epochs':args.max_epochs,'initial_lr':args.lr,
            'weight_decay':args.weight_decay,'optimizer':'Adam recreated every training epoch',
            'loss':'MSE(DM)+MSE(NC)+MSE(NU)','lr_patience':'accu_l>20 (21 non-improving epochs)',
            'lr_factor':0.98,'early_stop':False,'dropout':0.0,
            'monitor_every':args.eval_every,
            'historical_r2':'r2_score(y_pred_scaled,y_true_scaled) (reversed; selection metric)',
            'standard_r2':'r2_score(y_true_physical,y_pred_physical)',
            'checkpoint_metric':'historical reversed R2 on held-out VALIDATION partition',
            'image_scaling':'full-dataset maximum of selected reconstructed spectral source, matching historical genuine-HSI scaling method',
            'target_scaling':'full-dataset min-max, matching historical script',
            'training_source':TRAINING_SOURCE,'data_dir':str(data_dir)}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print('HISTORICAL REPRODUCTION: fold 1 ONLY. Training and validation, no separate test.',flush=True)
    print('GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    print('train/test sections',len(outer_train),len(test),'train/test parent plots',len(set(groups[outer_train])),len(set(groups[test])),flush=True)
    print('image scalar',scalar,'target minimum',ymin,'target range',yrange,flush=True)
    np.savez(out/'normalization.npz',xmean=xm,xstd=xstd,ymin=ymin,yrange=yrange,train_indices=outer_train,test_indices=test)
    best={name:-np.inf for name in TARGET_NAMES}
    best_epoch={name:None for name in TARGET_NAMES}
    history=[]
    lr_net=args.lr
    accu_l=0
    def evaluate_source(source,loader,epoch,phase):
        pred,truth,inds=evaluate(model,loader,ymin,yrange,device)
        scaled_pred=(pred-ymin)/yrange
        scaled_truth=(truth-ymin)/yrange
        records=[]
        for j,name in enumerate(TARGET_NAMES):
            # Historical, reversed argument order; faithful to original code.
            historical=float(r2_score(scaled_pred[:,j],scaled_truth[:,j]))
            standard=float(r2_score(truth[:,j],pred[:,j]))
            mse_scaled=float(np.mean((scaled_pred[:,j]-scaled_truth[:,j])**2))
            records.append({'epoch':epoch,'phase':phase,'source':source,'trait':name,
                            'lr':lr_net,'historical_reversed_r2':historical,
                            'standard_r2':standard,'mse_scaled':mse_scaled,
                            'rmse_physical':float(np.sqrt(np.mean((pred[:,j]-truth[:,j])**2))),
                            'mae_physical':float(np.mean(np.abs(pred[:,j]-truth[:,j])))})
        return records,pred,truth,inds
    # Historical loop evaluates model at the START of epoch, then trains once.
    # These logged checkpoints therefore correspond to weights after epoch-1 updates.
    for epoch0 in range(args.max_epochs):
        epoch=epoch0+1
        model.eval()
        # Perform historical three-target test evaluation on every epoch because
        # test R2 controls checkpoint selection and the old learning-rate counter.
        testrows,_,_,_=evaluate_source(TRAINING_SOURCE,test_loaders[TRAINING_SOURCE],epoch0,'validation')
        improved=False
        for row in testrows:
            trait=row['trait']
            r=row['historical_reversed_r2']
            if np.isfinite(r) and r>=best[trait]:
                improved=True;best[trait]=r;best_epoch[trait]=epoch0
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     'trait':trait,'fold':fold,'epoch':epoch0,'lr':lr_net,
                     'historical_reversed_test_R2':r,'standard_test_R2':row['standard_r2'],
                     'train_indices':outer_train,'test_indices':test,
                     'xmean':xm,'xstd':xstd,'ymin':ymin,'yrange':yrange,'config':config},
                     out/f'historical_best_{trait}.pth')
        if improved: accu_l=0
        else: accu_l+=1
        if epoch0==0 or epoch%args.eval_every==0 or epoch==args.max_epochs:
            monitored=list(testrows)
            for source,loader in test_loaders.items():
                if source!=TRAINING_SOURCE:
                    rs,_,_,_=evaluate_source(source,loader,epoch0,'validation');monitored+=rs
            rs,_,_,_=evaluate_source(TRAINING_SOURCE,eval_train_loader,epoch0,'train');monitored+=rs
            for row in monitored:
                row['best_epoch_DM']=best_epoch['DM']
                row['best_epoch_NC']=best_epoch['NC']
                row['best_epoch_NU']=best_epoch['NU']
                row['nonimprove_counter']=accu_l
            history.extend(monitored)
            pd.DataFrame(history).to_csv(out/'evaluation_history.csv',index=False)
            pd.DataFrame([{'trait':name,'selected_epoch':best_epoch[name],
                           'best_historical_test_R2':best[name],
                           'checkpoint':str(out/f'historical_best_{name}.pth')}
                           for name in TARGET_NAMES]).to_csv(out/'selected_checkpoints.csv',index=False)
            print(f'pre-update epoch {epoch0:04d}, LR={lr_net:.4g}, non-improve={accu_l}, best={best_epoch}, '
                  +' '.join(f'{r["trait"]}:hist={r["historical_reversed_r2"]:.3f}/standard={r["standard_r2"]:.3f}' for r in testrows),flush=True)
        # Historical optimizer reset: new Adam instance for ONE full-batch step.
        model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=lr_net,weight_decay=args.weight_decay)
        for xb,yb,_ in train_loader:
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted=model(xb)
            loss=sum(criterion(predicted[i].squeeze(-1),yb[:,i]) for i in range(3))
            loss.backward()
            # No gradient clipping, consistent with historical training file.
            optimizer.step()
        if not np.isfinite(loss.item()):
            raise RuntimeError(f'Nonfinite training loss at epoch {epoch}')
        if accu_l>20:
            lr_net*=0.98
            accu_l=0
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.max_epochs:
            print(f'  train update {epoch}: three-target SUM MSE={loss.item():.7g}; next LR={lr_net:.4g}',flush=True)
    # Final independently named predictions from the target-specific historical checkpoints.
    predictions=[];fold_metrics=[]
    for source,x in xs.items():
        loader=test_loaders[source]
        target_preds=[]
        true0=None;indexes0=None
        for trait in TARGET_NAMES:
            checkpoint=torch.load(out/f'historical_best_{trait}.pth',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            yhat,ytrue,idx=evaluate(model,loader,ymin,yrange,device)
            if true0 is None:true0=ytrue;indexes0=idx
            else:
                assert np.array_equal(indexes0,idx) and np.allclose(true0,ytrue)
            target_preds.append(yhat[:,TARGET_NAMES.index(trait)])
        final_pred=np.stack(target_preds,axis=1)
        met=metrics(true0,final_pred)
        for j,trait in enumerate(TARGET_NAMES):
            fold_metrics.append({'fold':1,'source':source,'trait':trait,
                         'selected_epoch':best_epoch[trait],**met[trait]})
        for j,ix in enumerate(indexes0):
            row={'fold':1,'source':source,'sample_index':int(ix),'parent_plot':str(groups[ix])}
            for k,trait in enumerate(TARGET_NAMES):
                row[trait+'_true']=float(true0[j,k])
                row[trait+'_pred']=float(final_pred[j,k])
                row[trait+'_checkpoint_epoch']=best_epoch[trait]
            predictions.append(row)
    pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'oof_predictions_fold1_diagnostic.csv',index=False)
    print('FOLD 1 FINISHED. Validation-selected results; no independent test.',flush=True)
    print(pd.DataFrame(fold_metrics).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Historical ResTrans21 reproduction: ONE FOLD, exploratory test-guided selection.')
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-epochs',type=int,default=2500)
    parser.add_argument('--eval-every',type=int,default=10)
    parser.add_argument('--eval-batch-size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=5e-5)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    assert args.max_epochs>=1 and args.eval_every>=1
    run(args)