#!/usr/bin/env python3
"""Frozen genuine model transfer, five original folds, 802-only reconstructed 40-band source."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from train_nir802_reconstructed_5fold import Net,load_data,get_fertilizer,get_loader,evaluate,metrics,TARGET_NAMES,seed_all
ap=argparse.ArgumentParser()
ap.add_argument('--data-dir',required=True)
ap.add_argument('--genuine-folds-root',required=True,help='Contains fold_1,...,fold_5 each with standard_best_DM/NC/NU.pth')
ap.add_argument('--out-dir',required=True)
a=ap.parse_args();seed_all(42)
d=Path(a.data_dir);root=Path(a.genuine_folds_root);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
xs,y,groups,manifest=load_data(d);fert=get_fertilizer(d,manifest)
assert 'nir802_recon' in xs
splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=3).split(xs['genuine'],fert))
rows=[];preds=[];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type!='cuda':raise RuntimeError('GPU required')
model=Net(40).to(device)
for fold,(tr,te) in enumerate(splits,1):
    folder=root/f'fold_{fold}'
    target_preds=[];truth=None;ix=None
    for j,trait in enumerate(TARGET_NAMES):
        ckpt=folder/f'standard_best_{trait}.pth'
        if not ckpt.exists(): raise FileNotFoundError(f'Genuine-HSI fold-specific standard checkpoint missing: {ckpt}')
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert np.array_equal(state['train_indices'],tr) and np.array_equal(state['test_indices'],te),f'fold {fold}: split mismatch'
        model.load_state_dict(state['state_dict'],strict=True)
        loader=get_loader(xs['nir802_recon'],y,te,state['xmean'],state['xstd'],state['ymin'],state['yrange'],128,False)
        yh,yt,ii=evaluate(model,loader,state['ymin'],state['yrange'],device)
        if truth is None:truth=yt;ix=ii
        else: assert np.array_equal(ix,ii) and np.allclose(truth,yt)
        target_preds.append(yh[:,j])
    final=np.stack(target_preds,axis=1);stats=metrics(truth,final)
    for trait in TARGET_NAMES:rows.append({'fold':fold,'source':'nir802_recon','trait':trait,**stats[trait]})
    for k,idx in enumerate(ix):
        row={'fold':fold,'sample_index':int(idx)}
        for j,t in enumerate(TARGET_NAMES):row.update({t+'_true':float(truth[k,j]),t+'_pred':float(final[k,j])})
        preds.append(row)
pd.DataFrame(rows).to_csv(out/'frozen_transfer_fold_metrics.csv',index=False)
pd.DataFrame(preds).to_csv(out/'frozen_transfer_oof.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
