#!/usr/bin/env python3
"""Consolidate source-matched historical and standard-selected validation results."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--base',default='/trinity/home/jizh/HSI_SR_prediction/results_dual_checkpoint_5fold_fresh')
a=p.parse_args()
base=Path(a.base)
source_for={'genuine':'genuine','rgb':'rgb_only','rgb802':'rgb_plus_802'}
metrics=[];predictions=[];epochs=[]
for group,source in source_for.items():
    for fold in range(1,6):
        folder=base/group/f'fold_{fold}'
        for variant,metricfile,predfile in (
            ('historical','fold_metrics.csv','oof_predictions_fold1_diagnostic.csv'),
            ('standard','standard_selected_fold_metrics.csv','standard_selected_validation_predictions.csv')
        ):
            for filename in (metricfile,predfile):
                if not (folder/filename).is_file():
                    raise FileNotFoundError(f'Missing completed output: {folder/filename}')
            m=pd.read_csv(folder/metricfile).query('source == @source').copy()
            x=pd.read_csv(folder/predfile).query('source == @source').copy()
            assert len(m)==3 and set(m.trait)=={'DM','NC','NU'}
            assert m.fold.eq(fold).all() and x.fold.eq(fold).all()
            m.insert(0,'representation',group);m.insert(1,'selection',variant)
            x.insert(0,'representation',group);x.insert(1,'selection',variant)
            metrics.append(m);predictions.append(x)
        e=pd.read_csv(folder/'dual_selected_checkpoints.csv')
        e.insert(0,'representation',group);e.insert(1,'fold',fold)
        epochs.append(e)
allm=pd.concat(metrics,ignore_index=True)
allp=pd.concat(predictions,ignore_index=True)
alle=pd.concat(epochs,ignore_index=True)
for (rep,sel),part in allp.groupby(['representation','selection']):
    if part.sample_index.nunique()!=148 or part.sample_index.duplicated().any():
        raise ValueError(f'Invalid section coverage: {rep}/{sel}')
for fold in range(1,6):
    expected=None
    for rep in source_for:
        got=set(allp.query('fold == @fold and representation == @rep and selection == "standard"').sample_index)
        if expected is None: expected=got
        elif expected != got:raise ValueError(f'Split mismatch fold {fold}')
summary=(allm.groupby(['representation','selection','trait'],as_index=False)
    .agg(R2_mean=('R2','mean'),R2_sd=('R2','std'),
         RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),
         MAE_mean=('MAE','mean'),MAE_sd=('MAE','std')))
allm.to_csv(base/'dual_all_fold_metrics.csv',index=False)
allp.to_csv(base/'dual_all_oof_predictions.csv',index=False)
alle.to_csv(base/'dual_checkpoint_epochs.csv',index=False)
summary.to_csv(base/'dual_fivefold_mean_sd.csv',index=False)
print(summary.to_string(index=False))
print(f'Saved aggregated CSV files in {base}')
