#!/usr/bin/env python3
"""Exact 802-only section extraction using original September 2026 manifest geometry.
No image-template matching. Retains all original section rows and ordering;
ResTrans21 load_data selects the labeled subset exactly as before.
"""
from pathlib import Path
import argparse, json, shutil
import numpy as np
import pandas as pd
import h5py

p=argparse.ArgumentParser()
p.add_argument('--sections-dir',default='/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/cross_sensor_functional_validation_sections')
p.add_argument('--reconstructed',default='/trinity/home/jizh/nir802_reconstruction/saved_mapper_export/nir802_reconstructed_full.h5')
p.add_argument('--output-dir',default='/trinity/home/jizh/nir802_reconstruction/sections_data')
a=p.parse_args()
src=Path(a.sections_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(src/'matched_sections_manifest.csv')
req=('global_y0','global_y1','global_x0','global_x1')
if any(c not in m for c in req):raise RuntimeError('Missing original global section coordinates; do not guess.')
with h5py.File(src/'matched_sections_40band.h5') as f:
    n=f['genuine_mjolnir'].shape[0]
    if f['genuine_mjolnir'].shape[1:] != (15,100,40):raise RuntimeError('Unexpected original section shape')
if len(m)!=n:raise RuntimeError(f'Manifest {len(m)} vs original HDF5 {n} mismatch')
if 'section_index' in m and not np.array_equal(m.section_index.to_numpy(),np.arange(n)):
    raise RuntimeError('Original section_index ordering differs from HDF5, cannot align safely')
with h5py.File(a.reconstructed,'r') as f:
    for key in ('reconstructed_hsi','nir802_reconstructed'):
        if key in f:
            cube=np.asarray(f[key],dtype=np.float32);break
    else: raise KeyError(f'No recognized reconstruction key; available={list(f.keys())}')
if cube.shape!=(1000,1000,40):raise RuntimeError(f'Unexpected cube shape {cube.shape}')
patches=[]
for i,row in m.iterrows():
    y0,y1,x0,x1=[int(row[k]) for k in req]
    if not (0<=y0<y1<=1000 and 0<=x0<x1<=1000 and y1-y0==15 and x1-x0==100):
        raise RuntimeError(f'Section {i} has invalid original coordinates {(y0,y1,x0,x1)}')
    if all(c in m for c in ('stack_row','stack_col','plot_y0','plot_y1','plot_x0','plot_x1')):
        expected=(int(row.stack_row)*100+int(row.plot_y0),int(row.stack_row)*100+int(row.plot_y1),int(row.stack_col)*100+int(row.plot_x0),int(row.stack_col)*100+int(row.plot_x1))
        if (y0,y1,x0,x1)!=expected:raise RuntimeError(f'Section {i}: global/plot geometry inconsistent')
    patch=cube[y0:y1,x0:x1,:]
    if patch.shape!=(15,100,40) or not np.isfinite(patch).all():raise RuntimeError(f'Invalid section {i}')
    patches.append(patch)
patches=np.stack(patches).astype('float32')
# Write atomically. Never overwrite original dataset.
target=out/'matched_sections_40band.h5';tmp=out/'matched_sections_40band.h5.tmp'
with h5py.File(src/'matched_sections_40band.h5','r') as f,h5py.File(tmp,'w') as g:
    for name in f:
        f.copy(name,g)
    if 'nir802_reconstructed' in g:raise RuntimeError('Original HDF5 unexpectedly contains nir802_reconstructed')
    g.create_dataset('nir802_reconstructed',data=patches,compression='gzip',compression_opts=4)
    g.attrs['nir802_source']='Existing saved mapper reconstruction, original manifest global windows'
tmp.replace(target)
for fname in ('matched_sections_manifest.csv','groundtruth_sections_n.csv'):
    if (src/fname).exists():shutil.copy2(src/fname,out/fname)
# Reuse the historical 148-sample selection exactly, rather than dropping arbitrary rows.
labels_path=out/'groundtruth_sections_n.csv'
if not labels_path.exists():labels_path=Path('/trinity/home/jizh/RGB2msi/groundtruth/groundtruth_sections_n.csv')
labels=pd.read_csv(labels_path)
targets=('dm_gpermsq','n-content_perc','n-uptake_gpermsq')
ids=list(labels.columns[:3]);lookup={}
for _,r in labels.iterrows():
    key=tuple(int(float(r[c])) for c in ids)
    lookup[key]=np.asarray([r[c] for c in targets],dtype=float)
keep=[]
for _,r in m.iterrows():
    key=tuple(int(r[c]) for c in ('label_id_1','label_id_2','label_id_3'))
    v=lookup.get(key)
    keep.append(v is not None and np.all(np.isfinite(v)))
selected=np.flatnonzero(keep)
info={'original_sections':n,'prediction_sections':len(selected),'excluded_original_indices_0based':np.flatnonzero(~np.asarray(keep)).tolist(),'patch_shape':[15,100,40],'geometry':'Original matched_sections_manifest.csv global_y0:y1 and global_x0:x1','source_cube':str(a.reconstructed),'output_h5':str(target)}
(out/'extraction_verified.json').write_text(json.dumps(info,indent=2))
print('SUCCESS: original manifest geometry used. '+json.dumps(info,indent=2),flush=True)
if len(selected)!=148:raise RuntimeError(f'Expected 148 labeled sections in historical experiment; found {len(selected)}. Do not launch training.')
