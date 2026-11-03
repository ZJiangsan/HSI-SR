#!/usr/bin/env python3
import argparse,csv
import numpy as np,h5py
ap=argparse.ArgumentParser()
ap.add_argument('--reconstructed-full',required=True)
ap.add_argument('--reference-h5',required=True)
ap.add_argument('--output',required=True)
a=ap.parse_args();y=np.load(a.reconstructed_full,mmap_mode='r')
with h5py.File(a.reference_h5) as f:
    ref=np.asarray(f['hsi_hr'][:y.shape[0],:y.shape[1],:],dtype=np.float32)
    valid=np.asarray(f['valid_mask_hr'][:y.shape[0],:y.shape[1]],dtype=bool) if 'valid_mask_hr' in f else np.ones(y.shape[:2],bool)
assert y.shape==ref.shape,(y.shape,ref.shape)
valid &= np.isfinite(ref).all(-1)&np.isfinite(y).all(-1)
rows=[]
for name,sel in [('all',np.arange(40)),('anchor_excluded',np.array([i for i in range(40) if i!=26]))]:
    a=ref[valid][:,sel].astype(np.float64);b=np.asarray(y[valid][:,sel],dtype=np.float64)
    rmse=np.sqrt(np.mean((a-b)**2));psnr=20*np.log10(1.0/max(rmse,1e-12))
    cos=np.einsum('ij,ij->i',a,b)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)
    sam=float(np.degrees(np.arccos(np.clip(cos,-1,1))).mean())
    rows.append(dict(group=name,rmse=rmse,psnr_unit_range=psnr,sam_deg=sam,n_pixels=len(a)))
with open(a.output,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(rows)
