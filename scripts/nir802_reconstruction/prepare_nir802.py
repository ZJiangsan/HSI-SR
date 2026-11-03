#!/usr/bin/env python3
"""Extract the native unregistered 802.320nm HR NIR guide from Mjolnir's 40band prepared data.
Does not fabricate reconstructed HSI. Output must enter ORIGINAL abundance-Gram mapping code.
"""
import argparse,json
from pathlib import Path
import h5py,numpy as np
ap=argparse.ArgumentParser()
ap.add_argument('--prepared-h5',default='/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/mjolnir_prepared/mjolnir_field_40band.h5')
ap.add_argument('--out-dir',required=True)
a=ap.parse_args(); out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
with h5py.File(a.prepared_h5) as f:
    print('HDF5 datasets:',[(k,f[k].shape) for k in f if isinstance(f[k],h5py.Dataset)])
    w=np.asarray(f['selected_wavelengths_nm']).ravel()
    idx=int(np.argmin(abs(w-802.320)))
    assert idx==26,(idx,w[idx],'40-band index/order differs from verified CSV')
    assert abs(float(w[idx])-802.320)<0.05, (idx,w[idx])
    hr=f['hsi_hr']; lr=f['hsi_lr']; assert hr.ndim==3 and hr.shape[-1]==40 and lr.shape[-1]==40
    guide=np.asarray(hr[:1000,:1000,idx],dtype=np.float32)
    assert np.isfinite(guide).all()
    np.save(out/'nir802_hr_guide.npy',guide)
    with h5py.File(out/'nir802_reconstruction_input.h5','w') as o:
        o.create_dataset('hr_nir802',data=guide,compression='gzip')
        o.create_dataset('hsi_lr',data=lr[:],compression='gzip')
        o.create_dataset('selected_wavelengths_nm',data=w)
        o.attrs['nir_band_index_0based']=26
        o.attrs['nir_wavelength_nm']=float(w[idx])
        o.attrs['guidance']='Mjolnir derived, not registered to Sony RGB; same field support'
    print(f'HR NIR shape: {guide.shape}; LR HSI: {lr.shape}; index={idx}; wavelength={w[idx]:.6f} nm')
(out/'guide_metadata.json').write_text(json.dumps({'index_0based':26,'wavelength_nm':float(w[idx]),'hr_nir_shape':guide.shape,'source_h5':a.prepared_h5,'not_spatially_registered_to_sony':True},indent=2))
print('GUIDE PREPARED. The 40-band RECONSTRUCTION has not been performed.')
