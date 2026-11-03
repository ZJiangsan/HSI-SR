#!/usr/bin/env python3
"""Mjolnir native 802.320 nm single-guidance abundance-Gram reconstruction.
Ported from ZJiangsan/SparseHSR/scripts/gram_multiseed_robustness_seeds1_2.py
Uses original Mjolnir 40-band encoder/decoder weights and prepared Mjolnir HR/LR HSI.
Selection by UNSUPERVISED objective only. Full-HSI GT used for reporting AFTER optimization.
"""
import argparse,csv,json,random,time
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn

class Encoder(nn.Module):
 def __init__(self):
  super().__init__();self.conv11=nn.Linear(40,5,bias=False);self.conv12=nn.Linear(45,5,bias=False);self.conv13=nn.Linear(50,40,bias=False);self.relu=nn.ReLU()
 def forward(self,x):
  x1=torch.cat((x,self.relu(self.conv11(x))),dim=1)
  return self.relu(self.conv13(torch.cat((x1,self.relu(self.conv12(x1))),dim=1)))
class Decoder(nn.Module):
 def __init__(self):
  super().__init__();self.conv31=nn.Linear(40,40,bias=False)
 def forward(self,x):return self.conv31(x)
class Mapper(nn.Module):
 def __init__(self):
  super().__init__();self.conv11=nn.Linear(1,40,bias=False)
 def forward(self,x):return self.conv11(x)+x.repeat(1,40)
def sticks(v):
 p=torch.cumprod(1-v,dim=1);return v*torch.cat([torch.ones_like(v[:,:1]),p[:,:-1]],dim=1)
def gram(x):
 g=x.T@x/x.shape[0];return g-g.mean(dim=0,keepdim=True),torch.diag(g).unsqueeze(0)
def l21(x,y):return (((x-y).square()).sum(dim=1).sqrt()).sum()
def sam(x,y):
 num=(x*y).sum(1);den=x.square().sum(1).sqrt()*y.square().sum(1).sqrt();return torch.rad2deg(torch.acos((num/(den+1e-12)).clamp(-1+1e-8,1-1e-8))).mean()
def ergas(x,y):
 means=x.mean(0);mse=(x-y).square().mean(0);return 100*torch.sqrt((mse/(means.square()+1e-8)).mean())
def loss(mapped,recon,hg,hd,lg,ld):
 gl=l21(hg,lg)+ergas(hg,lg)+10*sam(hg,lg)+sam(hg.T,lg.T)
 dl=l21(hd,ld)+10*sam(hd,ld)
 return .1*l21(mapped+1e-8,recon)+10*dl+100*gl

def load_weights(model,p,device):
 c=torch.load(p,map_location='cpu',weights_only=False);sd=c.get('state_dict',c);model.load_state_dict(sd,strict=True);model.to(device).eval()
 for q in model.parameters():q.requires_grad_(False)

@torch.no_grad()
def infer(model,enc,dec,x,lr_mean,chunk=80000):
 blocks=[]
 for xx in x.split(chunk):
  m=model(xx).clamp(-1+1e-8,1-1e-8);a=sticks(enc(m)).clamp(1e-8,1-1e-8)
  blocks.append((dec(a).clamp(-1+1e-8,1-1e-8)+lr_mean).cpu())
 return torch.cat(blocks,0).numpy()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepared-h5',default='/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/mjolnir_prepared/mjolnir_field_40band.h5');ap.add_argument('--encoder',required=True);ap.add_argument('--decoder',required=True);ap.add_argument('--out-dir',required=True);ap.add_argument('--epochs',type=int,default=20000);ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--print-every',type=int,default=50);ap.add_argument('--device',default='cuda');args=ap.parse_args()
 out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 device=torch.device(args.device)
 with h5py.File(args.prepared_h5) as f:
  w=f['selected_wavelengths_nm'][:];idx=int(np.argmin(abs(w-802.320)));assert idx==26 and abs(w[idx]-802.320)<.05,(idx,w[idx]);
  hr=f['hsi_hr'];lr=f['hsi_lr'];assert lr.shape==(125,125,40),lr.shape
  # 1000x1000 is the original evaluation field, excluding the extra padded row/column.
  nir=np.asarray(hr[:1000,:1000,idx],dtype=np.float32);low=np.asarray(lr[:],dtype=np.float32)
 assert np.isfinite(nir).all() and np.isfinite(low).all()
 print('INFO HR',nir.shape,'LR',low.shape,'nm',float(w[idx]),'HR minmax',float(nir.min()),float(nir.max()),'LR minmax',float(low.min()),float(low.max()),flush=True)
 enc=Encoder();dec=Decoder();load_weights(enc,args.encoder,device);load_weights(dec,args.decoder,device)
 low_flat=torch.from_numpy(low.reshape(-1,40)).to(device)
 lr_mean=low_flat.mean(0,keepdim=True)
 with torch.no_grad():
  la=sticks(enc(low_flat-lr_mean)).clamp(1e-8,1-1e-8);lg,ld=gram(la)
 high=torch.from_numpy(nir.reshape(-1,1)).to(device);hr_mean=high.mean(0,keepdim=True);high=high-hr_mean
 mapper=Mapper().to(device);opt=torch.optim.Adam(mapper.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 best=float('inf');csv_path=out/'history.csv';ck=out/'best_mapper_unsupervised.pth';last=out/'latest_mapper.pth'
 with csv_path.open('w',newline='') as ff:
  writer=csv.writer(ff);writer.writerow(['epoch','unsupervised_loss','best_loss','seconds']);start=time.time()
  for epoch in range(1,args.epochs+1):
   opt.zero_grad(set_to_none=True)
   mapped=mapper(high).clamp(-1+1e-8,1-1e-8);ha=sticks(enc(mapped)).clamp(1e-8,1-1e-8)
   recon=dec(ha).clamp(-1+1e-8,1-1e-8);hg,hd=gram(ha)
   objective=loss(mapped,recon,hg,hd,lg,ld);value=float(objective.detach())
   if not np.isfinite(value):raise RuntimeError('Nonfinite loss at epoch '+str(epoch))
   objective.backward();opt.step()
   if value<best:
    best=value;torch.save({'epoch':epoch,'best_loss':best,'state_dict':mapper.state_dict(),'optimizer':opt.state_dict(),'seed':args.seed,'band_index':26,'band_nm':float(w[idx]),'selection_metric':'original_composite_loss','gt_metrics_used_for_selection':False},ck)
   writer.writerow([epoch,value,best,round(time.time()-start,2)])
   if epoch%args.print_every==0 or epoch==1:
    ff.flush();print(f'epoch={epoch} objective={value:.6f} best={best:.6f} runtime={time.time()-start:.1f}s',flush=True)
 torch.save({'epoch':epoch,'state_dict':mapper.state_dict()},last)
 best_state=torch.load(ck,map_location='cpu',weights_only=False);mapper.load_state_dict(best_state['state_dict']);mapper.eval()
 recon=infer(mapper,enc,dec,high,lr_mean).reshape(1000,1000,40)
 np.save(out/'nir802_reconstructed_full.npy',recon.astype(np.float32))
 with h5py.File(out/'nir802_reconstructed_full.h5','w') as h:
  h.create_dataset('reconstructed_hsi',data=recon.astype(np.float32),compression='gzip');h.create_dataset('selected_wavelengths_nm',data=w)
  h.attrs['guidance']='single unregistered native Mjolnir 802nm';h.attrs['band_index_zero_based']=26;h.attrs['best_epoch']=int(best_state['epoch']);h.attrs['selection_metric']='unsupervised_composite_loss'
 (out/'run_metadata.json').write_text(json.dumps({'best_epoch':best_state['epoch'],'best_objective':best,'wavelength':float(w[idx]),'seed':args.seed,'epochs':args.epochs,'source_h5':args.prepared_h5,'encoder':args.encoder,'decoder':args.decoder},indent=2))
 print('SUCCESS reconstruction',out/'nir802_reconstructed_full.npy','shape',recon.shape,flush=True)
if __name__=='__main__':main()
