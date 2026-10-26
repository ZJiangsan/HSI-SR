# Public archival copy of the original training script.
# NOTE: Legacy research code, retained for methodological traceability; not validated as a standalone run.
# Input path can be set through HSI_SR_MJOLNIR_H5. No raw data/checkpoints are included.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Historical Mjolnir 40-band encoder–decoder pretraining script.
Machine/user identifiers removed for public reproducibility.
"""








from __future__ import division
from scipy import interpolate
import random
import os
import os.path
import h5py
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import PIL
import pandas
import scipy.io
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy import genfromtxt
from IPython.display import clear_output 

import torchvision.utils as utils
###
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as udata
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.utils as utils
import time
import scipy.io as sio
import logging
import datetime
from math import sqrt
# Notebook-only matplotlib magic removed for Python compatibility.
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pylab import cm
import torch.optim
import random
from skimage.metrics import peak_signal_noise_ratio
from scipy.signal import savgol_filter

from torch.nn.parameter import Parameter
#
def get_random_SRF(vector_length, noise_type='u', var=1./10):
    x_shape = (3, vector_length)
    x = torch.zeros(x_shape)
    #
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    x *= var     
    return x.numpy()
#
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor





import numpy as np
import scipy.io as sio
import collections
import scipy.misc


def loadData(mname):
    return sio.loadmat(mname)

def readData(filename,num=2):
    input = loadData(filename)
    data = collections.namedtuple('data', ['hyperLR', 'multiHR', 'hyperHR',
                                           'dimLR', 'dimHR', 'srf','srfactor',
                                           'colLR','meanLR', 'reducedLR',
                                           'sphere','num'])
    
    data.hyperLR = np.array(input['hyperLR']).astype(np.float32)
    data.multiHR = np.array(input['multiHR']).astype(np.float32)
    data.hyperHR = np.array(input['hyperHR']).astype(np.float32)
    data.hyperLRI = np.array(input['hyperLRI']).astype(np.float32)
    data.dimLR = data.hyperLR.shape
    data.dimHR = data.multiHR.shape
    data.num = num
    
    # 3*31
    srf = [[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
            0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
            0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
            0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]]
    
    srf = np.array(srf).astype(np.float32)
    data.srf = srf.astype(np.float32)
    data.sr_factor = np.divide(data.dimHR[0],data.dimLR[0]).astype(np.int)
    
    data.col_lr_hsi = np.reshape(data.hyperLR,[data.dimLR[0]*data.dimLR[1],data.dimLR[2]])
    data.mean_lr_hsi = np.mean(data.col_lr_hsi,axis=0,keepdims=True)
    data.rcol_lr_hsi = np.subtract(data.col_lr_hsi,data.mean_lr_hsi)
    data.img_lr_hsi = np.reshape(data.rcol_lr_hsi,[data.dimLR[0],data.dimLR[1],data.dimLR[2]])
    
    data.col_hr_msi = np.reshape(data.multiHR, [data.dimHR[0] * data.dimHR[1], data.dimHR[2]])
    data.mean_hr_msi = np.mean(data.col_hr_msi, axis=0, keepdims=True)
    data.rcol_hr_msi = np.subtract(data.col_hr_msi,data.mean_hr_msi)
    
    data.multiLR = scipy.ndimage.zoom(data.multiHR, zoom=[1.0 / data.sr_factor, 1.0 / data.sr_factor, 1],order=0)
    data.col_lr_msi = np.reshape(data.multiLR, [data.dimLR[0] * data.dimLR[1], data.dimHR[2]])
    data.mean_lr_msi = np.mean(data.col_lr_msi, axis=0, keepdims=True)
    data.rcol_lr_msi = np.subtract(data.col_lr_msi,data.mean_lr_msi)
    
    
    data.col_hr_hsi = np.reshape(data.hyperHR,[data.dimHR[0]*data.dimHR[1],data.dimLR[2]])
    
    return data
##
canon_d500  = genfromtxt("specFunc_Canon500D.csv", delimiter=',')[1:32,:]/50
x1 = np.array(np.transpose(canon_d500, (1,0))[2,:]).reshape(1,-1)
x2 = np.array(np.transpose(canon_d500, (1,0))[1,:]).reshape(1,-1)
x3= np.array(np.transpose(canon_d500, (1,0))[0,:]).reshape(1,-1)
srf_nw = np.concatenate((x1,x2,x3), axis = 0)  #down_spc_or.conv11.weight.data.detach().cpu().numpy()
print(srf_nw.shape)






#### make it a pytorch network
# this is the ensemble encoding low resolution hsi
##
class encoder_lr_hsi_py(nn.Module):
    def __init__(self):
        super(encoder_lr_hsi_py, self).__init__()
        #
        #self.out_channels = out_channels
        self.conv11 = nn.Linear(40,5, bias = False)
        self.conv12 = nn.Linear(45,5, bias = False)
        self.conv13 = nn.Linear(50,40, bias = False)
        '''
        self.conv14 = nn.Linear(12,3, bias = False)
        self.conv15 = nn.Linear(15,3, bias = False)
        self.conv16 = nn.Linear(18,3, bias = False)
        self.conv17 = nn.Linear(21,3, bias = False)
        self.conv18 = nn.Linear(24,3, bias = False)
        self.conv19 = nn.Linear(27,3, bias = False)
        self.conv20 = nn.Linear(30,3, bias = False)
        self.conv21 = nn.Linear(33,33, bias = False)
        '''
        #self.conv22 = nn.Linear(36,3, bias = False)
        #self.conv23 = nn.Linear(39,3, bias = False)
        #self.conv24 = nn.Linear(42,3, bias = False)
        #self.conv25 = nn.Linear(45,45, bias = False)
        #self.conv13 = nn.Linear(9,9, bias = False)
        #
        self.relu = nn.ReLU()
        #
    def forward(self, x):
        layer_11 = self.conv11(x)
        layer_11 = self.relu(layer_11)
        stack_layer_11 = torch.cat((x,layer_11), dim = 1) #6
        layer_12 = self.conv12(stack_layer_11)
        layer_12 = self.relu(layer_12)
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim = 1) #9
        '''
        layer_13 = self.conv13(stack_layer_12)
        layer_13 = self.relu(layer_13)
        stack_layer_13 = torch.cat((stack_layer_12,layer_13), dim = 1) #12
        layer_14 = self.conv14(stack_layer_13)
        layer_14 = self.relu(layer_14)
        stack_layer_14 = torch.cat((stack_layer_13, layer_14), dim = 1) #15
        layer_15 = self.conv15(stack_layer_14)
        layer_15 = self.relu(layer_15)
        stack_layer_15 = torch.cat((stack_layer_14,layer_15), dim = 1) #18
        layer_16 = self.conv16(stack_layer_15)
        layer_16 = self.relu(layer_16)
        stack_layer_16 = torch.cat((stack_layer_15, layer_16), dim = 1) #21
        layer_17 = self.conv17(stack_layer_16)
        layer_17 = self.relu(layer_17)
        stack_layer_17 = torch.cat((stack_layer_16,layer_17), dim = 1) #24
        layer_18 = self.conv18(stack_layer_17)
        layer_18 = self.relu(layer_18)
        stack_layer_18 = torch.cat((stack_layer_17, layer_18), dim = 1) #27
        layer_19 = self.conv19(stack_layer_18)
        layer_19 = self.relu(layer_19)
        stack_layer_19 = torch.cat((stack_layer_18, layer_19), dim = 1) #30
        layer_20 = self.conv20(stack_layer_19)
        layer_20 = self.relu(layer_20)
        stack_layer_20 = torch.cat((stack_layer_19, layer_20), dim = 1) #33
        '''
        #layer_21 = self.conv21(stack_layer_20)
        #layer_21 = self.relu(layer_21)
        #stack_layer_21 = torch.cat((stack_layer_20, layer_21), dim = 1) #36
        #layer_22 = self.conv22(stack_layer_21)
        #layer_22 = self.relu(layer_22)
        #stack_layer_22 = torch.cat((stack_layer_21, layer_22), dim = 1) #39
        #layer_23 = self.conv23(stack_layer_22)
        #layer_23 = self.relu(layer_23)
        #stack_layer_23 = torch.cat((stack_layer_22, layer_23), dim = 1) #42
        #layer_24 = self.conv24(stack_layer_23)
        #layer_24 = self.relu(layer_24)
        #stack_layer_24 = torch.cat((stack_layer_23, layer_24), dim = 1) #45
        #
        uniform = self.conv13(stack_layer_12)
        uniform = self.relu(uniform)
        #
        return uniform
#
def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res
##
def stick_seg_sum(vsample, dim, hsize):
    
    remaining_stick = torch.ones(int(dim),).cuda()        
    for i in range(hsize):
        #[stick_segment, remaining_stick] = compute_latent_vars_break(i, remaining_stick, vsample)
        stick_segment = vsample[:, i] * remaining_stick
        remaining_stick *= (1 - vsample[:, i])
        [stick_segment, remaining_stick] = (stick_segment, remaining_stick)
        
        if i == 0:
            stick_segment_sum_lr = (stick_segment).unsqueeze(1)
        else:
            stick_segment_sum_lr = torch.cat([stick_segment_sum_lr, (stick_segment).unsqueeze(1)],1)
    return stick_segment_sum_lr


####

class decoder_hsi_py(nn.Module):
    def __init__(self):
        super(decoder_hsi_py, self).__init__()
        self.conv31 = nn.Linear(40, 40, bias = False)
        #self.conv32 = nn.Linear(120,200, bias = False)
    def forward(self, x):
        layer_final = self.conv31(x)
        #stack_layer_final_1 = torch.cat((x, layer_final_1), dim = 1)
        #layer_final = self.conv32(stack_layer_final_1)
        #layer_final = torch.sigmoid(layer_31)
        return layer_final

####
###
class mine_msi_py_121(nn.Module):
    def __init__(self):
        super(mine_msi_py_121, self).__init__()
        
        self.conv51 = nn.Linear(71,1, bias = True)
        #self.conv52 = nn.Linear(123,1, bias = False)
    def forward(self, x):
        layer_51 = self.conv51(x)
        layer_51 = torch.sigmoid(layer_51)
        #layer_52 = self.conv52(layer_51)
        #layer_53 = torch.sigmoid(layer_52)
        return layer_51
###
class mine_msi_py_12(nn.Module):
    def __init__(self):
        super(mine_msi_py_12, self).__init__()
        
        self.conv51 = nn.Linear(12,12, bias = True)
        self.conv52 = nn.Linear(12,1, bias = False)
    def forward(self, x):
        layer_51 = self.conv51(x)
        layer_51 = torch.sigmoid(layer_51)
        layer_52 = self.conv52(layer_51)
        layer_53 = torch.sigmoid(layer_52)
        return layer_53
###
class mine_msi_py(nn.Module):
    def __init__(self):
        super(mine_msi_py, self).__init__()
        
        self.conv51 = nn.Linear(13,13, bias = True)
        self.conv52 = nn.Linear(13,1, bias = False)
    def forward(self, x):
        layer_51 = self.conv51(x)
        layer_52 = self.conv52(layer_51)
        layer_53 = torch.sigmoid(layer_52)
        return layer_53
#
class L_Dspec_n_or(nn.Module):
    def __init__(self):
        super(L_Dspec_n_or, self).__init__()
        self.conv11 = nn.Linear(4,40, bias = False)
        #self.conv12 = nn.Linear(210,3, bias = False)
    def forward(self,x):
        #
        out = self.conv11(x)
        #out = torch.cat((x,out), dim = 1)
        #out = self.conv12(out)
        #out = torch.add(x[:,[40, 120, 180]], out)
        #out = self.conv12(out)
        return out
#
def Geometric_loss(x):
    y = x.abs()
    sparse_loss = y.sum(1).mean()
    return sparse_loss
##
def Entropy_loss(x): # x is the matrix, p is the order(L1, L2,..., loss)
    p = 1
    x_abs = x.abs()
    x_abs_p = x_abs.pow(p)
    x_abs_p_sum=x_abs_p.sum(1).unsqueeze(1)
    x_abs_p_norm = x_abs_p/x_abs_p_sum
    x_abs_p_norm[x_abs_p_norm==0]=1
    entropy = -(x_abs_p_norm*x_abs_p_norm.log()).sum(1).mean()
    ###
    return entropy
##
def psnr_ev_gen_old(x_true, x_pred):
    mse = torch.mean((x_true - x_pred) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))
#
def psnr_ev_gen(x_true, x_pred):
    msr = ((x_true - x_pred)**2).mean(1)
    max2 = (x_true**2).max()
    psnrall = 10*torch.log10(max2/msr)
    out_mean = psnrall.mean()
    return out_mean
#
def sam_ev_gen(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    nom = torch.sum( torch.mul(im_true, im_fake), dim=1)
    denom1 = torch.sqrt( torch.sum( torch.pow(im_true,2), dim=1))
    denom2 = torch.sqrt( torch.sum( torch.pow(im_fake,2), dim=1))
    sam = torch.acos(torch.div(nom, torch.mul(denom1, denom2)).clamp(-1.0 + 1e-8, 1.0 - 1e-8))
    sam = torch.mul(torch.div(sam, np.pi), 180)
    sam = torch.div(torch.sum(sam), N)
    return sam
##
def JSD_loss(x1,x2):
    net_1_probs =  torch.nn.functional.softmax(x1, dim = 1)
    net_2_probs =  torch.nn.functional.softmax(x2, dim = 1)
    
    m = 0.5 * (net_1_probs + net_1_probs)
    loss = 0.0
    loss += torch.nn.functional.kl_div(torch.nn.functional.log_softmax(x1, dim=1), m, reduction="batchmean") 
    loss += torch.nn.functional.kl_div(torch.nn.functional.log_softmax(x2, dim=1), m, reduction="batchmean") 
    
    return (0.5 * loss)
##
def mrae_loss_gen(im_true, im_fake):
    error = torch.abs(im_fake-im_true)/im_true
    rrmse = torch.mean(error.reshape(-1))
    return rrmse
#
def sid_loss_gen(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    denom1 = torch.sqrt( torch.sum( torch.pow(im_true,2), dim=1))
    denom2 = torch.sqrt( torch.sum( torch.pow(im_fake,2), dim=1))
    #
    unit_t = torch.div(im_true, denom1.unsqueeze(1))
    uint_f = torch.div(im_fake, denom2.unsqueeze(1))
    #
    sid = ((unit_t - uint_f)* (unit_t.log() - uint_f.log())).sum() / (N)
    return sid
#
def l_2_1_loss(im_true, im_fake):
    error = ((((im_fake-im_true).pow(2)).sum(1)).pow(1/2)).sum(0) 
    return error
#

#
class gen_lrhsi(nn.Module):
    def __init__(self):
        super(gen_lrhsi, self).__init__()
        
        self.encoder = encoder_lr_hsi_py()
        self.decoder = decoder_hsi_py()
    
    def forward(self, x):
        
        encoder_op = self.encoder_lr_hsi_py(x)
        decoder_op = self.decoder_hsi_py(encoder_op)
        
        return decoder_op
#

def psnr_ev_gen_map(x_true, x_pred):
    msr = ((x_true - x_pred)**2).mean(1)
    max2 = (x_true**2).max()
    psnrall = 10*torch.log10(max2/msr)
    return psnrall
#
def sam_ev_gen_map(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    nom = torch.sum( torch.mul(im_true, im_fake), dim=1)
    denom1 = torch.sqrt( torch.sum( torch.pow(im_true,2), dim=1))
    denom2 = torch.sqrt( torch.sum( torch.pow(im_fake,2), dim=1))
    sam = torch.acos(torch.div(nom, torch.mul(denom1, denom2)).clamp(-1.0 + 1e-8, 1.0 - 1e-8))
    #print(sam.size())
    sam = torch.mul(torch.div(sam, np.pi), 180)
    #print(("sam_max", sam.max()))
    #print(("sam_min", sam.min()))
    #sam = torch.div(torch.sum(sam), N)
    #print("")
    #print(("sam_final", sam.item()))
    #print("")
    return sam
#
def mrae_loss_gen_map(im_true, im_fake):
    error = (torch.abs(im_fake-im_true)/im_true).mean(1)
    return error
#
def truncated_normal(t, mean=0.0, std=0.1):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
      cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
      if not torch.sum(cond):
        break
      t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
    return t

lr_decoder_w1= truncated_normal(torch.empty(3, 6))
lr_decoder_w2= truncated_normal(torch.empty(6, 3))
#

class decoder_msi_py(nn.Module):
    def __init__(self,lr_decoder_w1, lr_decoder_w2):
        super(decoder_msi_py, self).__init__()
        self.w1 = Parameter(lr_decoder_w1) # make the P not learnable
        self.w2 = Parameter(lr_decoder_w2) # make the P not learnable
    def forward(self,x):
        out1 = torch.matmul(x, self.w1)
        out2 = torch.matmul(out1, self.w2)
        #out = torch.matmul(torch.sigmoid(self.P),out)
        return out2
##
def save_checkpoint_sp(model_path, epoch, LR, name, model, optimizer_gen):
    """Save the checkpoint."""
    state = {
            'epoch': epoch,
             'lr': LR,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer_gen.state_dict(),
            }
    torch.save(state, os.path.join(model_path, 'unsupervised_CRF_learning_{}.pth'.format(name)))
##
def ED(x,y):
    c_f = 0
    for c in range(x.size()[0]):
        c_i = (((x[c,:]- y)**2).sum())**(1/2)
        c_f += c_i
    return c_f.mean()

##


def hist_color_gen(source, template):
    
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    s_values, bin_idx, s_counts = torch.unique(source, return_inverse=True,
                                            return_counts=True)
    
    t_values, t_counts = torch.unique(template, return_counts=True)
    
    s_quantiles = torch.cumsum(s_counts,-1)
    s_quantiles_n = s_quantiles / s_quantiles[-1]
    t_quantiles = torch.cumsum(t_counts,-1)
    t_quantiles_n = t_quantiles/t_quantiles[-1]
    
    interp_t_values = torch.tensor(np.interp(s_quantiles_n.numpy(), t_quantiles_n.numpy(), t_values))
    
    return interp_t_values[bin_idx].reshape(oldshape)
#
def hist_color_loss(source, template):
    
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    
    s_values, bin_idx, s_counts = torch.unique(source, return_inverse=True,
                                            return_counts=True)
    
    t_values, t_counts = torch.unique(template, return_counts=True)
    
    s_quantiles = torch.cumsum(s_counts,-1)
    s_quantiles_n = s_quantiles / s_quantiles[-1]
    t_quantiles = torch.cumsum(t_counts,-1)
    t_quantiles_n = t_quantiles/t_quantiles[-1]
    
    value_diff = (s_quantiles_n - t_quantiles_n).abs().sum()
    return value_diff
### l1 loss
def L1_loss(x_true, x_pred):
    abs_dif = (x_true - x_pred).abs().sum()/x_true.size()[0]
    return abs_dif
##
def ergas_loss_gen(img_real, img_fake, scale, eps):
    N,C = img_real.size()
    means_real = img_real.mean(dim=0)
    mses = ((img_real - img_fake)**2).mean(dim=0)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
    return 100 / scale * torch.sqrt((mses / (means_real**2 + eps)).mean())





#
def stick_seg_sum(vsample, dim, hsize):
    
    remaining_stick = torch.ones(int(dim),).cuda()        
    for i in range(hsize):
        #[stick_segment, remaining_stick] = compute_latent_vars_break(i, remaining_stick, vsample)
        stick_segment = vsample[:, i].clone() * remaining_stick
        remaining_stick = remaining_stick*(1 - vsample[:, i].clone())
        [stick_segment, remaining_stick] = (stick_segment, remaining_stick)
        
        if i == 0:
            stick_segment_sum_lr = (stick_segment).unsqueeze(1)
        else:
            stick_segment_sum_lr = torch.cat([stick_segment_sum_lr, (stick_segment).unsqueeze(1)],1)
    return stick_segment_sum_lr
#
def get_stick_segments(v):
    # Vectorized stick-breaking — drop-in for the original O(K^2) CPU loop.
    # pi[:,k] = v[:,k] * prod_{j<k}(1 - v[:,j]); identical output, computed on
    # v's device (no per-k GPU->CPU copies), via a single cumulative product.
    one_minus = 1 - v
    cumprod = torch.cumprod(one_minus, dim=1)
    prev = torch.cat([torch.ones_like(cumprod[:, :1]), cumprod[:, :-1]], dim=1)
    return (v * prev).cuda()
# if you want to change a vector with the sum  = 1:
def get_stick_segments_3(v): # specifically designed for 3X3 matrix
    ndim = v.size()[0]
    pi = torch.zeros(ndim)
    for k in range(ndim):
        if k == 0:
            pi[k] = v[k]
        else:
            pi[k] = v[k] * torch.stack([(1 - v[j]) for j in range(ndim) if j < k]).prod()
    return pi.cuda()
#
class UnitNormClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'P'):
            #print("yes")
            w = module.P.data
            beta = 0.4 #model.beta.data
            wb_x = torch.nn.functional.softplus(torch.tensor([beta, beta, beta])).unsqueeze(1).cuda()
            concat_wb_x = wb_x
            for iter in range(2):
                concat_wb_x = torch.cat([concat_wb_x, wb_x], 1)
            srf_L_x = w.sigmoid()
            vsample_x = srf_L_x ** (1.0 / concat_wb_x) # uniform is the u values
            P_values = get_stick_segments(vsample_x)
            module.P.data = P_values
##
def Entropy_loss_n(x): # x is the matrix, p is the order(L1, L2,..., loss)
    eps = 0.00000001
    norm = x.sum(1, keepdim = True) 
    base = torch.div(x, (norm + eps))
    entropy = (-base*(base + eps).log()).mean()
    return entropy
# smoothness on 4 endmembers with 6 bands
def smoothness(xx): # xx should be a tensor
    for i in range(xx.size()[1]-1):
        xx_dif_i = (xx[:,i+1] - xx[:,i])**2
        if i ==0:
            xx_dif = xx_dif_i
        if i > 0:
            xx_dif = xx_dif + xx_dif_i
    return xx_dif.sum().item()

# dissimilarity among different endmembers
def dissimilarity(xx): # xx should be a tensor
    for i in range(xx.size()[0]-1):
        if i ==0:
            l1_final  = ((xx[i+1,:] - xx[i,:])**2).mean()
        if i > 0:
            for j in range(i+1):
                l1_j  = ((xx[i+1,:] - xx[j,:])**2).mean()
                
                if j ==0:
                    l1_j_or  = l1_j
                if j > 0:
                    l1_j_or = l1_j_or + l1_j
            l1_final = l1_final + l1_j_or
    return 1/l1_final
#
#def get_features(name):
#    def hook(model, input, output):
#        features[name] = output.detach()
#    return hook
def ergas_loss_gen(img_real, img_fake, scale, eps):
    N,C = img_real.size()
    means_real = img_real.mean(dim=0)
    mses = ((img_real - img_fake)**2).mean(dim=0)
    # Warning: There is a small value in the denominator for numerical stability.
    # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
    return 100 / scale * torch.sqrt((mses / (means_real**2 + eps)).mean())





#
def stick_seg_sum(vsample, dim, hsize):
    
    remaining_stick = torch.ones(int(dim),).cuda()        
    for i in range(hsize):
        #[stick_segment, remaining_stick] = compute_latent_vars_break(i, remaining_stick, vsample)
        stick_segment = vsample[:, i].clone() * remaining_stick
        remaining_stick = remaining_stick*(1 - vsample[:, i].clone())
        [stick_segment, remaining_stick] = (stick_segment, remaining_stick)
        
        if i == 0:
            stick_segment_sum_lr = (stick_segment).unsqueeze(1)
        else:
            stick_segment_sum_lr = torch.cat([stick_segment_sum_lr, (stick_segment).unsqueeze(1)],1)
    return stick_segment_sum_lr
#
def get_stick_segments(v):
    # Vectorized stick-breaking — drop-in for the original O(K^2) CPU loop.
    # pi[:,k] = v[:,k] * prod_{j<k}(1 - v[:,j]); identical output, computed on
    # v's device (no per-k GPU->CPU copies), via a single cumulative product.
    one_minus = 1 - v
    cumprod = torch.cumprod(one_minus, dim=1)
    prev = torch.cat([torch.ones_like(cumprod[:, :1]), cumprod[:, :-1]], dim=1)
    return (v * prev).cuda()
# if you want to change a vector with the sum  = 1:
def get_stick_segments_3(v): # specifically designed for 3X3 matrix
    ndim = v.size()[0]
    pi = torch.zeros(ndim)
    for k in range(ndim):
        if k == 0:
            pi[k] = v[k]
        else:
            pi[k] = v[k] * torch.stack([(1 - v[j]) for j in range(ndim) if j < k]).prod()
    return pi.cuda()
#
class UnitNormClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'P'):
            #print("yes")
            w = module.P.data
            beta = 0.4 #model.beta.data
            wb_x = torch.nn.functional.softplus(torch.tensor([beta, beta, beta])).unsqueeze(1).cuda()
            concat_wb_x = wb_x
            for iter in range(2):
                concat_wb_x = torch.cat([concat_wb_x, wb_x], 1)
            srf_L_x = w.sigmoid()
            vsample_x = srf_L_x ** (1.0 / concat_wb_x) # uniform is the u values
            P_values = get_stick_segments(vsample_x)
            module.P.data = P_values
##
def Entropy_loss_n(x): # x is the matrix, p is the order(L1, L2,..., loss)
    eps = 0.00000001
    norm = x.sum(1, keepdim = True) 
    base = torch.div(x, (norm + eps))
    entropy = (-base*(base + eps).log()).mean()
    return entropy
# smoothness on 4 endmembers with 6 bands
def smoothness(xx): # xx should be a tensor
    for i in range(xx.size()[1]-1):
        xx_dif_i = (xx[:,i+1] - xx[:,i])**2
        if i ==0:
            xx_dif = xx_dif_i
        if i > 0:
            xx_dif = xx_dif + xx_dif_i
    return xx_dif.sum().item()

# dissimilarity among different endmembers
def dissimilarity(xx): # xx should be a tensor
    for i in range(xx.size()[0]-1):
        if i ==0:
            l1_final  = ((xx[i+1,:] - xx[i,:])**2).mean()
        if i > 0:
            for j in range(i+1):
                l1_j  = ((xx[i+1,:] - xx[j,:])**2).mean()
                
                if j ==0:
                    l1_j_or  = l1_j
                if j > 0:
                    l1_j_or = l1_j_or + l1_j
            l1_final = l1_final + l1_j_or
    return 1/l1_final
#
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
#
def plotSelectedPixel(img, title=None, figsize=(5,5)):
    ''' plot selected pixels of the image '''
    #vmin = np.min(img)
    #vmax = np.max(img)
    hh = img.shape[0]
    hw = img.shape[1]
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    rad2 = axis.imshow(img) #, vmin=vmin, vmax=vmax
    axis.set_title(title)
    #divider = make_axes_locatable(axis)
    #cax = divider.append_axes("right", size="3%", pad=0.05)
    #fig.colorbar(rad2, cax=cax)
    circle_b = plt.Circle((int(hw/1.5),int(hh/1.5)), 10, linewidth=3, color='yellow', fill=False)
    circle_r = plt.Circle((int(hw/2),int(hh/2)), 10, linewidth=3, color='r', fill=False)
    circle_y = plt.Circle((int(hw/3),int(hh/3)), 10, linewidth=3, color='b', fill=False)
    axis.add_artist(circle_b)
    axis.add_artist(circle_r)
    axis.add_artist(circle_y)
    #
    axis.axes.xaxis.set_ticks([])
    axis.axes.yaxis.set_ticks([])
    #fig.savefig(os.path.join(iteration_path, "{}_test_fake_img_{}.png".format(epoch,i)))
    plt.tight_layout()
    plt.show()
    return fig, axis






##.  preparation of both hsi and msi in different sizes
import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np

def downsamplePSF(img,sigma,stride):
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    # generate filter same with fspecial('gaussian') function
    h = matlab_style_gauss2D((stride,stride),sigma)
    img_w,img_h,img_c = img.shape
    
    from scipy import signal
    out_img = np.zeros((img_w//stride, img_h//stride, img_c))
    for i in range(img_c):
        out = signal.convolve2d(img[:,:,i],h,'valid')
        out_img[:,:,i] = out[::stride,::stride]
    return out_img


def generate_LrHSI(img, scale_factor):
    img_lr = downsamplePSF(img, sigma=0.5, stride= scale_factor)
    return img_lr

def generate_MSI(img, sp_matrix):
    (h, w, c) = img.shape
    msi_channels = sp_matrix.shape[1]
    if sp_matrix.shape[0] == c:
        img_msi = np.dot(img.reshape(w*h,c), sp_matrix).reshape(h,w,sp_matrix.shape[1])
    else:
        raise Exception("The shape of sp matrix doesnot match the image")
    return img_msi






canon_d500  = genfromtxt("specFunc_Canon500D.csv", delimiter=',')[1:32,:]/50
x1 = np.array(np.transpose(canon_d500, (1,0))[2,:]).reshape(1,-1)
x2 = np.array(np.transpose(canon_d500, (1,0))[1,:]).reshape(1,-1)
x3= np.array(np.transpose(canon_d500, (1,0))[0,:]).reshape(1,-1)
srf_nw = np.concatenate((x1,x2,x3), axis = 0)  #down_spc_or.conv11.weight.data.detach().cpu().numpy()
print(srf_nw.shape)






#
# data_path = "balloons_ms.mat"
# data_in = readData(data_path)
# #
# srf = torch.tensor(data_in.srf)




xx = np.linspace(400, 1000, 200, endpoint=True) 
xx_n = xx.astype(np.int64)
##
xx_count = np.linspace(1, 200, 60, endpoint=True).astype(np.int64).tolist()
band_sample = [x_i - 1 for x_i in xx_count] # the position of 60 wavebands sampled from 200 bands

print("band_sample {}".format(band_sample))





xx_n_sample = xx_n[band_sample]
#xx_n_sample
#
band = [] # the bands for RGB visualization
x_wb = [480, 540, 650] # b, g, r wavebands
for i in range(len(x_wb)):
    xx_n_n  = np.absolute(xx_n_sample - x_wb[i])
    i_p = np.where(xx_n_n ==np.min(xx_n_n))
    print(i_p[0][0])
    print(xx_n_sample[i_p[0][0]])
    band.append(i_p[0][0])






# data preparation
srf_np = np.array([[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019,
            0.010, 0.004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007,
            0.012, 0.013, 0.015, 0.016, 0.017, 0.02, 0.013, 0.011, 0.009, 0.005,
            0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
           [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
            0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012, 0.013, 0.022,
            0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]])
#
# data preparation

data_path = os.environ.get("HSI_SR_MJOLNIR_H5", "RGB2msi/hyspex_mjolnir1024/mjolnir_prepared/mjolnir_field_40band.h5")


## read the already prepared HR-HSI, LR-HSI and the REAL selected wavelengths
mat = h5py.File(data_path, 'r')

hyper = np.float32(np.array(mat['hsi_hr']))      # H x W x 40
img_lr_hsi = np.float32(np.array(mat['hsi_lr'])) # h x w x 40
wavelengths = np.float32(np.array(mat['selected_wavelengths_nm']))

# Useful later for evaluation, but NOT used in decomposition training.
plot_mask_hr = np.uint8(np.array(mat['plot_mask_hr']))
valid_mask_hr = np.uint8(np.array(mat['valid_mask_hr']))

mat.close()



## remove the last row and the last column
hyper = hyper[:1000, :1000, :]
plot_mask_hr = plot_mask_hr[:1000, :1000]
valid_mask_hr = valid_mask_hr[:1000, :1000]



print(("hyper_shape", hyper.shape))
print(("img_lr_hsi_shape", img_lr_hsi.shape))
print(("wavelengths_shape", wavelengths.shape))
print("wavelength range: {:.3f} - {:.3f} nm".format(
    wavelengths[0], wavelengths[-1]
))

# The prepared HDF5 is already normalized by one global maximum.
# Do NOT reshape to 1000x1000: Mjolnir keeps its real field-mosaic dimensions.
hyper_in_n_n = hyper.copy()
print(("hyper_in_n", hyper_in_n_n.shape))
print("HR max: {}".format(np.max(hyper_in_n_n)))
print("LR max: {}".format(np.max(img_lr_hsi)))

# Find actual Mjolnir bands nearest the RGB-style display wavelengths.
r_idx = int(np.argmin(np.abs(wavelengths - 652.0)))
g_idx = int(np.argmin(np.abs(wavelengths - 549.0)))
b_idx = int(np.argmin(np.abs(wavelengths - 461.0)))

rgb_indices = [r_idx, g_idx, b_idx]  # R,G,B
bgr_indices = [b_idx, g_idx, r_idx]  # used below where old code flips [2,1,0]

print("RGB display bands:")
print("  R: index {}, {:.3f} nm".format(r_idx, wavelengths[r_idx]))
print("  G: index {}, {:.3f} nm".format(g_idx, wavelengths[g_idx]))
print("  B: index {}, {:.3f} nm".format(b_idx, wavelengths[b_idx]))

##
# visualize
xx_in_rgb = hyper_in_n_n[:, :, rgb_indices]
out_1 = xx_in_rgb[:, :, :]
nonzero = out_1[out_1 != 0]
i_min = np.quantile(nonzero.flatten(), 0.00001)
i_max = np.quantile(nonzero.flatten(), 0.99999)

# improve the image contrast
out_1_x = np.zeros((out_1.shape), dtype=np.float32)
for i in range(3):
    rgb_x_i = out_1[:, :, i]
    out_1_x[:, :, i] = (rgb_x_i - i_min) / (i_max - i_min)
    out_1_x[out_1_x < 0] = 0
    out_1_x[out_1_x > 1] = 1

out_1_x = out_1_x**(1/1.8)

plt.figure(figsize=(8,8))
plt.imshow(out_1_x[:, :, :])
plt.show()


plotSelectedPixel(out_1_x)


(h, w, c) = hyper_in_n_n.shape

# HR-HSI is kept for later mapping/evaluation.
img_hr_hsi_0 = hyper_in_n_n.copy()
img_hr_hsi = img_hr_hsi_0.copy()

print("np.max(img_hr_hsi_0): {}".format(np.max(img_hr_hsi_0)))
print("HR HSI shape: {}".format(img_hr_hsi.shape))

# IMPORTANT:
# The LR-HSI was already generated during Mjolnir data preparation using
# Gaussian 8x8, sigma=0.5, valid convolution, stride=8.
# Therefore DO NOT call generate_LrHSI() again here.
print("Prepared LR HSI shape: {}".format(img_lr_hsi.shape))






##.                   initiate all the networks
end_epoch = 300000
hsize = 40
LR_spc_dn = 1e-3
encoder_lr = 1e-3
mine_lr = 1e-3
decoder_lr = 1e-3
affine_x_lr = 1e-3
#
encoder = encoder_lr_hsi_py().cuda().float()
optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=encoder_lr, weight_decay=1e-4)
#
mine_msi = mine_msi_py_121().cuda() #mine_msi_py_9().cuda()
optimizer_mine_msi = torch.optim.Adam(mine_msi.parameters(), lr=mine_lr, weight_decay=1e-4)
##
decoder = decoder_hsi_py().cuda() #decoder_msi_py(lr_decoder_w1,lr_decoder_w2).cuda()
optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=decoder_lr, weight_decay=1e-4)
##
srf = torch.tensor(np.transpose(srf_np, (1,0)))
srf_learned_or = srf.cuda()
#
down_spc_or = L_Dspec_n_or().cuda().float() 
#down_spc = L_Dspec_n(3,3, srf_L).type(torch.cuda.FloatTensor).cuda() # 107 wavebands in total
optimizer_spc_or_dn = torch.optim.Adam(down_spc_or.parameters(), lr=LR_spc_dn, weight_decay=1e-4)
#
dtype = torch.cuda.FloatTensor
#
from torch.autograd import Variable
mse = torch.nn.MSELoss().type(dtype)





# #                                            PART I.  (ICVL)

start_ep = 0
LR = 1e-7
# control whether to include SID loss in style image decomposition
sid_cef = 100 # can be 0 or 100
#
end_epoch = 35000001

#
encoder_lr =  LR
decoder_lr = LR
mine_lr = LR
#
loss_save_mrae = np.inf
loss_save_sam = np.inf
loss_save_ergas = np.inf
loss_save_mrae_or = np.inf
loss_save_psnr = 0
#
# need to change the data range to 0-1 for easy learning !!!
l_h_size = img_lr_hsi.shape[0]
l_w_size = img_lr_hsi.shape[1]
c_size = img_lr_hsi.shape[2]
#
input_lr_hsi_raw = torch.tensor(img_lr_hsi.reshape(img_lr_hsi.shape[0]*img_lr_hsi.shape[1], img_lr_hsi.shape[2])).float()
input_lr_hsi_raw_mean = input_lr_hsi_raw.mean(0,keepdim = True)
input_lr_hsi = input_lr_hsi_raw - input_lr_hsi_raw_mean
#


##
# first only learn the abundance and endmember of HR-RGB, until fully convergence
# then only update the spectral response function learning network
##
for j in range(start_ep, end_epoch): 
    LR_spc_dn = LR
    mine_lr = LR
    encoder_lr = LR
    # load the data 
    #
    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=encoder_lr, weight_decay=1e-4)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=decoder_lr, weight_decay=1e-4)
    ##
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    #
    input_lr_hsi_var = Variable(input_lr_hsi.cuda(), requires_grad = True)
    ##
    out_LR_img_v = encoder(input_lr_hsi_var)
    out_LR_img_v = out_LR_img_v.clamp(1e-8, 1e+8)
    # apply sum to one constraints
    out_LR_img_s = get_stick_segments(out_LR_img_v)
    out_LR_img_s = out_LR_img_s.clamp(1e-8, 1.0-1e-8)
    ##
    entropy_loss = Entropy_loss(out_LR_img_s)
    #####. decoder
    # non negative
    decoder.conv31.weight.data = decoder.conv31.weight.data.clamp(-1 + 1e-8, 1 - 1e-8)
    # here decoder is to generate HR RGB image directly
    out_LR_img_de = decoder(out_LR_img_s) # 3, 15, 15, 31
    out_LR_img_de = out_LR_img_de.clamp(-1+ 1e-8, 1.0-1e-8)
    #
    loss_de_msi = l_2_1_loss(input_lr_hsi_var + torch.tensor(1e-8).cuda(), out_LR_img_de) #.type(dtype) # l_2_1_loss loss between reconstructed LR-HSI and GroundTruth LR-HSI
    loss_sid_msi = sid_loss_gen(input_lr_hsi_var + torch.tensor(1.0 +1e-8).cuda(), out_LR_img_de + torch.tensor(1.0 +1e-8).cuda()) #.type(dtype) # l_2_1_loss loss between reconstructed LR-HSI and GroundTruth LR-HSI
    sam_de_msi = sam_ev_gen(input_lr_hsi_var + torch.tensor(1e-8).cuda(), out_LR_img_de).type(dtype)
    psnr_de_msi = psnr_ev_gen(input_lr_hsi_var + torch.tensor(1e-8).cuda(), out_LR_img_de).type(dtype)
    ergas_de_msi = ergas_loss_gen(input_lr_hsi_var+ torch.tensor(1e-8).cuda(), out_LR_img_de+ torch.tensor(1e-8).cuda(),1, torch.tensor(1e-8).cuda()).type(dtype)
    ##
    ##   final loss from the decoder
    loss_de = 1.0*loss_de_msi #+ 0.9*loss_lr_de_msi
    loss_de_sid = 1.0*loss_sid_msi #+ 0.9*sid_lr_de_msi
    loss_de_sam = 1.0*sam_de_msi #+ 0.9*sid_lr_de_msi
    loss_de_pnsr = 1.0*psnr_de_msi #+ 0.9*psnr_lr_de_msi
    loss_de_ergas = 1.0*ergas_de_msi #+ 0.9*psnr_lr_de_msi
    #
    #else: #+ sid_a*loss_de_sid   + 100*loss_de_ergas
    loss_de_final = 10*loss_de + sid_cef*loss_de_sid  + 0.01*entropy_loss # - 500*mutual_loss + 0.001*hsi_volume_loss# + 0.001*dissimilarity_loss #10*sid_de_hsi + 10*sid_de_msi +  + 0.1*hsi_volume_loss + 6000*loss_de_sid
    # + 40*loss_de_sid
    #
    loss_out_mrae = loss_de_msi.item()
    loss_out_sam = sam_de_msi.item()
    loss_out_psnr = psnr_de_msi.item()
    loss_out_ergas = ergas_de_msi.item()
    ##
    print("current epoch {}".format(j))
    if loss_save_sam > loss_out_sam:
        loss_save_sam = loss_out_sam
        loss_save_sam_psnr = loss_out_psnr
        loss_save_sam_mrae = loss_out_mrae
        loss_save_sam_ergas = loss_out_ergas
        lr_save_sam = LR
        j_save_sam = j
        #
        print("Saving SAM {}".format(loss_save_sam))
        print("saving MRAE: {}, sam: {}, psnr: {}, ergas: {} epoch: {}, lr: {} ".format(loss_save_sam_mrae, loss_save_sam, loss_save_sam_psnr,loss_save_sam_ergas, j_save_sam, lr_save_sam))
        #print("loss_save {}".format(loss_save))
        #print("saving model {}".format(i))
        # CONSTRAINTS on encoder
        save_checkpoint_sp(os.getcwd(), 10000,lr_save_sam, "encoder_uav_mjolnir_CE_9_endmember_sam_4spc40_visNir_g54_01", encoder, optimizer_encoder) # 32233 is good enough
        save_checkpoint_sp(os.getcwd(), 10000,lr_save_sam, "decoder_uav_mjolnir_CE_9_endmember_sam_4spc40_visNir_g54_01", decoder, optimizer_decoder)
        #save_checkpoint_sp(os.getcwd(), 10000, lr_save_sam,"mine_msi_uav_pavia_CE_9_endmember_sam_3spc31_p3_01", mine_msi, optimizer_mine_msi)
        #save_checkpoint_sp(os.getcwd(), 10000, "down_spc_uav_bench_CE_20_endmember_1_layerDecoder_HRhsi_0", down_spc_or, optimizer_spc_or_dn)
    print("SAM: last saved mrae: {}, sam: {}, psnr: {}, ergas: {} at epoch: {} and lr:{}".format(loss_save_sam_mrae, loss_save_sam, loss_save_sam_psnr,loss_save_sam_ergas, j_save_sam, lr_save_sam))
    
    loss_de_final.backward()
    #
    optimizer_encoder.step()
    optimizer_decoder.step()
    # Log
    if j % 3000 == 0:
        #
        print("epoch {}, LR {}".format(j, LR))
        #########
        #
        print("loss_de {}".format(loss_de.item()))
        print("loss_de_sid {}".format(loss_de_sid.item()))
        print("psnr_de_hsi {}".format(psnr_de_msi.item()))
        print("entropy_loss: {}".format(entropy_loss.item()))
        #
        #out_LR_spa
        input_lr_hsi_np = input_lr_hsi_var.detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size,c_size) # input RGB
        out_LR_img_np = out_LR_img_de.detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size,c_size) # generated hr-rgb
        #
        out_1 = input_lr_hsi_np[:,:,[6,11,17]] 
        i_min = np.quantile(out_1[out_1!=0].flatten(), 0.00001)
        i_max = np.quantile(out_1[out_1!=0].flatten(), 0.99999)
        # improve teh image contrast
        out_1_x = np.zeros((out_1.shape), dtype=np.float32 )
        for i in range(3):
            rgb_x_i = out_1[:,:,i]
            out_1_x[:,:,i] = (rgb_x_i - i_min)/(i_max-i_min)
            out_1_x[out_1_x<0] = 0
            out_1_x[out_1_x>1] = 1
        #
        out_2 = out_LR_img_np[:,:,[6,11,17]] 
        i_min = np.quantile(out_2[out_2!=0].flatten(), 0.00001)
        i_max = np.quantile(out_2[out_2!=0].flatten(), 0.99999)
        # improve teh image contrast
        out_2_x = np.zeros((out_2.shape), dtype=np.float32 )
        for i in range(3):
            rgb_x_i = out_2[:,:,i]
            out_2_x[:,:,i] = (rgb_x_i - i_min)/(i_max-i_min)
            out_2_x[out_2_x<0] = 0
            out_2_x[out_2_x>1] = 1
        # plot the ground truth and output
        figure, axarr = plt.subplots(nrows=1,ncols=2, figsize=(6,4))
        plt.sca(axarr[0]); 
        plt.imshow(out_1_x[:,:,[2,1,0]]); plt.title("LR_rgb_gt, epoch_{}".format(j)) # HR_HSI
        plt.sca(axarr[1]); 
        plt.imshow(out_2_x[:,:,[2,1,0]]); plt.title("LR_rgb_rec, epoch_{}".format(j)) # HR_HSI
        figure.tight_layout()
        plt.show()        #
        ##.   error maps
        # mrae map
        hr_mrae_map = mrae_loss_gen_map(input_lr_hsi_var.cuda() + torch.tensor(1e-8).cuda(), out_LR_img_de).detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size)
        hr_mrae_map[hr_mrae_map>1.0] = 1.0
        # sam map
        hr_sam_map = sam_ev_gen_map(input_lr_hsi_var.cuda(), out_LR_img_de).detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size)
        # psnr map
        hr_psnr_map = psnr_ev_gen_map(input_lr_hsi_var.cuda(), out_LR_img_de).detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size)
        #
        figure, axarr = plt.subplots(nrows=1,ncols=3, figsize=(8,3))
        plt.sca(axarr[0]); 
        plt.imshow(hr_mrae_map); plt.title("hr_mrae_map,E_{}".format(j)) # HR_HSI
        plt.sca(axarr[1]); 
        plt.imshow(hr_sam_map); plt.title("hr_sam_map,E_{}".format(j)) # HR_HSI
        plt.sca(axarr[2]); 
        plt.imshow(hr_psnr_map); plt.title("hr_psnr_map,E_{}".format(j)) #out_HR_HSI
        figure.tight_layout()
        plt.show()        #
        # plot abundance map (randomly select 6 pixels for visualization)
        out_LR_img_s_np = out_LR_img_s.detach().cpu().squeeze().numpy().reshape(l_h_size,l_w_size,hsize)
        #s_number = Rand(0, hsize-1, 8)
        #
        figure, axarr = plt.subplots(nrows=1,ncols=9, figsize=(23,4))
        for x in range(9):
            plt.sca(axarr[x]); 
            plt.imshow(out_LR_img_s_np[:,:,x]); plt.title("Abundance LR,B{},E{}".format(x,j)) # HR_HSI
        figure.tight_layout()
        plt.show()        #
        print("")
        ##
        print("the actual values")
        x =np.linspace(363, 1018, 40, endpoint=True) 
        #
        plt.figure(figsize=(12,8))
        y1 =input_lr_hsi_np[int(l_h_size*0.2),int(l_w_size*0.2),:]
        y2 =out_LR_img_np[int(l_h_size*0.2),int(l_w_size*0.2),:]
        plt.plot(x, y1, color = 'b', linestyle = '-.', label = "gt1")
        plt.plot(x, y2, color = 'b', linestyle = '-', label = "rec1")
        y3 =input_lr_hsi_np[int(l_h_size*0.5),int(l_w_size*0.5),:]
        y4 =out_LR_img_np[int(l_h_size*0.5),int(l_w_size*0.5),:]
        plt.plot(x, y3, color = 'g', linestyle = '-.', label = "gt2")
        plt.plot(x, y4, color = 'g', linestyle = '-', label = "rec2")
        y5 =input_lr_hsi_np[int(l_h_size*0.8),int(l_w_size*0.8),:]
        y6 =out_LR_img_np[int(l_h_size*0.8),int(l_w_size*0.8),:]
        plt.plot(x, y5, color = 'r', linestyle = '-.', label = "gt3")
        plt.plot(x, y6,color = 'r', linestyle = '-',  label = "rec3")
        #plt.ylim(0, 1)
        plt.legend()
        plt.show()
        #
        ## visualize the smoothed endmembers
        #r_number = Rand(0,hsize-1,hsize-2)
        x =np.linspace(363, 1018, 40, endpoint=True) 
        plt.figure(figsize=(9,6))
        for i in range(hsize):
            y_i = decoder.conv31.weight.data.transpose(0,1).cpu().detach().numpy()[i,:]
            plt.plot(x, y_i, label = "ref {}".format(i))
        #plt.legend(prop={'size': 3})
        plt.show()
    
    if (j +1) % 5000 == 0:
        #LR_spc_dn = LR_spc_dn* 0.9
        #LR_spa_dn = LR_spa_dn* 0.9
        LR = LR* 0.98
        print("LR_gen: {}".format(LR))
        #print("mrae_a: {}".format(mrae_a))
        #print("sid_a: {}".format(sid_a))






# ## load the models if you have trained ones
resume_encoder = os.path.join(os.getcwd(), 'unsupervised_CRF_learning_encoder_uav_mjolnir_CE_9_endmember_sam_4spc40_visNir_g54_01.pth') # it is the wrong name only
resume_decoder = os.path.join(os.getcwd(), 'unsupervised_CRF_learning_decoder_uav_mjolnir_CE_9_endmember_sam_4spc40_visNir_g54_01.pth') # it is the wrong name only

checkpoint_encoder = torch.load(resume_encoder)
LR_tr = checkpoint_encoder['lr']
encoder.load_state_dict(checkpoint_encoder['state_dict'])
optimizer_encoder.load_state_dict(checkpoint_encoder['optimizer'])
##
checkpoint_decoder = torch.load(resume_decoder)
LR_tr = checkpoint_decoder['lr']
decoder.load_state_dict(checkpoint_decoder['state_dict'])
optimizer_decoder.load_state_dict(checkpoint_decoder['optimizer'])
















