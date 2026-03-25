import sys, os
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

files = glob.glob('/Volumes/EXTRA128/P2VM/fromIsabelle/*fits')

plot = True

if plot:
    plt.close(0); plt.figure(0, figsize=(12,6))
    plt.legend()

M,W = {}, {}

for i,file in enumerate(files):
    if i%100 == 0:
        print(i, len(files))
    f = fits.open(file)
    WL = {}
    FL = {}
    for i,h in enumerate(f):
        if 'EXTNAME' in h.header and h.header['EXTNAME']=='OI_WAVELENGTH':
            WL[h.header['INSNAME']] = h.data['EFF_WAVE']*1e6
            
        if 'EXTNAME' in h.header and h.header['EXTNAME']=='OI_FLUX':
            if 'FLUX' in [c.name for c in h.columns]:
                FL[h.header['INSNAME']] = h.data['FLUX']
            else:
                FL[h.header['INSNAME']] = h.data['FLUXDATA']
    f.close()
    for k in WL:
        if 'SC' in k:
            tmp = FL[k].mean(axis=0)
            tmp /= tmp[(WL[k]>2.05)*(WL[k]<2.15)].mean()
            if len(WL[k]) in M:
                M[len(WL[k])].append(tmp)
            else:
                M[len(WL[k])]= [tmp]
                W[len(WL[k])] = WL[k]
            if plot:
                plt.plot(WL[k], tmp, '-k', label=k+' %d'%len(WL[k]), alpha=0.1)
print([(k,len(M[k])) for k in M])

model = {'WL':np.linspace(1.9, 2.5, 2000)}

colors = {'LOW':'g', 'MEDIUM':'r', 'HIGH':'b'}

for i in M:
    if plot:
        plt.plot(W[i], np.mean(M[i], axis=0), '-k', label='AVG %d'%i, alpha=0.5)
    try:
        c0 = np.polyfit(W[i][W[i]<1.985], np.mean(M[i], axis=0)[W[i]<1.985], 1)
    except:
        c0 = [np.mean(M[i], axis=0)[0]]
        
    x = np.linspace(1.95, 2, 20)
    #plt.plot(x, np.maximum(0,np.polyval(c0, x)), ':r', alpha=0.2)
    try:
        c1 = np.polyfit(W[i][W[i]>2.32], np.mean(M[i], axis=0)[W[i]>2.32], 1)
    except:
        c1 = [np.mean(M[i], axis=0)[-1]]
    x = np.linspace(2.3, 2.5, 20)
    #plt.plot(x, np.polyval(c1, x), ':r', alpha=0.2)

    k='LOW'
    if i>200 and i<1000:
        k = 'MEDIUM'
    elif i>1000:
        k = 'HIGH'
    model[k] = np.interp(model['WL'], W[i], np.mean(M[i], axis=0), left=0, right=0)
    model[k][model['WL']<np.min(W[i])] = np.maximum(0, np.polyval(c0, model['WL'][model['WL']<np.min(W[i])]))
    model[k][model['WL']>np.max(W[i])] = np.maximum(0, np.polyval(c1, model['WL'][model['WL']>np.max(W[i])]))

    if plot:
        plt.plot(model['WL'], model[k], '-', color=colors[k], alpha=0.5, linewidth=2)
if plot:
    plt.ylim(0,1.5)

with open('./p2vm_flux.pckl', 'wb') as f:
    pickle.dump(model, f)
