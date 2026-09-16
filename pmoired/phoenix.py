import urllib.request
import os
import numpy as np
from astropy.io import fits
import pickle
from scipy.ndimage import gaussian_filter1d

URL = 'https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011'

# -- where PHOENIX data are saved
_dirdata = None
allFiles, allLogg, allTeff, maxTeff, WL = None, None, None, None, None
file_wavelength = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'

def changeDataDirectory(newdirdata=None, verbose=True): 
    global _dirdata, WL, file_wavelength, allFiles, allLogg, allTeff, maxTeff

    if newdirdata is None:
        if _dirdata is None:
            # -- default value
            newdirdata = os.path.join(os.path.expanduser('~'), '.pmrd', 'PHOENIX')
        else:
            newdirdata = _dirdata    
        
    if not os.path.exists(newdirdata):
        if verbose:
            print('creating directory', newdirdata)
        os.mkdir(newdirdata)
        
    # -- load         
    if not os.path.exists(os.path.join(newdirdata, file_wavelength)):
        if verbose:
            print('downloading', file_wavelength)
        url = URL.split('PHOENIX')[0]+'/'+file_wavelength
        data = urllib.request.urlopen(url).read()
        if verbose:
            print('  writing', os.path.join(newdirdata, file_wavelength))
        with open(os.path.join(newdirdata, file_wavelength), 'wb') as f:
            f.write(data)
        if verbose:
            print('  done')
    if verbose:
        print('loading', os.path.join(newdirdata, file_wavelength))
    with fits.open(os.path.join(newdirdata, file_wavelength)) as h:
        WL = h[0].data*1e-4 # in um

    # -- load or download Teff / logg grid
    filename = os.path.join(newdirdata, 'GRID_TEFF_LOGG.pckl')
    if not os.path.exists(filename):
        if verbose:
            print('getting Teff/Logg grid')
        allFiles = {0.0: getFilesList(0)}
        allLogg = {0.0: np.array(sorted(set([k[1] for k in allFiles[0.0]])))}
        allTeff = {0.0: np.array(sorted(set([k[0] for k in allFiles[0.0]])))}
        # -- max Teff for a given logg
        maxTeff = {0:{g:max([k[0] for k in allFiles[0.0] if k[1]==g]) for g in allLogg[0.0]}}
        if verbose:
            print('  saving as', filename)
        with open(filename, 'wb') as h:
            pickle.dump((allFiles, allLogg, allTeff, maxTeff), h)
    else:
        if verbose:
            print('  loading', filename)
        with open(filename, 'rb') as h:
            allFiles, allLogg, allTeff, maxTeff = pickle.load(h)
        
    _dirdata = newdirdata
    return

def file2Key(f, withMetal=False):
    """
    returns (Teff, logg)
    """
    k = [float(f.split('lte')[1].split('-')[0])]
    if '+' in f:
        k.append(float(f.split('-')[1].split('+')[0]))
        if withMetal:
            k.append(float(f.split('+')[1].split('.PH')[0]))
    else:
        k.append(float(f.split('-')[1])) 
        if withMetal:
            k.append(-float(f.split('-')[2].split('.PH')[0]))
    return tuple(k)

def makeFileName(Teff, logg, metal=0.0):
    s = '-' if metal<=0 else '_'
    return f"lte{Teff:05.0f}-{logg:.2f}{s}{np.abs(metal):.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"

def getFilesList(metal=0):
    """
    get all files from https://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011

    for a given metalicity
        only for metalicities in [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1.0]
    """
    if metal<=0:    
        url = URL+'/'+'Z-%.1f'%np.abs(metal)
    else:
        url = URL+'/'+'Z+%.1f'%metal
        
    data = urllib.request.urlopen(url).read()
    files = [l.split('href="')[1].split('"')[0] for l in data.decode().split('\n') if 'PHOENIX' in l and '.fits' in l]
    # -- key by Teff, logg
    return {file2Key(f):f for f in files}

    
def continuum(WL, SP, width=15e-4):
    """
    hacky way to compute continuum for *noiseless* synthetic spectra
    """
    res = np.zeros(len(WL))
    for i in range(len(WL)):
        res[i] = np.max(SP[np.abs(WL-WL[i])<width/2])
    return res

# -- default for CO bandheads for GRAVITY HR
_ip_wlmin = 2.25
_ip_wlmax = 2.4
_ip_R = 9000 
_ip_metal = 0.0
_ip_data = None
def initInterpolator(wlmin=None, wlmax=None, R=None, metal=None, dirdata=None, verbose=False):
    global _ip_wlmin, _ip_wlmax, _ip_R, _ip_metal, _ip_data, WL, _dirdata

    if _dirdata is None:
        changeDataDirectory(dirdata)

    _init = False
    if not (wlmin is None and wlmax is None and R is None):
        _ip_wlmin = wlmin
        _ip_wlmax = wlmax
        _ip_R = R
        _ip_metal = metal
        _init = True   

    savefile = 'GRID_'
    if _ip_metal is None:
        _ip_metal = 0.0
        
    if _ip_metal<=0:
        savefile += '-%.1f_'%np.abs(_ip_metal)
    else:
        savefile += '+%.1f_'%np.abs(_ip_metal)

    savefile += '%.4fum-%.4fum_R%.0f.pckl'%(_ip_wlmin, _ip_wlmax, _ip_R)

    if _ip_data is None:
        _init = True
    
    if _init:    
        if os.path.exists(os.path.join(_dirdata, savefile)):
            if verbose:
                print('restoring', savefile)
            with open(os.path.join(_dirdata, savefile), 'rb') as f:
                _ip_data = pickle.load(f)
        else:
            _ip_data = {'metal':_ip_metal,
                        'wlmin':_ip_wlmin,
                        'wlmax':_ip_wlmax,
                        'R':_ip_R,
                        'w':(WL>=_ip_wlmin)*(WL<=_ip_wlmax)}
            _ip_data['WL'] = WL[_ip_data['w']]
        
    files = os.listdir(_dirdata)
    files = [f for f in files if f.startswith('lte') and f.endswith('.fits') and file2Key(f, withMetal=True)[2]==_ip_metal]

    _add = 0
    for f in files:
        k = file2Key(f)
        if not k in _ip_data:
            if verbose:
                print('  adding', f)
            with fits.open(os.path.join(_dirdata, f)) as h:
                _ip_data[k] = h[0].data[_ip_data['w']]
            _ip_data[k] /= continuum(_ip_data['WL'], _ip_data[k])
            _ip_data[k] = gaussian_filter1d(_ip_data[k], 
                                            0.5*np.mean(_ip_data['WL']/np.gradient(_ip_data['WL']))/_ip_R)
            _add += 1
    if _add>0:
        if verbose:
            print('saving', savefile)
        with open(os.path.join(_dirdata, savefile), 'wb') as f:
            pickle.dump(_ip_data, f)
    return
        
def interpolator(Teff, logg, metal=0, verbose=False, dirdata=None):
    global allFiles, allLogg, allTeff, maxTeff, _ip_data, _dirdata

    if _dirdata is None:
        changeDataDirectory(dirdata)

    # -- possible metalicities:
    M = [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1.0]
    # -- closest metalicity
    m = M[np.argmin(np.abs(np.array(M)-metal))]
    if not m in allFiles:
        allFiles[m] = getFilesList(m)
        allLogg[m] = {0.0: np.array(sorted(set([k[1] for k in allFiles[m]])))}
        allTeff[m] = {0.0: np.array(sorted(set([k[0] for k in allFiles[m]])))}
        maxTeff[m] = {g:max([k[0] for k in allFiles[m] if k[1]==g]) for g in allLogg[m]}
    _logg = np.array([g for g in allLogg[m] if Teff<maxTeff[m][g]+200])
    dg = np.abs(logg - _logg)
    s = np.argsort(dg)
    logg1 = _logg[s[0]]
    logg2 = _logg[s[1]]
    
    Teff1 = np.array(list(set([k[0] for k in allFiles[m] if k[1]==logg1])))
    d1 = np.abs(Teff-Teff1)
    Teff1 = (Teff1[np.argsort(d1)[0]], Teff1[np.argsort(d1)[1]])
    
    Teff2 = np.array(list(set([k[0] for k in allFiles[m] if k[1]==logg2])))
    d2 = np.abs(Teff-Teff2)
    Teff2 = (Teff2[np.argsort(d2)[0]], Teff2[np.argsort(d2)[1]])

    K = [(Teff1[0], logg1, m), (Teff1[1], logg1, m), (Teff2[0], logg2, m), (Teff2[1], logg2, m)]
    addAny = False
    for k in K:
        filename = os.path.join(_dirdata, makeFileName(k[0], k[1], k[2]))
        if not os.path.exists(filename):
            if verbose:
                print('downloading...', end=' ')
            url = URL+'/Z%s%.1f/'%('-' if m<=0 else '+', np.abs(m))+makeFileName(k[0], k[1], k[2])
            if verbose:
                print(url)
            data = urllib.request.urlopen(url).read()
            if verbose:
                print('  writing', filename)
            with open(filename, 'wb') as f:
                f.write(data)
    if addAny or _ip_data is None:
        initInterpolator(verbose=verbose)
    
    if verbose:
        print(logg1, Teff1, logg2, Teff2)
        
    F1 = _ip_data[(Teff1[0], logg1)]+(Teff-Teff1[0])*(_ip_data[(Teff1[1], logg1)]-_ip_data[(Teff1[0], logg1)])/(Teff1[0]-Teff1[1])
    F2 = _ip_data[(Teff2[0], logg2)]+(Teff-Teff2[0])*(_ip_data[(Teff2[1], logg2)]-_ip_data[(Teff1[0], logg2)])/(Teff2[0]-Teff2[1])
    
    return F1 + (logg-logg1)*(F2-F1)/(logg2-logg1)
    