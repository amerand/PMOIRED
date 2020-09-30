import time
import copy
import multiprocessing
import random
import os
import platform, subprocess
import itertools
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.special
import scipy.interpolate
import scipy.stats

try:
    import pmoired.dpfit as dpfit
    import pmoired.dw as dw
except:
    import dpfit, dw

from astropy import constants

_c = np.pi**2/180/3600/1000*1e6


def Ssingle(oi, param, noLambda=False):
    """
    build spectrum fo Vsingle
    """
    if not noLambda:
        _param = computeLambdaParams(param)
    else:
        _param = param.copy()
    # -- flux (spectrum)
    f = np.zeros(oi['WL'].shape)

    # == continuum ===============
    if 'f0' in _param:
        f += _param['f0']
    elif 'f' in _param:
        f += _param['f']
    elif not 'spectrum' in _param:
        # -- no continuum is defined, assumes 1.0
        f += 1.

    # == polynomials ============================
    As = filter(lambda x: x.startswith('f') and x[1:].isdigit(), _param.keys())
    for a in As:
        i = int(a[1:])
        if i>0:
            f += _param[a]*(oi['WL']-np.min(oi['WL']))**i

    # == power law ================================
    if 'famp' in _param and 'fpow' in _param:
        f+= _param['famp']*(oi['WL']-np.min(oi['WL']))**_param['fpow']

    # == spectral lines ==============
    # -- list of fluxes of emission/absorption lines
    lines = filter(lambda x: x.startswith('line_') and x.endswith('_f'),
                    _param.keys())
    for l in lines:
        i = l.split('_')[1] # should not start with f!!!!
        wl0 = _param['line_'+i+'_wl0'] # in um
        if 'line_'+i+'_lorentzian' in _param.keys():
            dwl = _param['line_'+i+'_lorentzian'] # in nm
            f += _param[l]*1/(1+(oi['WL']-wl0)**2/(dwl/1000)**2)
        if 'line_'+i+'_gaussian' in _param.keys():
            dwl = _param['line_'+i+'_gaussian'] # in nm
            if 'line_'+i+'_power' in _param.keys():
                _pow = _param['line_'+i+'_power']
            else:
                _pow = 2.0
            f += _param[l]*np.exp(-4*np.log(2)*np.abs(oi['WL']-wl0)**_pow/
                                  (dwl/1000)**_pow)
    # == cubic splines fspl_wli, fspl_fi
    sw = sorted(list(filter(lambda x: x.startswith('fspl_wl'), _param.keys())))
    sf = sorted(list(filter(lambda x: x.startswith('fspl_f'), _param.keys())))
    if len(sw)>0:
        assert len(sw)==len(sf), "cubic spline flux is ill defined: N(fspl_wl)!=N(fspl_f)"
        X = np.array([_param[k] for k in sw])
        Y = np.array([_param[k] for k in sf])
        f += scipy.interpolate.interp1d(X, Y, kind='cubic', fill_value='extrapolate')(oi['WL'])

    # == linear flin_wli, flin_fi ===================
    sw = sorted(list(filter(lambda x: x.startswith('flin_wl'), _param.keys())))
    sf = sorted(list(filter(lambda x: x.startswith('flin_f'), _param.keys())))
    if len(sw)>0:
        assert len(sw)==len(sf), "linear flux is ill defined: N(flin_wl)!=N(flin_f)"
        X = np.array([_param[k] for k in sw])
        Y = np.array([_param[k] for k in sf])
        f += scipy.interpolate.interp1d(X, Y, kind='linear', fill_value='extrapolate')(oi['WL'])

    # == wavelets ==========================
    sw = sorted(list(filter(lambda x: x.startswith('fwvl'), _param.keys())))
    if len(sw)>0:
        assert np.log2(len(sw))%1==0, "wavelet flux must have 2**n nodes"
        n = int(np.log2(len(sw)))
        if 'fit' in oi and 'wl ranges' in oi['fit']:
            assert len(oi['fit']['wl ranges'])==1, "wavelet flux only work for single 'wl ranges'"

            X = np.linspace(max(oi['fit']['wl ranges'][0][0], min(oi['WL'])),
                            min(oi['fit']['wl ranges'][0][1], max(oi['WL'])), len(sw))
        else:
            X = np.linspace(min(oi['WL']), max(oi['WL']), len(sw))
        # -- inverse Haar wavelet transform
        Y = np.array([_param[k] for k in sw])
        Y = dw.oneD(Y, -n, order=8)*np.sqrt(2**n)
        #if min(X)<min(oi['WL']) or max(X)>max(oi['WL']):
        #    print('WL range !')
        f = np.interp(oi['WL'], X, Y)

    # == arbitrary, expressed as string ===================
    if 'spectrum' in _param.keys():
        sp = _param['spectrum']
        sp = sp.replace('$WL', 'oi["WL"]')
        for k in _param.keys():
            if k in sp:
                sp.replace('$'+k, str(_param[k]))
        f += eval(sp)

    return f

def VsingleOI(oi, param, noT3=False, fov=None, pix=None, dx=0, dy=0,
              timeit=False, indent=0, _ffrac=1.0, _dwl=0.0):
    """
    build copy of OI, compute VIS, VIS2 and T3 for a single object parametrized
    with param

    oi: result from oiutils.loadOI, or a list of results

    param: a dictionnary with the possible keys defined below.

    fov: field of view of synthetic image (in mas). if None (default), will
        not produce a sinthetic image
    pix: pixel size (in mas) for synthetic image
    dx, dy: coordinates of the center of the image (in mas). default is 0,0

    Possible keys in the parameters dictionnary:

    position:
    ---------
    'x', 'y': define position in the field, in mas

    size (will decide the model):
    -----------------------------
    'ud': uniform disk diameter (in mas)
        or
    'fwhm': Full width at half maximum for a Gaussian (in mas)
        or
    'udout': outside diameter for a ring (in mas)
    'thick': fractional thickness of the ring (0..1)

    if none of these is given, the component will be fully resolved (V=0)

    stretching:
    -----------
    object can be stretched along one direction, using 2 additional parameters:
    'projang': projection angle from N (positive y) to E (positive x), in deg
    'incl': inclination, in deg

    slant:
    ------
    'slant', 'slant projang'

    rings:
    ------
    rings are by default uniform in brightness. This can be altered by using
    different keywords:
    'profile': radial profile, can be 'uniform', 'doughnut' or 'power law'.
        if power law, 'power' gives the radial law
    'az ampi', 'az projangi'projan: defines the cos variation amplitude and phase for i nodes
        along the azimuth

    flux modeling:
    --------------
    if nothing is specified, flux is assume equal to 1 at all wavelengths

    'f' or 'f0': if the constant flux (as function of wavelength)
    'fi': polynomial amplitude in (wl-wl_min)**i (wl in um)
    'fpow', 'famp': famp*(wl-wl_min)**fpow (wl in um)
    spectral lines:
    'line_i_f': amplitude of line i (>0 for emission, <0 for absorption)
    'line_i_wl0': central wavelnegth of line i (um)
    'line_i_gaussian': fwhm for Gaussian profile (warning: in nm, not um!)
        or
    'line_i_lorentzian': width for Lorentzian (warning: in nm, not um!)

    """
    t0 = time.time()
    _param = computeLambdaParams(param)
    res = {}
    # -- what do we inherit from original data:
    for k in ['WL', 'header', 'insname', 'filename', 'fit']:
        if k in oi.keys():
            res[k] = oi[k]

    # -- small shift in wavelength (for bandwith smearing)
    if _dwl!=0:
        cwl = 1+_dwl/np.mean(res['WL'])
        res['WL'] += _dwl
    else:
        cwl = 1.0

    # -- model -> no telluric features
    res['TELLURICS'] = np.ones(res['WL'].shape)

    #print('->', fov, pix)
    if not fov is None and not pix is None:
        # -- image coordinates in mas
        X, Y = np.meshgrid(np.linspace(-fov/2, fov/2, int(fov/pix)+1),
                           np.linspace(-fov/2, fov/2, int(fov/pix)+1))
        if not dx is None:
            X += dx
        if not dy is None:
            Y += dy
        I = np.zeros(X.shape)
        #print('setting synthetic images')
    else:
        I = None

    # -- spectrum, fraction of it if needed for bandwith smearing
    f = Ssingle(res, _param, noLambda=True)*_ffrac

    # -- check negativity of spectrum
    negativity = np.sum(f[f<0])/np.sum(f[f>=0])

    if 'fit' in oi and 'ignore negative flux' in oi['fit'] and \
        oi['fit']['ignore negative flux']:
        negativity = 0.0
    if timeit:
        print(' '*indent+'VsingleOI > spectrum %.3fms'%(1000*(time.time()-t0)))
    ts = time.time()

    # -- where to get baseline infos?
    if 'OI_VIS' in oi.keys():
        key = 'OI_VIS'
    elif 'OI_VIS2' in oi.keys():
        key ='OI_VIS2'
    else:
        key = None

    if not key is None:
        baselines = oi[key].keys()
    else:
        baselines = []

    # -- max baseline
    Bwlmax = 0.0
    for k in baselines:
        Bwlmax = max(Bwlmax, np.max(oi[key][k]['B/wl']/cwl))

    # -- position of the element in the field
    if 'x' in _param.keys() and 'y' in _param.keys():
        x, y = _param['x'], _param['y']
    else:
        x, y = 0.0, 0.0

    # -- 'slant' i.e. linear variation of flux
    if 'slant' in list(_param.keys()) and 'slant projang' in list(_param.keys()):
        du, dv = 1e-6, 1e-6 # in meters / microns
        if _param['slant']<0:
            _param['slant'] = np.abs(_param['slant'])
            _param['slant projang'] = (_param['slant projang']+180)%360
    else:
        du, dv = 0.0, 0.0

    # -- do we need to apply a stretch?
    if 'projang' in _param.keys() and 'incl' in _param.keys():
        rot = -_param['projang']*np.pi/180
        _uwl = lambda z: np.cos(_param['incl']*np.pi/180)*\
                         (np.cos(rot)*z['u/wl']/cwl + np.sin(rot)*z['v/wl']/cwl)
        _vwl = lambda z: -np.sin(rot)*z['u/wl']/cwl + np.cos(rot)*z['v/wl']/cwl
        _Bwl = lambda z: np.sqrt(_uwl(z)**2 + _vwl(z)**2)

        if du:
            _udu = lambda z: (np.cos(rot)*z['u/wl']/cwl +\
                              np.sin(rot)*z['v/wl']/cwl)*np.cos(np.pi*_param['incl']/180)+du/res['WL']
            _vdu = lambda z: -np.sin(rot)*z['u/wl']/cwl + np.cos(rot)*z['v/wl']/cwl
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: (np.cos(rot)*z['u/wl']/cwl +\
                              np.sin(rot)*z['v/wl']/cwl)*np.cos(np.pi*_param['incl']/180)
            _vdv = lambda z: -np.sin(rot)*z['u/wl']/cwl + np.cos(rot)*z['v/wl']/cwl + dv/res['WL']
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            _X = (np.cos(rot)*(X-x) + np.sin(rot)*(Y-y))/np.cos(_param['incl']*np.pi/180)
            _Y = -np.sin(rot)*(X-x) + np.cos(rot)*(Y-y)
            R = np.sqrt(_X**2+_Y**2)
    else:
        _uwl = lambda z: z['u/wl']/cwl
        _vwl = lambda z: z['v/wl']/cwl
        _Bwl = lambda z: z['B/wl']/cwl
        if du:
            _udu = lambda z: z['u/wl']/cwl+du/res['WL']
            _vdu = lambda z: z['v/wl']/cwl
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: z['u/wl']/cwl
            _vdv = lambda z: z['v/wl']/cwl+dv/res['WL']
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            _X, _Y = X-x, Y-y
            R = np.sqrt(_X**2+_Y**2)

    # -- phase offset
    #phi = lambda z: -2j*_c*(z['u/wl']*x+z['v/wl']*y)
    PHI = lambda z: np.exp(-2j*_c*(z['u/wl']/cwl*x + z['v/wl']/cwl*y))
    if du:
        #dPHIdu = lambda z: -2j*_c*x*PHI(z)/oi['WL']
        #dPHIdv = lambda z: -2j*_c*y*PHI(z)/oi['WL']
        PHIdu = lambda z: np.exp(-2j*_c*((z['u/wl']/cwl+du/res['WL'])*x + z['v/wl']/cwl*y))
        PHIdv = lambda z: np.exp(-2j*_c*(z['u/wl']/cwl*x + (z['v/wl']/cwl+dv/res['WL'])*y))

    # -- guess which visibility function

    if 'ud' in _param.keys(): # == uniform disk ================================
        Rout = _param['ud']/2
        Vf = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bwl(z) + 1e-12)/(_c*_param['ud']*_Bwl(z)+ 1e-12)
        if _param['ud']<0:
            negativity += np.abs(_c*_param['ud']*Bwlmax)

        if du: # -- slanted
            Vfdu = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdu(z))/(_c*_param['ud']*_Bdu(z))
            Vfdv = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdv(z))/(_c*_param['ud']*_Bdv(z))
        if not I is None:
            # -- without anti aliasing
            #I = R<=_param['ud']/2
            # -- anti aliasing:
            na = 3
            for _dx in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                for _dy in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                    R2 = (_X-_dx)**2+(_Y-_dy)**2
                    I += R2<=(_param['ud']/2)**2
            I/=na**2
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)

    elif 'fwhm' in _param.keys(): # == gaussian ================================
        #if _param['fwhm']<0:
        #    negativity += np.max(np.abs(_c*_param['fwhm']*_Bwl(z)))

        a = 1./(2.*(_param['fwhm']/2.355)**2)
        Rout = _param['fwhm']/2
        Vf = lambda z: np.exp(-(_c*_Bwl(z))**2/a)
        if du:
            print('WARNING: slanted gaussian does not make sense!')
            Vfdu = lambda z: np.exp(-(_c*_Bdu(z))**2/a)
            Vfdv = lambda z: np.exp(-(_c*_Bdv(z))**2/a)
        if not I is None:
            I = np.exp(-R**2*a)
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)

    elif 'diamout' in _param or 'diam' in _param: # == F(r)*G(az) ========================
        # -- disk or ring with radial and az profile

        if not 'diamin' in _param and not 'thick' in _param:
            _param['thick'] = 1.0
        if 'thick' in _param:
            if 'diam' in _param:
                diamout = _param['diam']
            if 'diamout' in _param:
                diamout = _param['diamout']
            diamin = _param['diam']*(1-min(max(1e-9,_param['thick']),1))
        else:
            diamin = _param['diamin']
            diamout = _param['diamout']
            _param['thick'] = (diamout-diamin)/diamout

        if 'Nr' in oi['fit']:
            Nr = oi['fit']['Nr']
        else:
            Nr = max(10, int(100*_param['thick'])) # -- arbitrary !!!

        r = np.linspace(diamin/2, diamout/2, Nr)
        mu = np.sqrt(1-(2*r/diamout)**2)
        Rout = diamout/2

        if not 'profile' in _param:
            _param['profile'] = 'uniform'

        if '$' in _param['profile']:
            # -- generic formula
            tmp = _param['profile'].replace('$R', 'r')
            tmp = tmp.replace('$MU', 'mu')
            for k in _param.keys():
                if '$'+k in tmp:
                    tmp = tmp.replace('$'+k, str(_param[k]))
            Ir = eval(tmp)
        elif _param['profile']=='doughnut':
            Ir = 1-((r-np.mean(r))/np.ptp(r)*2)**2
        elif _param['profile']=='uniform': # uniform
            Ir = np.ones(r.shape)

        _n, _amp, _phi = [], [], []
        for k in _param.keys():
            if k.startswith('az amp'):
                _n.append(int(k.split('az amp')[1]))
                _phi.append(_param[k.replace('az amp', 'az projang')])
                _amp.append(_param[k])

        if 'projang' in _param.keys() and 'incl' in _param.keys():
            stretch = [np.cos(np.pi*_param['incl']/180), _param['projang']]
        else:
            stretch = [1,0]
        negativity += _negativityAzvar(_n, _phi, _amp)
        Vf = lambda z: _Vazvar(z['u/wl']/cwl, z['v/wl']/cwl, Ir, r, _n, _phi, _amp,
                                stretch=stretch)
        if du: # --slanted
            if len(_n):
                print('WARNING: slanted disk with azimutal variation not implemented properly!')
            Vfdu = lambda z: _Vazvar(_udu(z), _vdu(z), Ir, r, _n, _phi, _amp,)
            Vfdv = lambda z: _Vazvar(_udv(z), _vdv(z), Ir, r, _n, _phi, _amp,)
        if not I is None:
            I = _Vazvar(None, None, Ir, r, _n, _phi, _amp,
                        stretch=stretch, numerical=1, XY=(X-x,Y-y))
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)

    # elif 'udout' in _param.keys() and 'thick' in _param.keys():
    #     # == simple ring =======
    #     Rout = _param['udout']/2
    #     udin = _param['udout']*(1-_param['thick'])
    #     Fout, Fin = _param['udout']**2, udin**2,
    #     Vf = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bwl(z))/
    #                     (_c*_param['udout']*_Bwl(z))-
    #                     Fin*2*scipy.special.j1(_c*udin*_Bwl(z))/
    #                                     (_c*udin*_Bwl(z)))/(Fout-Fin)
    #     if du: # -- slanted
    #         Vfdu = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bdu(z))/
    #                         (_c*_param['udout']*_Bdu(z))-
    #                         Fin*2*scipy.special.j1(_c*udin*_Bdu(z))/
    #                                               (_c*udin*_Bdu(z)))/(Fout-Fin)
    #         Vfdv = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bdv(z))/
    #                         (_c*_param['udout']*_Bdv(z))-
    #                         Fin*2*scipy.special.j1(_c*udin*_Bdv(z))/
    #                                               (_c*udin*_Bdv(z)))/(Fout-Fin)
    #     if not I is None:
    #         # -- without anti aliasing
    #         #I = (R<=_param['udout']/2)*(R>=udin/2)
    #         # -- anti aliasing:
    #         na = 3
    #         for _dx in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
    #             for _dy in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
    #                 R2 = (_X-_dx)**2+(_Y-_dy)**2
    #                 I += (R2<=(_param['udout']/2)**2)*(R2>=(udin/2)**2)
    #             I/=na**2
    #             if np.sum(I)==0:
    #                 # -- unresolved -> single pixel
    #                 R2 = _X**2+_Y**2
    #                 I = R2==np.min(R2)
    else:
        # -- default == fully resolved flux == zero visibility
        Vf = lambda z: np.zeros(_Bwl(z).shape)
        if du:
            Vfdu = lambda z: np.zeros(_Bdu(z).shape)
            Vfdv = lambda z: np.zeros(_Bdv(z).shape)
        # -- image is uniformly filled
        if not I is None:
            I += 1

    # -- slant in the image
    if du and not I is None:
        normI = np.sum(I)
        I *= (1.0 + np.sin(_param['slant projang']*np.pi/180)*_param['slant']/Rout*_X +
                    np.cos(_param['slant projang']*np.pi/180)*_param['slant']/Rout*_Y)
        #I *= normI/np.sum(I)

    # -- cos or sin variation
    # -- https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms
    # -- item 115
    # ==> not implemented yet!
    if 'cos' in _param and 'cos pa' in _param:
        _C = np.cos(_param['cos pa']*np.pi/180)
        _S = np.sin(_param['cos pa']*np.pi/180)
        _xc = _S*_X + _C*_Y
        I *= 1 + _param['cos']*np.cos(_xc)
    if 'sin' in _param and 'sin pa' in _param:
        _C = np.cos(_param['sin pa']*np.pi/180)
        _S = np.sin(_param['sin pa']*np.pi/180)
        _xc = _S*_X + _C*_Y
        I *= 1 + _param['sin']*np.sin(_xc)

    # -- check that slant does not lead to negative flux
    if du and np.abs(_param['slant'])>1:
        negativity += (np.abs(_param['slant'])-1)

    if timeit:
        print(' '*indent+'VsingleOI > setup %.3fms'%(1000*(time.time()-ts)))

    res['OI_VIS'] = {}
    res['OI_VIS2'] = {}
    tv = time.time()
    for k in baselines:
        # -- for each baseline
        tmp = {}
        if du: # -- for slanted
            V = Vf(oi[key][k])
            # compute slant from derivative of visibility
            dVdu = (Vfdu(oi[key][k]) - V)/du
            dVdv = (Vfdv(oi[key][k]) - V)/dv
            dVdu /= 2*_c/res['WL']
            dVdv /= 2*_c/res['WL']
            # -- see https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms
            # -- relations 106 and 107
            V = V+1j*(np.sin(_param['slant projang']*np.pi/180)*_param['slant']/Rout*dVdu +
                      np.cos(_param['slant projang']*np.pi/180)*_param['slant']/Rout*dVdv)
            V *= PHI(oi[key][k])
        else:
            #print('!!', key, oi[key][k].keys())
            #print('!! x:', x, 'y', y)
            V = Vf(oi[key][k])*PHI(oi[key][k])

        tmp['|V|'] = np.abs(V)
        # -- force -180 -> 180
        tmp['PHI'] = (np.angle(V)*180/np.pi+180)%360-180
        # -- not needed, strictly speaking, takes a long time!
        for l in ['B/wl', 'FLAG']:
            tmp[l] = oi[key][k][l]
        if not fov is None:
            # -- slow dow the code!
            for l in ['u', 'v', 'u/wl', 'v/wl', 'B/wl', 'MJD']:
                tmp[l] = oi[key][k][l]
                if '/wl' in l:
                    tmp[l] /= cwl
            tmp['EV'] = np.zeros(tmp['|V|'].shape)
            tmp['EPHI'] = np.zeros(tmp['PHI'].shape)

        res['OI_VIS'][k] = tmp

        if 'OI_VIS2' in oi.keys():
            tmp = {}

            # -- not needed, strictly speaking, takes a long time!
            for l in ['B/wl', 'FLAG']:
                tmp[l] = oi['OI_VIS2'][k][l]
                if '/wl' in l:
                    tmp[l] /= cwl

            tmp['V2'] = np.abs(V)**2
            if not fov is None:
                # -- slow dow the code!
                for l in ['u', 'v', 'u/wl', 'v/wl',  'MJD']:
                    tmp[l] = oi['OI_VIS2'][k][l]
                    if '/wl' in l:
                        tmp[l] /= cwl
                tmp['EV2'] = np.zeros(tmp['V2'].shape)

            res['OI_VIS2'][k] = tmp

    if timeit:
        print(' '*indent+'VsingleOI > complex vis %.3fms'%(1000*(time.time()-tv)))
    if 'OI_T3' in oi.keys() and not noT3:
        t3 = time.time()
        res['OI_T3'] = {}
        for k in oi['OI_T3'].keys():
            res['OI_T3'][k] = {}
            for l in ['u1', 'u2', 'v1', 'v2', 'MJD', 'formula', 'FLAG', 'Bmax/wl', 'Bavg/wl']:
                res['OI_T3'][k][l] = oi['OI_T3'][k][l]
                if '/wl' in l:
                     res['OI_T3'][k][l] /= cwl

        res = computeT3fromVisOI(res)
        if timeit:
            print(' '*indent+'VsingleOI > T3 from V %.3fms'%(1000*(time.time()-t3)))

    if not I is None:
        # -- normalize image to total flux, useful for adding
        res['MODEL'] = {'image':I/np.sum(I), 'X':X, 'Y':Y}
    else:
        res['MODEL'] = {}
    res['MODEL']['totalflux'] = f
    res['MODEL']['negativity'] = negativity # 0 is image is >=0, 0<p<=1 otherwise
    return res

def VfromImageOI(oi):
    """
    oi dict must have key 'image' of 'cube' -> Vmodel with "fov" and "pix"
    """
    if type(oi)==list:
        return [VfromImageOI(o) for o in oi]

    if not 'MODEL' in oi.keys() and \
                not ('image' in oi['MODEL'].keys() or
                     'cube' in oi['MODEL'].keys()):
        print('WARNING: VfromImage cannot compute visibility from image')
        print('         run "Vmodel" with fov and pix values set before')
        return oi

    oi['IM_VIS'] = {}
    oi['IM_FLUX'] = {}

    for k in oi['OI_VIS'].keys():
        tmp = {}
        for l in ['u', 'v', 'u/wl', 'v/wl', 'B/wl', 'MJD', 'FLAG']:
            tmp[l] = oi['OI_VIS'][k][l]
        # -- dims: uv, wl, x, y
        phi = -2j*_c*(oi['MODEL']['X'][None,None,:,:]*tmp['u/wl'][:,:,None,None] +
                      oi['MODEL']['Y'][None,None,:,:]*tmp['v/wl'][:,:,None,None])
        # -- very CRUDE!
        if 'cube' in oi['MODEL'].keys():
            V = np.sum(oi['MODEL']['cube'][None,:,:,:]*np.exp(phi), axis=(2,3))/\
                       np.sum(oi['MODEL']['cube'], axis=(1,2))[None,:]
        else:
            V = np.sum(oi['MODEL']['image'][None,None,:,:]*np.exp(phi), axis=(2,3))/\
                       np.sum(oi['MODEL']['image'])
        tmp['|V|'] = np.abs(V)
        tmp['V2'] = np.abs(V)**2
        tmp['PHI'] = (np.angle(V)*180/np.pi+180)%360 - 180
        oi['IM_VIS'][k] = tmp
    for k in oi['OI_FLUX'].keys():
        oi['IM_FLUX'][k] = {'FLUX':oi['OI_FLUX'][k]['FLAG']*0 +
                             np.sum(oi['MODEL']['cube'], axis=(1,2))[None,:],
                            'RFLUX':oi['OI_FLUX'][k]['FLAG']*0 +
                              np.sum(oi['MODEL']['cube'], axis=(1,2))[None,:],
                            'MJD':oi['OI_FLUX'][k]['MJD'],
                            'FLAG':oi['OI_FLUX'][k]['FLAG'],
                            }
    oi = computeT3fromVisOI(oi)
    oi = computeDiffPhiOI(oi)
    oi = computeNormFluxOI(oi)
    return oi

def VmodelOI(oi, p, fov=None, pix=None, dx=0.0, dy=0.0, timeit=False, indent=0):
    param = computeLambdaParams(p)
    if type(oi) == list:
        # -- iteration on "oi" if a list
        return [VmodelOI(o, param, fov=fov, pix=pix, dx=dx, dy=dy,
                        timeit=timeit, indent=indent) for o in oi]

    # -- split in components if needed
    comp = set([x.split(',')[0].strip() for x in param.keys() if ',' in x])
    if len(comp)==0:
        # -- assumes single component
        return VsingleOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy,
                         timeit=timeit, indent=indent+1)

    tinit = time.time()
    res = {} # -- contains result
    t0 = time.time()
    # -- modify parameters list to handle bandwidth smearing
    """
    based on x,y, decide if smearing is necessary for a component.
    the idea is to duplicate parameters and encoding in the name the _dwl and _ffrac

    the half size of fringe packet is ~ wl**2/delta_lambda = lambda*R

    a component "sep" (mas) away has fringe offset of B*sep*pi/180/3600/1000 (m)

    so, B*sep*pi/180/3600/1000 ~= lambda*R -> we have complete smearing
    if sep > lambda*R/B*180*3600*1000/pi ~ 206.26*lambda*R/Bmax
        for sep in mas, lambda in um and B in m (R has no units)

    Smearing will only be on Vcomplex, so will not work well for V2, because:

    observed V2 = smeared(V**2) != (smeared(V))**2


    """
    D = {'|V|':('OI_VIS', 'B/wl'),
         'DPHI':('OI_VIS', 'B/wl'),
         'V2':('OI_VIS2', 'B/wl'),
         'T3PHI':('OI_T3', 'Bmax/wl'),
         'T3AMP':('OI_T3', 'Bmax/wl'),
        }
    assert 'fit' in oi, "'fit' should be defined!"
    # -- in m/um
    _bwlmax = 0.
    for e in oi['fit']['obs']:
        if e in D:
            for b in oi[D[e][0]].keys():
                _bwlmax = max(_bwlmax, np.max(oi[D[e][0]][b][D[e][1]]))
    R = np.mean(oi['WL'])/np.mean(np.diff(oi['WL']))
    _sep = 206.26*R/_bwlmax

    tmp = {k:param[k] for k in param.keys() if not ',' in k}
    smearing = {}
    for c in comp:
        if c+',x' in param.keys() and c+',y' in param.keys():
            sep = np.sqrt(param[c+',x']**2 + param[c+',y']**2)
        else:
            sep = 0.
        # -- n larger than need be to be on the safe side
        n = 2*int(2*sep/_sep)+1
        smearing[c] = n
        if n<2:
            tmp.update({k:param[k] for k in param.keys() if k.startswith(c+',')})
        else:
            kz = filter(lambda x: x.startswith(c+','), param.keys())
            dwl = np.diff(oi['WL']).mean()/(n+1)
            # -- duplicate parameters with wl offset and flux weighing in the key
            for k in kz:
                for x in np.linspace(-1,1,n+2)[1:-1]:
                    tmp[c+'&dwl%f&ffrac%f,'%(x*dwl, 1/n)+k.split(',')[1]] = param[k]
    if len(tmp)>len(param):
        param = tmp
        comp = set([x.split(',')[0].strip() for x in param.keys() if ',' in x])
    if timeit:
        print(' '*indent+'VmodelOI > smearing %.3fms'%(1000*(time.time()-t0)))

    t0 = time.time()
    for c in comp:
        tc = time.time()
        # -- this component.
        _param = {k.split(',')[1].strip():param[k] for k in param.keys() if
                  k.startswith(c+',')}
        # -- Assumes all param without ',' are common to all
        _param.update({k:param[k] for k in param.keys() if not ',' in k})
        if '&dwl' in c and '&ffrac' in c:
            _dwl = float(c.split('&dwl')[1].split('&ffrac')[0])
            _ffrac = float(c.split('&ffrac')[1])
        else:
            _dwl, _ffrac = 0., 1.0
        if res=={}:
            # -- initialise
            res = VsingleOI(oi, _param, fov=fov, pix=pix, dx=dx, dy=dy,
                            timeit=timeit, indent=indent+1, noT3=True,
                            _dwl=_dwl, _ffrac=_ffrac)
            if 'image' in res['MODEL'].keys():
                # -- for this component
                res['MODEL'][c+',image'] = res['MODEL']['image']
                # -- smearing
                if '&dwl' in c:
                    res['MODEL'][c.split('&dwl')[0]+',image'] = res['MODEL']['image']

                # -- total
                res['MODEL']['cube'] = res['MODEL']['image'][None,:,:]*res['MODEL']['totalflux'][:,None,None]
                res['MODEL']['image'] *= np.mean(res['MODEL']['totalflux'])
            # -- for this component:
            res['MODEL'][c+',flux'] = res['MODEL']['totalflux'].copy()
            res['MODEL'][c+',negativity'] = res['MODEL']['negativity']
            # -- smearing
            if '&dwl' in c:
                res['MODEL'][c.split('&dwl')[0]+',flux'] = res['MODEL']['totalflux'].copy()
                res['MODEL'][c.split('&dwl')[0]+',negativity'] = res['MODEL']['negativity']

            # -- total complex visibility
            res['MOD_VIS'] = {}
            for k in res['OI_VIS'].keys(): # for each baseline
                res['MOD_VIS'][k] = res['MODEL']['totalflux'][None,:]*res['OI_VIS'][k]['|V|']*\
                                 np.exp(1j*np.pi*res['OI_VIS'][k]['PHI']/180)
            m = {}
        else:
            # -- combine model with others
            m = VsingleOI(oi, _param, fov=fov, pix=pix, dx=dx, dy=dy,
                        timeit=timeit, indent=indent+1, noT3=True,
                        _dwl=_dwl, _ffrac=_ffrac)
            if 'image' in m['MODEL'].keys():
                res['MODEL'][c+',image'] = m['MODEL']['image']
                if '&dwl' in c:
                    _c = c.split('&dwl')[0]
                    if _c+',image' in res['MODEL']:
                        res['MODEL'][_c+',image'] += m['MODEL']['image']
                    else:
                        res['MODEL'][_c+',image'] = m['MODEL']['image']
                res['MODEL']['image'] += np.mean(m['MODEL']['totalflux'])*\
                                        m['MODEL']['image']
                res['MODEL']['cube'] += m['MODEL']['image'][None,:,:]*\
                                        m['MODEL']['totalflux'][:,None,None]
            res['MODEL'][c+',flux'] = m['MODEL']['totalflux']
            res['MODEL'][c+',negativity'] = m['MODEL']['negativity']
            if '&dwl' in c:
                _c = c.split('&dwl')[0]
                if _c+',flux' in res['MODEL']:
                    res['MODEL'][_c+',flux'] += m['MODEL']['totalflux']
                else:
                    res['MODEL'][_c+',flux'] = m['MODEL']['totalflux']

            res['MODEL']['totalflux'] += m['MODEL']['totalflux']
            res['MODEL']['negativity'] += m['MODEL']['negativity']
            # -- total complex Visibility
            for k in res['OI_VIS'].keys(): # for each baseline
                res['MOD_VIS'][k] += m['MODEL']['totalflux'][None,:]*m['OI_VIS'][k]['|V|']*\
                                np.exp(1j*np.pi*m['OI_VIS'][k]['PHI']/180)
        if timeit:
            print(' '*indent+'VmodelOI > VsingleOI "%s" %.3fms'%(c, 1000*(time.time()-tc)))

    t0 = time.time()
    # -- normalise by total flux, compute OI_VIS and OI_VIS2
    for k in res['MOD_VIS'].keys():
        res['MOD_VIS'][k] /= res['MODEL']['totalflux'][None,:]
        if 'OI_VIS' in res and k in res['OI_VIS']:
            res['OI_VIS'][k]['|V|'] = np.abs(res['MOD_VIS'][k])
            res['OI_VIS'][k]['PHI'] = (np.angle(res['MOD_VIS'][k])*180/np.pi+180)%360-180
        if 'OI_VIS2' in res and k in res['OI_VIS2']:
            res['OI_VIS2'][k]['V2'] = np.abs(res['MOD_VIS'][k])**2

    res['OI_FLUX'] = {}
    if 'OI_FLUX' in oi.keys():
        for k in oi['OI_FLUX'].keys():
            res['OI_FLUX'][k] = {'FLUX': oi['OI_FLUX'][k]['FLUX']*0 +
                                 res['MODEL']['totalflux'][None,:],
                                 'RFLUX': oi['OI_FLUX'][k]['FLUX']*0 +
                                   res['MODEL']['totalflux'][None,:],
                                 'EFLUX': oi['OI_FLUX'][k]['FLUX']*0,
                                 'FLAG': oi['OI_FLUX'][k]['FLAG'],
                                 'MJD': oi['OI_FLUX'][k]['MJD'],
            }
    if timeit:
        print(' '*indent+'VmodelOI > fluxes %.3fms'%(1000*(time.time()-t0)))

    t0 = time.time()
    if 'OI_T3' in oi.keys():
        res['OI_T3'] = {}
        for k in oi['OI_T3'].keys():
            res['OI_T3'][k] = {}
            for l in ['MJD', 'u1', 'u2', 'v1', 'v2', 'formula', 'FLAG', 'Bmax/wl', 'Bavg/wl']:
                res['OI_T3'][k][l] = oi['OI_T3'][k][l]
        res = computeT3fromVisOI(res)
        if timeit:
            print(' '*indent+'VmodelOI > T3 %.3fms'%(1000*(time.time()-t0)))

    t0 = time.time()
    if 'fit' in oi and 'obs' in oi['fit'] and 'DPHI' in oi['fit']['obs']:
        res = computeDiffPhiOI(res, param)
        if timeit:
            print(' '*indent+'VmodelOI > dPHI %.3fms'%(1000*(time.time()-t0)))
            t0 = time.time()
    if 'fit' in oi and 'obs' in oi['fit'] and 'NFLUX' in oi['fit']['obs']:
        res = computeNormFluxOI(res, param)
        if timeit:
            print(' '*indent+'VmodelOI > normFlux %.3fms'%(1000*(time.time()-t0)))

    for k in['telescopes', 'baselines', 'triangles']:
        if k in oi:
            res[k] = oi[k]
    res['param'] = computeLambdaParams(param)

    t0 = time.time()
    if 'fit' in res and 'spec res pix' in res['fit']:
        # -- convolve by spectral Resolution
        N = 2*int(2*oi['fit']['spec res pix'])+3
        x = np.arange(N)
        ker = np.exp(-(x-np.mean(x))**2/(2.*(oi['fit']['spec res pix']/2.355)**2))
        ker /= np.sum(ker)
        if 'NFLUX' in res.keys():
            for k in res['NFLUX'].keys():
                for i in range(res['NFLUX'][k]['NFLUX'].shape[0]):
                    res['NFLUX'][k]['NFLUX'][i] = np.convolve(
                                res['NFLUX'][k]['NFLUX'][i], ker, mode='same')
        for k in res['OI_VIS'].keys():
            for i in range(res['OI_VIS'][k]['|V|'].shape[0]):
                res['OI_VIS'][k]['|V|'][i] = np.convolve(
                            res['OI_VIS'][k]['|V|'][i], ker, mode='same')
                res['OI_VIS'][k]['PHI'][i] = np.convolve(
                            res['OI_VIS'][k]['PHI'][i], ker, mode='same')
                if 'DPHI' in res.keys():
                    res['DPHI'][k]['DPHI'][i] = np.convolve(
                                res['DPHI'][k]['DPHI'][i], ker, mode='same')
        for k in res['OI_VIS2'].keys():
            for i in range(res['OI_VIS2'][k]['V2'].shape[0]):
                res['OI_VIS2'][k]['V2'][i] = np.convolve(
                            res['OI_VIS2'][k]['V2'][i], ker, mode='same')
        for k in res['OI_T3'].keys():
            for i in range(res['OI_T3'][k]['MJD'].shape[0]):
                res['OI_T3'][k]['T3PHI'][i] = np.convolve(
                            res['OI_T3'][k]['T3PHI'][i], ker, mode='same')
                res['OI_T3'][k]['T3AMP'][i] = np.convolve(
                            res['OI_T3'][k]['T3AMP'][i], ker, mode='same')
        if timeit:
            print(' '*indent+'VmodelOI > convolve %.3fms'%(1000*(time.time()-t0)))
    if timeit:
        print(' '*indent+'VmodelOI > total %.3fms'%(1000*(time.time()-tinit)))
    res['smearing'] = smearing
    # -- TODO:

    return res

def computeDiffPhiOI(oi, param=None, order='auto'):
    if not param is None:
        _param = computeLambdaParams(param)
    else:
        _param = None
    if type(oi)==list:
        return [computeDiffPhiOI(o, _param, order) for o in oi]
    if 'param' in oi.keys() and param is None:
        _param = oi['param']

    if not 'OI_VIS' in oi.keys():
        #print('WARNING: computeDiffPhiOI, nothing to do')
        return oi

    # -- user-defined wavelength range
    fit = {'wl ranges':[(min(oi['WL']), max(oi['WL']))]}

    if not 'fit' in oi:
        oi['fit'] = fit.copy()
    elif not 'wl ranges' in oi['fit']:
        # -- weird but necessary to avoid a global 'fit'
        fit.update(oi['fit'])
        oi['fit'] = fit.copy()

    w = np.zeros(oi['WL'].shape)
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- user defined continuum
    if 'fit' in oi and 'cont ranges' in oi['fit']:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['cont ranges']:
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)

    # -- exclude where lines are in the models
    if not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in _param.keys():
                    dwl = 1.2*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in _param.keys():
                    dwl = 8*_param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-_param[k])>=dwl)

    if np.sum(w)==0:
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = oi['WL cont'].copy()

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.2)
        order = max(order, 1)

    if np.sum(oi['WL cont'])<order+1:
        print('ERROR: not enough WL to compute continuum!')
        return oi

    oi['DPHI'] = {}
    for k in oi['OI_VIS'].keys():
        data = []
        for i,phi in enumerate(oi['OI_VIS'][k]['PHI']):
            mask = w*~oi['OI_VIS'][k]['FLAG'][i,:]
            if np.sum(mask)>order:
                c = np.polyfit(oi['WL'][mask], phi[mask], order)
                data.append(phi-np.polyval(c, oi['WL']))
            else:
                data.append(phi-np.median(phi))

        data = np.array(data)
        oi['DPHI'][k] = {'DPHI':data,
                         'FLAG':oi['OI_VIS'][k]['FLAG'],
                         }
        if 'MJD' in oi['OI_VIS'][k]:
            oi['DPHI'][k]['MJD'] = oi['OI_VIS'][k]['MJD']
        if 'EPHI' in oi['OI_VIS'][k]:
            oi['DPHI'][k]['EDPHI'] = oi['OI_VIS'][k]['EPHI']

    if 'IM_VIS' in oi.keys():
        for k in oi['IM_VIS'].keys():
            data = []
            for i,phi in enumerate(oi['IM_VIS'][k]['PHI']):
                mask = w
                c = np.polyfit(oi['WL'][mask], phi[mask], order)
                data.append(phi-np.polyval(c, oi['WL']))
            data = np.array(data)
            oi['IM_VIS'][k]['DPHI'] = data
    return oi

def computeNormFluxOI(oi, param=None, order='auto'):
    if not param is None:
        _param = computeLambdaParams(param)
    else:
        _param = None

    if type(oi)==list:
        return [computeNormFluxOI(o, _param, order) for o in oi]

    if not 'OI_FLUX' in oi.keys():
        #print('WARNING: computeNormFluxOI, nothing to do')
        return oi

    if 'param' in oi.keys() and param is None:
        _param = oi['param']

    # -- user defined wavelength range
    w = oi['WL']>0
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- user defined continuum
    if 'fit' in oi and 'cont ranges' in oi['fit']:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['cont ranges']:
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)

    # -- exclude where lines are in the models
    if not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in _param.keys():
                    dwl = 1.2*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in _param.keys():
                    dwl = 8*_param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-_param[k])>=dwl)


    if np.sum(w)==0:
        #print('WARNING: no continuum! using all wavelengths')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.15)
        order = max(order, 1)

    if np.sum(oi['WL cont'])<order+1:
        print('ERROR: not enough WL to compute continuum!')
        return oi

    oi['NFLUX'] = {}
    # -- normalize flux in the data:
    for k in oi['OI_FLUX'].keys():
        data = []
        edata = []
        for i, flux in enumerate(oi['OI_FLUX'][k]['FLUX']):
            mask = w*~oi['OI_FLUX'][k]['FLAG'][i,:]
            # -- continuum
            if np.sum(mask)>order:
                c = np.polyfit(oi['WL'][w], flux[w], order)
                data.append(flux/np.polyval(c, oi['WL']))
            else:
                data.append(flux/np.median(flux))

            #edata.append(oi['OI_FLUX'][k]['EFLUX'][i]/np.polyval(c, oi['WL']))
            # -- err normalisation cannot depend on the mask nor continuum calculation!
            edata.append(oi['OI_FLUX'][k]['EFLUX'][i]/np.median(flux))

        data = np.array(data)
        edata = np.array(edata)
        oi['NFLUX'][k] = {'NFLUX':data,
                         'ENFLUX':edata,
                         'FLAG':oi['OI_FLUX'][k]['FLAG'],
                         'MJD':oi['OI_FLUX'][k]['MJD']}

    # -- flux computed from image cube
    if 'IM_FLUX' in oi.keys():
        for k in oi['IM_FLUX'].keys():
            data = []
            for i,flux in enumerate(oi['IM_FLUX'][k]['FLUX']):
                mask = w*~oi['IM_FLUX'][k]['FLAG'][i,:]
                # -- continuum
                if np.sum(mask)>order:
                    c = np.polyfit(oi['WL'][w], flux[w], order)
                    data.append(flux/np.polyval(c, oi['WL']))
                else:
                    data.append(flux/np.median(flux))
            data = np.array(data)
            oi['IM_FLUX'][k]['NFLUX'] = data

    if 'MODEL' in oi.keys() and 'totalflux' in oi['MODEL'].keys():
        mask = w
        c = np.polyfit(oi['WL'][w], oi['MODEL']['totalflux'][w], order)
        for k in list(oi['MODEL'].keys()):
            if k.endswith(',flux'):
                oi['MODEL'][k.replace(',flux', ',nflux')] = \
                    oi['MODEL'][k]/np.polyval(c, oi['WL'])
        oi['MODEL']['totalnflux'] = oi['MODEL']['totalflux']/np.polyval(c,oi['WL'])
    return oi

def computeLambdaParams(params):
    paramsI = params.copy()
    paramsR = {}
    loop = True
    nloop = 0
    s = '$' # special character to identify keywords
    while loop and nloop<10:
        loop = False
        for k in paramsI.keys():
            if type(paramsI[k])==str:
                # -- allow parameter to be expression of others
                tmp = paramsI[k]
                compute = False
                for _k in paramsI.keys():
                    if s+_k in paramsI[k]:
                        tmp = tmp.replace(s+_k, '('+str(paramsI[_k])+')')
                        if not s in tmp:
                            # -- no more replacement
                            compute = True
                # -- are there still un-computed parameters?
                for _k in paramsI.keys():
                    if s+_k in tmp:
                        # -- set function to another loop
                        loop = True
                        paramsI[k] = tmp
                if compute and not loop:
                    paramsR[k] = eval(tmp)
                else:
                    paramsR[k] = tmp
            else:
                paramsR[k] = paramsI[k]
        nloop+=1

    assert nloop<10, 'too many recurences in evaluating parameters!'+str(paramsI)
    return paramsR

def computeT3fromVisOI(oi):
    """
    oi => from oifits.loadOI()

    assumes OI contains the complex visibility as OI_VIS

    - no errors are propagated (used for modeling purposes)
    - be careful: does it in place (original OI_T3 data are erased)
    """
    if type(oi) == list:
        return [computeT3fromVisOI(o) for o in oi]
    if 'OI_T3' in oi.keys():
        for k in oi['OI_T3'].keys():
            s, t, w0, w1, w2 = oi['OI_T3'][k]['formula']
            if np.isscalar(s[0]):
                oi['OI_T3'][k]['T3PHI'] = s[0]*oi['OI_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1]*oi['OI_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2]*oi['OI_VIS'][t[2]]['PHI'][w2,:]
            else:
                oi['OI_T3'][k]['T3PHI'] = s[0][:,None]*oi['OI_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1][:,None]*oi['OI_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2][:,None]*oi['OI_VIS'][t[2]]['PHI'][w2,:]

            # -- force -180 -> 180 degrees
            oi['OI_T3'][k]['T3PHI'] = (oi['OI_T3'][k]['T3PHI']+180)%360-180
            oi['OI_T3'][k]['ET3PHI'] = np.zeros(oi['OI_T3'][k]['T3PHI'].shape)
            oi['OI_T3'][k]['T3AMP'] = np.abs(oi['OI_VIS'][t[0]]['|V|'][w0,:])*\
                                      np.abs(oi['OI_VIS'][t[1]]['|V|'][w1,:])*\
                                      np.abs(oi['OI_VIS'][t[2]]['|V|'][w2,:])
            oi['OI_T3'][k]['ET3AMP'] = np.zeros(oi['OI_T3'][k]['T3AMP'].shape)
    if 'IM_VIS' in oi.keys():
        oi['IM_T3'] = {}
        for k in oi['OI_T3'].keys():
            # -- inherit flags from Data,
            oi['IM_T3'][k] = {'FLAG':oi['OI_T3'][k]['FLAG'],
                               'MJD':oi['OI_T3'][k]['MJD'],
                               'Bmax/wl':oi['OI_T3'][k]['Bmax/wl'],
                               'Bavg/wl':oi['OI_T3'][k]['Bavg/wl'],
                               }
            s, t, w0, w1, w2 = oi['OI_T3'][k]['formula']
            if np.isscalar(s[0]):
                oi['IM_T3'][k]['T3PHI'] = s[0]*oi['IM_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1]*oi['IM_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2]*oi['IM_VIS'][t[2]]['PHI'][w2,:]
            else:
                oi['IM_T3'][k]['T3PHI'] = s[0][:,None]*oi['IM_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1][:,None]*oi['IM_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2][:,None]*oi['IM_VIS'][t[2]]['PHI'][w2,:]

            # -- force -180 -> 180 degrees
            oi['IM_T3'][k]['T3PHI'] = (oi['IM_T3'][k]['T3PHI']+180)%360-180

            oi['IM_T3'][k]['T3AMP'] = np.abs(oi['IM_VIS'][t[0]]['|V|'][w0,:])*\
                                       np.abs(oi['IM_VIS'][t[1]]['|V|'][w1,:])*\
                                       np.abs(oi['IM_VIS'][t[2]]['|V|'][w2,:])
    return oi


def testTelescopes(k, telescopes):
    """
    k: a telescope, baseline or triplet (str)
        eg: 'A0', 'A0G1', 'A0G1K0' etc.
    telescopes: single or list of telescopes (str)
        eg: 'G1', ['G1', 'A0'], etc.

    returns True if any telescope in k

    assumes all telescope names have same length!
    """
    if type(telescopes)==str:
        telescopes = [telescopes]
    test = False
    for t in telescopes:
        test = test or (t in k)
    return test

def testBaselines(k, baselines):
    """
    k: a telescope, baseline or triplet (str)
        eg: 'A0', 'A0G1', 'A0G1K0' etc.
    baselines: single or list of baselines (str)
        eg: 'G1A0', ['G1A0', 'A0C1'], etc.

    returns True if any baseline in k
        always False for telescopes
        order of telescope does not matter

    assumes all telescope names have same length!
    """
    if type(baselines)==str:
        baselines = [baselines]
    test = False
    for b in baselines:
        test = test or (b[:len(b)//2] in k and b[len(b)//2:] in k)
    return test

def computePrior(param, prior):
    res = []
    for p in prior.keys():
        form = p+''
        val = str(prior[p][1])+''
        for i in range(3):
            for k in param.keys():
                if k in form:
                    form = form.replace(k, str(param[k]))
                if k in val:
                    val = val.replace(k, str(param[k]))
        # -- residual
        resi = '('+form+'-'+str(val)+')/abs('+str(prior[p][2])+')'
        if prior[p][0]=='<' or prior[p][0]=='<=' or prior[p][0]=='>' or prior[p][0]=='>=':
            resi = '%s if 0'%resi+prior[p][0]+'%s else 0'%resi
        #print(resi)
        res.append(eval(resi))
    return res

def residualsOI(oi, param, timeit=False):
    """
    assumes dict OI has a key "fit" which list observable to fit:

    OI['fit']['obs'] is a list containing '|V|', 'PHI', 'DPHI', 'V2', 'T3AMP', 'T3PHI'
    OI['fit'] can have key "wl ranges" to limit fit to [(wlmin1, wlmax1), (), ...]

    """
    tt = time.time()
    res = np.array([])

    if type(oi)==list:
        for i,o in enumerate(oi):
            res = np.append(res, residualsOI(o, param, timeit=timeit))
        return res

    if 'fit' in oi:
        fit = oi['fit']
    else:
        fit = {'obs':[]}

    t0 = time.time()
    if 'DPHI' in fit['obs']:
        oi = computeDiffPhiOI(oi, param)
        if timeit:
            print('residualsOI > dPHI %.3fms'%(1000*(time.time()-t0)))
            t0 = time.time()
    if 'NFLUX' in fit['obs']:
        oi = computeNormFluxOI(oi, param)
        if timeit:
            print('residualsOI > normFlux %.3fms'%(1000*(time.time()-t0)))

    if 'ignore telescope' in fit:
        ignoreTelescope = fit['ignore telescope']
    else:
        ignoreTelescope = 'no way they would call a telescope that!'

    if 'ignore baseline' in fit:
        ignoreBaseline = fit['ignore baseline']
    else:
        ignoreBaseline = 'no way they would call a baseline that!'

    # -- compute model
    t0 = time.time()
    m = VmodelOI(oi, param, timeit=timeit)
    if timeit:
        print('residualsOI > VmodelOI: %.3fms'%(1000*(time.time()-t0)))

    ext = {'|V|':'OI_VIS',
            'PHI':'OI_VIS',
            'DPHI':'DPHI',
            'V2':'OI_VIS2',
            'T3AMP':'OI_T3',
            'T3PHI':'OI_T3',
            'NFLUX':'NFLUX', # flux normalized to continuum
            'FLUX':'OI_FLUX' # flux, corrected from tellurics
            }
    w = np.ones(oi['WL'].shape)

    if 'wl ranges' in fit:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    w = np.bool_(w)
    t0 = time.time()
    for f in fit['obs']:
        # -- for each observable:
        if f in ext.keys():
            if 'PHI' in f:
                rf = lambda x: ((x + 180)%360 - 180)
            else:
                rf = lambda x: x
            # -- for each telescope / baseline / triangle
            #print(f, end=' ')
            for k in oi[ext[f]].keys():
                test = testTelescopes(k, ignoreTelescope) or testBaselines(k, ignoreBaseline)
                if not test:
                    mask = w[None,:]*~oi[ext[f]][k]['FLAG']
                    err = oi[ext[f]][k]['E'+f].copy()
                    if 'max error' in oi['fit'] and f in oi['fit']['max error']:
                        # -- ignore data with large error bars
                        mask *= (err<oi['fit']['max error'][f])
                    if 'max relative error' in oi['fit'] and f in oi['fit']['max relative error']:
                        # -- ignore data with large error bars
                        mask *= (err<(oi['fit']['max relative error'][f]*
                                        np.abs(oi[ext[f]][k][f])))
                    if 'mult error' in fit and f in oi['fit']['mult error']:
                        # -- force error to a minimum value
                        err *= oi['fit']['mult error'][f]
                    if 'min error' in fit and f in oi['fit']['min error']:
                        # -- force error to a minimum value
                        err = np.maximum(oi['fit']['min error'][f], err)
                    if 'min relative error' in fit and f in oi['fit']['min relative error']:
                        # -- force error to a minimum value
                        err = np.maximum(oi['fit']['min relative error'][f]*
                                         np.abs(oi[ext[f]][k][f]), err)

                    tmp = rf(oi[ext[f]][k][f][mask] -
                             m[ext[f]][k][f][mask])/err[mask]
                    res = np.append(res, tmp.flatten())
                else:
                    pass
                    #print('ignoring', ext[f], k)
            #print('')
        else:
            print('WARNING: unknown observable:', f)
    if timeit:
        print('residualsOI > "res": %.3fms'%(1000*(time.time()-t0)))
        print('residualsOI > total: %.3fms'%(1000*(time.time()-tt)))
        print('-'*30)
    res = np.append(res, m['MODEL']['negativity']*len(res))
    if 'fit' in oi and 'prior' in oi['fit']:
        res = np.append(res, computePrior(param, oi['fit']['prior']))
    #print(len(res))
    return res

def fitOI(oi, firstGuess, fitOnly=None, doNotFit=None, verbose=2,
          maxfev=5000, ftol=1e-6, follow=None, prior=None,
          randomise=False, iter=-1, obs=None):
    """
    oi: a dict of list of dicts returned by oifits.loadOI

    firstGuess: a dict of parameters describring the model. can be numerical
        values or a formula in a string self-referencing to parameters. e.g.
        {'a':1, 'b':2, 'c':'3*a+b/2'}
        Parsing is very simple, parameters name should be ambiguous! for example
        using {'s':1, 'r':2, 'c':'sqrt(s*r)'} will fail, because 'c' will be
        interpreted as '1q2t(1*2)'

    fitOnly: list of parameters to fit. default is all of them

    doNotFit: list of parameters not to fit. default is none of them

    verbose: 0, 1 or 2 for various level of verbosity

    maxfev: maximum number of function evaluation (2000)

    ftol: stopping constraint for chi2 change (1e-4)

    follow: list of parameters to print in screen (for verbose>0)

    randomise: for bootstrapping

    prior: dict of priors.
        {'a+b':('=', 1.2, 0.1)} if a and b are parameters. a+b = 1.2 +- 0.1
        {'a':('<', 'sqrt(b)', 0.1)} if a and b are parameters. a<sqrt(b), and
            the penalty is 0 for a==sqrt(b) but rises significantly
            for a>sqrt(b)+0.1
    """
    if not prior is None:
        if type(oi)==dict:
            if 'fit' in oi:
                oi['fit']['prior'] = prior
            else:
                oi['fit'] = {'prior':prior}
        else:
            for o in oi:
                if 'fit' in o:
                    o['fit']['prior'] = prior
                else:
                    o['fit'] = {'prior':prior}

    if not obs is None:
        if type(oi)==dict:
            if 'fit' in oi:
                oi['fit']['obs'] = obs
            else:
                oi['fit'] = {'obs':obs}
        else:
            for o in oi:
                if 'fit' in o:
                    o['fit']['obs'] = obs
                else:
                    o['fit'] = {'obs':obs}

    if randomise is False:
        #print('all data')
        tmp = oi
    else:
        #print('randomise')
        tmp = randomiseData2(oi,  verbose=False)
    z = 0.0
    if fitOnly is None and doNotFit is None:
        fitOnly = list(firstGuess.keys())
    fit = dpfit.leastsqFit(residualsOI, tmp, firstGuess, z, verbose=verbose,
                           maxfev=maxfev, ftol=ftol, fitOnly=fitOnly,
                           doNotFit=doNotFit, follow=follow)
    fit['prior'] = prior
    return fit

def randomiseData2(oi, verbose=False):
    """
    based on "configurations per MJD". Basically draw data from MJDs and/or
    configuration:
    1) have twice as many oi's, with either:
        - half of the MJDs ignored
        - half of the config, for each MJD, ignored

    oi: list of data, from loadOI
    """
    res = []

    # -- build a dataset twice as big as original one
    for i in range(len(oi)*2):
        tmp = copy.deepcopy(oi[i%len(oi)])
        # -- exclude half of the spectral data vectors:
        mjd_t = []
        for k in tmp['configurations per MJD'].keys():
            mjd_t.extend([str(k)+str(c) for c in tmp['configurations per MJD'][k]])
        random.shuffle(mjd_t)
        ignore = mjd_t[:len(mjd_t)//2]

        exts = filter(lambda x: x in ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX'], tmp.keys())
        for l in exts:
            if list(tmp[l].keys()) == ['all']:
                for i,mjd in enumerate(tmp[l]['all']['MJD']):
                    if str(mjd)+tmp[l]['all']['NAME'][i] in ignore:
                        tmp[l]['all']['FLAG'][i,:] = True
            else:
                for k in tmp[l].keys():
                    for i,mjd in enumerate(tmp[l][k]['MJD']):
                        if str(mjd)+str(k) in ignore:
                            tmp[l][k]['FLAG'][i,:] = True
        res.append(tmp)
    return res

def randomiseData(oi, randomise='telescope or baseline', P=None, verbose=False):
    # -- make new sample of data
    if P is None:
        # -- evaluate reduction in amount of data
        Nall, Nt, Nb = 0, 0, 0
        for o in oi:
            for x in o['fit']['obs']:
                if x in ['NFLUX']:
                    Nall += len(o['telescopes'])
                    Nt += len(o['telescopes']) - 1
                    Nb += len(o['telescopes'])
                if x in ['|V|', 'DPHI', 'V2']:
                    Nall += len(o['baselines'])
                    Nt += (len(o['telescopes'])-1)*(len(o['telescopes'])-2)//2
                    Nb += len(o['baselines']) - 1
                if x in ['T3PHI']:
                    Nall += len(o['OI_T3'])
                    Nt += (len(o['telescopes'])-2)*(len(o['telescopes'])-3)//2
                    Nb += len(o['OI_T3']) - (len(o['telescopes'])-2)

        if str(randomise).lower()=='telescope or baseline':
            P = int(round(len(oi) * 2*Nall/(Nb+Nt)))
        elif 'telescope' in str(randomise).lower():
            P = int(round(len(oi) * Nall/Nt))
        elif 'baseline' in str(randomise).lower():
            P = int(round(len(oi) * Nall/Nb))
        else:
            P = len(oi)
    tmp = []
    # -- build new dataset of length P
    for k in range(P):
        i = np.random.randint(len(oi))
        tmp.append(copy.deepcopy(oi[i]))
        # -- ignore part of the data
        if str(randomise).lower()=='telescope or baseline':
            if np.random.rand()<0.5:
                j = np.random.randint(len(tmp[-1]['telescopes']))
                tmp[-1]['fit']['ignore telescope'] = tmp[-1]['telescopes'][j]
                tmp[-1]['fit']['ignore baseline'] = 'not a baseline name'
                if verbose:
                    print('   ', i, 'ignore telescope', tmp[-1]['telescopes'][j])
            else:
                j = np.random.randint(len(tmp[-1]['baselines']))
                tmp[-1]['fit']['ignore telescope'] = 'not a telescope name'
                tmp[-1]['fit']['ignore baseline'] = tmp[-1]['baselines'][j]
                if verbose:
                    print('   ', i, 'ignore baseline', tmp[-1]['baselines'][j])
        elif 'telescope' in str(randomise).lower():
            j = np.random.randint(len(tmp[-1]['telescopes']))
            tmp[-1]['fit']['ignore telescope'] = tmp[-1]['telescopes'][j]
            tmp[-1]['fit']['ignore baseline'] = 'not a baseline name'
            if verbose:
                print('   ', i, 'ignore telescope', tmp[-1]['telescopes'][j])
        elif 'baseline' in str(randomise).lower():
            j = np.random.randint(len(tmp[-1]['baselines']))
            tmp[-1]['fit']['ignore telescope'] = 'not a telescope name'
            tmp[-1]['fit']['ignore baseline'] = tmp[-1]['baselines'][j]
            if verbose:
                print('   ', i, 'ignore baseline', tmp[-1]['baselines'][j])
    return tmp

def get_processor_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        return subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip().decode()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        return subprocess.check_output(command, shell=True).strip().decode()
    return "unknown processor"

def bootstrapFitOI(oi, fit, N=None, fitOnly=None, doNotFit=None, maxfev=5000,
                   ftol=1e-6, sigmaClipping=4.5, multi=True, prior=None):
    """
    """
    if N is None:
        # count number of spectral vector data
        N = 0
        ext = {'|V|':'OI_VIS',
                'PHI':'OI_VIS',
                'DPHI':'OI_VIS',
                'V2':'OI_VIS2',
                'T3AMP':'OI_T3',
                'T3PHI':'OI_T3',
                'NFLUX':'NFLUX', # flux normalized to continuum
                'FLUX':'OI_FLUX' # flux, corrected from tellurics
                }
        if type(oi)==dict:
            oi = [oi]
        for o in oi:
            for p in o['fit']['obs']:
                if ext[p] in o:
                    for k in o[ext[p]].keys():
                        N += len(o[ext[p]][k]['MJD'])
        N *= 2

    fitOnly = fit['fitOnly']
    doNotFit = fit['doNotFit']
    maxfev = fit['maxfev']
    ftol = fit['ftol']/10
    prior = fit['prior']
    firstGuess = fit['best']

    kwargs = {'maxfev':maxfev, 'ftol':ftol, 'verbose':False,
              'fitOnly':fitOnly, 'doNotFit':doNotFit,
              'randomise':True, 'prior':prior, 'iter':-1}

    res = []
    if multi:
        if type(multi)!=int:
            Np = min(multiprocessing.cpu_count(), N)
        else:
            Np = min(multi, N)
        print('running', N, 'fits...')
        # -- estimate fitting time by running 'Np' fit in parallel
        t = time.time()
        pool = multiprocessing.Pool(Np)
        for i in range(min(Np, N)):
            kwargs['iter'] = i
            res.append(pool.apply_async(fitOI, (oi, firstGuess, ), kwargs))
        pool.close()
        pool.join()
        res = [r.get(timeout=1) for r in res]
        print('one fit takes ~%.2fs using %d threads'%(
                (time.time()-t)/min(Np, N), Np))

        # -- run the remaining
        if N>Np:
            pool = multiprocessing.Pool(Np)
            for i in range(max(N-Np, 0)):
                kwargs['iter'] = Np+i
                res.append(pool.apply_async(fitOI, (oi,firstGuess, ), kwargs))
            pool.close()
            pool.join()
            res = res[:Np]+[r.get(timeout=1) for r in res[Np:]]
    else:
        Np = 1
        t = time.time()
        res.append(fitOI(oi, firstGuess, **kwargs))
        print('one fit takes ~%.2fs'%(time.time()-t))
        for i in range(N-1):
            kwargs['iter'] = i
            if i%10==0:
                print('%s | bootstrap fit %d/%d'%(time.asctime(), i, N-1))
            res.append(fitOI(oi, firstGuess, **kwargs))
    print('it took %.1fs, %.2fs per fit on average'%(time.time()-t,
                                                    (time.time()-t)/N))
    try:
        res = analyseBootstrap(res, sigmaClipping=sigmaClipping)
        res['fit to all data'] = fit
        return res
    except:
        print('analysing bootstrap failed! returning all fits')
        return res

def analyseBootstrap(Boot, sigmaClipping=4.5, verbose=2):
    """
    Boot: a list of fits (list of dict from dpfit.leastsqFit)
    """
    # -- allow to re-analyse bootstrapping results:
    if type(Boot) == dict and 'all fits' in Boot.keys():
        fit = Boot['fit to all data']
        Boot = Boot['all fits']
    else:
        fit = None

    res = {'best':{}, 'uncer':{}, 'fitOnly':Boot[0]['fitOnly'],
           'all best':{}, 'all best ignored':{}, 'sigmaClipping':sigmaClipping,
           'all fits':Boot}
    if not fit is None:
        res['fit to all data'] = fit
    mask = np.ones(len(Boot), dtype=bool)
    # -- sigma clipping and global mask
    if not sigmaClipping is None:
        for k in res['fitOnly']:
            tmp = np.ones(len(Boot), dtype=bool)
            for j in range(3): # iterate a few times
                x = np.array([b['best'][k] for b in Boot])
                res['best'][k] = np.median(x[tmp])
                res['uncer'][k] = np.percentile(x[tmp], 84) - np.percentile(x[tmp], 16)
                tmp = np.abs(x-res['best'][k])<=sigmaClipping*res['uncer'][k]
            mask *= tmp

    for k in res['fitOnly']:
        for j in range(3):
            x = np.array([b['best'][k] for b in Boot])
            res['best'][k] = np.mean(x[mask])
            res['uncer'][k] = np.std(x[mask])
        res['all best'][k] = x[mask]
        res['all best ignored'][k] = x[~mask]
    for k in Boot[0]['best'].keys():
        if not k in res['best'].keys():
            res['best'][k] = Boot[0]['best'][k]
            res['uncer'][k] = 0.0

    res['mask'] = mask

    M = [[b['best'][k] for i,b in enumerate(Boot) if mask[i]] for k in res['fitOnly']]

    if len(res['fitOnly'])>1:
        res['cov'] = np.cov(M)
        cor = np.sqrt(np.diag(res['cov']))
        cor = cor[:,None]*cor[None,:]
    else:
        res['cov'] = np.array([[np.std(M)**2]])
        cor = np.array([[np.sqrt(res['cov'])]])

    res['cor'] = res['cov']/cor
    res['covd'] = {ki:{kj:res['cov'][i,j] for j,kj in enumerate(res['fitOnly'])}
                   for i,ki in enumerate(res['fitOnly'])}
    res['cord'] = {ki:{kj:res['cor'][i,j] for j,kj in enumerate(res['fitOnly'])}
                   for i,ki in enumerate(res['fitOnly'])}
    if verbose:
        if not sigmaClipping is None:
            print('using %d fits out of %d (sigma clipping %.2f)'%(
                    np.sum(mask), len(Boot), sigmaClipping))
        ns = max([len(k) for k in res['best'].keys()])
        print('{', end='')
        for k in sorted(res['best'].keys()):
            if res['uncer'][k]>0:
                n = int(np.ceil(-np.log10(res['uncer'][k])+1))
                fmt = '%.'+'%d'%n+'f'
                print("'"+k+"'%s:"%(' '*(ns-len(k))), fmt%res['best'][k], end=', ')
                print('# +/-', fmt%res['uncer'][k])
            else:
                print("'"+k+"'%s:"%(' '*(ns-len(k))), end='')
                if type(res['best'][k])==str:
                    print("'"+res['best'][k]+"'")
                else:
                    print(res['best'][k])
        print('}')
    if verbose>1 and np.size(res['cov'])>1:
        dpfit.dispCor(res)
    return res

def sigmaClippingOI(oi, sigma=4, n=5, param=None):
    if type(oi)==list:
        return [sigmaClippingOI(o, sigma=sigma, n=n, param=param) for o in oi]

    w = oi['WL']>0
    # -- user-defined wavelength ranges
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- user defined continuum
    if 'fit' in oi and 'cont ranges' in oi['fit']:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['cont ranges']:
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)

    # -- exclude where lines are in the models
    if not param is None:
        for k in param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in param.keys():
                    dwl = 1.2*param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in param.keys():
                    dwl = 8*param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-param[k])>=dwl)
    if np.sum(w)==0:
        #print('WARNING: no continuum! using all wavelengths')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    # -- do sigma clipping in the continuum
    w = oi['WL mask']*oi['WL cont']
    for k in oi['OI_FLUX'].keys():
        for i in range(len(oi['OI_FLUX'][k]['MJD'])):
            oi['OI_FLUX'][k]['FLUX'][i,w] = _sigmaclip(
                        oi['OI_FLUX'][k]['FLUX'][i,w], s=sigma)
    for k in oi['OI_VIS'].keys():
        for i in range(len(oi['OI_VIS'][k]['MJD'])):
            oi['OI_VIS'][k]['|V|'][i,w] = _sigmaclip(
                        oi['OI_VIS'][k]['|V|'][i,w], s=sigma)
            oi['OI_VIS'][k]['PHI'][i,w] = _sigmaclip(
                        oi['OI_VIS'][k]['PHI'][i,w], s=sigma)
            oi['OI_VIS2'][k]['V2'][i,w] = _sigmaclip(
                        oi['OI_VIS2'][k]['V2'][i,w], s=sigma)
    for k in oi['OI_T3'].keys():
        for i in range(len(oi['OI_T3'][k]['MJD'])):
            oi['OI_T3'][k]['T3PHI'][i,w] = _sigmaclip(
                        oi['OI_T3'][k]['T3PHI'][i,w], s=sigma)
            oi['OI_T3'][k]['T3AMP'][i,w] = _sigmaclip(
                        oi['OI_T3'][k]['T3AMP'][i,w], s=sigma)
    return oi

def _sigmaclip(x, s=4.0, n=3, maxiter=5):
    N, res = len(x), np.copy(x)
    nc = True
    iter = 0
    while nc and iter<maxiter:
        nc = 0
        c = np.polyfit(np.arange(N), res, 1)
        c = [0]
        std = (np.percentile(res-np.polyval(c, np.arange(N)), 84) -
               np.percentile(res-np.polyval(c, np.arange(N)), 16))/2.
        med = np.median(res-np.polyval(c, np.arange(N)))
        for i in range(N):
            #med = np.median(res[max(0,i-n):min(N-1, i+n)])
            if abs(med-res[i])>s*std:
                res[i] = med + np.polyval(c, i)
                nc += 1
        iter += 1
    return np.array(res)

# def showUV(oi, fig=0, polar=False):
#     plt.close(fig)
#     plt.figure(fig)
#     ax = plt.subplot(111, aspect='equal', polar=polar)
#     if not type(oi)==list:
#         oi = [oi]
#     for o in oi:
#         key = 'OI_VIS'
#         if not key in o.keys():
#             key = 'OI_VIS2'
#         for k in o[key].keys():
#             R =  o[key][k]['u/wl']**2 + o[key][k]['v/wl']**2
#             PA = np.arctan2(o[key][k]['v/wl'], o[key][k]['u/wl'])
#             if polar:
#                 plt.plot(PA, R, '.k')
#                 plt.plot(PA+np.pi, R, '.k')
#             else:
#                 plt.plot(o[key][k]['u/wl'], o[key][k]['v/wl'], '.k', alpha=0.5)
#                 plt.plot(-o[key][k]['u/wl'], -o[key][k]['v/wl'], '.k', alpha=0.5)
#     if polar:
#         ax.spines['polar'].set_visible(False)
#         X = [0, np.pi/2, np.pi, 3*np.pi/2]
#         ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
#         ax.set_xticklabels(['270', '0', '90', '180'])
#     return

ai1mcB = {'i':0} # initialize global marker/color for baselines
ai1mcT = {'i':0} # initialize global marker/color for triangles
ai1ax = {} # initialise global list of axes

def showOI(oi, param=None, fig=0, obs=None, showIm=False, fov=None, pix=None,
           imPow=1., imWl0=None, cmap='bone', dx=0.0, dy=0.0, debug=False,
           showChi2=True, wlMin=None, wlMax=None, spectro=None, imMax=None,
           figWidth=None, figHeight=None, logB=False, logV=False, color=(1.0,0.2,0.1),
           checkImVis=False, showFlagged=False, onlyMJD=None, showUV=False,
           allInOne=False, vLambda0=None):
    """
    oi: result from oifits.loadOI
    param: dict of parameters for model (optional)
    obs: observable to show (default oi['fit']['obs'] if found, else all available)
        list of ('V2', '|V|', 'PHI', 'T3PHI', 'T3AMP', 'DPHI', 'FLUX')
    fig: figure number (default 1)
    figWidth: width of the figure (default 9)
    allInOne: plot all data files in a single plot, as function of baseline
        if not, show individual files in plots as function of wavelength (default)
        logB: baseline in log scale (default False)
        logV: V2 and |V| in log scale (default False)

    showIm: show image (False)
        fov: field of view (in mas)
        pix: pixel size (in mas)
        imPow: show image**imPow (default 1.0)
        imWl0: wavelength(s) at which images are shown
        imMax: max value for image
        cmap: color map ('bone')
        dx: center of FoV (in mas, default:0.0)
        dy: center of FoV (in mas, default:0.0)

    """
    global ai1ax, ai1mcB, ai1mcT

    if type(oi)==list:
        if allInOne:
            ai1mcB = {'i':0} # initialize global marker/color for baselines
            ai1mcT = {'i':0} # initialize global marker/color for triangles
            ai1ax = {} # initialise global list of axes
        if fig is None:
            fig = 0
        for i,o in enumerate(oi):
            if allInOne:
                f = fig
            else:
                f = fig+i
            _showIm = (showIm or not fov is None ) and i==(len(oi)-1)
            #print('=>', showIm, fov, pix, '->', _showIm)
            showOI(o, param=param, fig=f, obs=obs, fov=fov, pix=pix, dx=dx, dy=dy,
                   checkImVis=checkImVis, showIm=_showIm, allInOne=allInOne,
                   imWl0=imWl0, imPow=imPow, cmap=cmap, figWidth=figWidth,
                   wlMin=wlMin, wlMax=wlMax, spectro=spectro, imMax=imMax,
                   logB=logB, logV=logV, color=color, showFlagged=showFlagged,
                   onlyMJD=onlyMJD, showUV=showUV, figHeight=figHeight,
                   showChi2=showChi2 and not allInOne,
                   debug=debug, vLambda0=vLambda0
                   )
        if allInOne:
           ai1mcB = {'i':0} # initialize global marker/color for baselines
           ai1mcT = {'i':0} # initialize global marker/color for triangles
           ai1ax = {} # initialise global list of axes
           title = []
           for o in oi:
               title.extend([os.path.basename(f) for f in o['filename'].split(';')])
           title = sorted(title)
           if len(title)>2:
               title = title[0]+' ... '+title[-1]
               title += ' (%d files)'%len(title)
               fontsize = 8
           elif len(title)>1:
               title = ', '.join(title)
               fontsize = 8
           else:
               fontsize = 10
           plt.suptitle(title, fontsize=fontsize)
        return

    #print('->', computeLambdaParams(param))

    if not vLambda0 is None:
        um2kms = lambda um: (um-vLambda0)/um*2.998e5
        kms2um = lambda kms: vLambda0*(1 + kms/2.998e5)

    if spectro is None:
        spectro = len(oi['WL'])>10
    if not spectro:
        showChi2=False

    if not onlyMJD is None and \
        not any([mjd in onlyMJD for mjd in oi['configurations per MJD'].keys()]):
        # -- nothing to plot!
        return

    if param is None:
        showChi2=False
    else:
        if type(param)==dict and 'best' in param.keys() and 'fitOnly' in param.keys():
            param = param['best']

    if ai1ax == {}:
        plt.close(fig)

    # -- user-defined wavelength range
    fit = {'wl ranges':[(min(oi['WL']), max(oi['WL']))]}
    if not 'fit' in oi:
        oi['fit'] = fit.copy()
    elif not 'wl ranges' in oi['fit']:
        # -- weird but necessary to avoid a global 'fit'
        fit.update(oi['fit'])
        oi['fit'] = fit.copy()

    if 'ignore telescope' in oi['fit']:
        ignoreTelescope = oi['fit']['ignore telescope']
    else:
        ignoreTelescope = 'no way they would call a telescope that!'

    if 'ignore baseline' in oi['fit']:
        ignoreBaseline = oi['fit']['ignore baseline']
    else:
        ignoreBaseline = '**'

    w = np.zeros(oi['WL'].shape)
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    if not 'obs' in oi['fit'] and obs is None:
            obs = []
            if 'OI_T3' in oi:
                obs.append('T3PHI')
            if 'OI_VIS2' in oi:
                obs.append('V2')
            if 'OI_VIS' in oi:
                obs.append('|V|')
            if 'OI_FLUX' in oi:
                obs.append('FLUX')
    elif not obs is None:
        pass
    else:
        obs = oi['fit']['obs']

    # -- force recomputing differential quantities
    if 'WL cont' in oi.keys():
        oi.pop('WL cont')
    if 'DPHI' in obs:
        oi = computeDiffPhiOI(oi, param)
    if 'NFLUX' in obs:
        oi = computeNormFluxOI(oi, param)
    if wlMin is None:
        wlMin = min(oi['WL'][oi['WL mask']])
    if wlMax is None:
        wlMax = max(oi['WL'][oi['WL mask']])

    if not param is None:
        #print('compute model V (analytical)')
        m = VmodelOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy)
        if not fov is None and checkImVis:
            #print('compute V from Image, fov=', fov)
            m = VfromImageOI(m)
        print('bandwidth smearing spectral channel(s):', m['smearing'])
    else:
        m = None

    c = 1 # column
    ax0 = None
    data = {
            'FLUX':{'ext':'OI_FLUX', 'var':'FLUX'},
            'NFLUX':{'ext':'NFLUX', 'var':'NFLUX', 'unit':'normalized'},
            #'T3PHI':{'ext':'OI_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl'},
            #'T3AMP':{'ext':'OI_T3', 'var':'T3AMP', 'X':'Bmax/wl'},
            'T3PHI':{'ext':'OI_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bavg/wl'},
            'T3AMP':{'ext':'OI_T3', 'var':'T3AMP', 'X':'Bavg/wl'},
            'DPHI':{'ext':'DPHI', 'var':'DPHI', 'unit':'deg', 'X':'B/wl', 'C':'PA'},
            'PHI':{'ext':'OI_VIS', 'var':'PHI', 'X':'B/wl', 'C':'PA'},
            '|V|':{'ext':'OI_VIS', 'var':'|V|', 'X':'B/wl', 'C':'PA'},
            'V2':{'ext':'OI_VIS2', 'var':'V2', 'X':'B/wl', 'C':'PA'},
            }
    imdata = {
             'FLUX':{'ext':'IM_FLUX', 'var':'FLUX'},
             'NFLUX':{'ext':'IM_FLUX', 'var':'NFLUX', 'unit':'normalized'},
             #'T3PHI':{'ext':'IM_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl'},
             #'T3AMP':{'ext':'IM_T3', 'var':'T3AMP', 'X':'Bmax/wl'},
             'T3PHI':{'ext':'IM_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bavg/wl'},
             'T3AMP':{'ext':'IM_T3', 'var':'T3AMP', 'X':'Bavg/wl'},
             'DPHI':{'ext':'IM_VIS', 'var':'DPHI', 'unit':'deg', 'X':'B/wl', 'C':'PA'},
             'PHI':{'ext':'IM_VIS', 'var':'PHI', 'X':'B/wl', 'C':'PA'},
             '|V|':{'ext':'IM_VIS', 'var':'|V|', 'X':'B/wl', 'C':'PA'},
             'V2':{'ext':'IM_VIS', 'var':'V2', 'X':'B/wl', 'C':'PA'},
             }
    # -- plot in a certain order
    obs = list(filter(lambda x: x in obs,
            ['FLUX', 'NFLUX', 'T3PHI', 'PHI', 'DPHI', 'T3AMP', '|V|', 'V2']))
    ncol = len(obs)

    if showUV:
        obs = obs[::-1]
        obs.append('UV')
        obs = obs[::-1]
        if not any(['FLUX' in o for o in obs]):
            ncol += 1

    mcB = {} # marker/color for baselines (showUV=True)
    mcT = {'i':0} # marker/color for triangles (showUV=True)

    if allInOne:
        # -- use globals to have consistent markers/colors
        mcB.update(ai1mcB)
        mcT.update(ai1mcT)

    # -- 8 markers
    markers = ['d', 'o', '*', 'h', '^', 'v', '>', '<']
    if not spectro:
        # -- 9 colors
        #colors = [(1,.2,.2), (.2,1,.2), (.2,.2,1)]# 'm', 'orange', 'cyan', '0.5']
        colors = list(itertools.permutations([0.1, 0.5, 0.9])) + ['0.5']

        #colors = matplotlib.cm.Paired(np.linspace(0,1,12)[1::2])
    else:
        colors = matplotlib.cm.nipy_spectral(np.linspace(0, .9, len(oi['baselines'])))

    if figWidth is None and figHeight is None:
        figHeight =  min(max(ncol, 10), 6)
        figWidth = min(figHeight*ncol, 9.5)
    if figWidth is None and not figHeight is None:
        figWidth = min(figHeight*ncol, 9.5)
    if not figWidth is None and figHeight is None:
        figHeight =  max(figWidth/ncol, 6)

    plt.figure(fig, figsize=(figWidth, figHeight))
    i_flux = 0
    i_col = 0
    yoffset = 0
    for c,l in enumerate(obs):
        # -- for each observable to plot
        if l=='UV': # special case for UV plot
            i = 0 # indexes for color
            if 'i' in mcB:
                # -- keep track of index
                i = mcB['i']
            bmax = []
            n_flux = np.sum(['FLUX' in o for o in obs])
            if allInOne and 'UV' in ai1ax:
                ax = ai1ax['UV']
            else:
                if len(oi['baselines'])>8 and n_flux>0:
                    ax = plt.subplot(n_flux+2, ncol, 1, aspect='equal')
                else:
                    ax = plt.subplot(n_flux+1, ncol, 1, aspect='equal')

            # -- for each observables per baselines
            ext = [e for e in ['OI_VIS', 'OI_VIS2'] if e in oi]
            for e in ext:
                if debug:
                    print(e, sorted(oi[e].keys()))
                for k in sorted(oi[e].keys()):
                    if debug:
                        print(len(oi[e][k]['MJD']))
                    if not k in mcB:
                        mcB[k] = markers[i%len(markers)], colors[i%len(colors)]
                        label = k
                        i+=1
                        if allInOne:
                            ai1mcB[k] = mcB[k]
                            mcB['i'] = i
                            ai1mcB['i'] = i
                    else:
                        label = ''
                    mark, col = mcB[k]

                    # -- for each MJD:
                    allMJDs = oi[e][k]['MJD']
                    if onlyMJD is None:
                        MJDs = allMJDs
                    else:
                        MJDs = [m for m in allMJDs if m in onlyMJD]
                    for mjd in MJDs:
                        w = allMJDs==mjd
                        bmax.append(np.sqrt(oi[e][k]['u'][w]**2+
                                            oi[e][k]['v'][w]**2))
                        ax.plot(oi[e][k]['u'][w], oi[e][k]['v'][w],
                                color=col, marker=mark, label=label,
                                linestyle='none', markersize=5)
                        ax.plot(-oi[e][k]['u'][w], -oi[e][k]['v'][w],
                                color=col, marker=mark,
                                linestyle='none', markersize=5)
                        label = ''

            bmax = 1.05*max(bmax)
            ax.legend(fontsize=6, loc='upper left', ncol=2)
            ax.set_title('u,v (m)', fontsize=10)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            if ax.get_xlim()[1]<bmax or ax.get_ylim()[1]<bmax:
                ax.set_xlim(-bmax, bmax)
                ax.set_ylim(-bmax, bmax)

            if allInOne and not 'UV' in ai1ax:
                ai1ax['UV'] = ax
                t = np.linspace(0, 2*np.pi, 100)
                for b in [50, 100, 150, 200, 250, 300]:
                     ax.plot(b*np.cos(t), b*np.sin(t), ':k', alpha=0.2)
            elif not allInOne:
                t = np.linspace(0, 2*np.pi, 100)
                for b in [50, 100, 150, 200, 250, 300]:
                     ax.plot(b*np.cos(t), b*np.sin(t), ':k', alpha=0.2)

            i_flux += 1
            if not any(['FLUX' in o for o in obs]):
                i_col+=1
            else:
                # do NOT incremente i_col!
                pass
            continue

        if not data[l]['ext'] in oi.keys():
            i_col += 1
            continue

        N = len(oi[data[l]['ext']].keys())
        # -- for each telescope / baseline / triplets
        keys = [k for k in oi[data[l]['ext']].keys()]

        # -- average normalized flux
        if l=='NFLUX' and 'UV' in obs:
            tmp = np.zeros(len(oi['WL']))
            etmp = 1.e6*np.ones(len(oi['WL']))
            weight = np.zeros(len(oi['WL']))
            for k in keys:
                allMJDs = list(oi[data[l]['ext']][k]['MJD'])
                if onlyMJD is None:
                    MJDs = allMJDs
                else:
                    MJDs = [m for m in allMJDs if m in onlyMJD]
                for mjd in MJDs:
                    j = allMJDs.index(mjd)
                    mask = ~oi[data[l]['ext']][k]['FLAG'][j,:]
                    tmp[mask] += oi[data[l]['ext']][k][l][j,mask]/\
                                 oi[data[l]['ext']][k]['E'+l][j,mask]
                    # todo: better estimation of error of average flux
                    etmp[mask] = np.minimum(etmp[mask],
                               oi[data[l]['ext']][k]['E'+l][j,mask])
                    weight[mask] += 1/oi[data[l]['ext']][k]['E'+l][j,mask]
            w = weight>0
            tmp[w] /= weight[w]
            oi[data[l]['ext']][''] = {
                l:np.array([tmp]),
                'E'+l:np.array([etmp]),
                'FLAG':np.array([~w]), 'MJD':[MJDs[0]],
                    }
            keys = ['']

        for i,k in enumerate(sorted(keys)):
            # -- for each telescope / baseline / triangle
            X = lambda r, j: r['WL']
            Xlabel = r'wavelength ($\mu$m)'
            Xscale = 'linear'
            if not spectro:
                if 'X' in data[l]:
                    X = lambda r, j: r[data[l]['ext']][k][data[l]['X']][j,:]
                    Xlabel = data[l]['X']
                    if logB:
                        Xscale = 'log'
                yoffset = 0.0

            if allInOne and l+str(i) in ai1ax:
                ax = ai1ax[l+str(i)]
            else:
                if 'UV' in obs and 'FLUX' in l:
                    if i==0:
                        if len(oi['baselines'])<=8:
                            if ax0 is None:
                                ax0 = plt.subplot(n_flux+2, ncol, ncol*(i_flux+1)+1)
                                ax = ax0
                            else:
                                ax = plt.subplot(n_flux+2, ncol, ncol*(i_flux+1)+1,
                                                sharex=ax0)
                        else:
                            if ax0 is None:
                                ax0 = plt.subplot(n_flux+1, ncol, ncol*(i_flux)+1)
                                ax = ax0
                            else:
                                ax = plt.subplot(n_flux+1, ncol, ncol*(i_flux)+1,
                                                sharex=ax0)
                else:
                    if not spectro:
                        if i==0:
                            ax = plt.subplot(1, ncol, i_col+1)
                    else:
                        if ax0 is None:
                            ax0 = plt.subplot(N, ncol, ncol*i+i_col+1)
                            ax = ax0
                        else:
                            ax = plt.subplot(N, ncol, ncol*i+i_col+1, sharex=ax0)
                if allInOne:
                    ai1ax[l+str(i)] = ax

            if not ('UV' in obs and 'FLUX' in l):
                yoffset = 0.0

            if not 'UV' in l and not vLambda0 is None and i==0:
                axv = ax.secondary_xaxis('top', functions=(um2kms, kms2um),
                                         color='0.6')
                axv.tick_params(axis='x', labelsize=8,
                                labelbottom=True,
                                labeltop=False,)
            else:
                axv = None

            chi2 = 0.0
            resi = np.array([])
            ymin, ymax = 1e6, -1e6
            setylim = False

            # -- for each MJD:
            allMJDs = list(oi[data[l]['ext']][k]['MJD'])
            if onlyMJD is None:
                MJDs = allMJDs
            else:
                MJDs = [m for m in allMJDs if m in onlyMJD]
            for mjd in MJDs:
                j = allMJDs.index(mjd)
                mask = ~oi[data[l]['ext']][k]['FLAG'][j,:]*oi['WL mask']
                flagged = oi[data[l]['ext']][k]['FLAG'][j,:]*oi['WL mask']
                # -- data:
                y = oi[data[l]['ext']][k][data[l]['var']][j,:]
                if 'PHI' in l and any(mask):# and np.ptp(y[mask])>300:
                    y[mask] = np.unwrap(y[mask]*np.pi/180)*180/np.pi
                    y[mask] = np.mod(y[mask]-np.mean(y[mask])+180, 360)+np.mean(y[mask]-180)%360-360
                    #y[flagged] = np.unwrap(y[flagged]*np.pi/180)*180/np.pi
                    #y[flagged] = np.mod(y[flagged]+180, 360)-180
                    pass

                err = oi[data[l]['ext']][k]['E'+data[l]['var']][j,:].copy()
                ign = (~mask).copy()
                showIgn = False
                if 'max error' in oi['fit'] and \
                        data[l]['var'] in oi['fit']['max error']:
                    # -- ignore data with large error bars
                    ign *= err>=oi['fit']['max error'][data[l]['var']]
                    mask *= err<oi['fit']['max error'][data[l]['var']]
                    showIgn = True
                if 'max relative error' in oi['fit'] and \
                        data[l]['var'] in oi['fit']['max relative error']:
                    # -- ignore data with large error bars
                    ign *= err>=oi['fit']['max relative error'][data[l]['var']]*np.abs(y)
                    mask *= err<oi['fit']['max relative error'][data[l]['var']]*np.abs(y)
                    showIgn = True
                if 'mult error' in oi['fit'] and \
                            data[l]['var'] in oi['fit']['mult error']:
                    err *= oi['fit']['mult error'][data[l]['var']]
                if 'min error' in oi['fit'] and data[l]['var'] in oi['fit']['min error']:
                    err[mask] = np.maximum(oi['fit']['min error'][data[l]['var']], err[mask])

                if 'min relative error' in oi['fit'] and \
                            data[l]['var'] in oi['fit']['min relative error']:
                    # -- force error to a minimum value
                    err[mask] = np.maximum(oi['fit']['min relative error'][data[l]['var']]*y[mask],
                                           err[mask])

                # -- show data
                test = testTelescopes(k, ignoreTelescope) or testBaselines(k, ignoreBaseline)

                showLabel = False
                # -- set T3 color if allInOne
                #if allInOne and 'T3' in l:
                if 'T3' in l:
                    if not k in mcT:
                        mcT[k]  = (markers[mcT['i']%len(markers)],
                                    colors[mcT['i']%len(colors)] )
                        mcT['i'] += 1
                        ai1mcT = mcT
                        showLabel = True

                if not spectro:
                    if k in mcB: # known baseline
                        mark, col = mcB[k]
                    elif k in mcT: # known triangle
                        mark, col = mcT[k]
                    else:
                        mark, col = '.', 'k'

                    # -- data dots
                    ax.plot(X(oi, j)[mask], y[mask], mark,
                            color=col if not test else '0.5',
                            alpha=0.5, label=k if j==0 else '',)
                    # -- data error bars
                    ax.errorbar(X(oi, j)[mask], y[mask], yerr=err[mask],
                                color=col if not test else '0.5',
                                alpha=0.2, linestyle='none', linewidth=1)
                    # -- line
                    #ax.plot(X(oi, j)[mask], y[mask], '-',
                    #        color=col if not test else '0.5',
                    #        alpha=0.1, label=k if j==0 else '')

                    if showFlagged:
                        ax.plot(X(oi, j)[flagged], y[flagged], 'x',
                                color = 'm' if not test else '0.5',
                                alpha=0.5, label='flag' if j==0 else '')
                        ax.errorbar(X(oi, j)[flagged], y[flagged], yerr=err[flagged],
                                    color='m', alpha=0.2, linestyle='None')
                else:
                    # -- data spectra
                    ax.step(X(oi, j)[mask], y[mask]+yoffset*i,
                            '-', color='k' if not test else '0.5',
                            alpha=0.5, label=k if j==0 else '', where='mid')
                    # -- data errors
                    ax.fill_between(X(oi, j)[mask],
                                    y[mask]+err[mask]+yoffset*i,
                                    y[mask]-err[mask]+yoffset*i,
                                    step='mid', color='k',
                                    alpha=0.1 if not test else 0.05)
                    if showFlagged:
                        ax.plot(X(oi, j)[flagged], y[flagged]+yoffset*i,
                                'mx' if not test else '0.5',
                                alpha=0.5, label='flag' if j==0 else '')
                        ax.errorbar(X(oi, j)[flagged], y[flagged]+yoffset*i,
                                    yerr=err[flagged], color='m', alpha=0.2, linestyle='None')

                if showIgn and showFlagged:
                    # -- show ignored data (filtered on error for example)
                    plt.plot(X(oi,j)[ign], y[ign]+yoffset*i, 'xy', alpha=0.5)

                maskp = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                try:
                    ymin = min(ymin, np.percentile((y+yoffset*i-err)[maskp], 1))
                    ymax = max(ymax, np.percentile((y+yoffset*i+err)[maskp], 99))
                    setylim = True
                except:
                    pass

                # -- show model (analytical)
                if not m is None and any(mask) and data[l]['ext'] in m.keys():
                    if not k in m[data[l]['ext']].keys():
                        k = list(m[data[l]['ext']].keys())[0]
                    ym = m[data[l]['ext']][k][data[l]['var']][j,:]
                    ym[mask] = np.unwrap(ym[mask]*np.pi/180)*180/np.pi
                    ym[mask] = np.mod(ym[mask]+180-np.mean(ym[mask]), 360)+(np.mean(ym[mask])-180)%360-360

                    # -- computed chi2 *in the displayed window*
                    maskc2 = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                    err[err<=0] = 1
                    if 'PHI' in l:
                        _resi = ((y-ym+180)%360-180)/err
                    else:
                        _resi = (y-ym)/err

                    # -- build residuals array
                    resi = np.append(resi, _resi[maskc2].flatten())
                    # -- show predictions from model
                    if not spectro:
                        ax.plot(X(m,j)[maskc2], ym[maskc2],
                                '-', alpha=0.5 if not test else 0.3,
                                color=color if col=='k' else '0.5',
                                linewidth=2)
                        if col!='k':
                            ax.plot(X(m,j)[maskc2], ym[maskc2],
                                    '--', alpha=0.7, color=col,
                                    linewidth=2)
                    else:
                        ax.step(X(m,j)[maskc2], ym[maskc2]+yoffset*i,
                                '-', alpha=0.4 if not test else 0.2,
                                where='mid', color=color)
                    try:
                        ymin = min(ymin, np.min(ym[maskp]+yoffset*i))
                        ymax = max(ymax, np.max(ym[maskp]+yoffset*i))
                        setylim = True
                    except:
                        pass
                else:
                    #resi = []
                    pass

                # -- show model: numerical FT from image
                if checkImVis:
                    if not spectro: #allInOne:
                        ax.plot(X(m, j)[mask],
                             m[imdata[l]['ext']][k][imdata[l]['var']][j,mask]+yoffset*i,
                            '1b', label='from image' if j==0 else '', alpha=0.4)
                    else:
                        ax.step(X(m, j)[mask],
                             m[imdata[l]['ext']][k][imdata[l]['var']][j,mask]+yoffset*i,
                            '--b', label='from image' if j==0 else '',
                            alpha=0.4, linewidth=2, where='mid')

                # -- show continuum for differetial PHI and normalized FLUX
                if (l=='DPHI' or l=='NFLUX') and 'WL cont' in oi:
                    maskc = ~oi[data[l]['ext']][k]['FLAG'][j,:]*\
                                    oi['WL cont']*oi['WL mask']
                    _m = np.mean(oi[data[l]['ext']][k][data[l]['var']][j,maskc])
                    cont = np.ones(len(oi['WL']))*_m
                    cont[~maskc] = np.nan
                    ax.plot(X(oi, j), cont+yoffset*i, ':', color='c', linewidth=3)

                # -- show phase based on OPL
                if l=='PHI' and 'OPL' in oi.keys():
                    dOPL = oi['OPL'][k[2:]] - oi['OPL'][k[:2]]
                    print(k, dOPL)
                    wl0 = 2.2
                    cn = np.polyfit(oi['WL']-wl0, oi['n_lab'], 8)
                    cn[-2:] = 0.0
                    ax.plot(oi['WL'],
                            -360*dOPL*np.polyval(cn, oi['WL']-wl0)/(oi['WL']*1e-6)+yoffset*i,
                            ':c', linewidth=2)
            # -- end loop on MJDs
            # if allInOne and not resi is None:
            #     allInOneResiduals[l].extend(list(resi))
            #     test =  i == len(oi[data[l]['ext']].keys())-1
            #     if test:
            #         resi = np.array(allInOneResiduals[l])
            # else:
            #     test = True

            if showChi2 and not resi is None:
                resi = resi[np.isfinite(resi)]
                try:
                    chi2 = np.mean(resi**2)
                except:
                    print(resi)

                if len(resi)<24:
                    rms = np.std(resi)
                    f = 'rms'
                else:
                    rms = 0.5*(np.percentile(resi, 84)-np.percentile(resi, 16))
                    f = '%%rms'
                fmt = '$\chi^2$=%.2f '
                if rms<=0 or not np.isfinite(np.log10(rms)):
                    n = 1
                else:
                    n = max(int(np.ceil(-np.log10(rms)+1)), 0)
                fmt += f+'=%.'+str(n)+'f'+r'$\sigma$'
                if k in mcB:
                    mark, col = mcB[k]
                else:
                    mark, col = 'o', color
                ax.text(0.02, 0.02, fmt%(chi2, rms),
                        color=col, fontsize=6,
                        transform=ax.transAxes, ha='left', va='bottom')

            if spectro:
                maskp = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                if setylim:
                    yamp = ymax-ymin
                    ax.set_ylim(ymin - 0.2*yamp-i*yoffset, ymax + 0.2*yamp)
                if 'UV' in obs and 'FLUX' in l and i==0:
                    yoffset = yamp
                ax.set_xlim(wlMin, wlMax)
                if k in mcB:
                    mark, col = mcB[k]
                else:
                    mark, col = 'o', 'k'
                if not 'UV' in obs or not 'FLUX' in l:
                    ax.text(0.02, 0.98, k, transform=ax.transAxes,
                            ha='left', va='top', fontsize=6, color=col)
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
            else:
                if l in ['V2', '|V|']:
                    if logV:
                        ax.set_ylim(1e-4,1)
                        ax.set_yscale('log')
                    else:
                        ax.set_ylim(0,1)
            if i==N-1:
                ax.set_xlabel(Xlabel)
                if Xscale=='log':
                    ax.set_xscale('log')
            if i==0:
                title = l
                if 'unit' in data[l]:
                    title += ' (%s)'%data[l]['unit']
                ax.set_title(title)
                if not vLambda0 is None:
                    axv.set_xlabel('velocity (km/s)')
                if l=='NFLUX':
                    ax.set_xlabel(Xlabel)
            if allInOne or l=='T3PHI':
                ax.legend(fontsize=6, ncol=3)

        i_col += 1

    plt.subplots_adjust(hspace=0, wspace=0.2, left=0.06, right=0.99)
    if 'filename' in oi.keys() and not allInOne:
        title = [os.path.basename(f) for f in oi['filename'].split(';')]
        title = sorted(title)
        if len(title)>2:
            title = title[0]+'...'+title[-1]
            fontsize = 8
        elif len(title)>1:
            title = ', '.join(title)
            fontsize = 8
        else:
            title = title[0]
            fontsize = 10
        plt.suptitle(title, fontsize=fontsize)

    if showIm and not param is None:
        showModel(oi, param, m=m, fig=fig+1, imPow=imPow, cmap=cmap, figWidth=figWidth,
                fov=fov, pix=pix, dx=dx, dy=dy, imWl0=imWl0, imMax=imMax)
    return

def showModel(oi, param, m=None, fig=0, figHeight=4, figWidth=None, fov=None,
              pix=None, imPow=1.0, dx=0., dy=0., imWl0=None, cmap='bone', imMax=None):
    """
    oi: result from loadOI for mergeOI,
        or a wavelength vector in um (must be a np.ndarray)
    param: parameter dictionnary, describing the model
    m: result from Vmodel, if none given, computed from 'oi' and 'param'
    fig: which figure to plot on (default 1)
    figHeight: height of the figure, in inch

    fov: field of view in mas
    pix: pixel size in mas
    imPow: power law applied to image for display. use 0<imPow<1 to show low
        surface brightness features
    dx, dy: center of image (in mas)
    imWl0: list of wavelength (um) to show the image default (min, max)
    cmap: color map (default 'bone')
    """
    param = computeLambdaParams(param)
    # -- catch case were OI is just a wavelength vector:
    if type(oi)==np.ndarray:
        oi = {'WL':oi, # minimum required
              'fit':{'obs':['NFLUX']} # force computation of continuum
              }

    if m is None:
        m = VmodelOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy)
        #print(m['MODEL'].keys())
        #if not 'WL mask' in oi and 'WL mask' in m:
        #    oi['WL mask'] = m['WL mask'].copy()
        if not 'WL cont' in oi and 'WL cont' in m:
            oi['WL cont'] = m['WL cont'].copy()
        #print(oi['WL mask'])

    # -- show synthetic images ---------------------------------------------
    if imWl0 is None or imWl0==[]:
        if 'cube' in m['MODEL'].keys():
            if not 'WL mask' in oi.keys():
                imWl0 = np.min(oi['WL']),  np.max(oi['WL'])
            else:
                imWl0 = [np.min(oi['WL'][oi['WL mask']]),
                         np.max(oi['WL'][oi['WL mask']]),]
        else:
            imWl0 = [np.mean(oi['WL'])]
    if 'cube' in m['MODEL']:
        nplot = len(imWl0)+1
    else:
        nplot = len(imWl0)

    if figWidth is None and figHeight is None:
        figHeight =  min(max(nplot, 8), 5)
        figWidth = min(figHeight*nplot, 9.5)
    if figWidth is None and not figHeight is None:
        figWidth = min(figHeight*nplot, 9.5)
    if not figWidth is None and figHeight is None:
        figHeight =  max(figWidth/nplot, 6)

    plt.close(fig)
    plt.figure(fig, figsize=(figWidth, figHeight))

    if 'cube' in m['MODEL'].keys():
        normIm = np.max(m['MODEL']['cube'])**imPow
    else:
        normIm = 1

    if imMax is None:
        imMax = 1.0

    # -- components
    comps = set([k.split(',')[0].strip() for k in param.keys() if ',' in k and not '&dwl' in k])
    # -- peak wavelengths to show components with color code
    wlpeak = {}
    allpeaks = []
    for c in comps:
        # -- lines in the components
        #lines = list(filter(lambda x: x.startswith(c+',line_') and x.endswith('wl0'), param))
        # if len(lines):
        #     # -- weighted wavelength of lines:
        #     wlpeak[c] = np.sum([param[k]*param[k.replace('_wl0', '_f')] for k in lines])/\
        #                 np.sum([param[k.replace('_wl0', '_f')] for k in lines])
        #     allpeaks.append(wlpeak[c])
        # else:
        #     wlpeak[c] = None
        wlpeak[c] = None
    symbols = {}
    a, b, c = 0.9, 0.6, 0.1
    colors = [(a,b,c), (b,c,a), (c,a,b),
              (a,c,b), (c,b,a), (b,a,c),
              (a,c,c), (c,a,c), (c,c,a),
              (b,b,c), (b,c,b), (c,b,b),
              (b,b,b)]
    _ic = 0
    for c in comps:
        if wlpeak[c] is None:
            symbols[c] = {'m':'x', 'c':colors[_ic%len(colors)]}
            _ic+=1
        else:
            if len(allpeaks)==1:
                # -- only one componenet with line
                symbols[c] = {'m':'+', 'c':'orange'}
            else:
                symbols[c] = {'m':'+',
                              'c':matplotlib.cm.nipy_spectral(0.1+0.8*(wlpeak[c]-min(allpeaks))/np.ptp(allpeaks))
                             }

    if 'WL mask' in oi.keys():
        mask = oi['WL mask']
    else:
        mask = np.isfinite(oi['WL'])

    axs = []
    for i,wl in enumerate(imWl0):
        # -- for each wavelength for which we need to show the image
        if axs ==[]:
            axs = [plt.subplot(1, nplot, i+1, aspect='equal')]
        else:
            axs.append(plt.subplot(1, len(imWl0)+1, i+1, aspect='equal',
                       sharex=axs[0], sharey=axs[0]))
        _j = np.argmin(np.abs(oi['WL'][mask]-wl))
        _wl = oi['WL'][mask][_j]
        j = np.arange(len(oi['WL']))[mask][_j]
        if not imPow == 1:
            plt.title('Image$^{%.2f}$ $\lambda$=%.3f$\mu$m'%(imPow, _wl),
                    fontsize=9)
        else:
            plt.title('Image $\lambda$=%.3f$\mu$m'%(_wl), fontsize=9)

        if 'cube' in m['MODEL'].keys():
            im = m['MODEL']['cube'][j,:,:]
            if np.min(im)<0:
                print('WARNING: negative image! wl=%.4fum'%_wl, end=' ')
                print(' fraction of <0 flux:',
                      '%.2e'%(-np.sum(im[im<0])/np.sum(im[im>=0])), end=' ')
                print(' "negativity" in model:', m['MODEL']['negativity'])

            im = np.maximum(im, 0)**imPow
            im /= normIm
        elif 'image' in m['MODEL'].keys():
            im = m['MODEL']['image']/m['MODEL']['image'].max()
        else:
            print('!!! no imaging data !!!')
            print(m['MODEL'].keys())

        vmin, vmax = 0, imMax**imPow
        print('debug: im min,max =', im.min(), ',', im.max())
        print('    vmin, vmax =', vmin, ',', vmax)
        print('    Xmin , Xmax =', m['MODEL']['X'].min(), ',', m['MODEL']['X'].max())
        print('    Ymin , Ymax =', m['MODEL']['Y'].min(), ',', m['MODEL']['Y'].max())


        pc = plt.pcolormesh(m['MODEL']['X'], m['MODEL']['Y'],
                            im, vmin=vmin, vmax=vmax,
                             cmap=cmap, shading='auto')
        cb = plt.colorbar(pc, ax=axs[-1])
        #Xcb = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])*imMax
        Xcb = np.linspace(0,1,11)*imMax**imPow
        XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
        XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
        cb.set_ticks(Xcb)
        cb.set_ticklabels(XcbL)
        cb.ax.tick_params(labelsize=7)
        plt.xlabel(r'$\leftarrow$ E (mas)')
        if i==0:
            plt.ylabel(r'N $\rightarrow$ (mas)')

        # -- show position of each components
        for c in sorted(comps):
            if c+',x' in param.keys():
                x = param[c+',x']
            else:
                x = 0.0
            if c+',y' in param.keys():
                y = param[c+',y']
            else:
                y = 0.0
            plt.plot(x, y, symbols[c]['m'], color=symbols[c]['c'], label=c)
            if i==0:
                plt.legend(fontsize=5, ncol=2)
        axs[-1].tick_params(axis='x', labelsize=7)
        axs[-1].tick_params(axis='y', labelsize=7)

    axs[-1].invert_xaxis()
    if not 'cube' in m['MODEL']:
        plt.tight_layout()
        return

    ax = plt.subplot(1, len(imWl0)+1, len(imWl0)+1)
    if 'totalnflux' in m['MODEL']:
        key = 'nflux'
        plt.title('spectra, normalized\nto total continuum', fontsize=8)
    else:
        key = 'flux'
        plt.title('spectra', fontsize=8)
    if len(m['WL'][mask])>20:
        plt.step(m['WL'][mask], m['MODEL']['total'+key][mask],
                '-k', label='total', where='mid')
    else:
        plt.plot(m['WL'][mask], m['MODEL']['total'+key][mask],
                '.-k', label='total')

    if 'WL cont' in oi:
        cont = np.ones(oi['WL'].shape)
        cont[~oi['WL cont']] = np.nan
        if len(m['WL'][mask])>20:
            plt.step(m['WL'][mask], (m['MODEL']['total'+key]*cont)[mask],
                    'c', label='continuum', where='mid', alpha=0.7,
                    linewidth=3, linestyle='dotted')
        else:
            plt.plot(m['WL'][mask], (m['MODEL']['total'+key]*cont)[mask], '.-',
                    label='continuum', alpha=0.7,
                    linewidth=3, linestyle='dotted')

    # -- show spectra of each components
    KZ = filter(lambda x: not '&dwl' in x, m['MODEL'].keys())
    for k in sorted(KZ):
        if k.endswith(','+key):
            if len(m['WL'][mask])>20:
                plt.step(m['WL'][mask], m['MODEL'][k][mask],
                         label=k.split(',')[0].strip(), where='mid',
                         color=symbols[k.split(',')[0].strip()]['c'])
            else:
                plt.plot(m['WL'][mask], m['MODEL'][k][mask], '.-',
                         label=k.split(',')[0].strip(),
                         color=symbols[k.split(',')[0].strip()]['c'])

    plt.grid(color=(0.2, 0.4, 0.7), alpha=0.2)

    plt.legend(fontsize=5)
    plt.xlabel('wavelength ($\mu$m)')
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

    plt.ylim(0)
    plt.tight_layout()
    return

def _callbackAxes(ax):
    global _AX, _AY
    i = None
    for k in _AY.keys():
        if ax==_AY[k]:
            i = k
    #print('callback:', i)
    if not i is None:
        #print('callback:', i, ax.get_ylim())
        _AX[i].set_xlim(ax.get_ylim())
        #_AX[i].figure.canvas.draw()
    else:
        print('could not find axes')
    return

def showBootstrap(b, fig=0, figWidth=None, showRejected=False,
                  combParam={}, sigmaClipping=4.5):
    """
    you can look at combination of parameters:

    combParam: {'sep':'np.sqrt($x**2 + $y**2)'} assuming 'x' and 'y' are parameters from the model

    """
    global _AX, _AY
    boot = copy.deepcopy(b)
    if len(combParam)>0:
        s = '$'
        for k in combParam:
            if not k in boot['fitOnly']:
                boot['fitOnly'].append(k)
            for i,f in enumerate(boot['all fits']):
                tmp = combParam[k]+''
                j = 0
                while s in tmp and j<5:
                    for x in f['best'].keys():
                        if s+x in tmp:
                            tmp = tmp.replace(s+x, '('+str(f['best'][x])+')')
                boot['all fits'][i]['best'][k] = eval(tmp)
                boot['all fits'][i]['uncer'][k] = 0.0
        print('analyse')
        boot = analyseBootstrap(boot, verbose=2, sigmaClipping=sigmaClipping)
        print('done')

    if figWidth is None:
        figWidth = min(9.5, 1+2*len(boot['fitOnly']))

    fontsize = max(min(4*figWidth/len(boot['fitOnly']), 14), 6)
    plt.close(fig)
    fig = plt.figure(fig, figsize=(figWidth, figWidth))
    _AX = {}

    color1 = 'orange'
    color2 = (0.2, 0.4, 1.0)
    color3 = (0.8, 0.2, 0.4)

    combi = False
    # -- for each fitted parameters, show histogram
    for i1, k1 in enumerate(sorted(boot['fitOnly'])):
        _AX[i1] = plt.subplot(len(boot['fitOnly']),
                              len(boot['fitOnly']),
                              1+i1*len(boot['fitOnly'])+i1)
        bins = int(3*np.ptp(boot['all best'][k1])/boot['uncer'][k1])
        bins = min(bins, len(boot['mask'])//5)
        bins = max(bins, 5)
        h = plt.hist(boot['all best'][k1], bins=bins,
                     color='k', histtype='step', alpha=0.9)
        h = plt.hist(boot['all best'][k1], bins=bins,
                     color='k', histtype='stepfilled', alpha=0.05)
        if k1 in boot['fit to all data']['best']:
            plt.errorbar(boot['fit to all data']['best'][k1],
                        0.4*max(h[0]), markersize=fontsize/2,
                        xerr=boot['fit to all data']['uncer'][k1], color=color1,
                        fmt='s', capsize=fontsize/2, label='fit to all data')
            combi = False
        else:
            combi = True
        plt.errorbar(boot['best'][k1], 0.5*max(h[0]), xerr=boot['uncer'][k1],
                    color=color3 if combi else color2, fmt='d',
                    capsize=fontsize/2, label='bootstrap', markersize=fontsize/2)

        plt.legend(fontsize=6)
        # -- title
        n = int(np.ceil(-np.log10(boot['uncer'][k1])+1))
        fmt = '%s=\n'+'%.'+'%d'%n+'f'+'$\pm$'+'%.'+'%d'%n+'f'
        plt.title(fmt%(k1, boot['best'][k1], boot['uncer'][k1]),
                    fontsize=fontsize)
        _AX[i1].yaxis.set_visible(False)
        if i1!=(len(boot['fitOnly'])-1):
            _AX[i1].xaxis.set_visible(False)
        else:
            _AX[i1].tick_params(axis='x', labelsize=fontsize*0.8)
            _AX[i1].set_xlabel(k1, fontsize=fontsize)
    _AY = {}
    # -- show density plots
    for i1, k1 in enumerate(sorted(boot['fitOnly'])):
        for i2 in range(i1+1, len(boot['fitOnly'])):
            k2 = sorted(boot['fitOnly'])[i2]
            if i1==0:
                _AY[i2] = plt.subplot(len(boot['fitOnly']),
                            len(boot['fitOnly']),
                            1+i2*len(boot['fitOnly'])+i1,
                            sharex=_AX[i1])
                ax = _AY[i2]
            else:
                ax = plt.subplot(len(boot['fitOnly']),
                            len(boot['fitOnly']),
                            1+i2*len(boot['fitOnly'])+i1,
                            sharex=_AX[i1],
                            sharey=_AY[i2])

            plt.plot(boot['all best'][k1], boot['all best'][k2], '.',
                     alpha=np.sqrt(2/len(boot['mask'])), color='k')
            if showRejected:
                plt.plot(boot['all best ignored'][k1],
                         boot['all best ignored'][k2], 'xr', alpha=0.3)

            if k1 in boot['fit to all data']['best'] and k2 in boot['fit to all data']['best']:
                plt.plot(boot['fit to all data']['best'][k1],
                         boot['fit to all data']['best'][k2], 'x', color='0.5')
                x, y = dpfit.errorEllipse(boot['fit to all data'], k1, k2)
                plt.plot(x, y, '-', color=color1)#, label='c=%.2f'%boot['cord'][k1][k2])

            # -- combined parameters function of the other one?
            if (k1 in combParam and k2 in combParam[k1]) or \
                (k2 in combParam and k1 in combParam[k2]):
                _c = color3
            else:
                _c = color2
            plt.plot(boot['best'][k1], boot['best'][k2], '+',
                    color=_c)
            x, y = dpfit.errorEllipse(boot, k1, k2)
            plt.plot(x, y, '-', color=_c, label='c=%.2f'%boot['cord'][k1][k2])

            plt.legend(fontsize=6)
            if i2==(len(boot['fitOnly'])-1):
                plt.xlabel(k1, fontsize=fontsize)
                ax.tick_params(axis='x', labelsize=fontsize*0.8)
            else:
                ax.xaxis.set_visible(False)
            if i1==0:
                plt.ylabel(k2, fontsize=fontsize)
                _AY[i2].callbacks.connect('ylim_changed', _callbackAxes)
                _AY[i2].tick_params(axis='y', labelsize=fontsize*0.8)
            else:
                ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0, # 0.65*fig.subplotpars.hspace,
                        wspace = 0, # 0.65*fig.subplotpars.wspace
                        )
    return

def Oi2Obs(oi, obs=[]):
    """
    obs can be 'V' (complex), '|V|', 'PHI', 'DPHI', 'V2', 'T3AMP', 'CP'

    return OBS format:
    (uv, wl, t3formula, obs, data, error)
    uv = ((u1,v1), (u2,v2), ..., (un, vn)) in meters
    wl = wavelength vector in um
    mjd = date vector.
    t3mat = formalae to compute t3 from (u_i, v_i)

    limitation:
    - does not work if OI does not contain OI_VIS or OI_VIS2
    - assume each u,v (and triangle) have same MJD dimension
    """
    if type(oi)==list:
        return [Oi2Obs(o) for o in oi]
    res = []
    # -- u,v
    uv = []
    key = 'OI_VIS'
    if not key in oi:
        key = 'OI_VIS2'
    for k in oi[key].keys():
        uv.append((oi[key][k]['u'], oi[key][k]['v']))
    res.append(uv)

    # -- wavelength
    res.append(oi['WL'])

    # -- T3 formula (if needed)
    formula = None
    if 'T3AMP' in obs or 'T3PHI' in obs:
        formula = []
        for k in oi['OI_T3'].keys():
            pass

def _negativityAzvar(n, phi, amp, N=200):
    x = np.linspace(0, 2*np.pi, N)
    y = np.ones(N)
    for i in range(len(n)):
        y += amp[i]*np.cos(n[i]*(x + 3*np.pi/2 + phi[i]*np.pi/180))
    return np.sum(y<0)/np.sum(y>=0)


def _Vazvar(u, v, I, r, n, phi, amp, stretch=None, V0=None, numerical=False,
            XY=None, nB=30, numVis=False):
    """
    complex visibility of aritrary radial profile with cos AZ variations.

    u,v: baseline/wavelength coordinates in meters/microns (can be 1D or 2D)

    I(r)*(1+amp*cos(n*(PA+phi)))
        -> I, r: 1D arrays in mas
        -> PA: projection angle, computed from u,v
        -> n: an integer giving the number of cos max in PA
        -> phi: is a PA rotation (deg from N to E)

    n, phi, amp can be lists to have a harmonic decomposition

    stretch=[e, PA]: stretch/PA (optional, default is None) with 0<e<=1 and PA
        the rotation (N>E) of the major axis. If the stretch is defined, the
        azimutal variations rotates with it! PA in degrees

    V0 is V(u,v) for I(r). If not given, will be computed numerically (slow).
    It is the Hankel0 transform of I(r), e.g. 2J1(x)/x for UD.

    nB = 50 # number of baselines for Hankel transform

    numerical=True: compute numerically the Visibility. XY (square regular mesh
    grid in mas) can be given, or computed automatically. In that later case, the
    pitch of the image is set to ensure good accuracy in Visibility. Returns:

        ([Vis], Image, X, Y) if XY=None
            Vis: the visibility for U,V
            Image: the 2D image (total flux of 1.0)
            X, Y: the 2D meshgrid of spatial coordinates of the image (in mas)
        ([Vis], Image) is XY=X,Y where X,Y are square regular grids (in mas)
        Vis only returned is numVis is ste to True
    ---

    https://www.researchgate.net/publication/241069465_Two-Dimensional_Fourier_Transforms_in_Polar_Coordinates
    section 3.2.1 give the expression for the Fourier transform of a function
    in the form f(r,theta). It f(r, theta) can be written as g(r)*h(theta), the final
    Fourier Transform TF(rho, psi) can be written as:

    F(rho, psi) = sum_{n=-inf}^{inf} 2pi j^-n e^{j*n*psi} int_0^infty fn(r) Jn(rho*r) r dr

    with:

    fn(r) = 1/2pi g(r) int_0^2pi h(theta) e^{-j*n*theta} dtheta

    In the simple case of h(theta)=cos(n*theta), fn is non-0 only for -n and n

    fn = f-n = 1/2*g(r) and noticing that J-n = -Jn

    F(rho, psi) =  (-j)^n * e^{j*n*psi} int_0^infty[g(r)*J-n(rho*r) r*dr] +
                  - (-j)^n * e^{-j*n*psi} int_0^infty[g(r)*Jn(rho*r) r*dr] +
                 = -2*(-j)^n * cos(n*psi) * int_0^infty[g(r)*Jn(rho*r) r*dr]
                 = 2*(-j)^n * cos(n*psi) * Gn(rho)

    where Gn is the nth order Hankel transform of g.

    see Also: https://en.wikipedia.org/wiki/Hankel_transform#Relation_to_the_Fourier_transform_(general_2D_case)
    """

    #c = np.pi/180./3600./1000.*1e6
    if not isinstance(n, list):
        # -- scalar case
        n = [n]
        phi = [phi]
        amp = [amp]
    if not stretch is None:
        rot = stretch[1]*np.pi/180
    else:
        rot = 0.0

    if numerical:
        if u is None or v is None:
            u, v = np.array([1.]), np.array([1])
        B = np.sqrt(u**2+v**2)
        if XY is None:
            # -- to double check
            pix = 1/(B.max()*_c*20)
            Nx = int(np.ptp(r)/pix)+1
            Ny = int(np.ptp(r)/pix)+1
            Nmax = 200 # RAM requirement balloons quickly
            if Nx>Nmax:
                print("WARNING: synthetic image size is too large >Nmax=%d"%Nmax)
                return
            x = np.linspace(-np.max(r), np.max(r), Nx)
            X, Y = np.meshgrid(x, x)
            retFull = True
        else:
            X,Y = XY
            pix = np.diff(X).mean()
            Nx = X.shape[0]
            Ny = X.shape[1]
            retFull = False

        if not stretch is None:
            # -- apply stretch in direct space
            Xr = np.cos(-rot)*X + np.sin(-rot)*Y
            Yr =-np.sin(-rot)*X + np.cos(-rot)*Y
            Xr /= stretch[0]
        else:
            Xr, Yr = X, Y

        Im = np.zeros((Nx, Ny))
        #print('azvar image:', X.shape, Y.shape, Im.shape)
        r2 = np.array(r)**2
        # -- 2D intensity profile with anti-aliasing
        # ns = 3
        # for i in range(Ny):
        #     for j in range(Nx):
        #         for dx in np.linspace(-pix/2, pix/2, ns+2)[1:-1]:
        #             for dy in np.linspace(-pix/2, pix/2, ns+2)[1:-1]:
        #                 _r2 = (Xr[j,i]+dx)**2+(Yr[j,i]+dy)**2
        #                 Im[j,i] += 1./ns**2*np.interp(_r2,r2,I,right=0.0,left=0.0)
        Im = np.interp(Xr**2+Yr**2, r2, I, right=0, left=0)
        # -- azimutal variations in image
        PA = np.arctan2(Yr, Xr)

        PAvar = np.ones(PA.shape)
        for k in range(len(n)):
            PAvar += amp[k]*np.cos(n[k]*(PA + 3*np.pi/2 + phi[k]*np.pi/180))
        Im *= PAvar
        # -- normalize image to total flux
        Im /= np.sum(Im)

        # -- numerical Visibility
        if numVis:
            if len(u.shape)==2:
                Vis = Im[:,:,None,None]*np.exp(-2j*_c*(u[None,None,:,:]*X[:,:,None,None] +
                                                            v[None,None,:,:]*Y[:,:,None,None]))
            elif len(u.shape)==1:
                Vis = Im[:,:,None]*np.exp(-2j*_c*(u[None,None,:,:]*X[:,:,None] +
                                                       v[None,None,:,:]*Y[:,:,None]))
            Vis = np.trapz(np.trapz(Vis, axis=0), axis=0)/np.trapz(np.trapz(Im, axis=0), axis=0)
        else:
            Vis = None

        if retFull:
            if numVis:
                return Vis, Im, X, Y
            else:
                return Im, X, Y
        else:
            if numVis:
                return Vis, Im
            else:
                return Im

    if not stretch is None:
        # -- apply stretch in u,v space
        _u = np.cos(-rot)*u + np.sin(-rot)*v
        _v =-np.sin(-rot)*u + np.cos(-rot)*v
        _u *= stretch[0]
        _B = np.sqrt(_u**2 + _v**2)
        _PA = np.arctan2(_v,_u)
    else:
        _B = np.sqrt(u**2 + v**2)
        _PA = np.arctan(v,u)

    # use subsets of baseline for interpolation (faster)
    if _B.size>nB:
        Bm = np.linspace(np.min(_B), np.max(_B), nB)
    else:
        Bm = None

    # -- define Hankel transform of order n
    def Hankel(n):
        if not Bm is None:
            H = np.trapz(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*_c*Bm[None,:]*r[:,None]),
                         r, axis=0)/np.trapz(I*r, r)
            return np.interp(_B, Bm, H)
        else:
            H = np.trapz(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*_c*_B.flatten()[None,:]*r[:,None]),
                         r, axis=0)/np.trapz(I*r, r)
            return np.reshape(H, _B.shape)

    # -- visibility without PA variations -> Hankel0
    if V0 is None:
        Vis = Hankel(0)*(1.+0j) # -- force complex
    else:
        Vis = V0*(1.+0j) # -- force complex

    for i in range(len(n)):
        if np.abs(amp[i])>0:
            Vis += amp[i]*(-1j)**n[i]*Hankel(n[i])*\
                    np.cos(n[i]*(_PA+3*np.pi/2+phi[i]*np.pi/180))

    # -- return complex visibility
    return Vis

def testAzVar():
    """
    """
    #c = np.pi/180./3600./1000.*1e6

    wl0 = 1.6 # in microns
    diam = 10 # in mas
    bmax = 140 # in meters

    # -- U,V grid
    Nuv = 51
    b = np.linspace(-bmax,bmax,Nuv)
    U, V = np.meshgrid(b, b)

    stretch = None
    # == image parameters =================================================
    # ~FS CMa
    thick = .7 # uniform disk for thick==1, uniform ring for 0<f<1
    n = [1,2,3] # int number of maxima in cos azimutal variation (0==no variations)
    phi = [90,60,30] # PA offset in azimutal var (deg)
    amp = [0.33,0.33,0.33] # amplitude of azimutal variation: 0<=amp<=1
    stretch = [.5, 90] # eccentricity and PA of large

    # n = [1] # int number of maxima in cos azimutal variation (0==no variations)
    # phi = [90] # PA offset in azimutal var (deg)
    # amp = [1.0] # amplitude of azimutal variation: 0<=amp<=1

    # =====================================================================

    # == Semi-Analytical =======================================
    t0 = time.time()
    # -- 1D intensity profiles
    Nr = min(int(30/thick), 200)
    _r = np.linspace(0, diam/2, Nr)

    if False:
        # -- uniform ring ------------------------
        _i = 1.0*(_r<=diam/2.)*(_r>=diam/2.*(1-thick))
        # -- analytical V of radial profile, and semi-analytical azimutal variation
        if not stretch is None:
            rot = stretch[1]*np.pi/180
            _u = np.cos(-rot)*U + np.sin(-rot)*V
            _v =-np.sin(-rot)*U + np.cos(-rot)*V
            _u *= stretch[0]
            x = np.pi*c*np.sqrt(_u**2+_v**2)*diam/wl0
        else:
            x = np.pi*c*np.sqrt(U**2+V**2)*diam/wl0
        VisR = (2*scipy.special.j1(x+1e-6)/(x+1e-6) -
               (1-thick)**2*2*scipy.special.j1(x*(1-thick)+1e-6)/(x*(1-thick)+1e-6))/(1-(1-thick)**2)
        #VisR = None # does not work
        Visp = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, V0=VisR, stretch=stretch)
    else:
        # -- "doughnut" -------------------------
        r0 = 0.5*(1 + (1-thick))*diam/2
        width = thick*diam/4
        _i = np.maximum(1-(_r-r0)**2/width**2, 0)

        # -- truncated gaussian: r0 as FWHM = 2.355*sigma
        #_i = np.exp(-(_r/r0*2.355)**2)*(_r<=diam/2)*(_r>=(1-thick)*diam/2)

        # -- do not give analytical visibility of radial profile (slower)
        Visp = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, stretch=stretch)

    tsa = time.time()-t0
    print('semi-analytical: %.4fs'%(time.time()-t0))

    # == Numerical ============================================
    t0 = time.time()
    Vis, I, X, Y = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, numerical=1,
                           stretch=stretch, numVis=1)
    Nx = X.shape[0]
    tn = time.time()-t0
    print('numerical:       %.4fs'%(tn))
    print('Image min/max = %f / %f'%(np.min(I), np.max(I)))
    # == Show result =======================================
    print('speedup: x%.0f'%(tn/tsa))
    plt.figure(0, figsize=(11,4))
    plt.clf()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93,
                        bottom=0.12, hspace=0.25, wspace=0)
    ax = plt.subplot(1,4,1)
    ax.set_aspect('equal')
    plt.pcolormesh(X, Y, I, cmap='inferno', vmin=0)
    # plt.imshow(I, cmap='gist_heat', vmin=0, origin='lower',
    #             extent=[_r[0], _r[-1], _r[0], _r[-1]])
    title = r'image %dx%d, $\theta$=%.2fmas'%(Nx, Nx, diam)
    plt.title(title)
    plt.xlim(plt.xlim()[1], plt.xlim()[0])
    plt.xlabel('E <- (mas)')
    plt.ylabel('-> N (mas)')
    x, y = np.array([0,  0]), np.array([-diam/2, diam/2])

    # -- label
    # if not stretch is None:
    #     rot = stretch[1]*np.pi/180
    #     x, y = np.cos(rot)*x+np.sin(rot)*y, -np.sin(rot)*x+np.cos(rot)*y
    # else:
    #     rot = 0
    # plt.plot(x, y, 'o-w', alpha=0.5)
    # plt.text(0, 0, r'$\theta$=%.2fmas'%(diam), rotation=90+rot*180/np.pi,
    #         color='w', ha='center', va='bottom', alpha=0.5)

    ax0 = plt.subplot(2,4,2)
    plt.title('|V| numerical')
    ax0.set_aspect('equal')
    pvis = plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, np.abs(Vis),
                          cmap='gist_stern', vmin=0, vmax=1)
    plt.colorbar(pvis)

    ax = plt.subplot(2,4,3, sharex=ax0, sharey=ax0)
    plt.title('|V| semi-analytical')
    ax.set_aspect('equal')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, np.abs(Visp),
                    cmap='gist_stern', vmin=0, vmax=1)

    ax = plt.subplot(2,4,4, sharex=ax0, sharey=ax0)
    dyn = 1. # in 1/100 visibility
    plt.title('$\Delta$|V| (1/100)')
    ax.set_aspect('equal')
    res = 100*(np.abs(Vis)-np.abs(Visp))
    pv = plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0,
                    res, cmap='RdBu',
                    vmin=-np.max(np.abs(res)),
                    vmax=np.max(np.abs(res)),
                    )
    plt.colorbar(pv)
    print('median  |V|  residual (abs.) = %.3f'%np.median(np.abs(res)), '%')
    print('90perc  |V|  residual (abs.) = %.3f'%np.percentile(np.abs(res), 90), '%')

    ax = plt.subplot(2,4,6, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title('$\phi$ numerical')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, 180/np.pi*np.angle(Vis),
                    cmap='hsv', vmin=-180, vmax=180)
    plt.colorbar()

    ax = plt.subplot(2,4,7, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title('$\phi$ semi-analytical')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, 180/np.pi*np.angle(Visp),
                    cmap='hsv', vmin=-180, vmax=180)

    ax = plt.subplot(2,4,8, sharex=ax0, sharey=ax0)
    dyn = 1
    plt.title('$\Delta\phi$ (deg)')
    ax.set_aspect('equal')
    res = 180/np.pi*((np.angle(Vis)-np.angle(Visp)+np.pi)%(2*np.pi)-np.pi)
    pp = plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, res,
                        cmap='RdBu', vmin=-dyn, vmax=dyn)
    print('median phase residual (abs.) = %.3f'%np.median(np.abs(res)), 'deg')
    print('90perc phase residual (abs.) = %.3f'%np.percentile(np.abs(res), 90), 'deg')

    plt.colorbar(pp)
    ax0.set_xlim(ax0.get_xlim()[1], ax0.get_xlim()[0])
    return

def _smoothUnwrap(s, n=5):
    """
    s: signal in degrees

    unwrap if the phase jump is smooth, i.e. over many samples
    """
    offs = np.zeros(len(s)) # total offsets

    for i in np.arange(len(s))[n:-n]:
        if (np.median(s[i-n:i])-np.median(s[i:i+n]))<-180:
            s[i+1:]-=360
        if (np.median(s[i-n:i])-np.median(s[i:i+n]))>180:
            s[i+1:]+=360
    return s
