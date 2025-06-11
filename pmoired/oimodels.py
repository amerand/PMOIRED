import time
import copy
import multiprocessing
import random
import os
import platform, subprocess
import itertools
import sys
import pickle

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import matplotlib.cm # deprecated in 3.8
#import matplotlib.colormaps

import scipy.special
import scipy.interpolate
import scipy.stats

import pmoired.dpfit as dpfit
import pmoired.dw as dw
import pmoired.oifits as oifits
# import pmoired.ulensBin2 as ulensBin2

from astropy import constants

_c = np.pi**2/180/3600/1000*1e6

# -- default max number of processes
MAX_THREADS = multiprocessing.cpu_count()
US_SPELLING = False

def Ssingle(oi, param, noLambda=False):
    """
    build spectrum for Vsingle
    """
    if not noLambda:
        _param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))
    else:
        _param = param.copy()
    # -- flux (spectrum)
    f = np.zeros(oi['WL'].shape)

    # == continuum ===============
    if 'f0' in _param:
        f += _param['f0']
    elif 'f' in _param:
        f += _param['f']
    elif not 'spectrum' in _param and not any([x.startswith('fwvl') for x in _param]):
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
    #print('lines', list(lines))
    for l in lines:
        i = l.split('_')[1] # should not start with f!!!!
        wl0 = _param['line_'+i+'_wl0'] # in um
        if 'line_'+i+'_lorentzian' in _param.keys():
            dwl = _param['line_'+i+'_lorentzian'] # in nm
            #f += _param[l]*1/(1+(oi['WL']-wl0)**2/(dwl/1000)**2)
            f += _param[l]*(0.5*dwl/1000)**2/((oi['WL']-wl0)**2 + (0.5*dwl/1000)**2)
        if 'line_'+i+'_truncexp' in _param.keys():
            dwl = _param['line_'+i+'_truncexp'] # in nm
            if 'line_'+i+'_pow' in _param:
                pow = _param['line_'+i+'_pow']
            else:
                pow = 1.0
            f += _param[l]*np.exp(-np.abs(oi['WL']-wl0)**pow/(dwl/1000)**pow)*(oi['WL']>=wl0)
            f += _param[l]*np.exp(-np.abs(oi['WL']-wl0)**(2*pow)/(dwl/1000/10)**(2*pow))*(oi['WL']<wl0)
        if 'line_'+i+'_gaussian' in _param.keys():
            dwl = _param['line_'+i+'_gaussian'] # in nm
            if 'line_'+i+'_power' in _param.keys():
                _pow = _param['line_'+i+'_power']
            else:
                _pow = 2.0
            _tmp = _param[l]*np.exp(-np.abs(oi['WL']-wl0)**_pow/
                                  (2*(dwl/1000/2.35482)**_pow))
            #print('_tmp', _tmp.max(), _param[l])
            #print(_param[l])
            f += _tmp
    #print('f', f.min(), f.max())

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
        # -- inverse Daubechies wavelet transform
        Y = np.array([_param[k] for k in sw])
        Y = dw.oneD(Y, -n, order=8)
        #if min(X)<min(oi['WL']) or max(X)>max(oi['WL']):
        #    print('WL range !')
        f += np.interp(oi['WL'], X, Y)

    # == arbitrary, expressed as string ===================
    if 'spectrum' in _param.keys():
        sp = _param['spectrum']
        sp = sp.replace('$WL', 'oi["WL"]')
        for k in _param.keys():
            if k in sp:
                sp.replace('$'+k, str(_param[k]))
        try:
            f += eval(sp)
        except:
            print('!!! cannot evaluate', sp)
    return f

def _Kepler3rdLaw(a=None, P=None, M1M2=None):
    """
    - a in solar radii (scalar or None)
    - P in days (scalar or None)
    - M1M2 = M1+M2 in solar masses (scalar or None)

    at least 2 of the 3 should be input, then the function returns the
    third one in the proper unit. If not, check if the 3 are
    consistent (boolean).

    REF: 'The call to adopt a nominal set of astrophysical parameters
    and constants to improve the accuracy of fundamental physical
    properties of stars.' from P. Harmanec [1106.1508v1.pdf in
    Astro-Ph]
    """
    G = 6.67428e-11
    Msol = 1.988419e30
    Rsol = 6.95508e8
    C = G*Msol*(86400)**2/(4*np.pi**2*Rsol**3)
    eC = 0.0075
    if a is None:
        return (P**2*M1M2*C)**(1/3.)
    elif P is None:
        return (a**3/(M1M2*C))**(1/2.)
    elif M1M2 is None:
        return a**3/(P**2*C)
    else:
        return abs(a**3/(P**2*M1M2) - C)<=eC

def _ti2campbell(ti, c=None, deg=True):
    """
    Thiele-Innes parameters {'A':,'B':,'G':,'F':} to Campbell {'incl':,'omega':,'OMEGA':,'a':}
    https://arxiv.org/pdf/2206.05726.pdf Appendix A
    https://gitlab.obspm.fr/gaia/nsstools/-/blob/main/nsstools/classes.py?ref_type=heads
    """
    # -- semi-major axis
    u = (ti['A']**2 + ti['B']**2 + ti['F']**2 + ti['G']**2)/2
    v = ti['A']*ti['G'] - ti['B']*ti['F']
    a = np.sqrt(u + np.sqrt(u**2-v**2))

    # -- omega ± OMEGA
    opO = np.arctan2(ti['B']-ti['F'], ti['G']+ti['A'])
    if not c is None:
        print('opO: %.3f (%.3f) delta=%.3f'%(opO, c['omega']+c['OMEGA'], opO-c['omega']-c['OMEGA']))
        print('   sign(B-F)=%2.0f sign(sin(opO))=%2.0f'%(np.sign(ti['B']-ti['F']),
                                                         np.sign(np.sin(opO)) ))
    if np.sign(ti['B']-ti['F'])==-1 and np.sign(np.sin(opO))==1:
        opO -= np.pi
    if np.sign(ti['B']-ti['F'])==1 and np.sign(np.sin(opO))==-1:
        opO += np.pi
    if not c is None:
        print('   opO: %.3f (%.3f) delta=%.3f'%(opO, c['omega']+c['OMEGA'], opO-c['omega']-c['OMEGA']))

    omO = np.arctan2(ti['B']+ti['F'], ti['G']-ti['A'])
    if not c is None:
        print('omO: %.3f (%.3f) delta=%.3f'%(omO, c['omega']-c['OMEGA'], omO-c['omega']+c['OMEGA']))
        print('   sign(-B-F)=%2.0f sign(sin(omO))=%2.0f'%(np.sign(-ti['B']-ti['F']),
                                                         np.sign(np.sin(omO)) ))

    if np.sign(-ti['B']-ti['F'])==-1 and np.sign(np.sin(omO))==1:
        omO -= np.pi
    if np.sign(-ti['B']-ti['F'])==1 and np.sign(np.sin(omO))==-1:
        omO += np.pi
    if not c is None:
        print('   omO: %.3f (%.3f) delta=%.3f'%(omO, c['omega']-c['OMEGA'], omO-c['omega']+c['OMEGA']))

    omega = ((opO+omO)/2)#%np.pi
    if omega<0:
        omega += 2*np.pi
    OMEGA = ((opO-omO)/2)#%np.pi
    if not c is None:
        print('omega: %.3f (%.3f) delta=%.3f'%(omega, c['omega'], omega-c['omega']))
        print('OMEGA: %.3f (%.3f) delta=%.3f'%(OMEGA, c['OMEGA'], OMEGA-c['OMEGA']))

    if OMEGA<0:
        omega += 3*np.pi
        omega = omega%(2*np.pi)
        OMEGA += np.pi
        OMEGA = OMEGA%(2*np.pi)
        if not c is None:
            print('correct for OMEGA<0')
            print('omega: %.3f (%.3f) delta=%.3f'%(omega, c['omega'], omega-c['omega']))
            print('OMEGA: %.3f (%.3f) delta=%.3f'%(OMEGA, c['OMEGA'], OMEGA-c['OMEGA']))

    #print(np.sign(np.sin(opO)), np.sign(ti['B']-ti['F']))
    # -- inclination
    d1 = np.abs((ti['A']+ti['G'])*np.cos(omO))
    d2 = np.abs((ti['F']-ti['B'])*np.sin(omO))
    d3 = np.abs((ti['A']-ti['G'])*np.cos(opO))
    d4 = np.abs((ti['F']+ti['B'])*np.sin(opO))
    if d1>=d2:
        incl = 2*np.arctan2(np.sqrt(d3), np.sqrt(d1))
    else:
        incl = 2*np.arctan2(np.sqrt(d4), np.sqrt(d2))
    if not c is None:
        print('d1,2,3,4: %.3f, %.3f, %.3f, %.3f'%(d1,d2,d3,d4))
        print('incl %.3f (%.3f) delta=%.3f'%(incl, c['incl'], incl-c['incl']))
        print('  %swith d3/d1: %.3f'%('*' if d1>=d2 else ' ', 2*np.arctan(np.sqrt(d3/d1))))
        print('  %swith d4/d2: %.3f'%(' ' if d1>=d2 else '*',2*np.arctan(np.sqrt(d4/d2))))

    if deg:
       return {'a':a, 'omega':omega*180/np.pi, 'OMEGA':OMEGA*180/np.pi, 'incl':incl*180/np.pi}
    else:
        return {'a':a, 'omega':omega, 'OMEGA':OMEGA, 'incl':incl}

def _campbell2ti(c):
    """
    Campbell {'incl':,'omega':,'OMEGA':,'a':} to Thiele-Innes parameters {'A':,'B':,'G':,'F':}

    angles in radians
    """
    return {'A': c['a']*(np.cos(c['omega'])*np.cos(c['OMEGA']) - np.sin(c['omega'])*np.sin(c['OMEGA'])*np.cos(c['incl'])),
            'B': c['a']*(np.cos(c['omega'])*np.sin(c['OMEGA']) + np.sin(c['omega'])*np.cos(c['OMEGA'])*np.cos(c['incl'])),
            'F':-c['a']*(np.sin(c['omega'])*np.cos(c['OMEGA']) + np.cos(c['omega'])*np.sin(c['OMEGA'])*np.cos(c['incl'])),
            'G':-c['a']*(np.sin(c['omega'])*np.sin(c['OMEGA']) - np.cos(c['omega'])*np.cos(c['OMEGA'])*np.cos(c['incl'])),
           }

newOrbitalParameters = False

#vSign = -1 # checked against orbitize! on Jul23 2024

def _orbit(t, P, Vrad=False, verbose=False, withZ=False):
    """
    position or Vrad for binary:  𝑉𝑟(𝑡)=𝐾[cos(𝜔+𝜈(𝑡))+𝑒cos𝜔]+𝛾
    # K1 ≡ radial velocity semi ‐ amplitude of host star = 2pi a_1*sin(i)/(P*sqrt(1−e**2))
    # a1 = a*m2/(m1+m2)
    # mass function = m2**3 sin(i)**3 / (m1+m2)**2

    parameters: e, P, MJD0, a, omega, OMEGA, incl (angles in degrees, dates in days)
                or
                e, P, MJD0, A, B, G, F (4 Thiele-Innes parameters, dates in days)

    alternatively, if M (Msun) and plx (mas) are given, a is computed from Kepler third law.

    omega, the argument of periastron, is the one of secondary wrt to primary! (checked definition against orbitize!)

    t = array of MJDs

    return x,y by default. If Vrad='a', 'b', 'a-b' return velocity of the primary, secondary, or
    the velocity dfference. For velocities, 'gamma' should be given, as well as:
    - 'Ka', 'Kb'
    - or 'Ma' and 'Mb'
    - or 'M' and 'q'==Mb/Ma

    a slope in gamma, adding (t-MJD0)*'gamma/d'

    """
    # -- Thiele-Innes parameters
    if 'A' in P and 'B' in P and 'F' in P and 'G' in P:
        P.update(_ti2campbell(P))

    # -- Kepler Third Law
    if not 'P' in P and ('a' in P and 'plx' in P and 'M' in P):
        P['P'] = np.sqrt((P['a']/P['plx'])**3/P['M'])*365.25
        print('_orbit: setting P')
    if not 'P' in P and ('a' in P and 'plx' in P and 'Ma' in P and 'Mb' in P):
        P['P'] = np.sqrt((P['a']/P['plx'])**3/(P['Ma']+P['Mb']))*365.25
        print('_orbit: setting P')
    try:
        M = 2*np.pi*(t-P['MJD0'])/P['P']
    except:
        M = 2*np.pi*(np.array(t)-P['MJD0'])/P['P']

    E = 0
    for i in range(20):
        E = M + np.abs(P['e'])*np.sin(E)
    cos_nu = (np.cos(E)-np.abs(P['e']))/(1-np.abs(P['e'])*np.cos(E))
    nu = np.arccos(cos_nu)
    nu[np.sin(M)<0] = 2*np.pi-nu[np.sin(M)<0]

    if 'Ma' in P and 'Mb' in P:
        P['M'] = P['Ma']+P['Mb']
        P['q'] = P['Mb']/P['Ma']
        print(P)
        print('_orbit: setting M, q')
    a_au = None
    if 'M' in P and 'P' in P:
        a_au = (P['M']*(P['P']/365.25)**2)**(1/3.) # in AU

    if not 'plx' in P and ('a' in P and 'M' in P):
        P['plx'] = P['a']/a_au
        #print('_orbit: setting plx')

    if not 'M' in P and ('a' in P and 'plx' in P):
        P['M'] = (P['a']/P['plx'])**3/(P['P']/365.25)**2 # in Msun
        a_au = (P['M']*(P['P']/365.25)**2)**(1/3.) # in AU
        #print('_orbit: setting M')

    if not 'a' in P and 'plx' in P:
        P['a'] = a_au*P['plx']
        #print('_orbit: setting a', P['a'])

    if not 'K' in P and (not 'Ka' in P and not 'Kb' in P) and not a_au is None:
        P['K'] = 2*np.pi*a_au*1.495978707e8*np.sin(P['incl']*np.pi/180)/(P['P']*24*3600*np.sqrt(1-np.abs(P['e'])**2)) # km/s
        #print('_orbit: setting K', P['K'])

    if not 'Ka' in P and not 'Kb' in P and 'q' in P: # q = Mb/Ma
        P['Ka'] = P['K']/(1+1/P['q'])
        P['Kb'] = P['K']/(1+P['q'])
        #print('_orbit: setting Ka, Kb', P['Ka'], P['Kb'])

    # separation
    if 'a' in P:
        r = P['a']*(1-np.abs(P['e'])**2)/(1+np.abs(P['e'])*np.cos(nu))
        _o, _O, _i = P['omega']*np.pi/180, P['OMEGA']*np.pi/180, P['incl']*np.pi/180

        if newOrbitalParameters:
            # -- Householder & Weiss, 2023
            # -- https://arxiv.org/pdf/2212.06966 eq. 3
            # -- DOES NOT SEEM TO WORK...
            x = r*(np.cos(_O)*np.cos(nu+_o) - np.sin(_O)*np.sin(nu+_o)*np.cos(_i) )
            y = r*(np.sin(_O)*np.cos(nu+_o) + np.cos(_O)*np.sin(nu+_o)*np.cos(_i) )
            z = r*np.sin(nu+_o)*np.sin(_i)
        else:
            # -- Antoine's implementation, same as Alex...
            x, y, z = r*np.cos(nu), r*np.sin(nu), 0
            # -- omega: argument of peri-passage, "-" because x to y is counter clockwise on sky
            x, y, z = x*np.cos(-_o) + y*np.sin(-_o), \
                     -x*np.sin(-_o) + y*np.cos(-_o), \
                      z
            # -- inclination
            x, y, z = x, \
                      y*np.cos(_i+np.pi) + z*np.sin(_i+np.pi), \
                     -y*np.sin(_i+np.pi) + z*np.cos(_i+np.pi)
            # -- OMEGA: longitude of ascending node (z <0 -> >0), ref is North==y, not x!
            x, y, z = x*np.cos(_O-np.pi/2) + y*np.sin(_O-np.pi/2), \
                     -x*np.sin(_O-np.pi/2) + y*np.cos(_O-np.pi/2), \
                      z
    VBVA = None
    if 'Ka' in P and 'Kb' in P:
        #VA =  vSign*P['Ka']*(np.cos((P['omega'])*np.pi/180+nu) + np.abs(P['e'])*np.cos((P['omega'])*np.pi/180))
        #VB = -vSign*P['Kb']*(np.cos((P['omega'])*np.pi/180+nu) + np.abs(P['e'])*np.cos((P['omega'])*np.pi/180))

        # -- sign calibrated assuming omega secondary wrt to primary (same as orbitize!)
        VA = -P['Ka']*(np.cos((P['omega'])*np.pi/180+nu) + np.abs(P['e'])*np.cos((P['omega'])*np.pi/180))
        VB =  P['Kb']*(np.cos((P['omega'])*np.pi/180+nu) + np.abs(P['e'])*np.cos((P['omega'])*np.pi/180))
        if 'gamma' in P:
            VA += P['gamma']
            VB += P['gamma']
        # -- slope in gamma velocity
        if 'gamma/d' in P:
            try:
                VA += (t-P['MJD0'])*P['gamma/d']
                VB += (t-P['MJD0'])*P['gamma/d']
            except:
                VA += (np.array(t)-P['MJD0'])*P['gamma/d']
                VB += (np.array(t)-P['MJD0'])*P['gamma/d']
        VBVA = VB-VA
    if 'K' in P and VBVA is None:
        # -- "Vb-Va"
        VBVA = P['K']*(np.cos((P['omega'])*np.pi/180+nu) + np.abs(P['e'])*np.cos((P['omega'])*np.pi/180))

    if not Vrad is False:
        if Vrad is True:
            return VBVA
        if type(Vrad) == str and Vrad=='a':
            return VA
        if type(Vrad) == str and Vrad=='b':
            return VB
        if type(Vrad) == str and Vrad=='b-a':
            return VBVA
        if type(Vrad) == str and Vrad=='a-b':
            return -VBVA
    else:
        if withZ:
            return (x,y,z)
        else:
            return (x,y)

def _expandVec(x, n:int, dx=None):
    """
    x : vector of increasing values
    n : expansion factor (integer) should be >1
    dx: full width at value x, used incase x is scalar

    add n values in between values of x using linear interpolation, and n values before
    and after the first and last values

    return vector of length 2n+1 for the scalar case (from x=dx/2 to x+dx/2)
    and len(x)*(n+1) + n in case x is a vector.

    examples:

    _expandVec([0, 0.3, 0.9], 2)
    > array([-0.2, -0.1,  0. ,  0.1,  0.2,  0.3,  0.5,  0.7,  0.9,  1.1,  1.3])

    _expandVec(0.4, 4, dx=0.1)
    > array([0.35 , 0.375, 0.4  , 0.425, 0.45 ])

    """
    if dx is None:
        res = np.interp(np.linspace(0, 1, (n+1)*(len(x)-1)+1), np.linspace(0, 1, len(x)), x)
        t = np.linspace(0, 1, n+2)
        res = np.append(res, (x[-1]+t*(x[-1]-x[-2]))[1:-1])
        res = np.append(res[::-1], (x[0]-t*(x[1]-x[0]))[1:-1])[::-1]
    elif type(x) == float or len(x)==1:
        res = np.linspace(x-dx/2, x+dx/2, 2*n+1)
    return res


def VsingleOI(oi, param, noT3=False, imFov=None, imPix=None, imX=0, imY=0, imMJD=None,
              timeit=False, indent=0, _ffrac=1.0, _dwl=0.0, fullOutput=False):
    """
    build copy of OI, compute VIS, VIS2 and T3 for a single object parametrized
    with param

    oi: result from oiutils.loadOI (a dict), or a list of results (list of dict)

    param: a dictionnary with the possible keys defined below.

    imFov: field of view of synthetic image (in mas). if None (default), will
        not produce a sinthetic image
    imPix: imPixel size (in mas) for synthetic image
    imX, imY: coordinates of the center of the image (in mas). default is 0,0

    Possible keys in the parameters dictionnary:

    position:
    ---------
    'x', 'y': define position in the field, in mas
    Note that 'x' and 'y' can be expressed in terms of the MJD, e.g.:
        'x': '$x0 + ($MJD-57200)*$dx' which requires 'x0' and 'dx' to be defined

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
    'az ampi', 'az projangi': defines the cos variation amplitude and phase for
        i-th nodes along the azimuth

    crescent:
    ---------
    crout, crin: the diameter of the outer and inner disk
    croff: offset in fraction of (crout-crin) -1...1, where 0 is a ring
    crprojang: the PA of the thick part of the crescent (in deg)
    the orientation of the crescent is defined by 'incl' and 'projang', defining
        the large axis

    flux modeling:
    --------------
    if nothing is specified, flux is assume equal to 1 at all wavelengths

    'f' or 'f0': if the constant flux (as function of wavelength)
    'fi': polynomial amplitude in (wl-wl_min)**i (wl in um)
    'fpow', 'famp': famp*(wl-wl_min)**fpow (wl in um)
    spectral lines:
    'line_i_f': amplitude of line i (>0 for emission, <0 for absorption)
    'line_i_wl0': central wavelength of line i (um)
    'line_i_gaussian': fwhm for Gaussian profile (warning: in nm, not um!)
        or
    'line_i_lorentzian': width for Lorentzian (warning: in nm, not um!)
    """
    t0 = time.time()
    modErr = 1 # set to 0 so errs are 0's

    # -- compute self-referenced parameters
    _param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))
    res = {}
    if fullOutput:
        for k in['telescopes', 'baselines', 'triangles']:
            if k in oi:
                res[k] = oi[k]

    # -- what do we inherit from original data:
    for k in ['MJD', 'WL', 'fit']:
        if k in oi.keys():
            res[k] = oi[k].copy()
    for k in ['insname', 'filename']:
        if k in oi.keys():
            res[k] = oi[k]
    if fullOutput:
        for k in ['header']:
            if k in oi.keys():
                res[k] = oi[k].copy()

    # -- small shift in wavelength (for bandwith smearing)
    # -- offset
    res['WL'] += _dwl
    # -- scaling
    cwl = 1.0 + _dwl/np.mean(res['WL'])

    if 'fit' in oi and 'smear' in oi['fit']:
        smear = oi['fit']['smear']
    else:
        smear = None

    _debug = False
    # -- copy u,v, etc, exampdin WL range in case of smearing

    if not smear is None:
        _debug = False
        if _debug:
            print('DBG> VsingleOI: init smearing')
        # expand the WL table -> re bin *after* combining components!
        res['binWL'] = res['WL']*1.0
        #res['WL'] = np.linspace(res['WL'].min(), res['WL'].max(), len(res['WL'])*smear)
        #res['WL'] = np.linspace(res['WL'][0], res['WL'][-1], len(res['WL'])*smear)
        if len(res['WL'])==1:
            res['WL'] = _expandVec(res['WL'][0], smear-1, dx=res['dWL'][0])
        else:
            res['WL'] = _expandVec(res['WL'], smear-1)

        if _debug:
            print('DBG> old', _WL.shape, 'new', res['WL'].shape)
            #print('')
        if not 'dWL' in res:
            res['dWL'] = np.gradient(res['WL'])
        else:
            res['dWL'] = np.interp(res['WL'], _WL, res['dWL'])

    if 'OI_FLUX' in oi:
        res['OI_FLUX'] = {}
        for k in oi['OI_FLUX']:
            res['OI_FLUX'][k] = {}
            res['OI_FLUX'][k]['MJD'] = oi['OI_FLUX'][k]['MJD']
            res['OI_FLUX'][k]['MJD2'] = oi['OI_FLUX'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_FLUX'][k]['FLAG'] = np.bool(0.0*res['OI_FLUX'][k]['MJD2'])
            res['OI_FLUX'][k]['FLUX'] = 1+0.0*res['OI_FLUX'][k]['MJD2']
            res['OI_FLUX'][k]['EFLUX'] = 1+0.0*res['OI_FLUX'][k]['MJD2']

    for e in ['OI_VIS', 'OI_VIS2', 'OI_CF']:
        if e in oi:
            res[e] = {}
            for k in oi[e]:
                res[e][k] = {}
                res[e][k]['MJD'] = oi[e][k]['MJD']
                res[e][k]['MJD2'] = oi[e][k]['MJD'][:,None] + 0*res['WL'][None,:]
                res[e][k]['FLAG'] = np.bool(0.0*res[e][k]['MJD2'])
                res[e][k]['u/wl'] = oi[e][k]['u'][:,None]/res['WL'][None,:]
                res[e][k]['v/wl'] = oi[e][k]['v'][:,None]/res['WL'][None,:]
                res[e][k]['B/wl'] = np.sqrt(oi[e][k]['u'][:,None]**2+
                                            oi[e][k]['v'][:,None]**2)/res['WL'][None,:]
    if 'OI_T3' in oi:
        res['OI_T3'] = {}
        for k in oi['OI_T3']:
            res['OI_T3'][k] = {}
            res['OI_T3'][k]['MJD'] = oi['OI_T3'][k]['MJD']
            res['OI_T3'][k]['u1'] = oi['OI_T3'][k]['u1']
            res['OI_T3'][k]['u2'] = oi['OI_T3'][k]['u2']
            res['OI_T3'][k]['v1'] = oi['OI_T3'][k]['v1']
            res['OI_T3'][k]['v2'] = oi['OI_T3'][k]['v2']
            res['OI_T3'][k]['B1'] = oi['OI_T3'][k]['B1']
            res['OI_T3'][k]['B2'] = oi['OI_T3'][k]['B2']
            res['OI_T3'][k]['B3'] = oi['OI_T3'][k]['B3']
            res['OI_T3'][k]['MJD2'] = oi['OI_T3'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_T3'][k]['FLAG'] = np.bool(0.0*res['OI_T3'][k]['MJD2'])
            res['OI_T3'][k]['u1/wl'] = oi['OI_T3'][k]['u1'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['v1/wl'] = oi['OI_T3'][k]['v1'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['u2/wl'] = oi['OI_T3'][k]['u2'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['v2/wl'] = oi['OI_T3'][k]['v2'][:,None]/res['WL'][None,:]
            bmax = np.maximum(oi['OI_T3'][k]['B1'], oi['OI_T3'][k]['B2'])
            bmax = np.maximum(oi['OI_T3'][k]['B3'], bmax)
            bmin = np.minimum(oi['OI_T3'][k]['B1'], oi['OI_T3'][k]['B2'])
            bmin = np.minimum(oi['OI_T3'][k]['B3'], bmin)
            res['OI_T3'][k]['Bmax/wl'] = bmax[:,None]/res['WL'][None,:]
            res['OI_T3'][k]['Bmin/wl'] = bmin[:,None]/res['WL'][None,:]
            res['OI_T3'][k]['Bavg/wl'] = (oi['OI_T3'][k]['B1'][:,None]+
                                        oi['OI_T3'][k]['B2'][:,None]+
                                        oi['OI_T3'][k]['B3'][:,None])/res['WL'][None,:]/3

    # -- model -> no telluric features
    res['TELLURICS'] = np.ones(res['WL'].shape)

    #print('->', imFov, imPix)
    if not imFov is None and not imPix is None:
        # -- image coordinates in mas
        X, Y = np.meshgrid(np.linspace(-imFov/2, imFov/2, 2*int(imFov/imPix/2)+1),
                           np.linspace(-imFov/2, imFov/2, 2*int(imFov/imPix/2)+1))
        if not imX is None:
            X += imX
        if not imY is None:
            Y += imY
        I = np.zeros(X.shape)
        #print('setting synthetic images')
    else:
        X, Y = 0., 0.
        I = None

    # -- spectrum, fraction of it if needed for bandwith smearing
    # -- vector, same length as oi['WL']
    flux = Ssingle(res, _param, noLambda=True)*_ffrac

    if any(flux>0):
        # -- check negativity of spectrum
        negativity = np.sum(flux[flux<0])/np.sum(flux[flux>=0])
    elif all(flux<0):
        negativity = np.sum(flux[flux<0])
    else:
        negativity = 0

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
        baselines = list(oi[key].keys())
    else:
        baselines = []

    # -- max baseline
    Bwlmax = 0.0
    for k in baselines:
        Bwlmax = max(Bwlmax, np.max(res[key][k]['B/wl']/cwl))
    if Bwlmax==0:
        Bwlmax = 50.0

    # -- position of the element in the field
    if 'x' in _param.keys() and 'y' in _param.keys():
        _xs, _ys = _param['x'], _param['y']
        if type(_xs)==str and '$MJD' in _xs:
            x = lambda o: eval(_xs.replace('$MJD', "o['MJD2']"))
        elif type(_xs)==str and _xs=='orbit':
            # -- collect orbital parameters
            oP = {k.split('orb ')[1]:_param[k] for k in _param if k.startswith('orb ')}
            x = lambda o: _orbit(o['MJD2'], oP)[0]
        else:
            x = lambda o: np.ones(len(o['MJD']))[:,None]*_xs+0*res['WL'][None,:]

        if type(_ys)==str and '$MJD' in _ys:
            y = lambda o: eval(_ys.replace('$MJD', "o['MJD2']"))
        elif type(_ys)==str and _ys=='orbit':
            # -- collect orbital parameters
            oP = {k.split('orb ')[1]:_param[k] for k in _param if k.startswith('orb ')}
            y = lambda o: _orbit(o['MJD2'], oP)[1]
        else:
            y = lambda o: np.ones(len(o['MJD']))[:,None]*_ys+0*res['WL'][None,:]

    else:
        _xs, _ys = 0, 0
        x = lambda o: np.ones(len(o['MJD']))[:,None]*0+0*res['WL'][None,:]
        y = lambda o: np.ones(len(o['MJD']))[:,None]*0+0*res['WL'][None,:]

    #print('debug:', x, y)
    # -- 'slant' i.e. linear variation of flux
    if 'slant' in list(_param.keys()) and 'slant projang' in list(_param.keys()):
        du, dv = 1e-6, 1e-6 # in meters / microns
        if _param['slant']<0:
            _param['slant'] = np.abs(_param['slant'])
            _param['slant projang'] = (_param['slant projang']+180)%360
    else:
        du, dv = 0.0, 0.0

    # -- user-defined wavelength range
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        WLR = oi['fit']['wl ranges']
    else:
        WLR = [(min(res['WL']), max(res['WL']))]
    wwl = np.zeros(res['WL'].shape)
    for wlr in WLR:
        # -- compute a bit larger, to avoid border effects
        wwl += (res['WL']>=wlr[0]-0.05*(wlr[1]-wlr[0]))*\
               (res['WL']<=wlr[1]+0.05*(wlr[1]-wlr[0]))
    wwl = np.bool_(wwl)

    # -- do we need to apply a stretch?
    if 'projang' in _param.keys() and 'incl' in _param.keys():
        rot = -_param['projang']*np.pi/180
        _uwl = lambda z: np.cos(_param['incl']*np.pi/180)*\
                         (np.cos(rot)*z['u/wl'][:,wwl]/cwl +
                         np.sin(rot)*z['v/wl'][:,wwl]/cwl)
        _vwl = lambda z: -np.sin(rot)*z['u/wl'][:,wwl]/cwl + \
                          np.cos(rot)*z['v/wl'][:,wwl]/cwl
        _Bwl = lambda z: np.sqrt(_uwl(z)**2 + _vwl(z)**2)

        if du:
            _udu = lambda z: (np.cos(rot)*z['u/wl'][:,wwl]/cwl +
                              np.sin(rot)*z['v/wl'][:,wwl]/cwl)*\
                              np.cos(np.pi*_param['incl']/180)+du/res['WL'][wwl]
            _vdu = lambda z: -np.sin(rot)*z['u/wl'][:,wwl]/cwl + np.cos(rot)*z['v/wl'][:,wwl]/cwl
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: (np.cos(rot)*z['u/wl'][:,wwl]/cwl +
                              np.sin(rot)*z['v/wl'][:,wwl]/cwl)*np.cos(np.pi*_param['incl']/180)
            _vdv = lambda z: -np.sin(rot)*z['u/wl'][:,wwl]/cwl + \
                              np.cos(rot)*z['v/wl'][:,wwl]/cwl + \
                              dv/res['WL'][wwl]
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            if len(baselines):
                _X = (np.cos(rot)*(X-np.mean(x(res[key][baselines[0]]))) + \
                       np.sin(rot)*(Y-np.mean(y(res[key][baselines[0]]))))/np.cos(_param['incl']*np.pi/180)
                _Y = -np.sin(rot)*(X-np.mean(x(res[key][baselines[0]]))) + \
                       np.cos(rot)*(Y-np.mean(y(res[key][baselines[0]])))
            else:
                _X = (np.cos(rot)*(X-_xs) + np.sin(rot)*(Y-_ys))/np.cos(_param['incl']*np.pi/180)
                _Y = -np.sin(rot)*(X-_xs) + np.cos(rot)*(Y-_ys)

            R = np.sqrt(_X**2+_Y**2)
    else:
        _uwl = lambda z: z['u/wl'][:,wwl]/cwl
        _vwl = lambda z: z['v/wl'][:,wwl]/cwl
        _Bwl = lambda z: z['B/wl'][:,wwl]/cwl
        if du:
            _udu = lambda z: z['u/wl'][:,wwl]/cwl+du/res['WL'][wwl]
            _vdu = lambda z: z['v/wl'][:,wwl]/cwl
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: z['u/wl'][:,wwl]/cwl
            _vdv = lambda z: z['v/wl'][:,wwl]/cwl+dv/res['WL'][wwl]
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            if len(baselines):
                _X, _Y = X-np.mean(x(res[key][baselines[0]])), Y-np.mean(y(res[key][baselines[0]]))
            else:
                _X, _Y = X-_xs, Y-_ys

            R = np.sqrt(_X**2+_Y**2)

    # -- phase offset
    #phi = lambda z: -2j*_c*(z['u/wl']*x+z['v/wl']*y)
    PHI = lambda o: np.exp(-2j*_c*(o['u/wl'][:,wwl]/cwl*x(o)[:,wwl] +
                                   o['v/wl'][:,wwl]/cwl*y(o)[:,wwl]))

    if du:
        #dPHIdu = lambda z: -2j*_c*x*PHI(z)/oi['WL']
        #dPHIdv = lambda z: -2j*_c*y*PHI(z)/oi['WL']
        PHIdu = lambda o: np.exp(-2j*_c*((o['u/wl'][:,wwl]/cwl +
                                         du/res['WL'][:,wwl])*x(o)[:,wwl] +
                                          z['v/wl'][:,wwl]/cwl*y(o)[:,wwl]))
        PHIdv = lambda o: np.exp(-2j*_c*(o['u/wl'][:,wwl]/cwl*x(o)[:,wwl] +
                                        (o['v/wl'][:,wwl]/cwl+dv/res['WL'][wwl])*y(o)[:,wwl]))

    # -- guess which visibility function
    if 'sparse' in _param:
        # -- UGLY: duplicate code -> make sure to update "ul" below !
        Vf = lambda z: VsparseImage(z['u'], z['v'], res['WL'][wwl], _param, z['MJD'])
        if not I is None:
            #print('sparse image')
            if imFov is None:
                imN = None
            else:
                imN = int(np.sqrt(len(X.flatten())))
            if imMJD is None and type(_param['sparse'])==dict:
                if 'configurations per MJD' in oi:
                    imMJD = np.mean(list(oi['configurations per MJD'].keys()))
                    print('WARNING: synthetic image for <MJD>=', imMJD)
            if not imMJD is None and type(imMJD)!=np.ndarray:
                imMJD = np.array([imMJD])
            tmp = VsparseImage(np.array([1]), np.array([1]), res['WL'][wwl], _param,
                               imMJD, fullOutput=True, imFov=imFov, imPix=imPix,
                               imX=imX, imY=imY, imN=imN)
            I = tmp[1]/np.sum(tmp[1])
    elif 'Vin' in _param or 'V1mas' in _param or 'Vin_Mm/s' in _param: # == Keplerian disk ==
        # == TODO: make this faster by only computing for needed wavelength
        if imFov is None:
            imN = None
        else:
            imN = int(np.sqrt(len(X.flatten())))
        tmp = Vkepler([1], [1], res['WL'][wwl], _param, fullOutput=True,
                      imFov=imFov, imPix=imPix, imX=imX, imY=imY, imN=imN)
        flux = np.zeros(len(res['WL']))
        flux[wwl] = tmp[1]

        if not I is None:
            #print('shape', tmp[4].shape)
            I = np.zeros((len(res['WL']), tmp[4].shape[1], tmp[4].shape[2]))
            I[wwl,:,:] = tmp[4]
            I *= np.sum(flux[:,None,None])/np.sum(I)
        Vf = lambda z: Vkepler(z['u'], z['v'], res['WL'][wwl], _param)
        # -- TEST
        #kt = list(oi['OI_VIS2'].keys())[0]
        #print('test', kt, np.abs(Vf(oi['OI_VIS2'][kt])))
    elif 'ud' in _param.keys(): # == uniform disk ================================
        Rout = _param['ud']/2
        if any(flux>0):
            Vf = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bwl(z) + 1e-12)/(_c*_param['ud']*_Bwl(z)+ 1e-12)
        else:
            # -- save time
            Vf = lambda z: np.zeros(_Bwl(z).shape)

        if _param['ud']<0:
            negativity += np.abs(_c*_param['ud']*Bwlmax)

        if du: # -- slanted
            Vfdu = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdu(z))/(_c*_param['ud']*_Bdu(z))
            Vfdv = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdv(z))/(_c*_param['ud']*_Bdv(z))
        if not I is None: # -- Compute image
            # -- without anti aliasing
            #I = R<=_param['ud']/2
            # -- anti aliasing:
            na = 3
            for _imX in np.linspace(-imPix/2, imPix/2, na+2)[1:-1]:
                for _imY in np.linspace(-imPix/2, imPix/2, na+2)[1:-1]:
                    R2 = (_X-_imX)**2+(_Y-_imY)**2
                    I += R2<=(_param['ud']/2)**2
            I/=na**2
            if np.sum(I)==0:
                # -- unresolved -> single imPixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)
    elif 'fwhm' in _param.keys(): # == gaussian ================================
        #if _param['fwhm']<0:
        #    negativity += np.max(np.abs(_c*_param['fwhm']*_Bwl(z)))

        if np.abs(_param['fwhm'])>1e-3: # UGLY KLUDGE!
            a = 1./(2.*(_param['fwhm']/(2*np.sqrt(2*np.log(2))))**2)
            Rout = np.abs(_param['fwhm'])/2
            Vf = lambda z: np.exp(-(_c*_Bwl(z))**2/a)
        else:
            a = None
            Vf = lambda z: 1 + 0*_Bwl(z)

        if not any(flux>0):
            # -- save time
            Vf = lambda z: np.zeros(_Bwl(z).shape)

        if du:
            print('WARNING: slanted gaussian does not make sense!')
            Vfdu = lambda z: np.exp(-(_c*_Bdu(z))**2/a)
            Vfdv = lambda z: np.exp(-(_c*_Bdv(z))**2/a)
        if not I is None:
            if not a is None:
                I = np.exp(-R**2*a)
            if np.sum(I)==0:
                # -- unresolved -> single imPixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)
    elif 'fwhmin' in _param and 'fwhmout' in _param:
        # -- bi-gaussian ring
        fwhmin, fwhmout = np.abs(_param['fwhmin']), np.abs(_param['fwhmout'])
        if fwhmin>fwhmout:
            fwhmin, fwhmout = fwhmout, fwhmin
        ain = 1./(2.*(fwhmin /(2*np.sqrt(2*np.log(2))))**2)
        aou = 1./(2.*(fwhmout/(2*np.sqrt(2*np.log(2))))**2)

        # -- az variation (if any)
        _n, _amp, _phi = [], [], []
        for k in _param.keys():
            if k.startswith('az amp'):
                _n.append(int(k.split('az amp')[1]))
                _phi.append(_param[k.replace('az amp', 'az projang')])
                _amp.append(_param[k])
        if len(_n)>0:
            if 'projang' in _param.keys() and 'incl' in _param.keys():
                stretch = [np.cos(np.pi*_param['incl']/180), _param['projang']]
            else:
                stretch = [1,0]
            if 'Nr' in oi['fit']:
                Nr = oi['fit']['Nr']
            else:
                Nr = min(int(20*fwhmout/fwhmin), 100)
            _r = np.linspace(0, 5*fwhmout, Nr)
            Ir = np.exp(-_r**2*aou)-np.exp(-_r**2*ain)
            negativity += np.sum(flux)*_negativityAzvar(_n, _phi, _amp)
            Vf = lambda z: _Vazvar(z['u/wl'][:,wwl]/cwl, z['v/wl'][:,wwl]/cwl,
                                   Ir, _r, _n, _phi, _amp, stretch=stretch)
            if not I is None:
                if len(baselines):
                    XY = (X-np.mean(x(res[key][baselines[0]])),
                          Y-np.mean(y(res[key][baselines[0]])))
                else:
                    XY = (X-_xs, Y-_ys)

                I = _Vazvar(None, None, Ir, _r, _n, _phi, _amp,
                            stretch=stretch, numerical=1, XY=XY)
                if np.sum(I)==0:
                    # -- unresolved -> single imPixel
                    R2 = _X**2+_Y**2
                    I = R2==np.min(R2)
        else:
            Vf = lambda z: (1/aou*np.exp(-(_c*_Bwl(z))**2/aou)-
                            1/ain*np.exp(-(_c*_Bwl(z))**2/ain))/\
                            (1/aou-1/ain)
            if not I is None:
                I = np.exp(-R**2*aou)-np.exp(-R**2*ain)

    elif 'crin' in _param and 'crout' in _param and 'croff' in _param: # crecsent
        #print('crescent')
        #if _param['crin']>_param['crout']:
        #    _crin=0.9999*_param['crout']
        #else:
        _crin = _param['crin']
        _crout = _param['crout']

        # -- offset of inner disk / outer disk
        _off = _param['croff']*(_crout-_crin)/2
        if 'crprojang' in _param:
            crpa = _param['crprojang']*np.pi/180
        else:
            crpa = 0.0
        if True:
            # == share offset between the 2 disks
            # -- offset of outer disk
            _offXo = -_off/2*np.sin(crpa)
            _offYo = -_off/2*np.cos(crpa)
            # -- offset of inner disk
            _offXi = _off/2*np.sin(crpa)
            _offYi = _off/2*np.cos(crpa)
        else:
            # == offset only inner disk
            # -- offset of outer disk
            _offXo = 0
            _offYo = 0
            # -- offset of inner disk
            _offXi = _off*np.sin(crpa)
            _offYi = _off*np.cos(crpa)

        Vf = lambda z: (np.exp(-2j*_c*_uwl(z)*_offXo
                               -2j*_c*_vwl(z)*_offYo
                               )*
                        _crout**2*2*scipy.special.j1(_c*_crout*_Bwl(z) + 1e-12)/
                                (_c*_crout*_Bwl(z)+ 1e-12) -
                        np.exp(-2j*_c*_uwl(z)*_offXi
                               -2j*_c*_vwl(z)*_offYi
                               )*
                        _crin**2*2*scipy.special.j1(_c*_crin*_Bwl(z) + 1e-12)/
                                (_c*_crin*_Bwl(z) + 1e-12))/\
                                                 (_crout**2-_crin**2)
        if not I is None:
            # -- outer disk
            R2 = (_X - _offXo)**2 + (_Y - _offYo)**2
            I = np.float64(R2<=(_crout**2/4))
            # -- remove inner disk
            R2 = (_X - _offXi)**2 + (_Y - _offYi)**2
            I -= (R2<=(_crin**2/4))
            if np.sum(I)==0:
                I = np.float64(np.abs(R2-_crin**2/4)<=imPix)
        if 'surf bri' in _param:
            # -- use this to compute flux, can be function of wavelength
            flux = np.pi*(_crout**2 - _crin**2)/4*_ffrac
            #print('diamin', diamin, 'diamout', diamout, 'Nr', Nr,
            #      'int(r*Ir)', f)
            if not '$WL' in _param['surf bri']:
                flux *= np.ones(len(res['WL']))*_param['surf bri']
            else:
                flux *= eval(_param['surf bri'].replace('$WL', 'res["WL"]'))
            if any(flux>0):
                # -- check negativity of spectrum
                negativity = np.sum(flux[flux<0])/np.sum(flux[flux>=0])
            elif all(flux<0):
                negativity = np.sum(flux[flux<0])
            else:
                negativity = 0
            if 'fit' in oi and 'ignore negative flux' in oi['fit'] and \
                oi['fit']['ignore negative flux']:
                negativity = 0.0
    elif 'diamout' in _param or 'diam' in _param: # == F(r)*G(az) ========================
        # -- disk or ring with radial and az profile

        if not 'diamin' in _param and not 'thick' in _param:
            _param['thick'] = 1.0
        if 'thick' in _param:
            if 'diam' in _param:
                diamout = _param['diam'] #*(1+min(max(1e-9,_param['thick']),1))
            if 'diamout' in _param:
                diamout = _param['diamout']
            diamin = _param['diam']*(1-min(max(1e-5,_param['thick']),1))
        else:
            diamin = _param['diamin']
            diamout = _param['diamout']
            _param['thick'] = (diamout-diamin)/diamout

        if 'Nr' in oi['fit']:
            Nr = oi['fit']['Nr']
        else:
            Nr = max(10, int(100*_param['thick'])) # -- arbitrary !!!
            #print('diamin', diamin, 'diamout', diamout, 'Nr', Nr)

        _r = np.linspace(diamin/2, diamout/2, Nr)
        _d = np.linspace(diamin, diamout, Nr)
        _mu = np.sqrt(1-(2*_r/diamout)**2)
        Rout = diamout/2

        if not 'profile' in _param:
            _param['profile'] = 'uniform'
        if '$' in _param['profile']:
            # -- generic formula
            tmp = _param['profile'].replace('$RMIN', str(np.min(_r)))
            tmp = tmp.replace('$RMAX', str(np.max(_r)))
            tmp = tmp.replace('$R', '_r')
            tmp = tmp.replace('$DMIN', str(np.min(_d)))
            tmp = tmp.replace('$DMAX', str(np.max(_d)))
            tmp = tmp.replace('$D', '_d')

            tmp = tmp.replace('$MU', '_mu')
            for k in _param.keys():
                if '$'+k in tmp:
                    tmp = tmp.replace('$'+k, str(_param[k]))
            Ir = eval(tmp)
        elif _param['profile']=='doughnut':
            Ir = 1-((_r-np.mean(_r))/np.ptp(_r)*2)**2
        elif _param['profile'].startswith('doughnut'):
            p = float(_param['profile'].split('doughnut')[1])
            Ir = 1-np.abs((_r-np.mean(_r))/np.ptp(_r)*2)**p
        elif _param['profile']=='uniform': # uniform
            Ir = np.ones(_r.shape)

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

        if 'surf bri' in _param:
            if len(_n)>0:
                # -- make sure max az var is 1
                # -- important for BB surface brightness
                psi = np.linspace(0, 360, 91)[:-1]
                psi = np.array(_n)[:,None]*(psi[None,:] + np.array(_phi)[:,None])
                tmp = 1 + np.array(_amp)[:,None]*np.cos(psi*np.pi/180)
                tmp = np.sum(tmp, axis=0)
                Ir /= max(tmp)

            # -- use this to compute flux, can be function of wavelength
            flux = 2*np.pi*np.trapezoid(Ir*_r, _r)*_ffrac
            if not '$WL' in _param['surf bri']:
                flux *= np.ones(len(res['WL']))*_param['surf bri']
            else:
                flux *= eval(_param['surf bri'].replace('$WL', 'res["WL"]'))
            if any(flux>0):
                # -- check negativity of spectrum
                negativity += np.sum(flux[flux<0])/np.sum(flux[flux>=0])
            elif all(f<0):
                negativity += np.sum(flux[flux<0])
            else:
                negativity += 0
            if 'fit' in oi and 'ignore negative flux' in oi['fit'] and \
                oi['fit']['ignore negative flux']:
                negativity = 0.0

        negativity += np.sum(flux)*_negativityAzvar(_n, _phi, _amp)
        Vf = lambda z: _Vazvar(z['u/wl'][:,wwl]/cwl, z['v/wl'][:,wwl]/cwl,
                               Ir, _r, _n, _phi, _amp, stretch=stretch)

        if not any(flux>0):
            # -- save time
            Vf = lambda z: np.zeros(_Bwl(z).shape)

        if du: # --slanted
            if len(_n):
                print('WARNING: slanted disk with azimutal variation not implemented properly!')
            Vfdu = lambda z: _Vazvar(_udu(z), _vdu(z), Ir, _r, _n, _phi, _amp,)
            Vfdv = lambda z: _Vazvar(_udv(z), _vdv(z), Ir, _r, _n, _phi, _amp,)
        if not I is None:
            if len(baselines):
                XY = (X-np.mean(x(res[key][baselines[0]])),
                      Y-np.mean(y(res[key][baselines[0]])))
            else:
                XY = (X-_xs, Y-_ys)

            I = _Vazvar(None, None, Ir, _r, _n, _phi, _amp,
                        stretch=stretch, numerical=1, XY=XY)
            if np.sum(I)==0:
                # -- unresolved -> single imPixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)
    elif any([k.startswith('ul ') for k in _param]):
        # -- only computed 1 image per night!
        MJD = []
        for _mjd in sorted(set(np.int_(res['MJD']+0.5))):
            MJD.append(np.round(np.mean(res['MJD'][np.int_(res['MJD']+0.5)==_mjd]),2))
        #try:
        sparse_param = VuLensBin(MJD, _param)
        #except:
        #    print('ERROR! for MJD=', MJD)
        #    print(_param)
        # == this does not work (yet)
        # MJDs = sorted(list(sparse_param['sparse'].keys()))
        # if len(MJDs)==1:
        #     flux *= np.sum(sparse_param['sparse'][MJDs[0]]['I'])
        # else:
        #     _param['spectrum(mjd)'] = 'np.interp($MJD, %s, %s)'%(str(MJDs),
        #         str([np.sum(sparse_param['sparse'][_mjd]['I']) for _mjd in MJDs]))

        #flux *= np.sum(sparse_param['I'])

        # -- UGLY: duplicate code -> make sure to update "sparse" below !
        Vf = lambda z: VsparseImage(z['u'], z['v'], res['WL'][wwl], sparse_param, z['MJD'])
        if not I is None:
            #print('ul synthetic image ("sparseImage")')
            if imFov is None:
                imN = None
            else:
                imN = int(np.sqrt(len(X.flatten())))
            if imMJD is None and type(sparse_param['sparse'])==dict:
                if 'configurations per MJD' in oi:
                    imMJD = np.mean(list(oi['configurations per MJD'].keys()))
                    print('WARNING: synthetic image for <MJD>=', imMJD)
            if not imMJD is None and type(imMJD)!=np.ndarray:
                imMJD = np.array([imMJD])
            tmp = VsparseImage(np.array([1]), np.array([1]), res['WL'][wwl],
                               sparse_param, imMJD, fullOutput=True,
                               imFov=imFov, imPix=imPix, imX=imX, imY=imY, imN=imN)
            I = tmp[1]#/np.sum(tmp[1])

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
    # if 'cos' in _param and 'cos pa' in _param:
    #     _C = np.cos(_param['cos pa']*np.pi/180)
    #     _S = np.sin(_param['cos pa']*np.pi/180)
    #     _xc = _S*_X + _C*_Y
    #     I *= 1 + _param['cos']*np.cos(_xc)
    # if 'sin' in _param and 'sin pa' in _param:
    #     _C = np.cos(_param['sin pa']*np.pi/180)
    #     _S = np.sin(_param['sin pa']*np.pi/180)
    #     _xc = _S*_X + _C*_Y
    #     I *= 1 + _param['sin']*np.sin(_xc)

    # -- check that slant does not lead to negative flux
    if du and np.abs(_param['slant'])>1:
        negativity += (np.abs(_param['slant'])-1)

    if timeit:
        print(' '*indent+'VsingleOI > setup %.3fms'%(1000*(time.time()-ts)))

    if 'spatial kernel' in _param:
        kfwhm = _param['spatial kernel']
        s = kfwhm/(2*np.sqrt(2*np.log(2)))

        # -- multiply complex visibilities by spatial kernel's visibility
        _ka = 1./(2*s**2)
        _ka /= (np.pi**2/180/3600/1000*1e6)**2

        # -- convolve image by spatial kernel
        if not imFov is None:
            ker = np.exp(-((X-imX)**2+(Y-imY)**2)/(2*s**2))/(s*np.sqrt(2*np.pi))
            I = scipy.signal.fftconvolve(I, ker, mode='same')
    else:
        kfwhm = None

    # == build observables: ===================================
    # res['OI_VIS'] = {}
    # res['OI_VIS2'] = {}
    # if 'OI_CF' in oi:
    #     res['OI_CF'] = {}
    tv = time.time()
    for k in baselines: # -- for each baseline
        tmp = {}
        #print('debug', oi[key][k]['u/wl'].shape)
        V = np.zeros((res[key][k]['u/wl'].shape[0],
                      res[key][k]['u/wl'].shape[1]), dtype=complex)
        if du: # -- for slanted
            V[:,wwl] = Vf(oi[key][k])
            # -- compute slant from derivative of visibility
            dVdu = (Vfdu(res[key][k]) - V[:,wwl])/du
            dVdv = (Vfdv(res[key][k]) - V[:,wwl])/dv
            dVdu /= 2*_c/res['WL'][wwl]
            dVdv /= 2*_c/res['WL'][wwl]
            # -- see https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms
            # -- relations 106 and 107
            V[:,wwl] = V[:,wwl]+1j*(np.sin(_param['slant projang']*np.pi/180)*_param['slant']/Rout*dVdu +
                                    np.cos(_param['slant projang']*np.pi/180)*_param['slant']/Rout*dVdv)
            V[:,wwl] *= PHI(res[key][k])
        else:
            #print('debug: MJD=%.3f, X=%.3f, Y=%.3f'%(np.mean(oi[key][k]['MJD']),
            #                    np.mean(x(oi[key][k])), np.mean(y(oi[key][k]))))
            V[:,wwl] = Vf(res[key][k]) * PHI(res[key][k])

        if not kfwhm is None:
            V *= (1+0j)*np.exp(-(res[key][k]['B/wl'])**2/_ka)

        tmp['|V|'] = np.abs(V)
        #print(k, Vf(oi[key][k]), tmp['|V|'])
        # -- force -180 -> 180
        tmp['PHI'] = (np.angle(V)*180/np.pi+180)%360-180
        # -- not needed, strictly speaking, takes a long time!
        for l in ['u', 'v', 'MJD',]: #'FLAG', 'MJD2' # 'B/wl']
            if '/wl' in l and cwl!=1:
                tmp[l] = oi[key][k][l].copy()/cwl
            else:
                tmp[l] = oi[key][k][l].copy()
        if 'NAME' in oi[key][k]:
            tmp['NAME'] = oi[key][k]['NAME'].copy()

        if fullOutput or not imFov is None:
            # -- slow down the code!
            for l in ['u', 'v',]:# 'u/wl', 'v/wl']:
                if '/wl' in l and cwl!=1:
                    tmp[l] = oi[key][k][l].copy()/cwl
                else:
                    tmp[l] = oi[key][k][l].copy()
            tmp['E|V|'] = np.zeros(tmp['|V|'].shape) + modErr
            tmp['EPHI'] = np.zeros(tmp['PHI'].shape) + modErr
        res['OI_VIS'][k].update(tmp)

        if 'OI_VIS2' in oi.keys():
            tmp = {}
            # -- not needed, strictly speaking, takes a long time!
            for l in ['u', 'v', 'MJD',]: #  'FLAG', 'MJD2', # 'B/wl']:
                if '/wl' in l and cwl!=1:
                    tmp[l] = oi['OI_VIS2'][k][l]/cwl
                else:
                    tmp[l] = oi['OI_VIS2'][k][l].copy()
            if 'NAME' in oi['OI_VIS2'][k]:
                tmp['NAME'] = oi['OI_VIS2'][k]['NAME'].copy()

            tmp['V2'] = np.abs(V)**2
            if fullOutput or not imFov is None:
                # -- slow down the code!
                for l in ['u', 'v', ]: #'u/wl', 'v/wl']:
                    if '/wl' in l and cwl!=1:
                        tmp[l] = oi['OI_VIS2'][k][l]/cwl
                    else:
                        tmp[l] = oi['OI_VIS2'][k][l].copy()
                tmp['EV2'] = np.zeros(tmp['V2'].shape) + modErr
            res['OI_VIS2'][k].update(tmp)
        if 'OI_CF' in oi.keys() and k in oi['OI_CF']:
            tmp = {}
            for l in [ 'u', 'v', 'MJD', ]:# 'FLAG', 'MJD2', #'B/wl',
                if '/wl' in l and cwl!=1:
                    tmp[l] = oi['OI_CF'][k][l]/cwl
                else:
                    tmp[l] = oi['OI_CF'][k][l].copy()
            if 'NAME' in oi['OI_CF'][k]:
                tmp['NAME'] = oi['OI_CF'][k]['NAME'].copy()

            tmp['CF'] = np.abs(flux[None,:]*V)
            if fullOutput or not imFov is None:
                # -- slow down the code!
                for l in ['u', 'v',]:# 'u/wl', 'v/wl']:
                    if '/wl' in l and cwl!=1:
                        tmp[l] = oi['OI_CF'][k][l]/cwl
                    else:
                        tmp[l] = oi['OI_CF'][k][l].copy()
                tmp['ECF'] = np.zeros(tmp['CF'].shape) + modErr
            res['OI_CF'][k].update(tmp)

    if timeit:
        print(' '*indent+'VsingleOI > complex vis %.3fms'%(1000*(time.time()-tv)))
    if 'OI_T3' in oi.keys() and not noT3:
        t3 = time.time()
        # res['OI_T3'] = {}
        for k in oi['OI_T3'].keys():
            #res['OI_T3'][k] = {}
            for l in ['u1', 'u2', 'v1', 'v2', 'MJD', 'formula', #'FLAG', 'Bmax/wl', 'Bmin/wl', 'Bavg/wl', 'MJD2',
                      'B1', 'B2', 'B3']:
                if '/wl' in l and cwl!=1:
                    res['OI_T3'][k][l] = oi['OI_T3'][k][l]/cwl
                else:
                    res['OI_T3'][k][l] = oi['OI_T3'][k][l].copy()
            if 'NAME' in oi['OI_T3'][k]:
                res['OI_T3'][k]['NAME'] = oi['OI_T3'][k]['NAME'].copy()

        res = computeT3fromVisOI(res)
        if timeit:
            print(' '*indent+'VsingleOI > T3 from V %.3fms'%(1000*(time.time()-t3)))

    if not I is None:
        # -- normalize image to total flux, useful for adding
        if len(I.shape)==2:
            res['MODEL'] = {'image':I/np.sum(I), 'X':X, 'Y':Y}
        else:
            res['MODEL'] = {'cube':I, 'X':X, 'Y':Y,
                            'image':np.mean(I, axis=0)}
            #print('X', X.shape, 'I', I.shape, 'image', np.mean(I, axis=0).shape)
    else:
        res['MODEL'] = {}
    res['MODEL']['negativity'] = negativity # 0 is image is >=0, 0<p<=1 otherwise
    res['MODEL']['totalflux'] = flux
    if 'OI_FLUX' in res:
        for k in res['OI_FLUX']:
            #print('@@@', res['OI_FLUX'][k]['FLUX'].shape, flux.shape)
            res['OI_FLUX'][k]['FLUX'] = 0*res['OI_FLUX'][k]['MJD'][:,None] + flux[None,:]

    if False: # -- this is not working properly :(
        if 'spectrum(mjd)' in _param:
            # WARNING: this will ignore any other "flux" or "spectrum" definition!!!
            tmp = _param['spectrum(mjd)']
            if '$MJD' in tmp:
                if '$WL' in tmp:
                    tmp.replace('$WL', '$WL[None,:]')
                    #tmp.replace('$MJD', '$MJD[:,None]')
                    res['MODEL']['totalflux(MJD)'] = tmp
                else:
                    #tmp.replace('$MJD', '$MJD[:,None]')
                    tmp += ' + $WL[None,:]'
                    res['MODEL']['totalflux(MJD)'] = tmp
                # -- KLUDGE: average out the flux over MJD
                #res['MODEL']['totalflux'] *= np.mean(tmp.replace('$WL', oi['WL']).replace('$MJD', oi['MJD']), axis=0)
            else:
                print('spectrum(mjd) should be as function of "$MJD"')
        else:
            # -- this works (I think)
            res['MODEL']['totalflux(MJD)'] = '$TFLUX[None,:] + 0*$MJD'
    res['param'] = param

    if not smear is None:
        if _debug:
            print('DBG> closing smearing')
            if not k in res['OI_VIS2'].keys():
                k = list(res['OI_VIS2'].keys())[0]
            print('DBG> OI_VIS2[V2]', k, res['OI_VIS2'][k]['V2'].shape, np.min(res['OI_VIS2'][k]['V2']))
            print('DBG> res.keys()', res.keys())
        # -- should de-bin after combining components!
        res['smear'] = smear
        if _debug:
            print('DBG> OI_VIS2[V2]', k, res['OI_VIS2'][k]['V2'].shape, np.min(res['OI_VIS2'][k]['V2']))
    return res

_sparse_image_file = ""
_sparse_image = {}
def VsparseImage(u, v, wl, param, mjd=None,  fullOutput=False,
                 imFov=None, imPix=None, imX=0, imY=0, imN=None):
    """
    complex visibility of sparse image from text file.
    u, v: 1D-ndarray spatial frequency (m). u and v must have same length N
    wl: 1D-ndarray wavelength (um). can have a different length from u and v

    param:
        'sparse': filename '.txt' should contain 3 columns (space separated)
            "x y intensity" x and y in mas, intensity is dimensionless
            Assumes each "x y" is a patch in the image. Each patch is assumed to
            have equal apparent surface, otherwise last colums must contain
            "surface * intensity".
        OR filename '.pickle': a pickled dict {'x':..., 'y':..., 'I':...} with each col
            is a 1D vector
        OR dict {'x':..., 'y':..., 'I':...} with each col
            is a 1D vector

    'sparse' can also be a dict keyed by MJD, and the closest image to "mjd" will
        be used. In this case, each element can be a txt, pickle or dict described
        above. See "VuLensBin" for a use of this

    returns:
    nd.array of shape (N,M) of complex visibilities
    """
    # -- param['sparse'] could be a dict keyed by MJD
    if not mjd is None:
        if type(param['sparse'])==dict:
            kmjd = np.array(list(param['sparse'].keys()))
            imjd = np.argmin(np.abs(mjd[None,:]-kmjd[:,None]), axis=0)
            res = np.zeros((len(u), len(wl)), dtype=complex)
            for i in set(imjd):
                #print(kmjd[i], param['sparse'][kmjd[i]])
                w = imjd==i
                p = param.copy()
                p['sparse'] = param['sparse'][kmjd[i]]
                if not fullOutput:
                    tmp = VsparseImage(u[w], v[w], wl, p, mjd=None)
                    res[w,:] = tmp
                else:
                    res = VsparseImage(u[w], v[w], wl, p, fullOutput=True, mjd=None,
                                imFov=imFov, imPix=imPix, imX=imX, imY=imY, imN=imN)
            return res
    else:
        # if type(param['sparse'])==dict:
        #     print('WARNING: VsparseImage was not given MJDs! using only image at MJD=',
        #             list(param['sparse'].keys())[0])
        #     param['sparse'] = param['sparse'][list(param['sparse'].keys())[0]]
        pass

    global _sparse_image_file, _sparse_image
    if type(param['sparse'])==str and _sparse_image_file!=param['sparse']:
        #print('loading new sparse image:', param['sparse'])
        if param['sparse'].endswith('.txt'):
            _sparse_image = {'x':[], 'y':[], 'I':[]}
            with open(param['sparse']) as f:
                for l in f.readlines():
                    if not l.strip().startswith('#'):
                        _sparse_image['x'].append(float(l.split()[0]))
                        _sparse_image['y'].append(float(l.split()[1]))
                        _sparse_image['I'].append(float(l.split()[2]))
            for k in ['x', 'y', 'I']:
                _sparse_image[k] = np.array(_sparse_image[k])
        elif param['sparse'].endswith('.pickle'):
            with open(param['sparse'], 'rb') as f:
                # -- pixel list "x(mas), y(mas), Intensity" as dict {'x':[], 'y':[], 'I':[]}
                _sparse_image = pickle.load(f)
        _sparse_image_file = param['sparse']
    else:
        _sparse_image = param['sparse']

    # == compute visibility
    c = np.pi/180/3600/1000/1e-6 # mas*m.um -> radians
    flipx, flipy = 1, 1
    if 'flip' in param:
        if param['flip']=='y':
            flipy = -1
        elif param['flip']=='x':
            flipx = -1

    if 'projang' in param:
        cpa = np.cos(np.pi/180*param['projang'])
        spa = np.sin(np.pi/180*param['projang'])
        _x = cpa*_sparse_image['x']*flipx - spa*_sparse_image['y']*flipy
        _y = spa*_sparse_image['x']*flipx + cpa*_sparse_image['y']*flipy
        if 'incl' in param:
            _y *= np.cos(np.pi/180*param['incl'])
    else:
        _x, _y = _sparse_image['x']*flipx, _sparse_image['y']*flipy

    if 'scale' in param:
        _x, _y = param['scale']*_x, param['scale']*_y
    if 'pow' in param:
        pow = param['pow']
    else:
        pow = 1

    #import sparsim
    #vis = sparsim.im2vis(u, v, wl, _x, _y, _sparse_image['I'])

    vis = np.exp(-2j*np.pi*c*(u[:,None,None]*_x[None,None,:]+
                             v[:,None,None]*_y[None,None,:])/wl[None,:,None])
    vis = np.sum(vis*_sparse_image['I'][None,None,:]**pow, axis=2)/\
         np.sum(_sparse_image['I']**pow)

    if not imFov is None:
        # -- compute cube
        if imN is None:
            imN = 2*int(imFov/imPix/2)+1
        X = np.linspace(imX-imFov/2, imX+imFov/2, imN)
        Y = np.linspace(imY-imFov/2, imY+imFov/2, imN)
        _X, _Y = np.meshgrid(X, Y)
        image = np.zeros((len(X), len(Y)))
        # -- interpolate -> slow:
        DX, DY = [-2,-1,0,1,2], [-2,-1,0,1,2]
        for k in range(len(_x)):
            i = np.argmin(np.abs(_x[k]-X))
            j = np.argmin(np.abs(_y[k]-Y))
            if False:
                # -- 4 quadrants
                if i==imN-1:
                    DX = [0,-1]
                elif _x[k]>X[i]:
                    DX = [0,1]
                else:
                    DX = [0,-1]
                if j==imN-1:
                    DY = [0,-1]
                elif _y[k]>Y[j]:
                    DY = [0,1]
                else:
                    DY = [0,-1]
                _w = {}
                for dx in DX:
                    for dy in DY:
                        if i+dx>=0 and j+dy>=0 and i+dx<imN and j+dy<imN:
                            _w[(dx, dy)] = 1-np.abs((_x[k]-X[i+dx])*(_y[k]-Y[j+dy]))/((imFov/(imN-1))**2)
                for dxy in _w:
                    image[j+dxy[1], i+dxy[0]] += _sparse_image['I'][k]**pow*_w[dxy]/3
            else:
                # -- gaussian
                _w = {}
                for dx in DX:
                    for dy in DY:
                        if i+dx>=0 and j+dy>=0 and i+dx<imN and j+dy<imN:
                            _w[(dx, dy)] = np.exp(-((_x[k]-X[i+dx])**2+
                                                    (_y[k]-Y[j+dy])**2)/(imFov/(imN-1))**2)
                nor = np.sum([_w[dxy] for dxy in _w])
                for dxy in _w:
                    image[j+dxy[1], i+dxy[0]] += _sparse_image['I'][k]**pow*_w[dxy]/nor

    else:
        _X, _Y, image = None, None, None

    if fullOutput:
        # -- visibility, image, x, y
        return vis, image, _X, _Y
    else:
        # -- only (complex) visibility
        return vis

def VuLensBin(mjd, param):
    """
    microlensing images for binary lens. Returns a dict valid for "VsparseImage"
    mjd is a list of dates.

    param: dictionnary
    - 'ul mjd0': date to minimum approach (MJD)
    - 'ul u0': minimum approach wrt lens' center of mass
    - 'ul tE': Einstein time (in days)
    - 'ul thetaE': Einstein radius (in mas)
    - 'ul rhos': angular radius of the source, in unit of thetaE
    - 'ul rotation': rotation of the whole scene (in deg, angle N->E)
    binary lens:

    - 'ul theta': approach angle of source wrt to binary orientation (in deg)
    - 'ul b': lens binary separation, in unit of thetaE
    - 'ul q': lens mass ratio m1/m2
        x1, y1 = -b*q/(1+q), 0
        x2, y2 = b*1/(1+q), 0
    - ul flip: optional. if ='y', then flip scene upside down
    """
    p = {k.split('ul ')[1]:param[k] for k in param if k.startswith('ul ')}
    p['theta'] *= np.pi/180
    for k in ['b', 'q']:
        if not k in p:
            p[k]=0
    # -- create parameter's dict for sparse image
    tmp = ulensBin2.computeSparseParam(mjd, p, parallel=False)
    tmp['scale'] = param['ul thetaE']
    tmp['projang'] = -param['ul rotation']

    if 'ul flip' in param and param['ul flip']=='y':
        tmp['flip']='y'
    return tmp

def Vkepler(u, v, wl, param, plot=False, _fudge=1.5, _p=1.5, fullOutput=False,
            imFov=None, imPix=None, imX=0, imY=0, imN=None):
    """
    _fudge=1.5, _p=1.5

    complex visibility of keplerian disk:
    u, v: 1D-ndarray spatial frequency (m). u and v must have same length N
    wl: 1D-ndarray wavelength (um). can have a different length from u and v

    param: parameters of disk:
        Geometry:
            Rin, Rout: inner and outer radii (mas)
                or
            diamin, diamout: inner and outer diameter (mas)
            [diamout/Rout can be ommited in case sizes are defined by fwhm ]

            Vin: velocity at Rin (km/s)
            beta: velocity radial power law (default keplerian=-0.5)
            incl: disk inclination (degrees, 0 is face-on)
            projang: projection angle (degrees, 0 is N and 90 is E)
            x, y: position in the field (mas, optional: default is 0,0)
            optional:
            [Vrad] radial velocity, in Km/s

        spectral lines (? can be anything)
            line_?_wl0: central wavelength of line (um)
            line_?_EW: equivalent width in nm (multiplied by total continuum)
            line_?_rpow: radial power law
             or line_?_fwhm: full width half max (mas)

        continuum flux (optional):
            cont_f: continuum total flux (at average wavelength)
            cont_rpow: power low variation of continuum (R/Rin). Should be <=0
             or cont_fwhm: full width half max (mas)

            cont_spx: spectral index of continuum (in [wl/mean(wl)]**cont_spx)

    returns:
    nd.array of shape (N,M) of complex visibilities
    """
    t0 = time.time()
    u = np.array(u)
    v = np.array(v)
    wl = np.array(wl)

    # -- wavelength resolution
    obs_dwl = np.median(np.abs(np.diff(wl)))

    P = [] # -- points in the disk

    # -- smallest angular resolution
    reso = np.min(1e-6*wl[None,:]/np.sqrt(u**2+v**2)[:,None])*180*3600*1000/np.pi
    #print('reso', reso, 'mas')
    # -- list of lines: enough point at Rin to cover +- Vin
    #As = list(filter(lambda x: x.startswith('wl0_'), param.keys()))
    As = list(filter(lambda x: x.endswith('_wl0'), param.keys()))

    if 'Rin' in param:
        Rin = np.abs(param['Rin'])
    elif 'diamin' in param:
        Rin = np.abs(param['diamin'])/2
    else:
        assert False, 'Rin or diamin must be defined for Keplerian disk'

    Rout = 0
    if 'Rout' in param:
        Rout = param['Rout']
    elif 'diamout' in param:
        Rout = param['diamout']/2

    # -- check in any FWHM are defined:
    fwhm = []
    if 'cont_fwhm' in param:
        fwhm.append(param['cont_fwhm'])
    for k in param:
        if k.startswith('line_') and k.endswith('fwhm'):
            fwhm.append(param[k])
    if len(fwhm)>0:
        Rout = max(Rout, 2*np.max(np.abs(fwhm)))

    assert Rout>0, 'Rout, diamout or fwhm must be defined: '+\
        'Rout='+str(Rout)+' fwhm='+str(fwhm)

    if 'beta' in param:
        beta = param['beta']
    else:
        beta = -0.5

    if 'Vin_Mm/s' in param:
        Vin = param['Vin_Mm/s']*1000
    elif 'Vin' in param:
        Vin = param['Vin']
    elif 'V1mas' in param:
        Vin = param['V1mas']*Rin**beta

    if 'Vrad' in param:
        Vrad = param['Vrad']
    else:
        Vrad = 0.0

    # -- delta wl corresponding to max velocity / max(line width, spectral resolution)
    tmp = []
    for a in As:
        if a.replace('wl0', 'gaussian') in param:
            tmp.append(param[a]*np.sqrt(Vin**2+Vrad**2)/2.998e5/
                       max(param[a.replace('wl0', 'gaussian')]/1000, obs_dwl))
        elif a.replace('wl0', 'lorentzian') in param:
            tmp.append(0.5*param[a]*np.sqrt(Vin**2+Vrad**2)/2.998e5/
                       max(param[a.replace('wl0', 'lorentzian')]/1000, obs_dwl))
        else:
            # -- default gaussian profile based on spectral resolution
            tmp.append(param[a]*np.sqrt(Vin**2+Vrad**2)/2.998e5/obs_dwl)

    # -- step size, in mas, as inner radius
    drin = min(reso, 2*np.pi*Rin/8)
    if tmp!=[]:
        drin = min(2*np.pi*Rin/max(tmp), drin)

    # -- safety margin: takes more points for _fudge>1
    drin /= _fudge

    # -- cos and sin values for rotations
    ci, si = np.cos(param['incl']*np.pi/180), np.sin(param['incl']*np.pi/180)
    cp, sp = np.cos(param['projang']*np.pi/180-np.pi/2), np.sin(param['projang']*np.pi/180-np.pi/2)

    # -- non uniform coverage in "r" (more points at inner radius, where range of velocity is widest)
    R = np.linspace(Rin**(1/_p), Rout**(1/_p),
                    max(int((Rout-Rin)/drin), 3))**_p
    dR = np.gradient(R)
    R = np.linspace(Rin**(1/_p), Rout**(1/_p),
                    max(int(dR[0]/drin*(Rout-Rin)/drin), 3))**_p
    dR = np.gradient(R)

    assert len(R)<500, 'too many points on disk ?! %d'%len(R)+' '+str(param)

    # -- TODO: this nested loop can be improved for speed!
    for i,r in enumerate(R):
        drt = dR[i] # step along the circle
        # -- PA angle (rad)
        T = np.linspace(0, 2*np.pi, int(2*np.pi*r/drt))[:-1]

        # -- avoid creating strong pattern in the mesh
        T += i*2*np.pi/len(R)

        # -- surface element
        dS = dR[i]*r*(T[1]-T[0]) # should make flux independent of grid density
        for t in T:
            # -- x, y, z, vx, vy, vz, rmin/r, PA, surface area
            # -- 0  1  2  3   4   5   6       7   8
            # -- include rotation in x (inclination)
            P.append([r*np.sin(t),
                      r*np.cos(t)*ci,
                      r*np.cos(t)*si,
                      Vin*(r/Rin)**beta*np.cos(t)+Vrad*np.sin(t),
                      (Vin*(r/Rin)**beta*np.sin(t)+Vrad*np.cos(t))*ci,
                      (Vin*(r/Rin)**beta*np.sin(t)+Vrad*np.cos(t))*si,
                      r/Rin, (t+3*np.pi/2)%(2*np.pi), dS])
            # -- rotation around z axis (projang)
            P[-1] = [P[-1][0]*cp+P[-1][1]*sp, -P[-1][0]*sp+P[-1][1]*cp, P[-1][2],
                     P[-1][3]*cp+P[-1][4]*sp, -P[-1][3]*sp+P[-1][4]*cp, P[-1][5],
                     P[-1][6], P[-1][7], P[-1][8]]
    P = np.array(P)
    t1 = time.time()
    c = np.pi/180/3600/1000/1e-6 # mas*m.um -> radians
    Vpoints = np.exp(-2j*np.pi*c*(u[:,None,None]*P[:,0][None,None,:]+
                                  v[:,None,None]*P[:,1][None,None,:])/wl[None,:,None])
    #print(P.shape, Vpoints.shape)
    flux = np.zeros((len(wl), P.shape[0]))
    # -- continuum
    if 'cont_f' in param:
        if 'cont_rpow' in param:
            cont_rpow = param['cont_rpow']
        else:
            cont_rpow = None

        if 'cont_fwhm' in param:
            cont_fwhm = param['cont_fwhm']
        else:
            cont_fwhm = None

        if 'cont_spx' in param:
            cont_spx = param['cont_spx']
        else:
            cont_spx = 0.0

        if not cont_rpow is None:
            cont = (P[:,6])**(cont_rpow)
            cont *= param['cont_f']/np.sum(cont*P[:,8])
            cont = cont[None,:]*(wl[:,None]/np.mean(wl))**cont_spx
        elif not cont_fwhm is None:
            # -- assume gaussian with FWHM based on disk's dimensions
            cont = np.exp(-(P[:,6]*Rin)**2/(2*(cont_fwhm/2.35482)**2))
            cont /= np.sum(cont*P[:,8])
            cont = param['cont_f']*cont[None,:]*(wl[:,None]/np.mean(wl))**cont_spx
    else:
        # -- no continuum
        cont = np.zeros((len(wl), P.shape[0]))

    flux = 1.*cont

    # -- lines profiles:
    #loren = lambda x, wl0, dwl: 1/(1+(x-wl0)**2/maximum(dwl/1000, obs_dwl/2)**2)
    #gauss = lambda x, wl0, dwl: np.exp(-(x-wl0)**2/(2*(np.maximum(dwl/1000, obs_dwl)/2.35482)**2))
    loren = lambda x, wl0, dwl: (0.5*dwl/1000)**2/((x-wl0)**2 + (0.5*dwl/1000)**2)
    gauss = lambda x, wl0, dwl: np.exp(-(x-wl0)**2/(2*dwl/1000/2.35482)**2)

    for a in As: # for each spectral lines
        # -- radial power law variation of the amplitude
        if a.replace('wl0', 'rpow') in param:
            Pin = param[a.replace('wl0', 'rpow')]
            fwhm = None
        elif a.replace('wl0', 'fwhm') in param:
            Pin = None
            fwhm = param[a.replace('wl0', 'fwhm')]
        else:
            # -- assume gaussian with FWHM based on disk's dimensions
            Pin = None
            fwhm = Rout

        if a.replace('wl0', 'gaussian') in param:
            dwl = param[a.replace('wl0', 'gaussian')] # in nm
        elif a.replace('wl0', 'lorentzian') in param:
            dwl = param[a.replace('wl0', 'lorentzian')] # in nm
        else:
            vel_dwl = np.abs(0.25*np.mean(wl)*Vin*P[:,6]**beta/2.998e5)
            #obs_dwl = np.maximum(obs_dwl, vel_dwl)
            obs_dwl = np.sqrt(obs_dwl**2 +  vel_dwl**2)
            dwl = obs_dwl*1000

        # -- line amplitude
        if a.replace('wl0', 'EW') in param:
            if not Pin is None:
                amp = (P[:,6])**(Pin)
            else: # -- gaussian radial profile
                amp = np.exp(-(P[:,6]*Rin)**2/(2*(fwhm/2.35482)**2))
            # -- equivalent width in nm: assumes gaussian profile!
            amp *= param[a.replace('wl0', 'EW')]/dwl/1.5053/np.sum(amp*P[:,8])
            # -- equivalent width in nm: assumes lorentzian profile!
            #amp *= param[a.replace('wl0', 'EW')]/dwl/1.54580/np.sum(amp*P[:,8])

        else:
            amp = P[:,6]**(Pin)

        # -- azimuthal variations?
        if a.replace('wl0', 'azamp1') in param and \
            a.replace('wl0', 'azprojang1') in param:
            azamp1 = param[a.replace('wl0', 'azamp1')]
            azpa1 = param[a.replace('wl0', 'azprojang1')]
        else:
            azamp1, azpa1 = 0, 0
        amp *= 1 + azamp1*np.cos(P[:,7]-azpa1*np.pi/180)

        if a.replace('wl0', 'gaussian') in param:
            flux += amp[None,:]*gauss(wl[:,None]*(1-P[:,5][None,:]/2.998e5),
                                      param[a], param[a.replace('wl0', 'gaussian')])
        elif a.replace('wl0', 'lorentzian') in param:
            flux += amp[None,:]*loren(wl[:,None]*(1-P[:,5][None,:]/2.998e5),
                                      param[a], param[a.replace('wl0', 'lorentzian')])
        else:
            # -- default: gaussian, based on spectral resolution
            flux += amp[None,:]*gauss(wl[:,None]*(1-P[:,5][None,:]/2.998e5),
                                     param[a], dwl)
            # -- default: lorentzian, based on spectral resolution
            #flux += amp[None,:]*loren(wl[:,None]*(1-P[:,5][None,:]/2.998e5),
            #                         param[a], dwl)

    if False:
        # -- trying to be clever integrating...
        flx = []
        vis = []
        # -- integrate each ring along PA
        for rminr in sorted(set(P[:,6])):
            w = np.where(P[:,6]==rminr)
            flx.append(np.trapezoid(flux[:,w[0]], P[:,7][w][None,:], axis=1))
            vis.append(np.trapezoid(flux[:,w[0]][None,:,:]*Vpoints[:,:,w[0]], P[:,7][w][None,None,:], axis=2))
        # -- integrate along radial dimension
        flx = np.trapezoid(np.array(flx)*R[:,None], R[:,None], axis=0)
        vis = np.trapezoid(np.array(vis)*R[:,None,None], R[:,None,None], axis=0)

        # -- for the case continuum is 0
        w0 = flx>1e-10*np.max(flx)
        vis[:,w0] = vis[:,w0]/flx[w0]
    else:
        # -- dumb integration
        # -- surface area of each point on disk
        flx = np.sum(flux*P[:,8][None,:], axis=1)
        vis = np.sum(Vpoints*flux[None,:,:]*P[:,8][None,None,:], axis=2)
        for _v in vis:
            _v[flx>0] /= flx[flx>0]
            _v[flx<=0] = 0.0
        vis = np.nan_to_num(vis, posinf=1, neginf=1)

    if 'x' in param and 'y' in param:
        x, y = param['x'], param['y']
    else:
        x, y = 0, 0

    vis *= np.exp(-2j*c*(u[:,None]/wl[None,:]*x +
                         v[:,None]/wl[None,:]*y))

    # -- debuging stuff:
    #print('_p=', _p, 'fugde=', fudge, 'len(P)=', len(P),
    #        '%.0fms'%(1000*(t1-t0)), ' + %.0fms'%(1000*(time.time()-t1)) )

    if plot:
        plt.close(0); plt.figure(0)
        ax = plt.subplot(211, aspect='equal')
        # -- show disk' dots on sky, with color is the velocit
        plt.scatter(P[:,0], P[:,1], marker='.',
                    #s=10*P[:,8]/min(P[:,8])*cont[0,:],
                    c=P[:,5], cmap='jet')
        plt.colorbar(label='velocity (km/s)')
        plt.xlabel('<- E (mas)'); plt.ylabel('-> N (mas)')
        ax.invert_xaxis()

        ax = plt.subplot(212, aspect='equal')
        # -- show disk' dots on sky, with color is the continuum flux
        plt.scatter(P[:,0], P[:,1], marker='.',
                    #s=10*P[:,8]/min(P[:,8])*cont[0,:],
                    c=np.sum(cont, axis=0), cmap='bone_r')
        plt.colorbar(label=r'continuum flux (avg over $\lambda$)')
        plt.xlabel('<- E (mas)'); plt.ylabel('-> N (mas)')
        ax.invert_xaxis()

    if not imFov is None:
        # -- compute cube
        if imN is None:
            imN = 2*int(imFov/imPix/2)+1
        X = np.linspace(imX-imFov/2-x, imX+imFov/2-x, imN)
        Y = np.linspace(imY-imFov/2-y, imY+imFov/2-y, imN)
        _X, _Y = np.meshgrid(X, Y)
        # -- deprojected coordinates
        _Xp, _Yp = cp*_X - sp*_Y, (sp*_X + cp*_Y)/ci
        R2 = _Xp**2+_Yp**2
        cube = np.zeros((len(wl), len(X), len(Y)))
        # -- interpolate for each WL:
        # @@@@@@@ THIS IS EXTREMELY SLOW! @@@@@@@@@@@@@@@@@@@@
        grP = [(p[0], p[1]) for p in P]
        gr = np.array([_X, _Y]).reshape(2, -1).T
        print('Vkepler images of shape', _X.shape, 'for', len(wl),
                'wavelength bins:', end=' ')
        t = time.time()
        for i in range(len(wl)):
            cube[i,:,:] = scipy.interpolate.RBFInterpolator(grP, flux[i,:],
                                                            kernel='linear', neighbors=3,
                                                            #kernel='thin_plate_spline',
                                                            #kernel='cubic',
                                                            )(gr).reshape((imN, imN))
            cube[i,:,:] *= (R2>=Rin**2)*(R2<=Rout**2)
            cube[i,:,:] = np.maximum(0, cube[i,:,:])
        print('in %.2fs'%(time.time()-t))
    else:
        _X, _Y, cube = None, None, None

    if fullOutput:
        # -- visibility, total spectrum, x, y, cube, points in disk
        return vis, flx, _X, _Y, cube, P, flux
    else:
        # -- only (complex) visibility
        return vis

def VfromImageOI(oi):
    """
    oi dict must have key 'image' of 'cube' -> Vmodel with "imFov" and "imPix"
    """
    if type(oi)==list:
        return [VfromImageOI(o) for o in oi]

    if not 'MODEL' in oi.keys() and \
                not ('image' in oi['MODEL'].keys() or
                     'cube' in oi['MODEL'].keys()):
        print('WARNING: VfromImage cannot compute visibility from image')
        print('         run "Vmodel" with imFov and imPix values set before')
        return oi

    oi['IM_VIS'] = {}
    oi['IM_FLUX'] = {}
    print('+++', oi['MODEL'].keys())
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
    if 'OI_FLUX' in oi:
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
    if 'OI_FLUX' in oi:
        oi = computeNormFluxOI(oi)
    return oi

SMEA = 7
def VmodelOI(oi, p, imFov=None, imPix=None, imX=0.0, imY=0.0, timeit=False, indent=0,
             v2smear=True, fullOutput=False, debug=False):
    modErr = 1
    global SMEA
    if type(oi) == list:
        # -- iteration on "oi" if a list
        _param = [computeLambdaParams(p, MJD=np.mean(o['MJD'])) for o in oi]
        return [VmodelOI(o, _param[i], imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                        timeit=timeit, indent=indent, fullOutput=fullOutput, v2smear=v2smear,
                        debug=debug) for i,o in enumerate(oi)]

    param = computeLambdaParams(p, MJD=np.mean(oi['MJD']))
    #print('param', param)
    # -- split in components if needed
    comp = set([x.split(',')[0].strip() for x in param.keys() if ',' in x and not x.startswith('#')])
    if len(comp)==0:
        # -- assumes single component
        res = VsingleOI(oi, param, imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                        timeit=timeit, indent=indent+1, fullOutput=fullOutput)
        res = _applyTF(res)
        res = _applyWlKernel(res, debug=debug)
        if 'smear' in res:
            res = oifits._binOI(res, binning=res['smear'], noError=True)
        return res

    # -- multiple components:
    tinit = time.time()
    res = {} # -- contains result
    t0 = time.time()
    # -- modify parameters list to handle bandwidth smearing
    """
    based on x,y and/or size, decide if smearing is necessary for a component.
    the idea is to duplicate parameters and encoding in the name the _dwl and _ffrac

    the half size of fringe packet is ~ wl**2/delta_lambda = lambda*R

    a component "sep" (mas) away has fringe offset of B*sep*pi/180/3600/1000 (m)

    so, B*sep*pi/180/3600/1000 ~= lambda*R -> we have complete smearing
    if sep > lambda*R/B*180*3600*1000/pi ~ 206.26*lambda*R/Bmax
        for sep in mas, lambda in um and B in m (R has no units)

    Smearing will only be on Vcomplex, so will not work well for V2, because:

    observed V2 = smeared(V**2) != (smeared(V))**2
    """
    D = {'|V|'  :('OI_VIS', 'B/wl'),
         'N|V|' :('OI_VIS', 'B/wl'),
         'DPHI' :('OI_VIS', 'B/wl'),
         'V2'   :('OI_VIS2', 'B/wl'),
         'T3PHI':('OI_T3', 'Bmax/wl'),
         'T3AMP':('OI_T3', 'Bmax/wl'),
         'CF'   :('OI_CF', 'B/wl'),
        }
    assert 'fit' in oi, "'fit' should be defined!"
    # -- in m/um
    _bwlmax = 1.0
    for e in oi['fit']['obs']:
        if e in D:
            for b in oi[D[e][0]].keys():
                _bwlmax = max(_bwlmax, np.max(oi[D[e][0]][b][D[e][1]]))
    R = np.mean(oi['WL'])/np.mean(np.diff(oi['WL']))

    # -- separation corresponding to the full fringe packet, in mas
    # -- if bwlmax = max(B/wl), B in m and wl in um
    _sep = 206.26*R/_bwlmax
    tmp = {k:param[k] for k in param.keys() if not ',' in k}
    smearing = {}
    for c in comp:
        if c+',x' in param.keys() and c+',y' in param.keys():
            try:
                sep = np.sqrt(param[c+',x']**2 + param[c+',y']**2)
            except:
                sep = 0.0
        else:
            sep = 0.0
        if c+',ud' in param.keys():
            sep += param[c+',ud']/2
        if c+',diam' in param.keys():
            sep += param[c+',diam']/2
        if c+',diamout' in param.keys():
            sep += param[c+',diamout']/2

        # -- gaussian is smooth, no smearing issues
        #if c+',fwhm' in param.keys():
        #    sep += param[c+',fwhm']/2

        # -- number of spectral channel within the band
        # -- larger than need be to be on the safe side
        try:
            n = 2*int(SMEA*sep/_sep)+1
        except:
            n = 1

        n = 1 # not working?
        smearing[c] = n

        if n<2:
            tmp.update({k:param[k] for k in param.keys() if k.startswith(c+',')})
        else:
            kz = filter(lambda x: x.startswith(c+','), param.keys())
            dwl = np.diff(oi['WL']).mean()
            # -- duplicate parameters with wl offset and flux weighing in the key
            for k in kz:
                for x in np.linspace(-0.5,0.5,n+2)[1:-1]:
                    tmp[c+'&dwl%.7f&ffrac%.7f,'%(x*dwl, 1/n)+k.split(',')[1]] = param[k]
    if debug:
        print('VmodelOI: R, _bwlmax, _sep=', R, _bwlmax, _sep)
        print('VmodelOI: smearing=', smearing)

    if len(tmp)>len(param):
        param = tmp
        comp = set([x.split(',')[0].strip() for x in param.keys() if ',' in x])
        comp = sorted(comp)
    if timeit:
        print(' '*indent+'VmodelOI > smearing %.3fms'%(1000*(time.time()-t0)))

    # -- for each components:
    t0 = time.time()
    for c in comp:
        #print(c)
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
            res = VsingleOI(oi, _param, imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                            timeit=timeit, indent=indent+1, noT3=True,
                            _dwl=_dwl, _ffrac=_ffrac, fullOutput=fullOutput)
            if _dwl!=0:
                # -- correct wavelength offset
                _cwl = 1.0 + _dwl/np.mean(res['WL'])
                res['WL'] -= _dwl
                for x in ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_CF']:
                    if not x in res:
                        continue
                    for k in res[x]:
                        for z in res[x][k]:
                            if z.endswith('/wl'):
                                res[x][k][z] *= _cwl
                # -- end correction

            if 'image' in res['MODEL'].keys():
                # -- for this component
                res['MODEL'][c+',image'] = res['MODEL']['image']
                # -- smearing
                if '&dwl' in c:
                    res['MODEL'][c.split('&dwl')[0]+',image'] = res['MODEL']['image']

                # -- total
                if not 'cube' in res['MODEL']:
                    # TODO: in case flux(MJD), this averages out the flux over time
                    res['MODEL']['cube'] = res['MODEL']['image'][None,:,:]*\
                                           res['MODEL']['totalflux'][:,None,None]
                res['MODEL']['image'] *= np.mean(res['MODEL']['totalflux'])

            # -- for this component:
            res['MODEL'][c+',flux'] = res['MODEL']['totalflux'].copy()
            if 'totalflux(MJD)' in res['MODEL']:
                res['MODEL'][c+',flux(MJD)'] = res['MODEL']['totalflux(MJD)']

            if 'OI_VIS' in res:
                res['MODEL'][c+',vis'] = res['OI_VIS'].copy()
            else:
                res['MODEL'][c+',vis'] = {}

            res['MODEL'][c+',negativity'] = res['MODEL']['negativity']
            # -- OLD smearing
            if '&dwl' in c:
                # -- TODO: flux(MJD)
                res['MODEL'][c.split('&dwl')[0]+',flux'] = res['MODEL']['totalflux'].copy()
                res['MODEL'][c.split('&dwl')[0]+',negativity'] = res['MODEL']['negativity']

            # -- total complex visibility
            res['MODEL']['WL'] = res['WL']
            res['MOD_VIS'] = {}
            for k in res['MODEL'][c+',vis'].keys(): # for each baseline
                # TODO: flux as function of time!
                if 'totalflux(MJD)' in res['MODEL']:
                    tmp = res['MODEL']['totalflux(MJD)'].replace('$WL', "res['MODEL']['WL']")
                    tmp = tmp.replace('$MJD', "res['OI_VIS']['%s']['MJD2']"%k)
                    tmp = tmp.replace('$TFLUX', "res['MODEL']['totalflux']")
                    res['MOD_VIS'][k] = eval(tmp)*res['OI_VIS'][k]['|V|']*\
                                        np.exp(1j*np.pi*res['OI_VIS'][k]['PHI']/180)
                else:
                    res['MOD_VIS'][k] = res['MODEL']['totalflux'][None,:]*\
                                        res['OI_VIS'][k]['|V|']*\
                                        np.exp(1j*np.pi*res['OI_VIS'][k]['PHI']/180)
            m = {}
        else:
            # -- combine model with other components
            m = VsingleOI(oi, _param, imFov=imFov, imPix=imPix, imX=imX, imY=imY,
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
                # -- TODO: MJD-averaged flux
                res['MODEL']['image'] += np.mean(m['MODEL']['totalflux'])*\
                                        m['MODEL']['image']
                if 'cube' in m['MODEL']:
                    res['MODEL']['cube'] += m['MODEL']['cube']
                else:
                    # -- TODO: MJD-averaged flux
                    res['MODEL']['cube'] += m['MODEL']['image'][None,:,:]*\
                                            m['MODEL']['totalflux'][:,None,None]

            res['MODEL'][c+',flux'] = m['MODEL']['totalflux']
            if 'totalflux(MJD)' in m['MODEL']:
                res['MODEL'][c+',flux(MJD)'] =  m['MODEL']['totalflux(MJD)']
            if 'OI_VIS' in m:
                res['MODEL'][c+',vis'] = m['OI_VIS'].copy()
            else:
                res['MODEL'][c+',vis'] = {}
            res['MODEL'][c+',negativity'] = m['MODEL']['negativity']
            if '&dwl' in c:
                # -- TODO: flux(MJD)
                _c = c.split('&dwl')[0]
                if _c+',flux' in res['MODEL']:
                    res['MODEL'][_c+',flux'] += m['MODEL']['totalflux']
                else:
                    res['MODEL'][_c+',flux'] = m['MODEL']['totalflux']

            # -- Add
            res['MODEL']['totalflux'] += m['MODEL']['totalflux']
            if 'totalflux(MJD)' in m['MODEL']:
                res['MODEL']['totalflux(MJD)'] += '+' + m['MODEL']['totalflux(MJD)']
            res['MODEL']['negativity'] += m['MODEL']['negativity']

            # -- total complex Visibility
            for k in res['MODEL'][c+',vis'].keys(): # for each baseline
                if 'totalflux(MJD)' in res['MODEL']:
                    tmp = m['MODEL']['totalflux(MJD)'].replace('$WL', "m['MODEL']['WL']")
                    tmp = tmp.replace('$MJD', "m['OI_VIS']['%s']['MJD2']"%k)
                    tmp = tmp.replace('$TFLUX', "m['MODEL']['totalflux']")
                    try:
                        res['MOD_VIS'][k] += eval(tmp)*m['OI_VIS'][k]['|V|']*\
                                        np.exp(1j*np.pi*m['OI_VIS'][k]['PHI']/180)
                    except:
                        print('ERROR!', tmp)
                else:
                    res['MOD_VIS'][k] += m['MODEL']['totalflux'][None,:]*\
                                         m['OI_VIS'][k]['|V|']*\
                                         np.exp(1j*np.pi*m['OI_VIS'][k]['PHI']/180)

        res['MODEL']['param'] = computeLambdaParams(p, MJD=np.mean(res['MJD']))

        if timeit:
            print(' '*indent+'VmodelOI > VsingleOI "%s" %.3fms'%(c, 1000*(time.time()-tc)))

    t0 = time.time()
    # -- compute OI_VIS and OI_VIS2 (and OI_CF if needed)
    for b in res['MOD_VIS'].keys():
        # -- correlated flux == abs(non normalised visibility)
        if 'OI_CF' in oi.keys() and b in oi['OI_CF']:
            if not 'MOD_CF' in res:
                res['MOD_CF'] = {b: np.abs(res['MOD_VIS'][b])}
            else:
                res['MOD_CF'][b] = np.abs(res['MOD_VIS'][b])
            if 'OI_CF' in oi and not 'OI_CF' in res:
                res['OI_CF'] = {}
            if not b in res['OI_CF']:
                res['OI_CF'][b] = oi['OI_CF'][b].copy()
            res['OI_CF'][b]['CF'] = res['MOD_CF'][b]
        else:
            pass
            #print('no OI_CF')

        # -- TODO: flux as function of MJD
        if 'totalflux(MJD)' in res['MODEL']:
            tmp = res['MODEL']['totalflux(MJD)'].replace('$WL', "res['MODEL']['WL']")
            tmp = tmp.replace('$MJD', "res['OI_VIS']['%s']['MJD2']"%b)
            tmp = tmp.replace('$TFLUX', "res['MODEL']['totalflux']")
            res['MOD_VIS'][b] /= eval(tmp)
        else:
            res['MOD_VIS'][b] /= res['MODEL']['totalflux'][None,:]

        if 'OI_VIS' in res and b in res['OI_VIS']:
            res['OI_VIS'][b]['|V|'] = np.abs(res['MOD_VIS'][b])
            res['OI_VIS'][b]['PHI'] = (np.angle(res['MOD_VIS'][b])*180/np.pi+180)%360-180
        if 'OI_VIS2' in res and b in res['OI_VIS2']:
            # -- assume OI_VIS and OI_VIS2 have same structure!!!
            res['OI_VIS2'][b]['V2'] = np.abs(res['MOD_VIS'][b])**2

    if 'OI_FLUX' in oi.keys():
        for k in oi['OI_FLUX'].keys():
            if 'totalflux(MJD)' in res['MODEL']:
                tmp = res['MODEL']['totalflux(MJD)'].replace('$WL', "res['MODEL']['WL']")
                tmp = tmp.replace('$MJD', "res['OI_FLUX']['%s']['MJD2']"%k)
                tmp = tmp.replace('$TFLUX', "res['MODEL']['totalflux']")
                totalflux = eval(tmp)
            else:
                totalflux = res['OI_FLUX'][k]['FLUX']*0 + res['MODEL']['totalflux'][None,:]

            res['OI_FLUX'][k] = {'FLUX':  totalflux,
                                 'RFLUX': totalflux,
                                 'EFLUX': totalflux*0 + modErr,
                                 'FLAG':  res['OI_FLUX'][k]['FLAG'],
                                 'MJD':   oi['OI_FLUX'][k]['MJD'],}
        if timeit:
            print(' '*indent+'VmodelOI > fluxes %.3fms'%(1000*(time.time()-t0)))
    t0 = time.time()

    if 'smear' in res:
        # needs to happen before T3PHI, differential phase and normalise flux
        if debug:
            print('VmodelOI: closing smearing (binning)')
        #print('smear: binning', res['WL'].shape, '->', end=' ')
        res = oifits._binOI(res, noError=True)
        #print(res['WL'].shape)

    if 'OI_T3' in oi.keys():
        res['OI_T3'] = {}
        for k in oi['OI_T3'].keys():
            res['OI_T3'][k] = {}
            for l in ['MJD', 'u1', 'u2', 'v1', 'v2', 'formula', 'FLAG', 'Bmax/wl', 'Bavg/wl', 'Bmin/wl',
                        'B1', 'B2', 'B3']:
                if not oi['OI_T3'][k][l] is None:
                    res['OI_T3'][k][l] = oi['OI_T3'][k][l].copy()
                else:
                    print('!!', l, 'not in OI_T3[%s]'%k)
                    res['OI_T3'][k][l] = None
            if 'NAME' in oi['OI_T3'][k]:
                res['OI_T3'][k]['NAME'] = oi['OI_T3'][k]['NAME'].copy()
            res['OI_T3'][k]['FLAG'] = np.bool(0*res['OI_T3'][k]['MJD'][:,None]+0*res['WL'][None,:])

        if debug:
            print('VmodelOI: computing T3')

        res = computeT3fromVisOI(res)

        if timeit:
            print(' '*indent+'VmodelOI > T3 %.3fms'%(1000*(time.time()-t0)))


    t0 = time.time()
    if 'fit' in oi and 'obs' in oi['fit'] and \
            ('DPHI' in oi['fit']['obs'] or 'N|V|' in oi['fit']['obs']):
        if debug:
            print('VmodelOI: differential phase')
        res = computeDiffPhiOI(res, param, debug=debug)
        if timeit:
            print(' '*indent+'VmodelOI > dPHI %.3fms'%(1000*(time.time()-t0)))
            t0 = time.time()
    else:
        if debug:
            print('VmodelOI: NO NEED for differential phase', oi['fit'])

    if 'fit' in oi and 'obs' in oi['fit'] and 'NFLUX' in oi['fit']['obs']:
        #print('normalised fluxes')
        res = computeNormFluxOI(res, param, debug=debug)
        if timeit:
            print(' '*indent+'VmodelOI > normFlux %.3fms'%(1000*(time.time()-t0)))

    res = _applyWlKernel(res, debug=debug)

    for k in['telescopes', 'baselines', 'triangles']:
        if k in oi:
            res[k] = oi[k]

    res['param'] = computeLambdaParams(param, MJD=np.mean(oi['MJD']))

    t0 = time.time()
    if timeit:
        print(' '*indent+'VmodelOI > total %.3fms'%(1000*(time.time()-tinit)))

    res = _applyTF(res)
    return res

def _applyWlKernel(res, debug=False):
    if not ('fit' in res and 'wl kernel' in res['fit']):
        return res

    #print('wl kernel!')
    # -- convolve by spectral Resolution
    N = 2*int(2*res['fit']['wl kernel'])+3
    x = np.arange(N)
    ker = np.exp(-(x-np.mean(x))**2/(2.*(res['fit']['wl kernel']/2.35482)**2))
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
                res['DVIS'][k]['DPHI'][i] = np.convolve(
                            res['DVIS'][k]['DPHI'][i], ker, mode='same')
    if 'OI_CF' in res:
        for k in res['OI_CF'].keys():
            for i in range(res['OI_CF'][k]['CF'].shape[0]):
                res['OI_CF'][k]['CF'][i] = np.convolve(
                            res['OI_CF'][k]['CF'][i], ker, mode='same')
                res['OI_CF'][k]['PHI'][i] = np.convolve(
                            res['OI_CF'][k]['PHI'][i], ker, mode='same')

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
    return res

def _applyTF(res):
    # == single target self-calibration -> assumes tel name have no '-'!!!
    # "#TF_|V|_U1U2_*" -> overall coefficient
    # "#TF_|V|_U1U2_+" -> overall coefficient
    if any(['#TF' in k for k in res['param'].keys()]):
        _debug = False
        if _debug:
            print('computing TF')
        # -- organise the TF coefficient
        TF = {}
        for k in res['param']:
            if k.startswith('#TF'):
                obs = k.split('_')[1]
                if not obs in TF:
                    TF[obs] = {}
                b = k.split('_')[2]
                if not b in TF[obs]:
                    TF[obs][b] = {}
                TF[obs][b][k.split('_')[3]] = res['param'][k]
        if _debug:
            print('TF:', TF)

        O = {'V2'   : 'OI_VIS2',
             '|V|'  : 'OI_VIS',
             'T3PHI': 'OI_T3'
            }
        for o in TF:
            if _debug:
                print(' -> applying TF to', o)
            for b in TF[o]:
                if not O[o] in res:
                    continue
                if 'all' in res[O[o]]:
                    w = res[O[o]]['all']['NAME']==b
                    if '+' in TF[o][b]:
                        res[O[o]]['all'][o][w] += TF[o][b]['+']
                    if '*' in TF[o][b]:
                        res[O[o]]['all'][o][w] *= TF[o][b]['*']
                    if 's' in TF[o][b]:
                        res[O[o]]['all'][o][w] *= 1 + (res['WL']-np.mean(res['WL']))[None,:]*TF[o][b]['s']
                    if 'wl0' in TF[o][b] and 'wl2' in TF[o][b]:
                        res[O[o]]['all'][o][w] *= 1 + (res['WL']-TF[o][b]['wl0'])[None,:]*TF[o][b]['wl2']
                else:
                    if '+' in TF[o][b] and b in res[O[o]]:
                        res[O[o]][b][o] += TF[o][b]['+']
                    if '*' in TF[o][b] and b in res[O[o]]:
                        res[O[o]][b][o] *= TF[o][b]['*']
                    if 's' in TF[o][b] and b in res[O[o]]:
                        res[O[o]][b][o] *= 1 + (res['WL']-np.mean(res['WL']))[None,:]*TF[o][b]['s']
                    if 'wl0' in TF[o][b] and 'wl2' in TF[o][b] and b in res[O[o]]:
                        res[O[o]][b][o] *= 1 + (res['WL']-TF[o][b]['wl0'])[None,:]*TF[o][b]['wl2']
    return res

def computeLambdaParams(params, MJD=0):
    """
    if MJD==None, will not evaluate MJDs, MJD==0 throws an error if any '$MJD' are present
    """
    if params is None:
        return None
    paramsI = params.copy()
    paramsR = {}
    loop = True
    nloop = 0
    s = '$' # special character to identify keywords
    while loop and nloop<10:
        loop = False
        # -- for each keyword
        for k in sorted(list(paramsI.keys()), key=lambda x: -len(x)):
            if type(paramsI[k])==str:
                # -- allow parameter to be expression of others
                tmp = paramsI[k]
                compute = False
                for _k in sorted(list(paramsI.keys()), key=lambda x: -len(x)):
                    if s+_k in paramsI[k]:
                        try:
                            repl = '%e'%paramsI[_k]
                        except:
                            repl = str(paramsI[_k])
                        if _k.startswith('_') and _k.endswith('_'):
                            tmp = tmp.replace(s+_k, repl)
                        else:
                            tmp = tmp.replace(s+_k, '('+repl+')')
                        if not s in tmp:
                            # -- no more replacement
                            compute = True
                        elif s+'MJD' in tmp and not s in tmp.replace(s+'MJD', ''):
                            # -- no more replacement
                            compute = True
                    for kp in sorted(list(paramsI.keys()), key=lambda x: -len(x)):
                        if kp in tmp and not s+kp in tmp:
                            raise Exception('missing '+s+' for '+kp+' in {'+
                                "'"+_k+'": "'+paramsI[k]+'"}?')
                    #if s in tmp:
                    #    raise Exception('unknow parameters definition in {'+
                    #        "'"+_k+'": "'+paramsI[k]+'"}?')

                # -- are there still un-computed parameters?
                for _k in paramsI.keys():
                    if s+_k in tmp:
                        # -- set function to another loop
                        loop = True
                        paramsI[k] = tmp
                if compute and not loop and tmp!='orbit':
                    if s+'MJD' in tmp and not ( k in ['x', 'y'] or any([t in k for t in [',x', ',y']])):
                        #print('MJD!')
                        if MJD==0:
                            raise Exception('MJD= not defined!')
                        else:
                            tmp = tmp.replace('$MJD', str(MJD))
                    if s+'MJD' in tmp and ( k in ['x', 'y'] or any([t in k for t in [',x', ',y']])):
                        paramsR[k] = tmp
                    else:
                        paramsR[k] = eval(tmp.replace('(nan)', '(np.nan)'))
                else:
                    paramsR[k] = tmp
            else:
                paramsR[k] = paramsI[k]
        nloop+=1

    assert nloop<10, 'too many recurences in evaluating parameters!'+str(paramsI)
    # for k in paramsR:
    #     for _k in paramsR:
    #         if type(paramsR[k])==str and _k in paramsR[k]:
    #             msg = 'reference to "%s" in definition of "%s" may need to be preceeded by "%s":'%(_k, k, s)
    #             msg+= '\n  possibly {"%s": "%s"}'%(k, paramsR[k].replace(_k, '\033[1m'+s+'\033[0m'+_k))
    #             msg+= ' instead of {"%s": "%s"} ?'%(k, paramsR[k])
    #             print('\033[41mWARNING:\033[0m '+msg)

    return paramsR

def computeDiffPhiOI(oi, param=None, order='auto', debug=False,
                    visamp=True):
    if not param is None:
        _param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))
    else:
        _param = None
    if type(oi)==list:
        return [computeDiffPhiOI(o, _param, order) for o in oi]
    if 'param' in oi.keys() and param is None:
        _param = oi['param']

    if not 'OI_VIS' in oi.keys():
        return oi

    # -- user-defined wavelength range
    fit = {'wl ranges':[(min(oi['WL']), max(oi['WL']))]}
    if not 'fit' in oi:
        oi['fit'] = fit.copy()
    elif not 'wl ranges' in oi['fit']:
        # -- weird but necessary to avoid a global 'fit'
        fit.update(oi['fit'])
        oi['fit'] = fit.copy()

    # -- range(s) to take into consideration
    w = np.zeros(oi['WL'].shape)
    closest = []
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        closest.append(np.argmin(np.abs(oi['WL']-0.5*(WR[0]+WR[1]))))
    oi['WL mask'] = np.bool_(w)
    if not any(oi['WL mask']):
        for clo in closest:
            oi['WL mask'][clo] = True

    # -- user defined continuum
    if 'fit' in oi and 'continuum ranges' in oi['fit']:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['continuum ranges']:
            #print(WR, (oi['WL']>=WR[0])*(oi['WL']<=WR[1]))
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)
        #print('user defined continuum:', np.sum(np.bool_(wc)), np.sum(w))

    # -- exclude where lines are in the models
    elif not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                dwl = 0
                if k.replace('_wl0', '_gaussian') in _param.keys():
                    dwl = 1.5*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('_wl0', '_lorentzian') in _param.keys():
                    dwl = 3*_param[k.replace('wl0', 'lorentzian')]/1000.

                vel = 0.0 # velocity, in km/s
                if ',' in k:
                    kv = k.split(',')[0]+','+'Vin'
                    fv = 1.5
                    if k.replace('Vin', 'incl') in _param:
                        fv *= np.sin(_param[k.replace('Vin', 'incl')]*np.pi/180)
                else:
                    kv = 'Vin'
                    fv = 1.5
                    if 'incl' in _param:
                        fv *= np.sin(_param['incl']*np.pi/180)

                if not kv in _param:
                    kv +='_Mm/s'
                    fv = 1000.
                if kv in _param:
                    vel = np.abs(_param[kv]*fv)

                if ',' in k:
                    kv = k.split(',')[0]+','+'V1mas'
                else:
                    kv = 'V1mas'
                if kv in _param:
                    if 'Rin' in _param:
                        vel = _param[kv]/np.sqrt(_param[kv.replace('V1mas', 'Rin')])
                    elif 'diamin' in _param:
                        vel = _param[kv]/np.sqrt(0.5*_param[kv.replace('V1mas', 'diamin')])
                dwl = np.sqrt(dwl**2 + (_param[k]*vel/2.998e5)**2)
                if k.replace('_wl0', '_truncexp') in _param.keys():
                    dwl = 2*_param[k.replace('wl0', 'truncexp')]/1000.
                    w *= ~(((oi['WL']-_param[k])<=1.3*dwl)*
                            (oi['WL']>_param[k]-dwl/8))

    if np.sum(w)==0:
        print('computeDiffPhiOI: no continuum!, use all band')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = oi['WL cont'].copy()

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.2)
        order = max(order, 2)
        vorder = 1
    if 'fit' in oi and 'DPHI order' in oi['fit']:
        order = oi['fit']['DPHI order']
    if 'fit' in oi and 'N|V| order' in oi['fit']:
        vorder = oi['fit']['N|V| order']
    #print('computeDiffPhiOI: order=%d'%order)
    if debug:
        print('computeDiffPhiOI: continuum', oi['WL cont'])

    if np.sum(oi['WL cont'])<order+1:
        print('WARNING: not enough WL to compute continuum!')
        return oi

    oi['DVIS'] = {}
    _checkEmpty = False
    if _checkEmpty and np.sum(w)==0:
        print('empty mask (before flags)')

    for k in oi['OI_VIS'].keys():
        data = []
        for i,phi in enumerate(oi['OI_VIS'][k]['PHI']):
            mask = w * ~oi['OI_VIS'][k]['FLAG'][i,:]
            if _checkEmpty and np.sum(mask)==0:
                print(k, 'empty mask! -> w=%.0f, flag=%.0f'%(np.sum(w), np.sum(~oi['OI_VIS'][k]['FLAG'][i,:])), end=' ')

            if 'EPHI' in oi['OI_VIS'][k]:
                err = oi['OI_VIS'][k]['EPHI'][i,:].copy()
                # if 'baseline ranges' in oi['fit']:
                #     for bmin, bmax in oi['fit']['baseline ranges']:
                #         if not 'T3' in data[l]['ext']:
                #                 mask *= (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']<=bmax)*\
                #                         (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']>=bmin)
                #         else:
                #             for b in ['B1', 'B2', 'B3']:
                #                 mask *= ((oi[data[l]['ext']][k][b][j]<=bmax)*
                #                          (oi[data[l]['ext']][k][b][j]>=bmin))
                # if 'MJD ranges' in oi['fit']:
                #     for mjdmin, mjdmax in oi['fit']['MJD ranges']:
                #         mask *= (oi['OI_VIS'][k]['MJD2'][i,:]<=mjdmax)*\
                #                 (oi['OI_VIS'][k]['MJD2'][i,:]>=mjdmin)
                if 'max error' in oi['fit'] and 'DPHI' in oi['fit']['max error']:
                    # -- ignore data with large error bars
                    mask *= (err<=oi['fit']['max error']['DPHI'])
                if 'max relative error' in oi['fit'] and 'DPHI' in oi['fit']['max relative error']:
                    # -- ignore data with large error bars
                    mask *= (err<=(oi['fit']['max relative error']['DPHI']*
                                    np.abs(oi['OI_VIS'][k]['PHI'][i,:])))
                if 'mult error' in fit and 'DPHI' in oi['fit']['mult error']:
                    # -- force error to a minimum value
                    err *= oi['fit']['mult error']['DPHI']
                if 'min error' in fit and 'DPHI' in oi['fit']['min error']:
                    # -- force error to a minimum value
                    err = np.maximum(oi['fit']['min error']['DPHI'], err)
                if 'min relative error' in fit and 'DPHI' in oi['fit']['min relative error']:
                    # -- force error to a minimum value
                    err = np.maximum(oi['fit']['min relative error']['DPHI']*
                                     np.abs(oi['OI_VIS'][k]['PHI'][i,:]), err)
                # -- ignore 0-errors
                mask *= err>0
            else:
                err = None

            if np.sum(mask)>order:
                if not err is None:
                    c = np.polyfit(oi['WL'][mask], phi[mask], order, w=1/err[mask])
                else:
                    c = np.polyfit(oi['WL'][mask], phi[mask], order)
                #print('debug:', np.mean(phi[mask]-np.polyval(c, oi['WL'][mask])))
                data.append(phi-np.polyval(c, oi['WL']))
            else:
                if _checkEmpty:
                    print('computeDiffPhiOI: removing median')
                # -- not polynomial fit, use median
                data.append(phi-np.median(phi))
        if visamp:
            vdata = []
            edata = []
            for i,vis in enumerate(oi['OI_VIS'][k]['|V|']):
                vmask = w * ~oi['OI_VIS'][k]['FLAG'][i,:]
                if 'E|V|' in oi['OI_VIS'][k]:
                    verr = oi['OI_VIS'][k]['E|V|'][i,:]
                    if 'max error' in oi['fit'] and 'N|V|' in oi['fit']['max error']:
                        # -- ignore data with large error bars
                        vmask *= (verr<oi['fit']['max error']['N|V|'])
                    if 'max relative error' in oi['fit'] and 'N|V|' in oi['fit']['max relative error']:
                        # -- ignore data with large error bars
                        vmask *= (verr<(oi['fit']['max relative error']['N|V|']*
                                        np.abs(oi['OI_VIS'][k]['|V|'][i,:])))
                    if 'mult error' in fit and 'N|V|' in oi['fit']['mult error']:
                        # -- multiply all errors by factor
                        verr *= oi['fit']['mult error']['N|V|']
                    if 'min error' in fit and 'N|V|' in oi['fit']['min error']:
                        # -- force error to a minimum value
                        verr = np.maximum(oi['fit']['min error']['N|V|'], verr)
                    if 'min relative error' in fit and 'N|V|' in oi['fit']['min relative error']:
                        # -- force error to a minimum value
                        verr = np.maximum(oi['fit']['min relative error']['N|V|']*
                                         np.abs(oi['OI_VIS'][k]['|V|'][i,:]), verr)
                    vmask *= verr>0
                else:
                    verr = None
                if np.sum(vmask)>vorder:
                    if not verr is None:
                        c = np.polyfit(oi['WL'][vmask], vis[vmask],
                                        vorder, w=1/verr[vmask])
                    else:
                        c = np.polyfit(oi['WL'][vmask], vis[vmask],
                                        vorder)
                    vdata.append(vis/np.polyval(c, oi['WL']))
                else:
                    # -- not polynomial fit, use median
                    if np.median(vis)!=0:
                        vdata.append(vis/np.median(vis))
                    else:
                        vdata.append(np.ones(len(vis)))
                edata.append(np.std(vdata[-1][vmask]))
        # -- end visamp

        oi['DVIS'][k] = {'DPHI':np.array(data),
                         'FLAG':oi['OI_VIS'][k]['FLAG'],
                         'B/wl':oi['OI_VIS'][k]['B/wl'],
                         }
        if 'NAME' in oi['OI_VIS'][k]:
            oi['DVIS'][k]['NAME'] = oi['OI_VIS'][k]['NAME']
        if 'MJD' in oi['OI_VIS'][k]:
            oi['DVIS'][k]['MJD'] = oi['OI_VIS'][k]['MJD']
        if 'MJD2' in oi['OI_VIS'][k]:
            oi['DVIS'][k]['MJD2'] = oi['OI_VIS'][k]['MJD2']
        if 'EPHI' in oi['OI_VIS'][k]:
            oi['DVIS'][k]['EDPHI'] = oi['OI_VIS'][k]['EPHI']
        if visamp:
            oi['DVIS'][k]['N|V|'] = np.array(vdata)
            if 'E|V|' in oi['OI_VIS'][k]:
                # -- very crude estimation
                #oi['DVIS'][k]['EN|V|'] = oi['OI_VIS'][k]['E|V|']
                oi['DVIS'][k]['EN|V|'] = np.array(edata)[:,None]+\
                                        0*oi['OI_VIS'][k]['E|V|']

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

def computeNormFluxOI(oi, param=None, order='auto', debug=False):
    if not param is None:
        _param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))
    else:
        _param = None

    if type(oi)==list:
        return [computeNormFluxOI(o, _param, order) for o in oi]

    if not 'OI_FLUX' in oi.keys():
        print('WARNING: computeNormFluxOI, nothing to do')
        return oi

    if 'param' in oi.keys() and param is None:
        _param = oi['param'].copy()
        _param = computeLambdaParams(_param, MJD=np.mean(oi['MJD']))

    # -- user defined wavelength range
    w = oi['WL']>0
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        w = np.zeros(oi['WL'].shape)
    #   for WR in oi['fit']['wl ranges']:
    #       w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        closest = []
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
            closest.append(np.argmin(np.abs(oi['WL']-0.5*(WR[0]+WR[1]))))
        w = np.bool_(w)
        if not any(w):
            for clo in closest:
                w[clo] = True
    oi['WL mask'] = np.bool_(w).copy()
    # -- user defined continuum
    if 'fit' in oi and 'continuum ranges' in oi['fit'] and \
        oi['fit']['continuum ranges']!=[]:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['continuum ranges']:
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)

    # -- continuum: exclude where lines are in the models
    if not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                dwl = 0.
                if k.replace('wl0', 'gaussian') in _param.keys():
                    dwl = 1.5*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in _param.keys():
                    dwl = 3*_param[k.replace('wl0', 'lorentzian')]/1000.
                vel = 0
                if ',' in k:
                    kv = k.split(',')[0]+','+'Vin'
                    fv = 1.5
                    if k.replace('Vin', 'incl') in _param:
                        fv *= np.sin(_param[k.replace('Vin', 'incl')]*np.pi/180)
                else:
                    kv = 'Vin'
                    fv = 1.5
                    if 'incl' in _param:
                        fv *= np.sin(_param['incl']*np.pi/180)

                # if ',' in k:
                #     kv = k.split(',')[0]+','+'Vin'
                #     fv = 1.0
                # else:
                #     kv = 'Vin'
                #     fv = 1.0
                if not kv in _param:
                    kv+='_Mm/s'
                    fv = 1000.0
                if kv in _param:
                    vel = np.abs(_param[kv]*fv)

                if ',' in k:
                    kv = k.split(',')[0]+','+'V1mas'
                else:
                    kv = 'V1mas'
                if kv in _param:
                    if 'Rin' in _param:
                        vel = _param[kv]/np.sqrt(_param[kv.replace('V1mas', 'Rin')])
                    elif 'diamin' in _param:
                        vel = _param[kv]/np.sqrt(0.5*_param[kv.replace('V1mas', 'diamin')])

                # -- effects of "vel" depends on the size of the disk and "fpow"
                #dwl = np.sqrt(dwl**2 + (.5*_param[k]*vel/2.998e5)**2)
                dwl = np.sqrt(dwl**2 + (_param[k]*vel/2.998e5 )**2)
                if any(np.abs(oi['WL']-_param[k])>=np.abs(dwl)):
                    w *= (np.abs(oi['WL']-_param[k])>=np.abs(dwl))
                if k.replace('wl0', 'truncexp') in _param.keys():
                    dwl = 2*_param[k.replace('wl0', 'truncexp')]/1000.
                    w *= ~((oi['WL']<=_param[k]+1.3*dwl)*
                           (oi['WL']>=_param[k]-dwl/8))

    if np.sum(w)==0:
        print('WARNING: no continuum! using all wavelengths for spectra')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    if debug:
        print('computeNormFluxOI: continuum', oi['WL cont'])
        #print(' ', oi['WL'][oi['WL cont']])

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.15)
        order = max(order, 1)

    if 'fit' in oi and 'NFLUX order' in oi['fit']:
        order = oi['fit']['NFLUX order']

    if np.sum(oi['WL cont'])<order+1:
        print('WARNING: not enough WL to compute continuum!')
        order = np.sum(oi['WL cont'])-1
        #return oi

    oi['NFLUX'] = {}
    # -- normalize flux in the data:
    for k in oi['OI_FLUX'].keys():
        data = []
        edata = []
        cont = []
        for i, flux in enumerate(oi['OI_FLUX'][k]['FLUX']):
            mask = w*~oi['OI_FLUX'][k]['FLAG'][i,:]
            # -- continuum
            if np.sum(mask)>order:
                c = np.polyfit(oi['WL'][w]-np.mean(oi['WL'][w]), flux[w], order)
                _cont = np.polyval(c, oi['WL']-np.mean(oi['WL'][w]))
            else:
                _cont = np.nanmedian(flux)*np.ones(len(flux))
            data.append(flux/_cont)
            cont.append(_cont)

            #edata.append(oi['OI_FLUX'][k]['EFLUX'][i]/np.polyval(c, oi['WL']))
            # -- err normalisation cannot depend on the mask nor continuum calculation!
            edata.append(oi['OI_FLUX'][k]['EFLUX'][i]/np.nanmedian(flux))
            # -- we have to do this here
            if 'fit' in oi and 'mult error' in oi['fit'] and\
                    'NFLUX' in oi['fit']['mult error']:
                edata[-1] *= oi['fit']['mult error']['NFLUX']
            if 'fit' in oi and 'min relative error' in oi['fit'] and\
                    'NFLUX' in oi['fit']['min relative error']:
                edata[-1] = np.maximum(edata[-1],
                    oi['fit']['min relative error']['NFLUX']*data[-1])
            if 'fit' in oi and 'min error' in oi['fit'] and\
                    'NFLUX' in oi['fit']['min error']:
                edata[-1] = np.maximum(edata[-1],
                    oi['fit']['min error']['NFLUX'])

        data = np.array(data)
        edata = np.array(edata)
        cont = np.array(cont)
        oi['NFLUX'][k] = {'NFLUX':data,
                          'ENFLUX':edata,
                          'FLAG':oi['OI_FLUX'][k]['FLAG'],
                          'MJD':oi['OI_FLUX'][k]['MJD'],
                          'CONT': cont,
                          }
        # -- for boostrapping
        if 'NAME' in oi['OI_FLUX'][k]:
            oi['NFLUX'][k]['NAME'] = oi['OI_FLUX'][k]['NAME']

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
    # -- TODO: same as above, but for "totalflux(MJD)"
    return oi

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
            if oi['OI_T3'][k]['formula'] is None:
                break
            s, t, w0, w1, w2 = oi['OI_T3'][k]['formula']
            # print('T3', k, s, t, w0, w1, w2)
            # print(' MJD:', oi['OI_T3'][k]['MJD'], end=' -> ')
            # print(oi['OI_VIS'][t[0]]['MJD'][w0], end=', ')
            # print(oi['OI_VIS'][t[1]]['MJD'][w1], end=', ')
            # print(oi['OI_VIS'][t[2]]['MJD'][w2])
            # print(' u1   :', oi['OI_T3'][k]['u1'], '->', s[0]*oi['OI_VIS'][t[0]]['u'][w0])
            # print(' v1   :', oi['OI_T3'][k]['v1'], '->', s[0]*oi['OI_VIS'][t[0]]['v'][w0])
            # print(' u2   :', oi['OI_T3'][k]['u2'], '->', s[1]*oi['OI_VIS'][t[1]]['u'][w1])
            # print(' v2   :', oi['OI_T3'][k]['v2'], '->', s[1]*oi['OI_VIS'][t[1]]['v'][w1])
            # print(' u3   :', -oi['OI_T3'][k]['u1']-oi['OI_T3'][k]['u2'], '->',
            #     s[2]*oi['OI_VIS'][t[2]]['u'][w2])
            # print(' v3   :', -oi['OI_T3'][k]['v1']-oi['OI_T3'][k]['v2'], '->',
            #     s[2]*oi['OI_VIS'][t[2]]['v'][w2])

            if np.isscalar(s[0]):
                oi['OI_T3'][k]['T3PHI'] = s[0]*oi['OI_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1]*oi['OI_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2]*oi['OI_VIS'][t[2]]['PHI'][w2,:]
                # print('test:', k, t,
                #         np.mean(s[0]*oi['OI_VIS'][t[0]]['PHI'][w0,:]), '+',
                #         np.mean(s[1]*oi['OI_VIS'][t[1]]['PHI'][w1,:]), '+',
                #         np.mean(s[2]*oi['OI_VIS'][t[2]]['PHI'][w2,:]), '=',
                #         np.mean(oi['OI_T3'][k]['T3PHI']))
            else:
                oi['OI_T3'][k]['T3PHI'] = s[0][:,None]*oi['OI_VIS'][t[0]]['PHI'][w0,:]+\
                                          s[1][:,None]*oi['OI_VIS'][t[1]]['PHI'][w1,:]+\
                                          s[2][:,None]*oi['OI_VIS'][t[2]]['PHI'][w2,:]

            # -- force -180 -> 180 degrees
            oi['OI_T3'][k]['T3PHI'] = (oi['OI_T3'][k]['T3PHI']+180)%360-180
            #oi['OI_T3'][k]['ET3PHI'] = np.zeros(oi['OI_T3'][k]['T3PHI'].shape)

            oi['OI_T3'][k]['T3AMP'] = np.abs(oi['OI_VIS'][t[0]]['|V|'][w0,:])*\
                                      np.abs(oi['OI_VIS'][t[1]]['|V|'][w1,:])*\
                                      np.abs(oi['OI_VIS'][t[2]]['|V|'][w2,:])
            #oi['OI_T3'][k]['ET3AMP'] = np.zeros(oi['OI_T3'][k]['T3AMP'].shape)

    if 'IM_VIS' in oi.keys():
        oi['IM_T3'] = {}
        for k in oi['OI_T3'].keys():
            # -- inherit flags from Data,
            oi['IM_T3'][k] = {'FLAG':oi['OI_T3'][k]['FLAG'],
                               'MJD':oi['OI_T3'][k]['MJD'],
                               'Bmax/wl':oi['OI_T3'][k]['Bmax/wl'],
                               'Bavg/wl':oi['OI_T3'][k]['Bavg/wl'],
                               'Bmin/wl':oi['OI_T3'][k]['Bmin/wl'],
                               'B1':oi['OI_T3'][k]['B1'],
                               'B2':oi['OI_T3'][k]['B2'],
                               'B3':oi['OI_T3'][k]['B3'],
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

def computeSlopeVisOI(oi, errfilt={}):
    """
    compute the slope of the visibility
    """
    pass

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

def autoPrior(param):
    prior = []
    for k in param:
        tmp = ['diam', 'ud', 'diamin', 'fwhm', 'crin']
        if k in tmp or (',' in k and (k.split(',')[1] in tmp)):
            # -- strict dimentional priors, with param in mas
            prior.append((k, '>=', 0.0, 1e-6))

        tmp = ['f']
        if k in tmp or (',' in k and (k.split(',')[1] in tmp)):
            # -- hard to set a tolerance, but we can assume 1% of initial guess
            if type(param[k])!=str and param[k]>0:
                prior.append((k, '>=', 0.0, param[k]/100))
            else:
                prior.append((k, '>=', 0.0))

        tmp = ['diamout', 'crout', 'fwhmout']
        if k in tmp or (',' in k and (k.split(',')[1] in tmp)):
            prior.append((k, '>', k[:-3]+'in', 1e-3))

        tmp = ['thick']
        if k in tmp or (',' in k and (k.split(',')[1] in tmp)):
            prior.append((k, '>', 0, 1e-6))
            prior.append((k, '<', 1, 1e-6))

        tmp = ['croff']
        if k in tmp or (',' in k and (k.split(',')[1] in tmp)):
            prior.append((k, '>', -1, 1e-6))
            prior.append((k, '<', 1, 1e-6))
    return prior

def computePriorD(param, prior):
    res = []
    for p in prior.keys():
        form = p+''
        val = str(prior[p][1])+''
        for i in range(3):
            for k in tmp.keys():
                if k in form:
                    form = form.replace(k, '('+str(tmp[k])+')')
                if k in val:
                    val = val.replace(k, '('+str(tmp[k])+')')
        # -- residual
        if len(prior[p])==2:
            resi = '('+form+'-'+str(val)+')'
        elif len(prior[p])==3:
            resi = '('+form+'-'+str(val)+')/abs('+str(prior[p][2])+')'

        if prior[p][0]=='<' or prior[p][0]=='<=' or prior[p][0]=='>' or prior[p][0]=='>=':
            resi = '%s if 0'%resi+prior[p][0]+'%s else 0'%resi
        res.append(eval(resi))
    return np.array(res)

def computePriorL(param, prior, weight=1.0):
    res = []
    for p in prior:
        form = p[0]
        val = str(p[2])
        for i in range(3):
            for k in param.keys():
                if k in form:
                    form = form.replace(k, '('+str(param[k])+')')
                if k in val:
                    val = val.replace(k, '('+str(param[k])+')')
        # -- residual
        if len(p)==3:
            resi = '('+form+'-'+str(val)+')'
        elif len(p)==4:
            # -- ignore weight
            resi = '('+form+'-'+str(val)+')/abs('+str(p[3])+')/%f'%(2*weight)
        if p[1]=='<' or p[1]=='<=' or p[1]=='>' or p[1]=='>=':
            resi = '%s if 0'%resi+p[1]+'%s else 0'%resi
        try:
            res.append(eval(resi.replace('(nan)', '(np.nan)')))
        except:
            res.append(0.0)
            print('WARNING: could not compute prior "'+resi+'"')
    return np.array(res)*weight

def residualsOI(oi, param, timeit=False, what=False, debug=False, fullOutput=False,
                correlations=None, ignoreErr=None, _i0=0):
    """
    assumes dict OI has a key "fit" which list observable to fit:

    OI['fit']['obs'] is a list containing '|V|', 'PHI', 'DPHI', 'V2', 'T3AMP', 'T3PHI'
    OI['fit'] can have key "wl ranges" to limit fit to [(wlmin1, wlmax1), (), ...]

    correlations is the optional correlation dict
    """
    tt = time.time()
    res = np.array([])

    if not correlations is None:
        fullOutput = True

    if fullOutput:
        allwl    = np.array([])
        alldata  = np.array([])
        allerr   = np.array([])
        allmodel = np.array([])
        allins   = np.array([])
        what     = True

    if what:
        wh = []

    if type(oi)==list:
        for i,o in enumerate(oi):
            if what:
                tmp = residualsOI(o, param, timeit=timeit, what=True,
                                  fullOutput=fullOutput, ignoreErr=ignoreErr,
                                  correlations=correlations)
                res = np.append(res, tmp[0])
                wh += tmp[1]
                if fullOutput:
                    allwl    = np.append(allwl,    tmp[2])
                    alldata  = np.append(alldata,  tmp[3])
                    allerr   = np.append(allerr,   tmp[4])
                    allmodel = np.append(allmodel, tmp[5])
                    allins   = np.append(allins,   tmp[6])

            else:
                res = np.append(res, residualsOI(o, param,
                                                timeit=timeit,
                                                what=what,
                                                debug=debug,
                                                ignoreErr=ignoreErr,
                                                _i0=len(res)))
        if fullOutput:
            return res, wh, allwl, alldata, allerr, allmodel, allins
        elif what:
            return res, wh
        else:
            return res

    if 'fit' in oi:
        fit = oi['fit']
    else:
        fit = {'obs':[]}

    t0 = time.time()
    if 'DPHI' in fit['obs'] or 'N|V|' in fit['obs']:
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
            'DPHI':'DVIS',
            'N|V|':'DVIS',
            'V2':'OI_VIS2',
            'T3AMP':'OI_T3',
            'T3PHI':'OI_T3',
            'NFLUX':'NFLUX', # flux normalised to continuum
            'FLUX':'OI_FLUX', # flux, corrected from tellurics
            'CF':'OI_CF', # correlated flux
            }
    w = np.ones(oi['WL'].shape)

    if debug:
        print('residualsOI: fit=', fit)

    if 'wl ranges' in fit:
        w = np.zeros(oi['WL'].shape)
        #for WR in oi['fit']['wl ranges']:
        #    w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        closest = []
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
            closest.append(np.argmin(np.abs(oi['WL']-0.5*(WR[0]+WR[1]))))
        w = np.bool_(w)
        if not any(w):
            for clo in closest:
                w[clo] = True

    w = np.bool_(w)
    t0 = time.time()
    i = _i0
    for f in fit['obs']:
        # -- for each observable:
        if f in sorted(ext.keys()):
            if 'PHI' in f:
                rf = lambda x: ((x + 180)%360 - 180)
            else:
                rf = lambda x: x

            for k in sorted(oi[ext[f]].keys()): # -- for each telescope / baseline / triangle
                test = testTelescopes(k, ignoreTelescope) or testBaselines(k, ignoreBaseline)
                if not test:
                    mask = np.logical_and(w[None,:], ~oi[ext[f]][k]['FLAG'])
                    err = oi[ext[f]][k]['E'+f].copy()
                    if 'baseline ranges' in oi['fit']:
                        #print('debug: baseline ranges')
                        bmask = np.zeros(mask.shape, dtype=bool)
                        for bmin, bmax in oi['fit']['baseline ranges']:
                            if 'FLUX' in ext[f]:
                                bmask = np.ones(mask.shape, dtype=bool)
                            elif 'T3' in ext[f]:
                                tmpmask = np.ones(mask.shape, dtype=bool)
                                for b in ['B1', 'B2', 'B3']:
                                    try:
                                        tmpmask = np.logical_and(tmpmask,
                                            (oi[ext[f]][k][b]<=bmax)*
                                            (oi[ext[f]][k][b]>=bmin))
                                    except:
                                        tmpmask = np.logical_and(tmpmask,
                                               ((oi[ext[f]][k][b]<=bmax)*
                                                (oi[ext[f]][k][b]>=bmin))[:,None])
                                bmask = np.logical_or(bmask, tmpmask)
                            else:
                                bmask = np.logical_or(bmask,
                                        (oi[ext[f]][k]['B/wl']*oi['WL']<=bmax)*
                                        (oi[ext[f]][k]['B/wl']*oi['WL']>=bmin))
                        mask *= bmask

                    if 'MJD ranges' in oi['fit']:
                        mjdmask = np.zeros(mask.shape, dtype=bool)
                        for mjdmin, mjdmax in oi['fit']['MJD ranges']:
                            mjdmask = np.logical_or(mjdmask,
                                (oi[ext[f]][k]['MJD2']<=mjdmax)*\
                                (oi[ext[f]][k]['MJD2']>=mjdmin))
                        mask *= mjdmask

                    if 'max error' in oi['fit'] and f in oi['fit']['max error']:
                        # -- ignore data with large error bars
                        mask *= (err<oi['fit']['max error'][f])
                    if 'max relative error' in oi['fit'] and f in oi['fit']['max relative error']:
                        # -- ignore data with large error bars
                        #mask *= (err<(oi['fit']['max relative error'][f]*
                        #                np.abs(oi[ext[f]][k][f])))
                        # -- above depends on computation, e.g. for NFLUX: can be an issue!
                        mask *= (err<(oi['fit']['max relative error'][f]*
                                 np.median(np.abs(oi[ext[f]][k][f][mask]))))
                    if 'mult error' in fit and f in oi['fit']['mult error']:
                        # -- multiply all errors by value
                        err *= oi['fit']['mult error'][f]
                    if 'min error' in fit and f in oi['fit']['min error']:
                        # -- force error to a minimum value
                        err = np.maximum(oi['fit']['min error'][f], err)
                    if 'min relative error' in fit and f in oi['fit']['min relative error']:
                        # -- force error to a minimum value
                        err = np.maximum(oi['fit']['min relative error'][f]*
                                         np.abs(oi[ext[f]][k][f]), err)

                    if ext[f] in oi and ext[f] in m:
                        try:
                            tmp = rf(oi[ext[f]][k][f][mask] - m[ext[f]][k][f][mask])
                        except:
                            print('!', oi[ext[f]][k][f].shape, mask.shape, m[ext[f]][k][f].shape)
                        if not ignoreErr is None:
                            _i = i+np.arange(len(tmp))
                            tmp /= (err[mask]*(1-ignoreErr[_i]) + ignoreErr[_i])
                            #print('test', i, len(res), len(tmp))
                            i += len(tmp)
                        else:
                            tmp /= err[mask]

                        res = np.append(res, tmp.flatten())

                        if fullOutput:
                            for _m in mask:
                                allwl = np.append(allwl, oi['WL'][_m])
                            alldata  = np.append(alldata, oi[ext[f]][k][f][mask])
                            allerr   = np.append(allerr, err[mask])
                            allmodel = np.append(allmodel, m[ext[f]][k][f][mask])

                            # print('!! residualsOI !!', ext[f], k, f, oi[ext[f]][k][f].shape,
                            #      m[ext[f]][k][f].shape, mask.shape, err.shape)

                        if what:
                            #wh.extend([f+':'+k]*len(tmp))
                            if f=='T3PHI':
                                if k!='all':
                                    kk='|'.join(['%s%s'%('+' if m[ext[f]][k]['formula'][0][ii]==1 else '-',
                                                        m[ext[f]][k]['formula'][1][ii]) for ii in range(3)])
                                else:
                                    kk = []
                                    for j in range(len(m[ext[f]][k]['formula'][0][0])):
                                        ss = []
                                        for ii in range(3):
                                            if m[ext[f]][k]['formula'][0][ii][j]==1:
                                                s='+'
                                            else:
                                                s='-'
                                            s+= m['OI_VIS']['all']['NAME'][m[ext[f]][k]['formula'][2+ii][j]]
                                            ss.append(s)

                                        for _m in mask[j]:
                                            if _m:
                                                kk.append('|'.join(ss))
                            else:
                                if k!='all':
                                    kk = k
                                else:
                                    kk = []
                                    if not 'NAME' in oi[ext[f]][k]:
                                        print('!', f, ext[f], k)
                                    for _j,_n in enumerate(oi[ext[f]][k]['NAME']):
                                        kk.extend([_n]*sum(mask[_j,:]))

                            if not type(kk) is list:
                                wh += [f+':'+kk+';MJD:%.6f'%x for x in oi[ext[f]][k]['MJD2'][mask].flatten()]
                            else:
                                for j,_k in enumerate(kk):
                                    wh.append(f+':'+_k+';MJD:%.6f'%oi[ext[f]][k]['MJD2'][mask].flatten()[j])
                    else:
                        res = np.append(res, (err[mask]*0+1).flatten())
                        if what:
                            wh.extend(['?']*len(err[mask].flatten()))

                else:
                    print('ignoring', ext[f], k)
                    pass

            #print('')
        else:
            print('WARNING: unknown observable:', f)

    # -- do not apply errors in case of correlations, will use ignoreErr later!
    if not correlations is None:
        for k in correlations['rho']:
            if correlations['rho'][k]>0:
                w = np.array(wh)==k
                res[w] *= allerr[w]

    if timeit:
        print('residualsOI > "res": %.3fms'%(1000*(time.time()-t0)))
        print('residualsOI > total: %.3fms'%(1000*(time.time()-tt)))
        print('-'*30)
    if 'fit' in oi:
        if 'ignore relative flux' in oi['fit'] and oi['fit']['ignore relative flux']:
            pass
        else:
            res = np.append(res, m['MODEL']['negativity']*np.sqrt(len(res)))
            if fullOutput:
                allwl    = np.append(allwl,    0.0*m['MODEL']['negativity'])
                alldata  = np.append(alldata,  0.0*m['MODEL']['negativity'])
                allerr   = np.append(allerr,   0.0*m['MODEL']['negativity']+1)
                allmodel = np.append(allmodel, 0.0*m['MODEL']['negativity'])

    else:
        res = np.append(res, m['MODEL']['negativity']*np.sqrt(len(res)))
        if fullOutput:
            allwl    = np.append(allwl,    0.0*m['MODEL']['negativity'])
            alldata  = np.append(alldata,  0.0*m['MODEL']['negativity'])
            allerr   = np.append(allerr,   0.0*m['MODEL']['negativity']+1)
            allmodel = np.append(allmodel, 0.0*m['MODEL']['negativity'])

    if what:
        wh.extend(['<0?'])
    if 'fit' in oi and 'prior' in oi['fit']:
        # -- add priors as additional residuals.
        # -- approximate equal weight as rest of data
        # if not correlations is None:
        #     tmp = computePriorL(computeLambdaParams(param, MJD=np.mean(oi['MJD'])),
        #                     oi['fit']['prior'], weight=1/np.sqrt(len(res)))
        # else:
        tmp = computePriorL(computeLambdaParams(param, MJD=np.mean(oi['MJD'])),
                            oi['fit']['prior'], weight=np.sqrt(len(res)))
        res = np.append(res, tmp)
        if fullOutput:
            allwl    = np.append(allwl,    0.0*tmp)
            alldata  = np.append(alldata,  0.0*tmp)
            allerr   = np.append(allerr,   0.0*tmp+1)
            allmodel = np.append(allmodel, 0.0*tmp)

        if what:
            wh.extend(['prior']*len(oi['fit']['prior']))

    if 'additional residuals' in param:
        tmp = param['additional residuals'](computeLambdaParams(param, MJD=np.mean(oi['MJD'])))
        res = np.append(res, tmp)
        if fullOutput:
            allwl    = np.append(allwl,    0.0*tmp)
            alldata  = np.append(alldata,  0.0*tmp)
            allerr   = np.append(allerr,   0.0*tmp+1)
            allmodel = np.append(allmodel, 0.0*tmp)
        if what:
            wh.extend(['add. res.']*len(tmp))

    if fullOutput:
        return res, wh, allwl, alldata, allerr, allmodel, np.array([oi['insname']]*len(res))
    elif what:
        return res, wh
    else:
        return res

def computeCorrelationsOI(oi, param, dt_s=60):
    """
    chi2 taking into correlations for baselines and CP

    consider correlations if data are separated by less that dt_s seconds

    * ±1/3 correlation between T3PHI sharing one baseline

    * correlations for a spectrum of V2, |V|, DPHI or T3PHI
    """
    # -- for data out of oifits.py
    if not 'OI_T3' in oi:
        return

    # -- for merged data
    #if 'NAME' in oi:
    pass

def sparseFitOI(oi, firstGuess, sparse=[], significance=4, fitOnly=None,
                doNotFit=None, maxfev=5000, ftol=1e-6, follow=None, epsfcn=1e-6,
                verbose=False, prior=[]):
    """
    sparse=[] list of parameters to sparse fit. Should also be fitted according
        to "fitOnly" and/or "doNotFit"!
    significance=3
    """
    test = True
    N = 1
    if not doNotFit is None and fitOnly is None:
        fitOnly = list(filter(lambda k: not k in doNotFit, firstGuess.keys()))

    while test:
        print('Sparse fit %s: running fit #%d, for %d parameters'%(time.asctime(),
                                                                   N, len(fitOnly)))
        fit = fitOI(oi, firstGuess, fitOnly=fitOnly, verbose=verbose,
                    maxfev=maxfev, ftol=ftol, follow=follow, epsfcn=epsfcn,
                    prior=prior)
        #print('   chi2=', fit['chi2'])
        if len(fit['not significant']):
            print('  WARNING, ['+','.join(fit['not significant'])+'] do not change chi2')
            for s in fit['not significant']:
                fitOnly.remove(s)
            print('Sparse fit %s: re-running fit #%d, for %d parameters'%(time.asctime(),
                                                                        N, len(fitOnly)))
            fit = fitOI(oi, firstGuess, fitOnly=fitOnly, verbose=verbose,
                        maxfev=maxfev, ftol=ftol, follow=follow, epsfcn=epsfcn,
                        prior=prior)
        N += 1
        # -- actually, 2 fits is enough: subsequent fit reject only 1 or 2 params...
        if N<=2:
            fitOnly = fit['fitOnly']
            sparse = list(filter(lambda x: x in fitOnly, sparse))
            ignore = []
            #print('   checking', sparse)
            for k in sparse:
                # -- must depart from 0 to be significant, otherwise 0
                if np.abs(fit['best'][k]/fit['uncer'][k])<significance:
                    n = int(2-np.log10(fit['uncer'][k]))
                    f = '%.'+str(n)+'f +- %.'+str(n)+'f'
                    if verbose>1:
                        print('   ', k, 'not significant', f%(fit['best'][k],
                                                              fit['uncer'][k]), '-> 0')
                    fit['best'][k] = 0.0
                    ignore.append(k)
            if len(ignore):
                print('   will ignore %d parameters'%len(ignore))
            for k in ignore:
                fitOnly.remove(k)
                sparse.remove(k)
            test = len(ignore)>0
        else:
            test = False

    fit['sparse'] = sparse
    return fit

def sparseFitFluxes(oi, firstGuess, N={}, initFlux={}, refFlux=None,
                significance=4, fitOnly=None, doNotFit=None,
                maxfev=5000, ftol=1e-6, epsfcn=1e-6, prior=[]):
    """
    use discrete wavelets (WVL) to model spectra of different components.

    firstGuess: initial multi-components model, as dict. Do not need to contain
        the WVL keys. Components with WVL spectra should not contain 'f' value,
        as it will compete with the WVL!

    N: {c1:N1, c2:N2, ...} number of WVL coef. "Ni" must be a power of two!
        Ni must be at least 4

    optional:

    significance (in sigma) of the WVL parameter for sparse fitting. default is 3

    initFlux: {c1:f1, c2:f2, ...} to define the starting flux value. If not present,
        will be set to 1

    refFlux: 'c1' will define a component with WVL as reference flux, i.e. its
        average flux will fized to its initiale value

    """
    assert type(firstGuess)==dict, "firstGuess must be a dictionnary"
    assert type(N)==dict, "N must be a dictionnary"
    for k in N:
        assert N[k]>=4 and 2**int(np.log2(N[k]))==N[k], 'N[%s]>=4 must be power of 2'%k
    if not refFlux is None:
        assert refFlux in N, "refFlux must be one of the components with wavelet spectra"

    if fitOnly is None and doNotFit is None:
        fitOnly = list(firstGuess.keys())
    elif fitOnly is None:
        fitOnly = list(filter(lambda x: not x in doNotFit, firstGuess.keys()))

    # -- initilalise fit loops
    for c in N.keys():
        if c==refFlux:
            fitOnly.append(c+',fwvl_0001')
        else:
            fitOnly.extend([c+',fwvl_0000', c+',fwvl_0001'])

    # -- initialise parameter dict:
    sparse = []
    param = firstGuess.copy()
    for c in N.keys():
        for i in range(N[c]):
            k = c+',fwvl_%04d'%i
            if c in initFlux:
                param[k] = (i==0)*initFlux[c]
            else:
                param[k] = (i==0)*1.0
            if k in fitOnly:
                sparse.append(k)

    # -- fit progressively more and more parameters:
    maxI = int(np.log2(max([N[c] for c in N])))
    for i in range(1, maxI):
        print('== LOOP', i, '/', maxI-1, '==', time.asctime())
        _fitOnly = []
        for c in N:
            if 2**(i+1)-1<=N[c]:
                for j in range(2**i, 2**(i+1)):
                    k = c+',fwvl_%04d'%(j)
                    _fitOnly.append(k)
                    if not k in sparse:
                        sparse.append(k)
        print('   adding', len(_fitOnly), 'parameters to the fit')
        fit = sparseFitOI(oi, param, sparse, ftol=ftol,
                        significance=significance, fitOnly=fitOnly+_fitOnly,
                        maxfev=maxfev, epsfcn=epsfcn)
        for k in sparse:
            if fit['uncer'][k]==0:
                #print(k, 'should be 0!')
                fit['best'][k] = 0
        # -- update sparse with next set of wavelets parameters
        sparse = fit['sparse']
        for c in N:
            for j in range(2**(i+1), N[c]):
                sparse.extend([c+',fwvl_%04d'%j])
        param = fit['best']
        fitOnly = fit['fitOnly']
    f = 100*sum(['wvl' in k and fit['uncer'][k]>0 for k in fit['uncer']])/\
        sum(['wvl' in k for k in fit['uncer']])
    print('-'*50)
    print('fraction of fitted wavelets coef: %.1f%%'%f)
    print('-'*50)
    return fit

def _nSigmas(chi2r_TEST, chi2r_TRUE, NDOF):
    """
    - chi2r_TEST is the hypothesis we test
    - chi2r_TRUE is what we think is what described best the data
    - NDOF: number of degres of freedom

    chi2r_TRUE <= chi2r_TEST

    returns the nSigma detection
    """
    # -- better version by Alex, not limited to 8sigma:
    p = scipy.stats.chi2.sf(NDOF*chi2r_TEST/chi2r_TRUE, NDOF)
    nsigma = np.sqrt(scipy.special.chdtri(1, p))
    return nsigma

    # q = scipy.stats.chi2.cdf(NDOF*chi2r_TEST/chi2r_TRUE, NDOF)
    # p = 1.0-q
    # nsigma = np.sqrt(scipy.stats.chi2.ppf(1-p, 1))
    # if isinstance(nsigma, np.ndarray):
    #     nsigma[p<1e-15] = np.sqrt(scipy.stats.chi2.ppf(1-1e-15, 1))
    # elif p<1e-15:
    #     nsigma = np.sqrt(scipy.stats.chi2.ppf(1-1e-15, 1))
    # return nsigma

def limitOI(oi, firstGuess, p, nsigma=3, chi2Ref=None, NDOF=None, debug=False):
    """
    firstguess: model dictionnary
    p: parameter to explore, assumes p==0 is the null hypothesis
    chi2Ref: reduced chi2 for null hypotesis. If None, will compute for p==0
    """
    if chi2Ref is None:
        pvalue = firstGuess[p]
        firstGuess[p]=0
        # -- assume one free parameter:
        chi2Ref = residualsOI(oi, firstGuess)**2
        NDOF = len(residualsOI(oi, firstGuess))
        chi2Ref = np.sum(chi2Ref)/NDOF
        firstGuess[p] = pvalue
        if debug:
            print('chi2Ref:', chi2Ref)
            print('NDOF:', NDOF)

    ftol = 1e-2
    fact = 0.5
    i = 0
    imax= 256
    tmp = _nSigmas(np.sum(residualsOI(oi, firstGuess)**2)/NDOF, chi2Ref, NDOF)
    if debug:
        print(firstGuess[p], tmp, fact)
    while np.abs(tmp-nsigma)>ftol and i<imax:
        if (tmp<nsigma and fact<1) or (tmp>nsigma and fact>1):
            fact = 1/np.sqrt(fact)
        firstGuess[p] *= fact
        tmp = _nSigmas(np.sum(residualsOI(oi, firstGuess)**2)/NDOF, chi2Ref, NDOF)
        if debug:
            print(firstGuess[p], tmp, fact)
        i+=1
    if i==imax:
        print('!')
    return firstGuess

def tryfitOI(oi, firstGuess, fitOnly=None, doNotFit=None, verbose=3,
          maxfev=5000, ftol=1e-6, follow=None, prior=None, factor=100,
          randomise=False, iter=-1, obs=None, epsfcn=1e-6, keepFlux=False,
          onlyMJD=None, lowmemory=False, additionalRandomise=None,
          correlations=None):
    #try:
        return fitOI(oi, firstGuess, fitOnly, doNotFit, verbose,
              maxfev, ftol, follow, prior, factor,
              randomise, iter, obs, epsfcn, keepFlux,
              onlyMJD, lowmemory, additionalRandomise,
              correlations=correlations)
    #except:
    #    return {}

def fitOI(oi, firstGuess, fitOnly=None, doNotFit=None, verbose=3,
          maxfev=5000, ftol=1e-6, follow=None, prior=None, factor=100,
          randomise=False, iter=-1, obs=None, epsfcn=1e-5, keepFlux=False,
          onlyMJD=None, lowmemory=False, additionalRandomise=None,
          correlations=None):
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

    prior: list of priors as tuples.
        [('a+b', '=', 1.2, 0.1)] if a and b are parameters. a+b = 1.2 +- 0.1
        [('a', '<', 'sqrt(b)', 0.1)] if a and b are parameters. a<sqrt(b), and
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
        tmp = oi
        if not additionalRandomise is None:
            additionalRandomise(False)
    else:
        if not additionalRandomise is None:
            additionalRandomise(True)
        if onlyMJD:
            MJD = []
            for o in oi: # for each datasets
                MJD.extend(o['MJD'])
            MJD = set(MJD)
            onlyMJD = sorted(random.sample(MJD, len(MJD)//2))
        else:
            onlyMJD = None
        tmp = randomiseData2(oi, verbose=False, keepFlux=keepFlux,
                             onlyMJD=onlyMJD)
        # -- TODO: track the resampling and apply it to the correlation "catg"
        # not sure how to do it, maybe have randomiseData2 returns a resampling vector?
    z = 0.0
    if fitOnly is None and doNotFit is None:
        fitOnly = list(firstGuess.keys())

    if not correlations is None:
        # -- need to ignore errors
        resi = residualsOI(tmp, firstGuess, fullOutput=True)
        ignoreErr = np.zeros(len(resi[0]))
        for k in correlations['rho']:
            if correlations['rho'][k]>0:
                w = np.where(np.array(resi[1])==k)
                ignoreErr[w] = 1.0
        addKwargs = {'ignoreErr':ignoreErr}
    else:
        addKwargs = {}

    #addKwargs = {}
    fit = dpfit.leastsqFit(residualsOI, tmp, firstGuess, z,
                           verbose=bool(verbose), correlations=correlations,
                           maxfev=maxfev, ftol=ftol, fitOnly=fitOnly,
                           doNotFit=doNotFit, follow=follow, epsfcn=epsfcn,
                           addKwargs=addKwargs)

    fit['prior'] = prior
    # -- fix az projangles and inclinations
    for k in fit['best']:
        if fit['uncer'][k]>0:
            if k=='projang' or (',' in k and k.split(',')[1]=='projang'):
                fit['best'][k] = (fit['best'][k]+180)%360-180
                fit['track'][k] = (fit['track'][k]+180)%360-180
            if k=='incl' or (',' in k and k.split(',')[1]=='incl'):
                fit['best'][k] = (fit['best'][k]+180)%360-180
                fit['track'][k] = (fit['track'][k]+180)%360-180

    # -- fix az projangles and amplitudes
    fit['best'], changed = _updateAzAmpsProjangs(fit['best'])
    for k in changed:
        if k.replace('az amp', 'az projang') in fit['track']:
            n = int(k.split('az amp')[1])
            fit['track'][k] = np.abs(fit['track'][k])
            fit['track'][k.replace('az amp', 'az projang')] -= 180/n

    if type(verbose)==int and verbose>=1:
        print('# -- degrees of freedom:', fit['ndof'])
        print('# -- reduced chi2:', fit['chi2'])
        dpfit.dispBest(fit)
    if type(verbose)==int and verbose>=2 and \
        len(fit['cord'].keys())>1 and any([fit['uncer'][k]>0 for k in fit['uncer']]):
        dpfit.dispCor(fit)
    if lowmemory:
        for k in ['x', 'y', 'track', 'model']:
            if k in fit:
                fit.pop(k)
        if 'info' in fit:
            for k in ['fvec', 'fjac']:
                if k in fit['info']:
                    fit['info'].pop(k)
    return fit

def _updateAzAmpsProjangs(params):
    # -- changing parameters can have some nasty side effects if "az amp" or
    # -- "az projang" are defined as a function of one another!
    comps = set([k.split(',')[0] for k in params if ',' in k])
    if len(comps)>0:
        _safe = {c:True for c in comps}
        for k in params:
            if ',az amp' in k:
                c = k.split(',')[0]
                if type(params[k])==str or \
                        type(params[k.replace('az amp', 'az projang')])==str:
                    _safe[c] = False
                for kp in params:
                    if type(params)==str and \
                         ('$'+k in params or
                          '$'+k.replace('az amp', 'az projang') in params):
                          _safe[c] = False
        safe = lambda k: _safe[k.split(',')[0]] if ',' in k else True
    else:
        _safe = True
        for k in params:
            if 'az amp' in k:
                if type(params[k])==str or \
                        type(params[k.replace('az amp', 'az projang')])==str:
                    _safe = False
                for kp in params:
                    if type(params)==str and \
                        ('$'+k in params or
                          '$'+k.replace('az amp', 'az projang') in params):
                          _safe = False
        safe = lambda k: _safe
    #print('_safe:', _safe)

    changed = []
    for k in params:
        if safe(k) and 'az amp' in k:
            if type(params[k])!=str and \
                    type(params[k.replace('az amp', 'az projang')])!=str and \
                    params[k]<0:
                # -- correct negative amplitudes
                params[k] = np.abs(params[k])
                n = int(k.split('az amp')[1])
                params[k.replace('az amp', 'az projang')] -= 180/n
                changed.append(k)
            if type(params[k.replace('az amp', 'az projang')])!=str:
                # -- force projection angles -180 -> 180
                #fit['best'][k.replace('az amp', ',az projang')] =\
                #    (fit['best'][k.replace('az amp', ',az projang')]+180)%360-180
                pass

    return params, changed

def _chi2(oi, param):
    res2 = residualsOI(oi, param)**2
    return np.mean(res2)

def randomiseData2(oi, verbose=False, keepFlux=False, onlyMJD=None):
    """
    based on "configurations per MJD". Basically draw data from MJDs and/or
    configuration:
    1) have twice as many oi's, with either:
        - half of the MJDs ignored
        - half of the config, for each MJD, ignored

    oi: list of data, from loadOI

    keepFlux=True: do not randomise flux (default = False)
    """
    res = []

    # -- build a dataset twice as big as original one, but with half the data...
    Nignored = 0
    Nall = 0
    for i in range(len(oi)*2):
        tmp = copy.deepcopy(oi[i%len(oi)])
        # -- collect all the "MJD+config"
        # where config covers the telescope / baselines / triangles
        mjd_c = []
        for k in tmp['configurations per MJD'].keys():
            mjd_c.extend(['%.5f'%k+'|'+str(c) for c in tmp['configurations per MJD'][k]])
        mjd_c = list(set(mjd_c))

        if not onlyMJD is None:
            ignore = list(filter(lambda f: any(['%.5f'%x in f for x in onlyMJD]), mjd_c))
        else:
            random.shuffle(mjd_c)
            ignore = mjd_c[:len(mjd_c)//2]

        Nignored += len(ignore)
        Nall += len(mjd_c)

        if keepFlux:
            # -- do not randomise flux
            exts = list(filter(lambda x: x in ['OI_VIS', 'OI_VIS2',
                                                'OI_T3',], tmp.keys()))

        else:
            # -- also randomise flux
            exts = list(filter(lambda x: x in ['OI_VIS', 'OI_VIS2', 'OI_T3',
                                               'OI_FLUX', 'NFLUX'], tmp.keys()))

        if verbose:
            print('in', i, exts, 'ignore', ignore)
        for l in exts:
            if list(tmp[l].keys()) == ['all']: # merged data
                for i,mjd in enumerate(tmp[l]['all']['MJD']):
                    if '%.5f'%mjd+'|'+tmp[l]['all']['NAME'][i] in ignore:
                        tmp[l]['all']['FLAG'][i,:] = True
            else:
                for k in tmp[l].keys():
                    for i,mjd in enumerate(tmp[l][k]['MJD']):
                        if '%.5f'%mjd+'|'+str(k) in ignore:
                            tmp[l][k]['FLAG'][i,:] = True
        res.append(tmp)
    return res

def randomiseData(oi, randomise='telescope or baseline', P=None, verbose=True):
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
                if x in ['T3AMP']:
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

PROG_UPDATE = 1 # update period, in seconds
_prog_N = 1
_prog_Nmax = 0
_prog_t0 = time.time()
_prog_last = time.time()

def progress(results=None, finish=False):
    global _prog_N, _prog_last
    if finish:
        _prog_N = _prog_Nmax
    if finish or time.time()-_prog_last >= PROG_UPDATE:
        _nb = 60 # length of the progress bar
        tleft = (time.time()-_prog_t0)/max(_prog_N, 1)*(_prog_Nmax-_prog_N)
        if tleft>100:
            tleft = '%3.0fmin'%(tleft/60)
        else:
            tleft = '%3.0fs  '%(tleft)
        fmt = '%'+'%d'%int(np.ceil(np.log10(_prog_Nmax)))+'d'
        fmt = '%s/%s'%(fmt, fmt)+' %s left'
        res = time.asctime()+': '+\
            '['+bytes((219,)).decode('cp437')*int(_nb*_prog_N/max(_prog_Nmax, 1))+\
            '.'*(_nb-int(_nb*_prog_N/max(_prog_Nmax, 1))) + ']'+\
            fmt%(_prog_N, _prog_Nmax, tleft)+'\r'
        #print(res)
        sys.stdout.write(res)
        _prog_last = time.time()
    _prog_N+=1

def gridFitOI(oi, param, expl, N=None, fitOnly=None, doNotFit=None,
              maxfev=5000, ftol=1e-6, multi=True, epsfcn=1e-6,
              dLimParam=None, dLimSigma=3, debug=False, constrain=None,
              prior=None, verbose=2, correlations=None):
    """
    perform "N" fit on "oi", starting from "param", with grid / randomised
    parameters. N can be determined from "expl" if

    expl = {'grid':{'p1':(0,1,0.1), 'p2':(-1,1,0.5), ...},
            'rand':{'p3':(0,1), 'p4':(-np.pi, np.pi), ...},
            'randn':{'p5':(0, 1), 'p6':(np.pi/2, np.pi), ...}}

    grid=(min, max, step): explore all values for "min" to "max" with "step"
    rand=(min, max): uniform randomized parameter
    randn=(mean, std): normaly distributed parameter
    parameters can only appear once in either grid, rand or randn

    if "grid" are defined, they will define N as:
    N = prod_i((max_i-min_i)/step_i + 1)


    constrain: list of conditions, with same syntax as priors (see computePriorL).
    """
    global _prog_N, _prog_Nmax, _prog_t0, _prog_last

    assert type(expl)==dict, "expl must be a dict"
    assert 'grid' in expl or 'rand' in expl or 'randn' in expl
    # -- compute N and vectors for grid parameters
    if 'grid' in expl:
        N = 1
        R = {k:np.array([0]) for k in expl['grid']}
        for k in expl['grid']:
            n = int((expl['grid'][k][1]-expl['grid'][k][0])/expl['grid'][k][2] + 1)
            N *= n
            for l in expl['grid']:
                R[l] = R[l][:,None]+(k==l)*np.linspace(expl['grid'][k][0], expl['grid'][k][1], n)[None,:]
                R[l] = R[l].flatten()
    assert not N is None, 'cannot assert N, must be explicitly given!'

    if constrain is None:
        constrain = []

    # -- Prepare list of starting parameters' dict:
    PARAM = []

    for i in range(N):
        tmp = param.copy()
        if 'grid' in expl:
            for k in expl['grid']:
                tmp[k] = R[k][i]
        if 'rand' in expl:
            for k in expl['rand']:
                tmp[k] = expl['rand'][k][0] + np.random.rand()*(expl['rand'][k][1]-
                                                         expl['rand'][k][0])
        if 'randn' in expl:
            for k in expl['randn']:
                tmp[k] = expl['randn'][k][0] + np.random.randn()*expl['randn'][k][1]
        res = [0]
        for p in constrain:
            form = p[0]
            val = str(p[2])
            for i in range(3):
                for k in tmp.keys():
                    if k in form:
                        form = form.replace(k, '('+str(tmp[k])+')')
                    if k in val:
                        val = val.replace(k, '('+str(tmp[k])+')')
            # -- residual
            if len(p)==3:
                resi = '('+form+'-'+str(val)+')'
            elif len(p)==4:
                resi = '('+form+'-'+str(val)+')/abs('+str(p[3])+')'
            if p[1]=='<' or p[1]=='<=' or p[1]=='>' or p[1]=='>=':
                resi = '%s if 0'%resi+p[1]+'%s else 0'%resi
            try:
                res.append(eval(resi))
            except:
                print('WARNING: could not compute constraint "'+resi+'"')
        if all(np.array(res)==0):
            PARAM.append(tmp)
    if len(PARAM)<N and verbose:
        print(N-len(PARAM), 'grid point%s not within constraints'%('' if (N-len(PARAM))==1 else 's'))

    N = len(PARAM)
    # -- run all fits
    kwargs = {'maxfev':maxfev, 'ftol':ftol, 'verbose':False,
              'fitOnly':fitOnly, 'doNotFit':doNotFit, 'epsfcn':epsfcn,
              'iter':-1, 'prior':prior, 'lowmemory':True,
              'correlations':correlations}
    res = []
    _prog_N = 1
    _prog_Nmax = N
    _prog_t0 = time.time()
    _prog_last = time.time()

    if multi:
        if type(multi)!=int:
            #Np = min(multiprocessing.cpu_count(), N)
            Np = min(MAX_THREADS, N)
        else:
            Np = min(multi, N)
        if verbose:
            print(time.asctime()+': running', N, 'fits on', Np, 'processes')
        # -- estimate fitting time by running 'Np' fit in parallel
        t = time.time()
        pool = multiprocessing.Pool(Np)
        for i in range(N):
            if dLimParam is None:
                kwargs['iter'] = i
                if verbose:
                    res.append(pool.apply_async(tryfitOI, (oi, PARAM[i], ), kwargs,
                                                callback=progress))
                else:
                    res.append(pool.apply_async(tryfitOI, (oi, PARAM[i], ), kwargs))

            else:
                kwargs = {'nsigma': dLimSigma}
                if verbose:
                    res.append(pool.apply_async(limitOI, (oi, PARAM[i], dLimParam, ), kwargs,
                                                 callback=progress))
                else:
                    res.append(pool.apply_async(limitOI, (oi, PARAM[i], dLimParam, ), kwargs))

        pool.close()
        pool.join()
        res = [r.get(timeout=1) for r in res]
        res = [r for r in res if r!={}]
        # -- make sure the progress bar finishes
        if verbose:
            progress(finish=True)
    else:
        if debug:
            print('single thread')
        t = time.time()
        for i in range(N):
            if dLimParam is None:
                kwargs['iter'] = i
                res.append(tryfitOI(oi, PARAM[i], **kwargs))
                if verbose:
                    progress()
            else:
                kwargs = {'nsigma': dLimSigma}
                res.append(limitOI(oi, PARAM[i], dLimParam, **kwargs))
                if verbose:
                    progress()
        # -- make sure the progress bar finishes
        progress(finish=True)
        res = [r for r in res if r!={}]
    if verbose:
        print() # clear progress bar

    if verbose:
        print(time.asctime()+': it took %.1fs, %.2fs per fit on average'%(time.time()-t,
                                                    (time.time()-t)/N),
                                                    end=' ')
        print('[%.1f fit/minute]'%( 60*N/(time.time()-t)))

    #if dLimParam is None:
    #    res = analyseGrid(res, expl, verbose=1)
    return res

def analyseGrid(fits, expl, debug=False, verbose=1, deltaChi2=None):
    global _prog_N, _prog_Nmax, _prog_t0, _prog_last
    res = []
    bad = []
    errTooLarge = []
    noUncer = []
    infos = []
    # -- remove bad fit (no uncertainties)
    chi2min = np.nanmin([f['chi2'] for f in fits])
    chi2TooLarge = []
    for i,f in enumerate(fits):
        if np.isnan(f['chi2']):
            bad.append(f.copy())
            bad[-1]['bad'] = True
            noUncer.append(i)
            infos.append(f['mesg'])
        elif not deltaChi2 is None and f['chi2']>chi2min+deltaChi2:
            bad.append(f.copy())
            bad[-1]['bad'] = True
            chi2TooLarge.append(i)
        elif np.sum([f['uncer'][k]**2 for k in f['uncer']]):
            # -- some uncertainties are >0
            # -- check errors are reasonable / grid:
            test = False
            if 'grid' in expl:
                for k in expl['grid']:
                    test = test or f['uncer'][k]>np.abs(expl['grid'][k][2])
            if test:
                bad.append(f.copy())
                bad[-1]['bad'] = True
                errTooLarge.append(i)
            else:
                res.append(f.copy())
                res[-1]['bad'] = False
        else:
            bad.append(f.copy())
            bad[-1]['bad'] = True
            noUncer.append(i)
            infos.append(f['mesg'])

    if debug or verbose:
        print('fits to be taken into account:', len(res), '/', len(fits))
        if len(bad)-len(errTooLarge):
            print(' ', len(bad)-len(errTooLarge)-len(chi2TooLarge), 'did not numerically converge',
                '(incl %d have no uncertainties)'%len(noUncer))
        #print(Counter(infos))
        if len(errTooLarge):
            print(' ', len(errTooLarge),
              "have uncertainties larger than the grid's step[s]")
        if len(chi2TooLarge):
            print(' ', len(chi2TooLarge), "have chi2r > %.3f + %f = %.3f"%(chi2min, deltaChi2,
                chi2min+deltaChi2))


    # -- unique fits:
    tmp = []
    ignore = []
    chi2 = np.array([r['chi2'] for r in res])
    map = {}
    uncer = [r['uncer'].copy() for r in res]
    fitOnly = [r['fitOnly'].copy() for r in res]

    # -- for grid parameters, if not fitted, uncer is the step
    if 'grid' in expl:
        for k in expl['grid']:
            for i,u in enumerate(uncer):
                if u[k]==0:# or uncer[i][k]>0.5*expl['grid'][k][2]:
                    # -- step of grid
                    uncer[i][k] = 0.5*expl['grid'][k][2]
                    if not k in fitOnly[i]:
                        # -- for distance computation later
                        fitOnly[i].append(k)

    # -- create list of unique minima
    if verbose or debug:
        print(time.asctime()+': making list of unique minima...')
    if len(res)>1000:
        _prog_N = 1
        _prog_Nmax = len(res)
        _prog_t0 = time.time()
        _prog_last = time.time()

    mask = np.array([True for i in range(len(res))])
    keep = list(range(len(res)))
    for j in ignore:
        keep.remove(j)
    for i,f in enumerate(res):
        if len(res)>1000:
            progress()
        if i in ignore:
            continue
        # -- compute distance between minima, based on fitted parameters
        d = [np.nanmean([(f['best'][k]-res[j]['best'][k])**2/(uncer[i][k]*uncer[j][k])
                     for k in fitOnly[i]]) for j in keep]
        # -- group solutions with closeby ones
        w = np.array(d)/len(fitOnly[i]) < 1
        if debug:
            print(i, np.round(d, 2), w)
            print(list(np.arange(len(res))[w]))

        # -- best solution from the bunch
        tmp.append(res[np.array(keep)[w][np.nanargmin(chi2[np.array(keep)[w]])]])

        tmp[-1]['index'] = i
        # -- which minima should be considered
        map[i] = list(np.array(keep)[w])

        ignore.extend(list(np.array(keep)[w]))

        for j in np.array(keep)[w]:
            keep.remove(j)
    if len(res)>1000 and verbose:
        # -- make sure the progress bar finishes
        progress(finish=True)
        print()

    if debug or verbose:
        print('unique minima:', len(tmp), '/', len(res), end=' ')
        print('[~%.1f first guesses / minima]'%(len(res)/len(tmp)))
        if len(tmp)<len(res)/4:
            print('  few unique minima -> grid too fine / Nfits too large?')
        elif len(tmp)<=len(res)/2:
            print('  number of minima is OK compared to grid coarseness')
    if len(tmp)>len(res)/2 and verbose:
        print('  \033[43mWARNING!\033[0m: too many unique minima -> grid too coarse / Nfits too small?', end=' ')
        print(' \033[33mfinding the global minimum may be unreliable\033[0m')

    # -- keep track of all initial values leading to the local minimum
    if verbose or debug:
        print(time.asctime()+': linking unique minima to first guess(es)')
    #_prog_N = 1
    #_prog_Nmax = len(tmp)
    #_prog_t0 = time.time()

    for i,t in enumerate(tmp):
        #progress() this is always very fast
        _fG = []
        for j in map[t['index']]:
            if type(res[j]['firstGuess'])==list:
                _fG.extend(res[j]['firstGuess'])
            else:
                _fG.append(res[j]['firstGuess'].copy())
        t['firstGuess'] = _fG
        # t['firstGuess'] = []
        # for j in map[t['index']]:
        #     if type(res[j]['firstGuess'])==list:
        #         t['firstGuess'].extend(res[j]['firstGuess'])
        #     else:
        #         t['firstGuess'].append(res[j]['firstGuess'])
    #print()

    res = tmp
    res = sorted(res, key=lambda r: r['chi2'])

    # -- add bad fits:
    for b in bad:
        res.append(b)
        if type(res[-1]['firstGuess'])!=list:
            res[-1]['firstGuess'] = [res[-1]['firstGuess'].copy()]
        res[-1]['bad'] = True
    if verbose:
        print(time.asctime()+': done')
        print('-'*12)
        print('best fit: chi2=', res[0]['chi2'])
        dpfit.dispBest(res[0])
    try:
        if type(verbose)==int and verbose>1:
            dpfit.dispCor(res[0])
    except:
        pass
    return res

def showGrid(res, px, py, color='chi2', logV=False, fig=0, aspect=None,
             vmin=None, vmax=None, cmap='gist_stern', interpolate=False,
             expl=None, tight=False, constrain=None, significance=False):
    """
    res: results from a gridFitOI
    px, py: the two parameters to show (default 2 first alphabetically)
    color: color for the plot (default is chi2)

    """
    #print('debug sG:', '"'+px+'"', '"'+py+'"')
    plt.close(fig)
    plt.figure(fig)

    if not aspect is None:
        ax = plt.subplot(111, aspect=aspect)
    if not significance is False:
        c = np.array([_nSigmas(significance, r['chi2'], r['ndof'])
                        for r in res if ~r['bad']])
        color = r'significance min($\sigma$, 8)'
    else:
        # -- color of local minima
        if not color=='chi2':
            c = np.array([r['best'][color] for r in res if ~r['bad']])
        else:
            c = np.array([r[color] for r in res if ~r['bad']])

    # -- coordinates: keep only valid minima
    x = [r['best'][px] for r in res if ~r['bad']]
    y = [r['best'][py] for r in res if ~r['bad']]

    # -- initial positions
    ix, iy = [], []
    if not (type(interpolate)==int and interpolate>1):
        for r in res:
            for f in r['firstGuess']:
                if r['bad']:
                    plt.plot(f[px], f[py], 'x', color='r', alpha=0.3)
                else:
                    if type(f)==dict:
                        plt.plot(f[px], f[py], '+', color='k', alpha=0.3)
                        plt.plot([f[px], r['best'][px]],
                                 [f[py], r['best'][py]], '-k', alpha=0.2)
                        ix.append(f[px])
                        iy.append(f[py])

                    elif type(f)==list:
                        for _f in f:
                            plt.plot(_f[px], _f[py], '+', color='k', alpha=0.3)
                            plt.plot([_f[px], r['best'][px]],
                                     [_f[py], r['best'][py]], '-k', alpha=0.2)
                            ix.append(_f[px])
                            iy.append(_f[py])

    S = 4 # interpolation factor
    if type(expl) is dict and 'grid' in expl and px in expl['grid']:
        dx = expl['grid'][px][2]
        IX = np.linspace(expl['grid'][px][0]-expl['grid'][px][2]/2,
                         expl['grid'][px][1]+expl['grid'][px][2]/2,
                         S*int((expl['grid'][px][1]-expl['grid'][px][0])/
                              expl['grid'][px][2]+2)
                         )
    else:
        dx = np.ptp(ix)/np.sqrt(len(ix))
        IX = np.linspace(np.min(ix)-dx/2, np.max(ix)+dx/2, S*int(np.ptp(ix)/dx+1))

    if type(expl) is dict and 'grid' in expl and py in expl['grid']:
        dy = expl['grid'][py][2]
        IY = np.linspace(expl['grid'][py][0]-expl['grid'][py][2]/2,
                         expl['grid'][py][1]+expl['grid'][py][2]/2,
                         S*int((expl['grid'][py][1]-expl['grid'][py][0])/
                              expl['grid'][py][2]+2))
    else:
        dy = np.ptp(iy)/np.sqrt(len(iy))
        IY = np.linspace(np.min(iy)-dy/2, np.max(iy)+dy/2, S*int(np.ptp(iy)/dy+1))

    if type(vmax)==str:
        vmax = np.percentile(c, float(vmax))
    if type(vmin)==str:
        vmin = np.percentile(c, float(vmin))

    if vmin is None:
        vmin = min(c)
    if vmax is None:
        vmax = max(c)

    if logV and (min(c)>0 or vmin>0):
        c = np.log10(np.maximum(c, vmin))
        color = 'log10['+color+']'
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    if not (type(interpolate)==int and interpolate>1):
        plt.scatter(x, y, c=c, vmin=vmin, vmax=vmax, cmap=cmap,
                    plotnonfinite=True)
        plt.colorbar(label=color)
        # -- global minimum
        plt.plot(x[0], y[0], marker=r'$\bigodot$',
             #color=matplotlib.cm.get_cmap(cmap)(0),
             color=plt.get_cmap(cmap)(0),
             markersize=20, alpha=0.5)

    if interpolate:
        nX, nY = len(IX), len(IY)
        IX, IY = np.meshgrid(IX, IY)
        if constrain is None:
            constrain = []
        mask = np.ones(IX.shape)
        for p in constrain:
            form = p[0]
            val = str(p[2])
            for i in range(3):
                if px in form:
                    form = form.replace(px, 'IX')
                if px in val:
                    val = val.replace(px, 'IX')
                if py in form:
                    form = form.replace(py, 'IY')
                if py in val:
                    val = val.replace(py, 'IY')
            # -- residual
            if len(p)==3:
                resi = '('+form+'-'+str(val)+')'
            elif len(p)==4:
                resi = '('+form+'-'+str(val)+')/abs('+str(p[3])+')'
            if p[1]=='<' or p[1]=='<=' or p[1]=='>' or p[1]=='>=':
                resi = '%s if 0'%resi+p[1]+'%s else 0'%resi
            #print(resi)
            #print(eval(resi)==0)

            # try:
            #     res.append(eval(resi))
            # except:
            #     print('WARNING: could not compute constraint "'+resi+'"')


        grP = np.array([(x[i], y[i]) for i in range(len(x))])
        gr = np.array([IX, IY]).reshape(2, -1).T
        tmp = scipy.interpolate.RBFInterpolator(grP, c,
                                                kernel='linear', neighbors=4,
                                                #kernel='thin_plate_spline',
                                                #kernel='cubic',
                                                #kernel='gaussian', epsilon=.15*np.sqrt(dx*dy),
                                                #kernel='multiquadric',epsilon=.1*np.sqrt(dx*dy),
                                                )(gr).reshape((nY, nX))
        if interpolate>1:
            alpha=1
        else:
            alpha=0.8
        plt.pcolormesh(IX, IY, tmp*mask, cmap=cmap, alpha=alpha, rasterized=True)
        if interpolate>1:
            plt.colorbar(label=color)

    if tight:
        plt.xlim(np.min(IX)-dx/2, np.max(IX)+dx/2)
        plt.ylim(np.min(IY)-dy/2, np.max(IY)+dy/2)

    plt.xlabel(px)
    plt.ylabel(py)
    plt.tight_layout()
    return

def bootstrapFitOI(oi, fit, N=None, maxfev=5000, ftol=1e-6, sigmaClipping=None, chi2MaxClipping=None,
                    multi=True, prior=None, keepFlux=False, verbose=2,
                    strongMJD=False, randomiseParam=True, additionalRandomise=None,
                    correlations=None):
    """
    randomised draw data and perform N fits. Some parameters of the fitting engine can be changed,
    but overall the fitting context is the same as the last fit which was run.

    see also: doFit
    """
    global _prog_N, _prog_Nmax, _prog_t0, _prog_last

    if N is None:
        # -- count number of spectral vector data
        N = 0
        ext = {'|V|':'OI_VIS',
                'PHI':'OI_VIS',
                'DPHI':'OI_VIS',
                'V2':'OI_VIS2',
                'T3AMP':'OI_T3',
                'T3PHI':'OI_T3',
                'NFLUX':'NFLUX', # flux normalised to continuum
                'FLUX':'OI_FLUX' # flux corrected from tellurics
                }
        if type(oi)==dict:
            oi = [oi]
        for o in oi: # for each datasets
            for p in o['fit']['obs']:
                if ext[p] in o:
                    for k in o[ext[p]].keys():
                        N += len(o[ext[p]][k]['MJD'])
        N *= 2

    if strongMJD:
        # -- list all MJDs:
        MJD = []
        for o in oi: # for each datasets
            MJD.extend(o['MJD'])
        MJD = set(MJD)
        if len(MJD)<5: # to get at least 100 combination
            print('WARNING: cannot randomise only on MJD for', len(MJD), 'dates,',
                  'randomising on MJD+baselines/triangles instead')
            strongMJD = False
        elif verbose:
            print('randomising on', len(MJD), 'different dates')

    fitOnly = fit['fitOnly']
    doNotFit = fit['doNotFit']
    maxfev = fit['maxfev']
    if ftol is None:
        ftol = fit['ftol']
    epsfcn= fit['epsfcn']
    prior = fit['prior']
    firstGuess = fit['best']
    uncer = fit['uncer']

    kwargs = {'maxfev':maxfev, 'ftol':ftol, 'verbose':False,
              'fitOnly':fitOnly, 'doNotFit':doNotFit, 'epsfcn':epsfcn,
              'randomise':True, 'prior':prior, 'iter':-1,
              'keepFlux':keepFlux, 'onlyMJD':strongMJD, 'lowmemory':True,
              'additionalRandomise': additionalRandomise,
              'correlations':correlations}
    res = []
    t = time.time()
    if multi:
        if type(multi)!=int:
            Np = min(MAX_THREADS, N)
        else:
            Np = min(multi, N)
        if verbose:
            print(time.asctime()+': running', N, 'fits on', Np, 'processes')

        # -- run the remaining
        pool = multiprocessing.Pool(Np)
        _prog_N = 1
        _prog_Nmax = N
        _prog_t0 = time.time()
        _prog_last = time.time()

        for i in range(N):
            kwargs['iter'] = i
            if randomiseParam:
                # -- todo: should use the covariance matrix!
                if type(randomiseParam)!= bool:
                    tmpfg = {k:firstGuess[k]+randomiseParam*np.random.randn()*uncer[k]
                            if uncer[k]>0 else firstGuess[k] for k in firstGuess}
                else:
                    tmpfg = {k:firstGuess[k]+np.random.randn()*uncer[k]
                            if uncer[k]>0 else firstGuess[k] for k in firstGuess}
            else:
                tmpfg = firstGuess
            if verbose:
                res.append(pool.apply_async(fitOI, (oi, tmpfg, ), kwargs, callback=progress))
            else:
                res.append(pool.apply_async(fitOI, (oi, tmpfg, ), kwargs))

        pool.close()
        pool.join()
        res = [r.get(timeout=1) for r in res]
        for r in res:
            r['y'] = None
            r['y'] = None
        if verbose:
            # -- make sure the progress bar finishes
            progress(finish=True)
    else:
        Np = 1
        t = time.time()
        _prog_N = 1
        _prog_Nmax = N
        _prog_t0 = time.time()
        _prog_last = time.time()

        for i in range(N):
            kwargs['iter'] = i
            res.append(fitOI(oi, firstGuess, **kwargs))
            if verbose:
                progress()
        # -- make sure the progress bar finishes
        if verbose:
            progress(finish=True)
    if verbose:
        print(time.asctime()+': it took %.1fs, %.2fs per fit on average'%(time.time()-t,
                                                    (time.time()-t)/N),
                                                    end=' ')
        try:
            print('[%.1f fit/minutes]'%( 60*N/(time.time()-t)))
        except:
            print(time)

    res = analyseBootstrap(res, sigmaClipping=sigmaClipping, verbose=verbose, chi2MaxClipping=chi2MaxClipping)
    res['fit to all data'] = fit
    return res

def analyseBootstrap(Boot, sigmaClipping=None, verbose=2, chi2MaxClipping=None):
    """
    Boot: a list of fits (list of dict from dpfit.leastsqFit)
    """
    # -- allow to re-analyse bootstrapping results:
    if type(Boot) == dict and 'all fits' in Boot.keys():
        fit = Boot['fit to all data']
        Boot = Boot['all fits']
    else:
        fit = None
    try:
        res = {'best':{}, 'uncer':{}, 'uncer-':{}, 'uncer+':{},
               'fitOnly':Boot[0]['fitOnly'],
               'all best':{}, 'all best ignored':{}, 'sigmaClipping':sigmaClipping,
               'all fits':Boot, 'chi2MaxClipping':chi2MaxClipping}
    except:
        print(Boot)

    if not fit is None:
        res['fit to all data'] = fit
    mask = np.ones(len(Boot), dtype=bool)
    # -- sigma clipping and global mask
    if not sigmaClipping is None:
        for k in res['fitOnly']:
            tmp = np.ones(len(Boot), dtype=bool)
            for j in range(3): # iterate a few times
                x = np.array([b['best'][k] for b in Boot])
                res['best'][k] = np.nanmedian(x[tmp])
                res['uncer'][k] = (np.nanpercentile(x[tmp], 84) - np.nanpercentile(x[tmp], 16))/2
                tmp = np.abs(x-res['best'][k])<=sigmaClipping*res['uncer'][k]
            mask *= tmp
    if not chi2MaxClipping is None:
        mask *= np.array([b['chi2']<=chi2MaxClipping for b in Boot])

    for k in res['fitOnly']:
        for j in range(3):
            x = np.array([b['best'][k] for b in Boot])
            #res['best'][k] = np.mean(x[mask])
            #res['uncer'][k] = np.std(x[mask])
            res['best'][k] = np.median(x[mask])
            res['uncer+'][k] = np.nanpercentile(x[mask], 84)-res['best'][k]
            res['uncer-'][k] = res['best'][k]-np.nanpercentile(x[mask], 16)
            res['uncer'][k] = 0.5*(res['uncer+'][k]+res['uncer-'][k])

        res['all best'][k] = x[mask]
        res['all best ignored'][k] = x[~mask]
    res['all chi2'] = np.array([b['chi2'] for b in Boot])[mask]
    res['all chi2 ignored'] = np.array([b['chi2'] for b in Boot])[~mask]

    for k in Boot[0]['best'].keys():
        if not k in res['best'].keys():
            res['best'][k] = Boot[0]['best'][k]
            res['uncer'][k] = 0.0
    res['chi2'] = Boot[0]['chi2']
    res['mask'] = mask

    fitOnly = res['fitOnly'].copy()
    fitOnly.append('chi2')
    M = [[b['chi2'] if k=='chi2' else b['best'][k]  for i,b in enumerate(Boot) if mask[i]] for k in fitOnly]

    if len(fitOnly)>1:
        res['cov'] = np.cov(M)
        cor = np.sqrt(np.diag(res['cov']))
        cor = cor[:,None]*cor[None,:]
    else:
        res['cov'] = np.array([[np.std(M)**2]])
        cor = np.array([[np.sqrt(res['cov'])]])

    res['cor'] = res['cov']/cor
    res['covd'] = {ki:{kj:res['cov'][i,j] for j,kj in enumerate(fitOnly)}
                   for i,ki in enumerate(fitOnly)}
    res['cord'] = {ki:{kj:res['cor'][i,j] for j,kj in enumerate(fitOnly)}
                   for i,ki in enumerate(fitOnly)}
    if verbose:
        if not sigmaClipping is None:
            print('using %d fits out of %d (sigma clipping:%.2f, chi2MaxClipping:%s)'%(
                    np.sum(mask), len(Boot), sigmaClipping, str(chi2MaxClipping)))
        ns = max([len(k) for k in res['best'].keys()])
        # print('{', end='')
        # for k in sorted(res['best'].keys()):
        #     if res['uncer'][k]>0:
        #         n = int(np.ceil(-np.log10(res['uncer'][k])+1))
        #         fmt = '%.'+'%d'%max(n, 0)+'f'
        #         print("'"+k+"'%s:"%(' '*(ns-len(k))), fmt%res['best'][k], end=', ')
        #         print('# +/-', fmt%res['uncer'][k])
        #     else:
        #         print("'"+k+"'%s:"%(' '*(ns-len(k))), end='')
        #         if type(res['best'][k])==str:
        #             print("'"+res['best'][k]+"',")
        #         else:
        #             print(res['best'][k], ',')
        # print('}')
        dpfit.dispBest(res)

    if verbose>1 and np.size(res['cov'])>1:
        dpfit.dispCor(res)
    return res

def sigmaClippingOI(oi, sigma=4, n=5, param=None):
    if type(oi)==list:
        return [sigmaClippingOI(o, sigma=sigma, n=n, param=param) for o in oi]

    w = oi['WL']>0
    # -- user-defined wavelength ranges
    # if 'fit' in oi and 'wl ranges' in oi['fit']:
    #     w = np.zeros(oi['WL'].shape)
    #     for WR in oi['fit']['wl ranges']:
    #         w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    # oi['WL mask'] = np.bool_(w)

    closest = []
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        closest.append(np.argmin(np.abs(oi['WL']-0.5*(WR[0]+WR[1]))))
    oi['WL mask'] = np.bool_(w)
    if not any(oi['WL mask']):
        for clo in closest:
            oi['WL mask'][clo] = True

    # -- user defined continuum
    if 'fit' in oi and 'continuum ranges' in oi['fit']:
        wc = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['continuum ranges']:
            wc += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        w *= np.bool_(wc)

    # -- exclude where lines are in the models
    if not param is None:
        for k in param.keys():
            if 'line_' in k and 'wl0' in k:
                dwl = 0
                if k.replace('wl0', 'gaussian') in param.keys():
                    dwl = 1.5*param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in param.keys():
                    dwl = 3*param[k.replace('wl0', 'lorentzian')]/1000.
                if k.replace('wl0', 'truncexp') in param.keys():
                    dwl = 2*param[k.replace('wl0', 'lorentzian')]/1000.
                vel = 0
                if ',' in k:
                    kv = k.split(',')[0]+','+'Vin'
                    fv = 1.5
                    if k.replace('Vin', 'incl') in _param:
                        fv *= np.sin(_param[k.replace('Vin', 'incl')]*np.pi/180)

                else:
                    kv = 'Vin'
                    fv = 1.5
                    if 'incl' in _param:
                        fv *= np.sin(_param['incl']*np.pi/180)

                # if ',' in k:
                #     kv = k.split(',')[0]+','+'Vin'
                #     fv = 1.0
                # else:
                #     kv = 'Vin'
                #     fv = 1.0
                if not kv in _param:
                    kv+='_Mm/s'
                    fv = 1000

                if kv in _param:
                    vel = np.abs(_param[kv]*fv)

                if ',' in k:
                    kv = k.split(',')[0]+','+'V1mas'
                else:
                    kv = 'V1mas'
                if kv in _param:
                    if 'Rin' in _param:
                        vel = _param[kv]/np.sqrt(param[kv.replace('V1mas', 'Rin')])
                    elif 'diamin' in _param:
                        vel = _param[kv]/np.sqrt(0.5*_param[kv.replace('V1mas', 'diamin')])

                dwl = np.sqrt(dwl**2 + (_param[k]*vel/2.998e5)**2)
                w *= (np.abs(oi['WL']-param[k])>=dwl)
    if np.sum(w)==0:
        print('WARNING: no continuum! using all wavelengths for clipping')
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

ai1mcB = {'i':0} # initialize global marker/color for baselines
ai1mcT = {'i':0} # initialize global marker/color for triangles
ai1ax = {} # initialise global list of axes
ai1i = [] # initialise global list of axes position
ai1fig = None

FIG_MAX_WIDTH = 9.5
FIG_MAX_HEIGHT = 6

def showOI(oi, param=None, fig=0, obs=None, showIm=False, imFov=None, imPix=None,
           imPow=1., imWl0=None, cmap='bone', imX=0.0, imY=0.0, debug=False,
           showChi2=False, wlMin=None, wlMax=None, spectro=None, imMax=None,
           figWidth=None, figHeight=None, logB=False, logV=False, logS=False,
           color=(1.0,0.2,0.1), checkImVis=False, showFlagged=False,
           onlyMJD=None, showUV=False, allInOne=False, vWl0=None,
           cColors={}, cMarkers={}, imoi=None, bckgGrid=True, barycentric=False,
           autoLimV=False, t3B='max'):
    """
    oi: result from oifits.loadOI
    param: dict of parameters for model (optional)
    obs: observable to show (default oi['fit']['obs'] if found, else all available)
        list of ('V2', '|V|', 'PHI', 'T3PHI', 'T3AMP', 'DPHI', 'N|V|', 'FLUX')
    fig: figure number (default 1)
    figWidth: width of the figure (default 9)
    allInOne: plot all data files in a single plot, as function of baseline
        if not, show individual files in plots as function of wavelength (default)
        logB: baseline in log scale (default False)
        logV: V2 and |V| in log scale (default False)

    showIm: show image (False)
        imFov: field of view (in mas)
        imPix: imPixel size (in mas)
        imPow: show image**imPow (default 1.0)
        imWl0: wavelength(s) at which images are shown
        imMax: max value for image
        cmap: color map ('bone')
        imX: center of FoV (in mas, default:0.0)
        imY: center of FoV (in mas, default:0.0)
    """
    #print('debug', checkImVis)
    global ai1ax, ai1mcB, ai1mcT, ai1i, ai1fig, US_SPELLING

    if type(oi)==list:
        # -- multiple data sets -> recursive call
        models = []
        if allInOne:
            ai1mcB = {'i':0} # initialize global marker/color for baselines
            ai1mcT = {'i':0} # initialize global marker/color for triangles
            ai1ax = {} # initialise global list of axes
            ai1i = [] # initialise global list of position of axes

        if fig is None:
            fig = 0
        allWLc = [] # -- continuum -> absolute flux
        allWLs = [] # -- with spectral lines -> normalised flux
        allMJD = []
        if obs is None and allInOne:
            obs = []
            for o in oi:
                if 'fit' in o and 'obs' in o['fit']:
                    obs.extend(o['fit']['obs'])
            if len(obs):
                obs = list(set(obs))
            else:
                obs = None
        #print('all obs', obs)
        for i,o in enumerate(oi):
            if allInOne:
                f = fig
            else:
                f = fig+i

            if not imoi is None:
                if type(imoi)==list and len(imoi)==len(oi):
                    _imoi = imoi[i]
                else:
                    _imoi = imoi
            else:
                _imoi = None

            # -- m is the model based in parameters
            m = showOI(o, param=param, fig=f, obs=obs,
                   imFov=imFov, imPix=imPix, imX=imX, imY=imY, imMax=imMax,
                   imWl0=imWl0, imPow=imPow, checkImVis=checkImVis,
                   showIm=False, # taken care below
                   allInOne=allInOne, cmap=cmap, figWidth=figWidth,
                   wlMin=wlMin, wlMax=wlMax, spectro=spectro,
                   logB=logB, logV=logV, color=color, showFlagged=showFlagged,
                   onlyMJD=onlyMJD, showUV=showUV, figHeight=figHeight,
                   showChi2=showChi2 and not allInOne,
                   debug=debug, vWl0=vWl0, bckgGrid=bckgGrid,
                   cColors=cColors, cMarkers=cMarkers, imoi=_imoi,
                   barycentric=barycentric, autoLimV=autoLimV, t3B=t3B)
            if not param is None:
                if 'fit' in o and 'obs' in o['fit'] and 'NFLUX' in o['fit']['obs']:
                    if 'WL mask' in m:
                        allWLs.extend(list(m['WL'][m['WL mask']]))
                    else:
                        allWLs.extend(list(m['WL']))
                else:
                    if 'WL mask' in m:
                        allWLc.extend(list(m['WL'][m['WL mask']]))
                    else:
                        allWLc.extend(list(m['WL']))
            allMJD.extend(list(o['MJD']))
            models.append(m)

        allWLc = np.array(sorted(list(set(allWLc))))
        allWLs = np.array(sorted(list(set(allWLs))))
        allMJD = np.array(sorted(list(set(allMJD))))

        if showIm and not imFov is None:
            fluxes = {}
            spectra = {}
            im = 1
            if len(allWLc):
                allWL = {'WL':allWLc,
                         'fit':{'obs':[]},
                         'MJD':allMJD} # minimum required
                tmp = showModel(allWL, param, fig=f+im, imPow=imPow, cmap=cmap,
                              imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                              imWl0=imWl0, imMax=imMax, logS=logS,
                              figWidth=figWidth, cColors=cColors, cMarkers=cMarkers)
                im+=1
                fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                                tmp['MODEL'].keys() if k.endswith(',flux')}
                #print('debug: computing fluxes dict')
                m['fluxes'] = fluxes
                m['fluxes WL'] = allWLc
            if len(allWLs):
                allWL = {'WL':allWLs, # minimum required
                         'fit':{'obs':['NFLUX']}, # force computation of continuum
                         'MJD': allMJD }

                tmp = showModel(allWL, param, fig=f+im, imPow=imPow, cmap=cmap,
                              imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                              imWl0=imWl0, imMax=imMax, logS=logS,
                              figWidth=figWidth, cColors=cColors, cMarkers=cMarkers)
                fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                                tmp['MODEL'].keys() if k.endswith(',flux')}
                #print('debug: computing fluxes dict')
                m['spectra'] = fluxes
                m['specral WL'] = allWLs

        if allInOne:
            title = []
            for o in oi:
                title.extend([os.path.basename(f) for f in o['filename'].split(';')])
            title = sorted(title)
            #print('title:', title, len(title))
            if len(title)>2:
                title = title[0]+' ... '+title[-1]+' (%d files)'%len(title)
                fontsize = 8
            elif len(title)>1:
                title = ', '.join(title)
                fontsize = 8
            else:
                fontsize = 10
            if showIm:
                plt.figure(f)
            #print('DEBUG: cleaning')
            # ai1mcB = {'i':0} # initialize global marker/color for baselines
            # ai1mcT = {'i':0} # initialize global marker/color for triangles
            # ai1ax = {} # initialise global list of axes
            # ai1i = [] # initialise global list of axes
        return models

    # == actual plotting starts here
    if debug:
        print('starting plotting in showOI')

    #print('->', computeLambdaParams(param))

    if not vWl0 is None:
        um2kms = lambda um: (um-vWl0)/um*2.998e5
        kms2um = lambda kms: vWl0*(1 + kms/2.998e5)

    if spectro is None:
        spectro = len(oi['WL'])>10
    if not spectro:
        showChi2=False

    if not onlyMJD is None and \
        not any([mjd in onlyMJD for mjd in oi['configurations per MJD'].keys()]):
        # -- nothing to plot!
        return None

    if param is None:
        showChi2=False
    else:
        if type(param)==dict and 'best' in param.keys() and 'fitOnly' in param.keys():
            param = param['best']

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
    closest = []
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
        closest.append(np.argmin(np.abs(oi['WL']-0.5*(WR[0]+WR[1]))))
    oi['WL mask'] = np.bool_(w)
    if not any(oi['WL mask']):
        for clo in closest:
            oi['WL mask'][clo] = True

    if not 'obs' in oi['fit'] or obs is None:
        # -- come up with all possible observations:
        obs = []
        if 'OI_T3' in oi:
            obs.append('T3PHI')
        if 'OI_VIS2' in oi:
            obs.append('V2')
        if 'OI_VIS' in oi:
            obs.append('|V|')
        if 'OI_CF' in oi:
            obs.append('CF')
        if 'OI_FLUX' in oi:
            obs.append('FLUX')
    elif 'obs' in oi['fit'] and obs is None:
        # -- these are the obs from the fit, because not obs are given
        obs = oi['fit']['obs']

    if 'obs' in oi['fit']:
        obsfit = oi['fit']['obs']
    else:
        obsfit = obs
    if debug:
        print(' obs:', obs)
        print(' obsfit:', obsfit)

    # -- force recomputing differential quantities
    if 'WL cont' in oi.keys():
        oi.pop('WL cont')
    if 'DPHI' in obs or 'N|V|' in obs:
        if debug:
            print(' (re-)compute DiffPhi in data')
        oi = computeDiffPhiOI(oi, computeLambdaParams(param, MJD=np.mean(oi['MJD'])))
    if 'NFLUX' in obs:
        if debug:
            print(' (re-)compute NormFlux in data')
        oi = computeNormFluxOI(oi, computeLambdaParams(param, MJD=np.mean(oi['MJD'])))

    if wlMin is None:
        wlMin = min(oi['WL'][oi['WL mask']])
    if wlMax is None:
        wlMax = max(oi['WL'][oi['WL mask']])

    if not param is None:
        remObs = False
        if not 'fit' in oi:
            oi['fit'] = {'obs':obsfit}
            remObs = True
        elif not 'obs' in oi['fit']:
            oi['fit']['obs'] = obsfit
            remObs = True
        if debug:
            print(' computing Vmodel')
        #if 'fit' in oi and 'Nr' in oi['fit']:
        #    print('model Nr:', oi['fit']['Nr'])
        #print('showOI:', oi['fit'] )
        m = VmodelOI(oi, param, imFov=imFov, imPix=imPix, imX=imX, imY=imY,
            debug=debug)

        #for k in oi['OI_T3']:
            #print(k, np.sum(np.isnan(m['OI_T3'][k]['T3PHI'])))
            #print(k, m['OI_T3'][k]['FLAG'])

        if remObs:
            oi['fit'].pop('obs')
        # if not imFov is None and checkImVis:
        #     # == FIX THIS! does not work anymore bcause images are computed elsewhere
        #     if debug:
        #         print(' computing V from Image, imFov=', imFov)
        #     if not _m is None:
        #         m = VfromImageOI(m)
        # if checkImVis and not _m is None:
        #         # -- this is ugly :(
        #         for k in ['OI_VIS', 'OI_T3']:
        #             m[k.replace('OI_', 'IM_')] = _m[k]
        #         for k in m['IM_VIS'].keys():
        #             m['IM_VIS'][k]['V2'] = _m['OI_VIS2'][k]['V2']

        #if 'smearing' in m and any([m['smearing'][k]>1 for k in m['smearing']]):
        #    print('bandwidth smearing spectral channel(s):', m['smearing'])
        #if not 'smearing' in m:
        #    print('! no smearing? !')
        if debug:
            print('using parameters')
    else:
        if debug:
            print('no parameters given')
        m = None

    c = 1 # column
    ax0 = None
    data = {'FLUX':{'ext':'OI_FLUX', 'var':'FLUX', 'unit':'detector counts'},
            'NFLUX':{'ext':'NFLUX', 'var':'NFLUX', 'unit':'normalised'},
            'T3PHI':{'ext':'OI_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl', 'xunit':'1e6'},
            'T3AMP':{'ext':'OI_T3', 'var':'T3AMP', 'X':'Bmax/wl', 'xunit':'1e6'},
            #'T3PHI':{'ext':'OI_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bavg/wl'},
            #'T3AMP':{'ext':'OI_T3', 'var':'T3AMP', 'X':'Bavg/wl'},
            'DPHI':{'ext':'DVIS', 'var':'DPHI', 'unit':'deg', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            'PHI':{'ext':'OI_VIS', 'var':'PHI', 'unit':'deg', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            '|V|':{'ext':'OI_VIS', 'var':'|V|', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            'N|V|':{'ext':'DVIS', 'var':'N|V|', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            'V2':{'ext':'OI_VIS2', 'var':'V2', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            'CF':{'ext':'OI_CF', 'var':'CF', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
            }
    imdata = {'FLUX':{'ext':'IM_FLUX', 'var':'FLUX', 'unit':'detector counts'},
             'NFLUX':{'ext':'IM_FLUX', 'var':'NFLUX', 'unit':'normalised'},
             'T3PHI':{'ext':'IM_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl', 'xunit':'1e6'},
             'T3AMP':{'ext':'IM_T3', 'var':'T3AMP', 'X':'Bmax/wl', 'xunit':'1e6'},
             #'T3PHI':{'ext':'IM_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bavg/wl'},
             #'T3AMP':{'ext':'IM_T3', 'var':'T3AMP', 'X':'Bavg/wl'},
             'DPHI':{'ext':'IM_VIS', 'var':'DPHI', 'unit':'deg', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             'PHI':{'ext':'IM_VIS', 'var':'PHI', 'unit':'deg', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             '|V|':{'ext':'IM_VIS', 'var':'|V|', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             'N|V|':{'ext':'IM_VIS', 'var':'N|V|', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             'V2':{'ext':'IM_VIS', 'var':'V2', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             'CF':{'ext':'IM_VIS', 'var':'CF', 'X':'B/wl', 'C':'PA', 'xunit':'1e6'},
             }
    if US_SPELLING:
        data['NFLUX']['unit'] = 'normalized'
        imdata['NFLUX']['unit'] = 'normalized'

    if t3B in ['max', 'avg', 'min']:
        data['T3PHI']['X'] = 'B%s/wl'%t3B
        imdata['T3PHI']['X'] = 'B%s/wl'%t3B
        data['T3AMP']['X'] = 'B%s/wl'%t3B
        imdata['T3AMP']['X'] = 'B%s/wl'%t3B

    # -- plot in a certain order
    obs = list(filter(lambda x: x in obs,
            ['FLUX', 'NFLUX', 'T3PHI', 'PHI', 'DPHI', 'T3AMP', '|V|', 'N|V|', 'V2', 'CF']))
    ncol = len(obs)

    if showUV:
        obs = obs[::-1]
        obs.append('UV')
        obs = obs[::-1]
        if not any(['FLUX' in o for o in obs]):
            ncol += 1
    if debug:
        print('showing:', obs)
    mcB = {'i':0} # marker/color for baselines (showUV=True)
    mcT = {'i':0} # marker/color for triangles (showUV=True)

    if allInOne:
        # -- use globals to have consistent markers/colors
        mcB.update(ai1mcB)
        mcT.update(ai1mcT)

    markers = ['d', 'o', '*', '^', 'v', '>', '<', 'P', 'X']
    if not spectro:
        colors = list(itertools.permutations([0.1, 0.6, 0.9])) + ['0.5']
        colors += [(.1,.1,.9), (.1,.9,.1), (.9,.1,.1)]
    else:
        #colors = matplotlib.cm.nipy_spectral(np.linspace(0, .9, len(oi['baselines'])))
        colors = matplotlib.colormaps['nipy_spectral'](np.linspace(0, .9, len(oi['baselines'])))

    if figWidth is None and figHeight is None:
        figHeight =  min(max(ncol, 10), FIG_MAX_HEIGHT)
        figWidth = min(figHeight*ncol, FIG_MAX_WIDTH)
    if figWidth is None and not figHeight is None:
        figWidth = min(figHeight*ncol, FIG_MAX_WIDTH)
    if not figWidth is None and figHeight is None:
        figHeight =  max(figWidth/ncol, FIG_MAX_HEIGHT)
    if not allInOne or ai1ax == {}:
        plt.close(fig)
        ai1fig = plt.figure(fig, figsize=(figWidth, figHeight))

    i_flux = 0
    i_col = 0
    yoffset = 0
    #print('#', os.path.basename(oi['filename']), obsfit, obs)
    for c,l in enumerate(obs):
        if debug:
            print('handling:', l)
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
                if len(obsfit)==0:
                    # -- plotting only UV
                    ax = plt.subplot(111, aspect='equal')
                else:
                    ax = plt.subplot(n_flux+1, ncol, 1, aspect='equal')
            # -- for each observables per baselines
            ext = [e for e in ['OI_VIS', 'OI_VIS2'] if e in oi]
            for e in ext:
                #if debug:
                #    print(e, sorted(oi[e].keys()))
                #for k in sorted(oi[e].keys()): # alphabetical order
                for k in sorted(oi[e].keys(), key=lambda x: np.mean(oi[e][x]['B/wl'])): # lengh
                    #if debug:
                    #    print(len(oi[e][k]['MJD']))
                    if not k in mcB.keys():
                        mcB[k] = markers[i%len(markers)], colors[i%len(colors)]
                        i+=1
                        #if allInOne:
                        ai1mcB[k] = mcB[k]
                        mcB['i'] = i
                        ai1mcB['i'] = i
                        showLabel = True
                    else:
                        showLabel = False

                    mark, col = mcB[k]
                    # -- for each MJD:
                    allMJDs = oi[e][k]['MJD']
                    if onlyMJD is None:
                        MJDs = allMJDs
                    else:
                        MJDs = [m for m in allMJDs if m in onlyMJD]
                    for mjd in MJDs:
                        w = allMJDs==mjd
                        # -- check if any valid observables?
                        # -- pretty tricky!!!

                        b = np.sqrt(oi[e][k]['u'][w]**2+
                                    oi[e][k]['v'][w]**2)
                        try:
                            tmp = len(b)
                            bmax.extend(list(b))
                        except:
                            bmax.append(b)

                        ax.plot(oi[e][k]['u'][w], oi[e][k]['v'][w],
                                color=col, marker=mark,
                                label=k if showLabel else '',
                                linestyle='none', markersize=5)
                        ax.plot(-oi[e][k]['u'][w], -oi[e][k]['v'][w],
                                color=col, marker=mark, alpha=0.1,
                                linestyle='none', markersize=5)
                        label = ''
                        showLabel=False
            # -- test T3 u,v recomputation
            if False and 'OI_T3' in oi:
                for k in oi['OI_T3']:
                    plt.plot(oi['OI_T3'][k]['u1'], oi['OI_T3'][k]['v1'],
                            '1m')
                    plt.plot(-oi['OI_T3'][k]['u1'], -oi['OI_T3'][k]['v1'],
                            '1m')
                    plt.plot(oi['OI_T3'][k]['u2'], oi['OI_T3'][k]['v2'],
                            '2r')
                    plt.plot(-oi['OI_T3'][k]['u2'], -oi['OI_T3'][k]['v2'],
                            '2r')
                    plt.plot(oi['OI_T3'][k]['u1']+oi['OI_T3'][k]['u2'],
                             oi['OI_T3'][k]['v1']+oi['OI_T3'][k]['v2'],
                            '3g')
                    plt.plot(-oi['OI_T3'][k]['u1']-oi['OI_T3'][k]['u2'],
                             -oi['OI_T3'][k]['v1']-oi['OI_T3'][k]['v2'],
                            '3g')

            bmax = np.array(bmax)
            Bc = []
            for b in [10, 20, 50, 100, 150, 200, 250, 300]:
                try:
                    if any(b>bmax) and any(b<bmax):
                        Bc.append(b)
                except:
                    print('bmax:', bmax)

            #if allInOne and not 'UV' in ai1ax:
            if not 'UV' in ai1ax:
                ai1ax['UV'] = ax
            t = np.linspace(0, 2*np.pi, 100)
            for b in Bc:
                 ax.plot(b*np.cos(t), b*np.sin(t), ':k', alpha=0.2)

            bmax = 1.1*np.max(bmax)
            ax.legend(fontsize=5, loc='upper left', ncol=3)
            #ax.set_title('u,v (m)', fontsize=10)
            #ax.set_xlabel('u')
            #ax.set_ylabel('v')
            ax.set_title(r'u [$\leftarrow$E], v [$\uparrow$N] (m)', fontsize=10)

            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            if np.abs(ax.get_xlim())[1]<bmax or ax.get_ylim()[1]<bmax:
                ax.set_xlim(-bmax, bmax)
                ax.set_ylim(-bmax, bmax)
            if ax.get_xlim()[0]<ax.get_xlim()[1]:
                ax.invert_xaxis()

            i_flux += 1
            if not any(['FLUX' in o for o in obs]):
                i_col+=1
                if debug:
                    print('no FLUX')
            else:
                if debug:
                    print('FLUX')
                pass
            continue

        if not data[l]['ext'] in oi.keys():
            i_col += 1
            if debug:
                print('adding one colum')
            continue

        N = len(oi[data[l]['ext']].keys())
        # -- for each telescope / baseline / triplets
        keys = [k for k in oi[data[l]['ext']].keys()]
        if 'X' in data[l] and spectro and \
                all([data[l]['X'] in oi[data[l]['ext']][x] for x in keys]):
            keys = sorted(keys,
                key=lambda x: np.mean(oi[data[l]['ext']][x][data[l]['X']]))
        else:
            keys = sorted(keys)

        # -- average normalised flux -> is this thing still needed?!
        if False and l=='NFLUX' and 'UV' in obs and l in obsfit:
            if debug:
                print('>>> NFLUX?!')
            tmp = np.zeros(len(oi['WL']))
            etmp = 1.e6*np.ones(len(oi['WL']))
            weight = np.zeros(len(oi['WL']))

            for k in keys: # for each telescopes:
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
                    # TODO: better estimation of error of average flux
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
        else:
            pass

        showLegend = False

        # -- required for allInOne to have all required subplots,
        # -- but not necessarly plot!
        if not l in obsfit:
            #print('  not ploting:', l)
            continue
        #print('  ploting', l, i_col, ncol)
        # -- actually plot things:
        for i,k in enumerate(keys):
            if debug:
                print(k, end=',')
            # -- for each telescope / baseline / triangle
            X = lambda r, j: r['WL']
            Xlabel = r'wavelength ($\mu$m)'
            if barycentric and 'barycorr_km/s' in oi:
                X = lambda r, j: r['WL']*(1+oi['barycorr_km/s']/2.998e5)
                Xlabel = r'barycentric wavelength ($\mu$m)'

            Xscale = 'linear'
            if not spectro:
                if 'X' in data[l]:
                    X = lambda _r, _j: _r[data[l]['ext']][k][data[l]['X']][_j,:]
                    Xlabel = data[l]['X']
                    if '/wl' in Xlabel:
                        Xlabel = Xlabel.replace('/wl', r'/$\lambda$')
                    if 'xunit' in data[l]:
                        Xlabel += ' (%s)'%data[l]['xunit']
                    if logB:
                        Xscale = 'log'
                yoffset = 0.0
            if spectro:
                kax = l+k
            else:
                kax = l
            if allInOne and kax in ai1ax.keys():
                ax = ai1ax[kax]
                if 'r'+kax in ai1ax.keys():
                    axr = ai1ax['r'+kax]
            else:
                if 'UV' in obs and 'FLUX' in l:
                    if i==0:
                        if True or len(oi['baselines'])<=8:
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
                        if bckgGrid:
                            ax.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
                else:
                    if not spectro:
                        # -- as function of baseline/wl
                        if i==0:
                            while i_col in ai1i:
                                i_col+=1
                            if param is None:
                                # -- no model
                                ax = plt.subplot(1, ncol, i_col+1)
                            else:
                                # -- TODO: use GridSpec?
                                ax = plt.subplot(2, ncol, i_col+1)
                                axr = plt.subplot(2, ncol, ncol+i_col+1, sharex=ax)
                                axr.set_title(r'residuals ($\sigma$)',
                                              color='0.5', fontsize=8, x=.5, y=.9)
                                ax.tick_params(axis='y', labelsize=8)
                                axr.tick_params(axis='y', labelsize=8)
                    else:
                        if ax0 is None:
                            ax0 = plt.subplot(N, ncol, ncol*i+i_col+1)
                            ax = ax0
                        else:
                            ax = plt.subplot(N, ncol, ncol*i+i_col+1, sharex=ax0)
                if allInOne:
                    # -- keep track of windows and positions
                    ai1ax[kax] = ax
                    ai1i.append(i_col)
                    if not spectro and not param is None and not 'FLUX' in l:
                        ai1ax['r'+kax] = axr

            if not ('UV' in obs and 'FLUX' in l):
                yoffset = 0.0

            if not 'UV' in l and not vWl0 is None and i==0:
                axv = ax.secondary_xaxis('top', functions=(um2kms, kms2um),
                                         color='0.6')
                axv.tick_params(axis='x', labelsize=5,
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

            # -- for each epoch
            for mjd in MJDs:
                j = allMJDs.index(mjd)
                mask = ~oi[data[l]['ext']][k]['FLAG'][j,:]*oi['WL mask']
                flagged = oi[data[l]['ext']][k]['FLAG'][j,:]*oi['WL mask']
                # -- data:
                y = oi[data[l]['ext']][k][data[l]['var']][j,:]
                if 'PHI' in l and any(mask):# and np.ptp(y[mask])>300:
                    y[mask] = np.unwrap(y[mask]*np.pi/180)*180/np.pi
                    y[mask] = np.mod(y[mask]-np.mean(y[mask])+180, 360)+\
                                np.mean(y[mask]-180)%360-360
                    #y[flagged] = np.unwrap(y[flagged]*np.pi/180)*180/np.pi
                    #y[flagged] = np.mod(y[flagged]+180, 360)-180
                    pass

                err = oi[data[l]['ext']][k]['E'+data[l]['var']][j,:].copy()
                ign = (~mask).copy()
                showIgn = False
                if 'baseline ranges' in oi['fit']:
                    # for bmin, bmax in oi['fit']['baseline ranges']:
                    #     if 'FLUX' in data[l]['ext']:
                    #         pass
                    #     elif 'T3' in data[l]['ext']:
                    #         for b in ['B1', 'B2', 'B3']:
                    #             mask *= ((oi[data[l]['ext']][k][b][j]<=bmax)*
                    #                         (oi[data[l]['ext']][k][b][j]>=bmin))
                    #     else:
                    #         mask *= (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']<=bmax)*\
                    #                 (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']>=bmin)

                    bmask = np.zeros(mask.shape, dtype=bool)
                    for bmin, bmax in oi['fit']['baseline ranges']:
                        if 'FLUX' in data[l]['ext']:
                            bmask = np.ones(mask.shape, dtype=bool)
                        elif 'T3' in data[l]['ext']:
                            tmpmask = np.ones(mask.shape, dtype=bool)
                            for b in ['B1', 'B2', 'B3']:
                                try:
                                    tmpmask = np.logical_and(tmpmask,
                                        (oi[data[l]['ext']][k][b][j]<=bmax)*
                                        (oi[data[l]['ext']][k][b][j]>=bmin))
                                except:
                                    tmpmask = np.logical_and(tmp,
                                           ((oi[data[l]['ext']][k][b][j]<=bmax)*
                                            (oi[data[l]['ext']][k][b][j]>=bmin))[:,None])
                            bmask = np.logical_or(bmask, tmpmask)
                        else:
                            bmask = np.logical_or(bmask,
                                (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']<=bmax)*
                                (oi[data[l]['ext']][k]['B/wl'][j,:]*oi['WL']>=bmin))
                    mask *= bmask

                # if 'MJD ranges' in oi['fit']:
                #     for mjdmin, mjdmax in oi['fit']['MJD ranges']:
                #         mask *= (oi[data[l]['ext']][k]['MJD2'][j,:]<=mjdmax)*\
                #                 (oi[data[l]['ext']][k]['MJD2'][j,:]>=mjdmin)
                if 'MJD ranges' in oi['fit']:
                    mjdmask = np.zeros(mask.shape, dtype=bool)
                    for mjdmin, mjdmax in oi['fit']['MJD ranges']:
                        mjdmask = np.logical_or(mjdmask,
                            (oi[data[l]['ext']][k]['MJD2'][j,:]<=mjdmax)*\
                            (oi[data[l]['ext']][k]['MJD2'][j,:]>=mjdmin))
                    mask *= mjdmask

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
                        ai1mcT.update(mcT)
                        showLabel = True and not spectro
                    else:
                        showLabel = False
                if  l in ['V2', '|V|', 'DPHI', 'N|V|', 'CF']:
                    if not k in mcB:
                        mcB[k]  = (markers[mcB['i']%len(markers)],
                                    colors[mcB['i']%len(colors)] )
                        mcB['i'] += 1
                        ai1mcB.update(mcB)
                        showLabel = True and not spectro
                    else:
                        showLabel = False

                if not spectro:
                    if k in mcB: # known baseline
                        mark, col = mcB[k]
                    elif k in mcT: # known triangle
                        mark, col = mcT[k]
                    else:
                        mark, col = '.', 'k'

                    # -- data dots
                    ax.plot(X(oi, j)[mask], y[mask], mark,
                            color=col if not test else '0.5', markersize=5,
                            alpha=0.5, label=k if showLabel else '',)
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
                                alpha=0.5, #label='flag' if j==0 else ''
                                )
                        ax.errorbar(X(oi, j)[flagged], y[flagged],
                                    yerr=np.abs(err[flagged]),
                                    color='m', alpha=0.2, linestyle='None')
                else:
                    # -- data spectra
                    ax.step(X(oi, j)[mask], y[mask]+yoffset*i,
                            '-', color='k' if not test else '0.5',
                            alpha=0.5, label=k if showLabel else '', where='mid')
                    # -- data errors
                    ax.fill_between(X(oi, j)[mask],
                                    y[mask]+err[mask]+yoffset*i,
                                    y[mask]-err[mask]+yoffset*i,
                                    step='mid', color='k',
                                    alpha=0.1 if not test else 0.05)
                    if showFlagged:
                        ax.plot(X(oi, j)[flagged], y[flagged]+yoffset*i,
                                'mx' if not test else '0.5',
                                alpha=0.5, #label='flag' if j==0 else ''
                                )
                        ax.errorbar(X(oi, j)[flagged], y[flagged]+yoffset*i,
                                    yerr=err[flagged], color='m', alpha=0.2,
                                    linestyle='None')

                showLegend = showLegend or showLabel

                if showIgn and showFlagged:
                    # -- show ignored data (filtered on error for example)
                    ax.plot(X(oi,j)[ign], y[ign]+yoffset*i, 'xy', alpha=0.5)

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
                    ym[mask] = np.mod(ym[mask]+180-np.mean(ym[mask]), 360)+\
                                (np.mean(ym[mask])-180)%360-360

                    # -- computed chi2 *in the displayed window*
                    maskc2 = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)

                    #print('dbg>', oi['WL'].min(), oi['WL'].max())
                    #print('   >', m['WL'].min(), m['WL'].max())

                    err[err<=0] = 1
                    if 'PHI' in l:
                        _resi = ((y[maskc2]-ym[maskc2]+180)%360-180)/err[maskc2]
                    else:
                        _resi = (y[maskc2]-ym[maskc2])/err[maskc2]

                    # -- build residuals array
                    resi = np.append(resi, _resi.flatten())
                    # -- show predictions from model
                    if not spectro:
                        if np.sum(maskc2)>1:
                            ax.plot(X(m,j)[maskc2], ym[maskc2],
                                    '-', alpha=0.5 if not test else 0.3,
                                    color=color if col=='k' else '0.5',
                                    linewidth=2)
                            if col!='k':
                                ax.plot(X(m,j)[maskc2], ym[maskc2],
                                        '--', alpha=0.7, color=col,
                                        linewidth=2)
                        else:
                            # -- not enough points to show lines
                            ax.plot(X(m,j)[maskc2], ym[maskc2],
                                    '_', color=col, alpha=0.5, ms=5)

                        # -- residuals
                        if not 'FLUX' in l:
                            if 'PHI' in l:
                                axr.plot(X(m,j)[maskc2],
                                        ((y[maskc2]-ym[maskc2]+180)%360-180)/err[maskc2],
                                        mark, color=col, markersize=4, alpha=0.4)
                            else:
                                axr.plot(X(m,j)[maskc2],
                                        (y[maskc2]-ym[maskc2])/err[maskc2],
                                        mark, color=col, markersize=4, alpha=0.4)
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
                        ax.plot(X(imoi, j)[mask],
                             imoi[data[l]['ext']][k][data[l]['var']][j,mask]+yoffset*i,
                            '1g', alpha=0.5)
                    else:
                        ax.step(X(imoi, j)[mask],
                             imoi[data[l]['ext']][k][data[l]['var']][j,mask]+yoffset*i,
                            '--g', alpha=0.5, linewidth=2, where='mid')

                # -- show continuum for differetial PHI and normalised FLUX
                if (l=='DPHI' or l=='NFLUX' or l=='N|V|') and 'WL cont' in oi:
                    maskc = ~oi[data[l]['ext']][k]['FLAG'][j,:]*\
                                    oi['WL cont']*oi['WL mask']
                    _m = np.mean(oi[data[l]['ext']][k][data[l]['var']][j,maskc])
                    cont = np.ones(len(oi['WL']))*_m
                    cont[~maskc] = np.nan
                    ax.plot(X(oi, j), cont+yoffset*i, ':', color='c', linewidth=3)

                # -- show phase based on OPL
                if l=='PHI' and 'OPL' in oi.keys():
                    dOPL = oi['OPL'][k[2:]] - oi['OPL'][k[:2]]
                    wl0 = oi['WL'].mean()
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
                    print('ERROR chi2! "resi":', resi)

                if len(resi)<24:
                    rms = np.std(resi)
                    f = 'rms'
                else:
                    rms = 0.5*(np.percentile(resi, 84)-np.percentile(resi, 16))
                    f = '%%rms'
                fmt = r'$\chi^2$=%.2f '
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
                    if yamp>0:
                        ax.set_ylim(ymin - 0.2*yamp-i*yoffset, ymax + 0.2*yamp)
                if 'UV' in obs and 'FLUX' in l and i==0:
                    #yoffset = yamp # -- offset spectra of each telescope
                    pass
                if wlMin<wlMax:
                    ax.set_xlim(wlMin, wlMax)
                if k in mcB:
                    mark, col = mcB[k]
                else:
                    mark, col = 'o', 'k'
                if not 'UV' in obs or not 'FLUX' in l:
                    # -- put name of configuration
                    ax.text(0.02, 0.98, k, transform=ax.transAxes,
                            ha='left', va='top', fontsize=6, color=col)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                if bckgGrid:
                    ax.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
            else:
                if l in ['V2', '|V|']:
                    if logV:
                        ax.set_ylim(1e-4,1)
                        ax.set_yscale('log')
                    elif not autoLimV:
                        ax.set_ylim(0,1)
                    else:
                        #ax.set_ylim(0)
                        pass

            #if l=='NFLUX' and 'TELLURICS' in oi:
            #    plt.plot(oi['WL'], oi['TELLURICS'])

            if i==N-1:
                if not spectro and not param is None and not 'FLUX' in l:
                    axr.set_xlabel(Xlabel)
                    #axr.hlines(0, axr.get_xlim()[0], axr.get_xlim()[1],
                    #            color='0.5', linewidth=1)
                else:
                    ax.set_xlabel(Xlabel)
                if Xscale=='log':
                    ax.set_xscale('log')
                    xlims = ax.get_xlim()
                    XTICKS = []
                    for _p in [-1,0,1,2]:
                        for xtick in [1, 2, 5]:
                            if xtick*10**_p>=xlims[0] and xtick*10**_p<=xlims[1]:
                                XTICKS.append(xtick*10**_p)
                    if len(XTICKS):
                        ax.set_xticks(XTICKS)
                        ax.set_xticklabels([str(x) for x in XTICKS])
                        #ax.minorticks_off()

                    if not spectro and not param is None:
                        axr.set_xscale('log')
                        if len(XTICKS):
                            axr.set_xticks(XTICKS)
                            axr.set_xticklabels([str(x) for x in XTICKS])
                            #axr.minorticks_off()
            if i==0:
                title = l
                if 'units' in oi and l in oi['units']:
                        title += ' (%s)'%oi['units'][l]
                elif 'unit' in data[l]:
                    title += ' (%s)'%data[l]['unit']

                ax.set_title(title)
                if not vWl0 is None:
                    axv.set_xlabel('velocity (km/s)', fontsize=6)
                if l=='NFLUX':
                    ax.set_xlabel(Xlabel)
            if (allInOne or l=='T3PHI') and showLegend:
                ax.legend(fontsize=5, ncol=4)
        i_col += 1
        if debug:
            print()

    plt.subplots_adjust(hspace=0, wspace=0.2, left=0.06, right=0.98)
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
        showModel(oi, param, m=m, fig=fig+1, imPow=imPow, cmap=cmap,
                 figWidth=None,#figWidth,
                 imFov=imFov, imPix=imPix, imX=imX, imY=imY,
                 imWl0=imWl0, imMax=imMax, cColors=cColors, cMarkers=cMarkers,
                 bckgGrid=bckgGrid)
    return m

def showModel(oi, param, m=None, fig=0, figHeight=4, figWidth=None, WL=None,
              imFov=None, imPix=None, imPow=1.0, imX=0., imY=0., imWl0=None,
              cmap='bone', imMax=None, logS=False, showSED=True, legend=True,
              cColors={}, cMarkers={}, imPlx=None, bckgGrid=True):
    """
    oi: result from loadOI or mergeOI,
        or a wavelength vector in um (must be a np.ndarray)
    param: parameter dictionnary, describing the model
    m: result from Vmodel, if none given, computed from 'oi' and 'param'
    fig: which figure to plot on (default 1)
    figHeight: height of the figure, in inch

    imFov: field of view in mas (AU if parallax is given)
    imPix: imPixel size in mas (AU if parallax is given)
    imMax: cutoff for image display (0..1) or in percentile ('0'..'100')
    imPow: power law applied to image for display. use 0<imPow<1 to show low
        surface brightness features
    imX, imY: center of image (in mas, AU if parallax is given)
    imWl0: list of wavelength (um) to show the image default (min, max)
    cmap: color map (default 'bone')
    """
    if imPlx is None:
        imScale = 1.0
    else:
        imScale = 1./imPlx

    param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))
    # -- catch case were OI is just a wavelength vector:
    if type(oi)==np.ndarray:
        oi = {'WL':oi, # minimum required
              'fit':{'obs':['NFLUX']} # force computation of continuum
              }

    if m is None:
        m = VmodelOI(oi, param, imFov=imFov/imScale, imPix=imPix,
                    imX=imX/imScale, imY=imY/imScale,
                    debug=False)
        #print(m['MODEL'].keys())
        #if not 'WL mask' in oi and 'WL mask' in m:
        #    oi['WL mask'] = m['WL mask'].copy()
        if not 'WL cont' in oi and 'WL cont' in m:
            oi['WL cont'] = m['WL cont'].copy()
        #print('continuum:', oi['WL cont'])

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

    if 'cube' in m['MODEL'] and showSED:
        nplot = len(imWl0)+1
    else:
        nplot = len(imWl0)

    #print('showModel: nplot=', nplot)
    #print('showModel: fighWidth, figHeight=', figWidth, figHeight)

    if figWidth is None and figHeight is None:
        figHeight =  min(max(nplot, 8), FIG_MAX_HEIGHT)
        figWidth = min(figHeight*nplot, FIG_MAX_WIDTH)
    if figWidth is None and not figHeight is None:
        figWidth = min(figHeight*nplot, FIG_MAX_WIDTH)
    if not figWidth is None and figHeight is None:
        figHeight =  max(figWidth/nplot, FIG_MAX_HEIGHT)

    #print('showModel: fighWidth, figHeight=', figWidth, figHeight)

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
    colors = [(c,a,b), (a,b,c), (b,c,a),
              (a,c,b), (c,b,a), (b,a,c),
              (a,c,c), (c,a,c), (c,c,a),
              (b,b,c), (b,c,b), (c,b,b),
              (b,b,b)]
    markers = ['1', '2', '3', '4'] # 3 branches crosses
    _ic = 0
    _im =  0
    for c in comps:
        if wlpeak[c] is None:
            symbols[c] = {'m':markers[_im%len(markers)],
                          'c':colors[_ic%len(colors)]}
            _ic+=1
            _im+=1
        else:
            if len(allpeaks)==1:
                # -- only one component with line
                symbols[c] = {'m':'+', 'c':'orange'}
            else:
                symbols[c] = {'m':markers[_im%len(markers)],
                              #'c':matplotlib.cm.nipy_spectral(0.1+0.8*(wlpeak[c]-min(allpeaks))/np.ptp(allpeaks))
                              'c':matplotlib.colormaps['nipy_spectral'](0.1+0.8*(wlpeak[c]-min(allpeaks))/np.ptp(allpeaks))
                             }
                _im+=1
        if c in cColors:
            symbols[c]['c'] = cColors[c]
        if c in cMarkers:
            symbols[c]['m'] = cMarkers[c]

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
            axs.append(plt.subplot(1, nplot, i+1, aspect='equal',
                       sharex=axs[0], sharey=axs[0]))
        _j = np.argmin(np.abs(oi['WL'][mask]-wl))
        _wl = oi['WL'][mask][_j]
        j = np.arange(len(oi['WL']))[mask][_j]

        if not imPow == 1:
            title = 'Image$^{%.2f}$ '%imPow
        else:
            title ='Image '
        n = 2-np.log10(np.median(np.abs(np.diff(oi['WL'][mask]))))

        title += r'$\lambda$=%.'+str(int(n))+r'f$\mu$m'
        title = title%_wl
        plt.title(title, fontsize=9)

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

        if type(imMax)==str:
            _imMax = np.percentile(im**(1/imPow), float(imMax))
        else:
            _imMax = imMax
        #print('_imMax', _imMax)

        vmin, vmax = 0, _imMax**imPow

        #print('debug: im min,max =', im.min(), ',', im.max())
        #print('    vmin, vmax =', vmin, ',', vmax)
        #print('    Xmin , Xmax =', m['MODEL']['X'].min(), ',', m['MODEL']['X'].max())
        #print('    Ymin , Ymax =', m['MODEL']['Y'].min(), ',', m['MODEL']['Y'].max())

        pc = plt.pcolormesh(m['MODEL']['X']*imScale, m['MODEL']['Y']*imScale,
                            im, vmin=vmin, vmax=vmax,
                            cmap=cmap, shading='auto', rasterized=True)
        cb = plt.colorbar(pc, ax=axs[-1],
                          orientation='horizontal' if len(imWl0)>1 else
                          'vertical')

        Xcb = np.linspace(0,1,5)*_imMax**imPow
        XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
        XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
        cb.set_ticks(Xcb)
        cb.set_ticklabels(XcbL)
        cb.ax.tick_params(labelsize=6)
        plt.xlabel(r'$\delta$RA $\leftarrow$ E (mas)')
        if i==0:
           plt.ylabel(r'$\delta$Dec N $\rightarrow$ (mas)')

        #plt.xlabel(r'$\leftarrow$ E (mas)')
        #if i==0:
        #    plt.ylabel(r'N $\rightarrow$ (mas)')

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

            if legend:
                plt.plot(x*imScale, y*imScale, symbols[c]['m'],
                        color=symbols[c]['c'], label=c,
                        markersize=8)
            #plt.plot(x, y, '.w', markersize=8, alpha=0.5)

            if i==0 and legend:
               plt.legend(fontsize=5, ncol=2)
        axs[-1].tick_params(axis='x', labelsize=6)
        axs[-1].tick_params(axis='y', labelsize=6)

    axs[-1].invert_xaxis()
    if not 'cube' in m['MODEL'] or not showSED:
        plt.tight_layout()
        return m

    # -- show spectra / SED
    ax = plt.subplot(1, nplot, nplot)
    fS = lambda x: x
    # if logS:
    #     fS = lambda x: np.log10(x)

    if 'totalnflux' in m['MODEL']:
        key = 'nflux'
        if US_SPELLING:
            plt.title('spectra, normalized\nto total continuum', fontsize=8)
        else:
            plt.title('spectra, normalised\nto total continuum', fontsize=8)
    else:
        key = 'flux'
        if logS:
            plt.title('log10(SED)', fontsize=8)
        else:
            plt.title('SED', fontsize=8)

    w0 = m['MODEL']['total'+key][mask]>0
    if len(m['WL'][mask])>20 and \
            np.std(np.diff(m['WL'][mask]))/np.mean(np.diff(m['WL'][mask]))<0.1:
        plt.step(m['WL'][mask][w0], fS(m['MODEL']['total'+key][mask][w0]),
                '-k', label='total', where='mid')
    else:
        plt.plot(m['WL'][mask][w0], fS(m['MODEL']['total'+key][mask][w0]),
                '.-k', label='total')

    if 'WL cont' in oi:
        cont = np.ones(oi['WL'].shape)
        cont[~oi['WL cont']] = np.nan
        if len(m['WL'][mask])>20 and\
            np.std(np.diff(m['WL'][mask]))/np.mean(np.diff(m['WL'][mask]))<0.1:
            plt.step(m['WL'][mask][w0], fS((m['MODEL']['total'+key]*cont)[mask][w0]),
                    'c', label='continuum', where='mid', alpha=0.7,
                    linewidth=3, linestyle='dotted')
        else:
            plt.plot(m['WL'][mask][w0], fS((m['MODEL']['total'+key]*cont)[mask][w0]), '.-',
                    label='continuum', alpha=0.7,
                    linewidth=3, linestyle='dotted')

    # -- show spectra of each components
    KZ = filter(lambda x: not '&dwl' in x, m['MODEL'].keys())
    for k in sorted(KZ):
        if k.endswith(','+key):
            if logS:
                w0 = m['MODEL'][k][mask]>0
            else:
                w0 = m['MODEL'][k][mask]!=0

            if len(m['WL'][mask])>20 and\
                np.std(np.diff(m['WL'][mask]))/np.mean(np.diff(m['WL'][mask]))<0.1:
                plt.step(m['WL'][mask][w0], fS(m['MODEL'][k][mask][w0]),
                         label=k.split(',')[0].strip(), where='mid',
                         color=symbols[k.split(',')[0].strip()]['c'])
            else:
                plt.plot(m['WL'][mask][w0], fS(m['MODEL'][k][mask][w0]), '.-',
                         label=k.split(',')[0].strip(),
                         color=symbols[k.split(',')[0].strip()]['c'])
    if bckgGrid:
        plt.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
    ax.legend(fontsize=5)
    plt.xlabel(r'wavelength ($\mu$m)')
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    if not logS:
        plt.ylim(0)
    else:
        plt.yscale('log')
    plt.tight_layout()
    #print('fS(1)=', fS(1))
    return m

def _callbackAxes(ax):
    i = None
    for k in _AY.keys():
        if ax==_AY[k]:
            i = k
    #print('callback:', i)
    if not i is None:
        _AX[i].set_xlim(ax.get_ylim())
    else:
        pass

    i = None
    for k in _AX.keys():
        if ax==_AX[k]:
            i = k
    if not i is None:
        _AY[i].set_ylim(ax.get_xlim())
    else:
        pass
    return

def halfLightRadiusFromParam(param, comp=None, fig=None, verbose=True):
    if type(param) is dict and 'best' in param and 'uncer' in param:
        # -- ramdomised params
        allP = dpfit.randomParam(param, N=100, x=None)['r_param']
        if comp is None:
            C = list(filter(lambda x: x.endswith(',profile'), param['best'].keys()))
            C = sorted([c.split(',')[0] for c in C])
        else:
            C = [comp]
        allr = {c:[] for c in C}
        for p in allP:
            tmp = halfLightRadiusFromParam(p)
            for c in C:
                allr[c].append(tmp[c])

        res = {'best':{c:np.mean(allr[c]) for c in C},
               'uncer':{c:np.std(allr[c]) for c in C},
               }
        res['covd'] = {}
        res['cord'] = {}
        res['cov'] = np.zeros((len(C), len(C)))
        res['cor'] = np.zeros((len(C), len(C)))
        res['fitOnly'] = C

        for i1,c1 in enumerate(C):
            res['covd'][c1] = {}
            res['cord'][c1] = {}
            for i2,c2 in enumerate(C):
                tmp = np.cov(allr[c1], allr[c2])
                res['covd'][c1][c1] = tmp[0,0]
                res['covd'][c1][c2] = tmp[0,1]
                res['cord'][c1][c1] = 1
                res['cord'][c1][c2] = tmp[0,1]/np.sqrt(tmp[0,0]*tmp[1,1])
                res['cov'][i1,i1] = res['covd'][c1][c1]
                res['cov'][i1,i2] = res['covd'][c1][c2]
                res['cor'][i1,i1] = res['cord'][c1][c1]
                res['cor'][i1,i2] = res['cord'][c1][c2]
        if verbose:
            ns = max([len(c) for c in C])
            print('# == Half-Light Radii from best fit:')
            for c in C:
                try:
                    n = int(2-np.log10(res['uncer'][c]))
                    f = '#  %'+str(ns)+'s: %.'+str(n)+'f +- '+'%.'+str(n)+'f (mas)'
                    print(f%(c,res['best'][c], res['uncer'][c]))
                except:
                    print('#', c, res['best'][c])
            try:
                dpfit.dispCor(res, pre='#  ')
            except:
                print('error showing correlation matrix!')

        #res = {k:res[k] for k in ['best', 'uncer', 'covd', 'cord']}
        return res
    else:
        param = computeLambdaParams(param, MJD=np.mean(oi['MJD']))

    if comp is None:
        C = filter(lambda x: x.endswith(',profile'), param.keys())
        return {c.split(',')[0]:halfLightRadiusFromParam(param, c.split(',')[0]) for c in C}
    diamin = 0
    if comp+',diam' in param and comp+',thick' in param:
        diamin  = param[comp+',diam']*(1-param[comp+',thick']/2)
        diamout = param[comp+',diam']*(1+param[comp+',thick']/2)

    if comp+',diamout' in param:
        diamout = param[comp+',diamout']
    if comp+',diamin' in param:
        diamin = param[comp+',diamin']

    _r = np.linspace(diamin/2, diamout/2, 100)
    _d = np.linspace(diamin, diamout, 100)

    if not comp+',profile' in param:
        _p = np.ones(len(_r))
    elif param[comp+',profile'] == 'doughnut':
        _p = 1-((_r-_r.mean())/_r.ptp()*2)**2
    elif param[comp+',profile'].startswith('doughnut'):
        tmp = float(param[comp+',profile'].split('doughnut')[1])
        _p = 1-np.abs((_r-np.mean(_r))/np.ptp(_r)*2)**tmp
    elif param[comp+',profile'] == 'uniform':
        _p = np.ones(len(_r))
    else:
        tmp = param[comp+',profile']
        if '$RMIN' in tmp:
            tmp = tmp.replace('$RMIN', str(np.min(_r)))
        if '$RMAX' in tmp:
            tmp = tmp.replace('$RMAX', str(np.max(_r)))
        if '$R' in tmp:
            tmp = tmp.replace('$R', '_r')
        if '$DMIN' in tmp:
            tmp = tmp.replace('$DMIN', str(np.min(_d)))
        if '$DMAX' in tmp:
            tmp = tmp.replace('$DMAX', str(np.max(_d)))
        if '$D' in tmp:
            tmp = tmp.replace('$D', '_d')
        _p = eval(tmp)

    _cf = np.array([np.trapezoid((_p*_r)[_r<=x], _r[_r<=x]) for x in _r])
    _cf /= _cf.max()
    rh = np.interp(0.5, _cf, _r)
    if not fig is None:
        plt.close(fig)
        plt.figure(fig)
        plt.plot(_r, _p)
    return rh

def halfLightRadiusFromImage(oi, icube, incl, projang, x0=None, y0=None, fig=None):
    """
    un-inclined half-light radius.

    oi: from PMOIRED.oi. Need to have "oi.show" ran with imFov set
    icube: i-th cube in model
    incl: key to inclination of the rim in model's param
    projang: key to inclination of the rim in model's param
    x0, y0: keys to the center of rim (optional, wil assume 0,0 otherwise)
    """
    # -- interpret parameters
    param = pmoired.oimodels.computeLambdaParams(oi._model['MODEL']['param'], MJD=np.mean(oi['MJD']))
    incl = param[incl]
    projang = param[projang]
    if not x0 is None:
        x0 = param[x0]
    else:
        x0 = 0
    if not y0 is None:
        y0 = param[y0]
    else:
        y0 = 0

    # -- re-rotate model according to I and PA:
    cpa, spa = np.cos(projang*np.pi/180), np.sin(projang*np.pi/180)
    ci = np.cos(incl*np.pi/180)
    # -- coordinates of the center of the rim:
    _X = ((oi._model['MODEL']['X']-x0)*cpa - (oi._model['MODEL']['Y']-y0)*spa)/ci + x0
    _Y = (oi._model['MODEL']['X']-x0)*spa + (oi._model['MODEL']['Y']-y0)*cpa

    R = np.sqrt(_X**2+_Y**2)
    I = oi._model['MODEL']['cube'][icube,:,:]
    # -- remove extended component as the "background"
    I -= I.min()

    # -- order by distance to to the center
    P = I.flatten()[np.argsort(R.flatten())]
    # -- remove (unresolved) star as brightest pixel
    P[np.argmax(P)] = 0
    R = R.flatten()[np.argsort(R.flatten())]

    # -- radius
    r = np.linspace(R.min(), R.max(), 100)
    # -- cumulative flux
    cF = np.array([np.trapezoid(P[R<x]*R[R<x], R[R<x]) for x in r])
    cF /= cF.max()
    # -- half ligh radius:
    rh = np.interp(0.5, cF, r)

    if not fig is None:
        plt.close(fig); plt.figure(fig, figsize=(9, 5))
        ax = plt.subplot(121, aspect='equal')
        plt.pcolormesh(_X, _Y, I, vmax=np.percentile(I, 99.5),
                        shading='auto', rasterized=True)
        plt.title('de-rotated model')
        t = np.linspace(0, 2*np.pi, 100)[:-1]
        plt.plot(rh*np.cos(t), rh*np.sin(t), ':y', alpha=0.5, label='half light radius')
        ax.legend()

        ax = plt.subplot(222)
        # -- radial flux,
        plt.plot(R, P/P.max())
        plt.ylabel('radial profile (arb. unit)')

        plt.subplot(224, sharex=ax)
        plt.plot(r, cF)
        plt.hlines(0.5, 0, r.max(), linestyle='dotted')
        plt.vlines(rh, 0, 1, linestyle='dotted')

        plt.ylabel('cumulative flux')
        plt.xlabel('radius (mas)')
        plt.tight_layout()
    return rh


def showBootstrap(b, fig=0, figWidth=None, showRejected=False, ignore=None,
                  combParam=None, sigmaClipping=None, showChi2=False,
                  alternateParameterNames=None, showSingleFit=True, chi2MaxClipping=None):
    """
    you can look at combination of parameters:

    combParam: {'sep':'np.sqrt($x**2 + $y**2)'} assuming 'x' and 'y' are parameters from the model
    """
    symUncer = False
    global _AX, _AY
    global _bootAxes
    #t0 = time.time()
    boot = copy.deepcopy(b)
    #print('deep copy', time.time()-t0, 's')

    if combParam is None:
        combParam = {}

    if alternateParameterNames is None:
        alternateParameterNames = {}

    if len(combParam)>0:
        s = '$'
        for k in combParam:
            if not k in boot['fitOnly']:
                boot['fitOnly'].append(k)

            for i,f in enumerate(boot['all fits']):
                tmp = combParam[k]+''
                j = 0
                while s in tmp and j<5:
                    for x in boot['best'].keys():
                        if s+x in tmp:
                            tmp = tmp.replace(s+x, '('+str(boot['all fits'][i]['best'][x])+')')
                    j+=1
                boot['all fits'][i]['best'][k] = eval(tmp)
                boot['all fits'][i]['uncer'][k] = 0.0
            boot['all best'][k] = np.array([b['best'][k] for b in boot['all fits']])
        boot = analyseBootstrap(boot, verbose=2, sigmaClipping=sigmaClipping, chi2MaxClipping=chi2MaxClipping)

    color1 = 'orange' # single fit
    color2 = (0.2, 0.4, 1.0) # bootstrap
    colorC2 = (0, 0.4, 0.2) # chi2
    colorComb = (0.3, 0.4, 0.1) # combined parameters

    if ignore is None:
        ignore = []

    # -- for each fitted parameters, show histogram
    showP  = sorted(filter(lambda k: k not in combParam.keys() and k not in ignore, boot['fitOnly']),
        key=lambda k: k if not k in alternateParameterNames else alternateParameterNames[k])
    showP += sorted(filter(lambda k: k in combParam.keys() and k not in ignore, boot['fitOnly']),
        key=lambda k: k if not k in alternateParameterNames else alternateParameterNames[k])

    if figWidth is None:
        figWidth = min(FIG_MAX_WIDTH, 1+2*len(showP))

    fontsize = max(min(9*figWidth/len(showP), 14), 6)
    plt.close(fig)
    plt.figure(fig, figsize=(figWidth, figWidth))
    _AX = {}

    if showChi2:
        showP.append('chi2')

    offs = {}
    amps = {}
    t0 = time.time()
    _bootAxes = {}
    for i1, k1 in enumerate(showP):
        # -- create histogram plot
        _AX[i1] = plt.subplot(len(showP),
                              len(showP),
                              1+i1*len(showP)+i1)
        _bootAxes[k1] = _AX[i1]
        if k1=='chi2':
            # -- assumes bins is already defined (chi2 cannot be first!)
            # -- show histogram: line and area
            h = plt.hist(boot['all chi2'], bins=bins,
                         color=colorC2, histtype='step', alpha=0.9)
            h = plt.hist(boot['all chi2'], bins=bins,
                         color=colorC2, histtype='stepfilled', alpha=0.05)
            if showSingleFit:
                plt.plot(boot['fit to all data']['chi2'],
                            0.4*max(h[0]), 's', markersize=fontsize/2,
                            color=colorC2, label='fit to all data')
            plt.title(r'$\chi_{\rm red}^2$', fontsize=fontsize)
            offs['chi2'] = 0
            amps['chi2'] = 1
            plt.xlim(min(min(h[1]),
                        boot['fit to all data']['chi2']-0.1*np.ptp(h[1])),
                    max(max(h[1]),
                        boot['fit to all data']['chi2']+0.1*np.ptp(h[1])))
            #print('Xlim:', min(h[1]), max(h[1]), boot['fit to all data']['chi2'])
        else:
            # -- guess number of bins
            if k1 in boot['uncer']:
                bins = int(3*np.ptp(boot['all best'][k1])/boot['uncer'][k1])
            else:
                bins = 10
            bins = min(bins, len(boot['mask'])//5)
            bins = max(bins, 5)

            nd = np.abs(np.mean(boot['all best'][k1])/np.ptp(boot['all best'][k1]))
            if nd>0:
                nd = int(np.log10(nd))

            if nd>=4:
                offs[k1] = np.round(np.mean(boot['all best'][k1]), nd)
                amps[k1] = 10**(nd+1)
            else:
                offs[k1] = 0.0
                amps[k1] = 1.0

            #print(k1, nd, offs[k1])
            # -- show histogram: line and area
            h = plt.hist(amps[k1]*(boot['all best'][k1]-offs[k1]), bins=bins,
                         color='k', histtype='step', alpha=0.9)
            h = plt.hist(amps[k1]*(boot['all best'][k1]-offs[k1]), bins=bins,
                         color='k', histtype='stepfilled', alpha=0.05)
            # -- fitted and bootstrap values and uncertainties
            if showSingleFit and k1 in boot['fit to all data']['best']:
                plt.errorbar(amps[k1]*(boot['fit to all data']['best'][k1]-offs[k1]),
                            0.4*max(h[0]), markersize=fontsize/2,
                            xerr=amps[k1]*boot['fit to all data']['uncer'][k1],
                            color=color1, fmt='s', capsize=fontsize/2,
                            label='fit to all data')
            _color = color2
            if k1 in combParam:
                _color = colorComb

            if k1 in alternateParameterNames:
                T1 = alternateParameterNames[k1]
            else:
                T1 = k1

            if symUncer:
                plt.errorbar(amps[k1]*(boot['best'][k1]-offs[k1]), 0.5*max(h[0]),
                           xerr=amps[k1]*boot['uncer'][k1],
                           color=_color, fmt='d',
                           capsize=fontsize/2, label='bootstrap', markersize=fontsize/2)

                n = int(np.ceil(-np.log10(boot['uncer'][k1])+1))
                fmt = '%s\n'+'%.'+'%d'%max(n,0)+'f\n'+r'$\pm$'+'%.'+'%d'%max(n,0)+'f'
                plt.title(fmt%(T1, boot['best'][k1], boot['uncer'][k1]),
                            fontsize=fontsize)

            else:
                xerr=np.array([[amps[k1]*boot['uncer-'][k1]],
                               [amps[k1]*boot['uncer+'][k1]]])
                plt.errorbar(amps[k1]*(boot['best'][k1]-offs[k1]), 0.5*max(h[0]),
                            xerr=xerr ,
                            color=_color, fmt='d',
                            capsize=fontsize/2, label='bootstrap', markersize=fontsize/2)

                n = max(int(np.ceil(-np.log10(boot['uncer+'][k1])+1)),
                        int(np.ceil(-np.log10(boot['uncer-'][k1])+1)))
                check = 2*np.abs(boot['uncer+'][k1]-boot['uncer-'][k1])/\
                          (boot['uncer+'][k1]+boot['uncer-'][k1]) < 0.2
                if check:
                    fmt = '%s\n'+'%.'+'%d'%max(n,0)+'f\n'+r'$\pm$'+'%.'+'%d'%max(n,0)+'f'
                    plt.title(fmt%(T1, boot['best'][k1], boot['uncer'][k1]),
                                fontsize=fontsize)
                else:
                    fmt = '%s\n'+'%.'+'%d'%max(n,0)+'f\n'
                    fmt += r'$^{+'+'%.'+'%d'%max(n,0)+'f}'
                    fmt += r'_{-'+'%.'+'%d'%max(n,0)+'f}$'
                    plt.title(fmt%(T1, boot['best'][k1], boot['uncer+'][k1], boot['uncer-'][k1]),
                                fontsize=fontsize)

        if showSingleFit and i1==0:
            plt.legend(fontsize=5)

        # -- title
        _AX[i1].yaxis.set_visible(False)
        if i1!=(len(showP)-1):
            _AX[i1].xaxis.set_visible(False)
        else:
            _AX[i1].tick_params(axis='x', labelsize=fontsize*0.8)
            _AX[i1].set_xlabel(k1, fontsize=fontsize)
            if offs[k1]<0:
                _AX[i1].set_xlabel(k1+'\n+%f (%.0e)'%(np.abs(offs[k1]), 1/amps[k1]),
                                fontsize=fontsize)
            elif offs[k1]>0:
                _AX[i1].set_xlabel(k1+'\n-%f (%.0e)'%(np.abs(offs[k1]), 1/amps[k1]),
                                    fontsize=fontsize)
            _AX[i1].callbacks.connect('ylim_changed', _callbackAxes)
        # -- end histogram
    #print('hist in', time.time()-t0, 's')
    _AY = {}

    # -- show density plots
    for i1, k1 in enumerate(showP):
        for i2 in range(i1+1, len(showP)):
            if k1 in alternateParameterNames:
                T1 = alternateParameterNames[k1]
            else:
                T1 = k1

            k2 = showP[i2]
            if k2 in alternateParameterNames:
                T2 = alternateParameterNames[k2]
            else:
                T2 = k2

            if i1==0:
                _AY[i2] = plt.subplot(len(showP),
                            len(showP),
                            1+i2*len(showP)+i1,
                            sharex=_AX[i1])
                ax = _AY[i2]
            else:
                ax = plt.subplot(len(showP),
                            len(showP),
                            1+i2*len(showP)+i1,
                            #sharex=_AX[i1],
                            sharey=_AY[i2])
            _bootAxes[(k1, k2)] = ax
            #if k1=='chi2' or k2=='chi2':
            #    continue
            #print(k1, k2)
            c, m = 'k', '.'
            if k1=='chi2':
                X1 = boot['all chi2']
                X1r = boot['all chi2 ignored']
                c = colorC2
            else:
                X1 = boot['all best'][k1]
                X1r = boot['all best ignored'][k1]
            if k2=='chi2':
                X2 = boot['all chi2']
                X2r = boot['all chi2 ignored']
                c = colorC2
            else:
                X2 = boot['all best'][k2]
                X2r = boot['all best ignored'][k2]

            if len(boot['mask'])<300:
                plt.plot(amps[k1]*(X1-offs[k1]), amps[k2]*(X2-offs[k2]), m,
                          alpha=np.sqrt(2/len(boot['mask'])), color=c)
            else:
                plt.hist2d(amps[k1]*(X1-offs[k1]), amps[k2]*(X2-offs[k2]),
                           cmap='bone_r', bins=int(np.sqrt(len(boot['mask'])/3))
                         )

            if showRejected:
                plt.plot(amps[k1]*(X1r-offs[k1]), amps[k2]*(X2r-offs[k2]),
                        'xr', alpha=0.3)

            # -- combined parameters function of the other one?
            #print(combParam, k1, k2)
            if k1 in combParam or k2 in combParam:
                _c = colorComb
            else:
                _c = color2

            #x, y = dpfit.errorEllipse(boot, k1, k2)
            if k1=='chi2':
                _i1 = len(boot['fitOnly'])
                _c = colorC2
            else:
                _i1 = boot['fitOnly'].index(k1)
            if k2=='chi2':
                _i2 = len(boot['fitOnly'])
                _c = colorC2
            else:
                _i2 = boot['fitOnly'].index(k2)
            t = np.linspace(0,2*np.pi,100)
            sMa, sma, a = dpfit._ellParam(boot['cov'][_i1,_i1],
                                    boot['cov'][_i2,_i2],
                                    boot['cov'][_i1,_i2])
            _X,_Y = sMa*np.cos(t), sma*np.sin(t)
            _X,_Y = _X*np.cos(a)+_Y*np.sin(a),-_X*np.sin(a)+_Y*np.cos(a)
            if ((k1.endswith(',x') and k2.endswith(',y')) or
                (k1.endswith(',y') and k2.endswith(',x'))) and \
               k1.split(',')[0]== k2.split(',')[0]:
                print('ellipse (emin, emax, PA) for %s/%s: %.4f %.4f %.1f'%(
                            k1, k2, sMa, sma, a*180/np.pi))

            if k1=='chi2':
                x = np.mean(boot['all chi2'])+_X
            else:
                x = boot['best'][k1]+_X
            if k2=='chi2':
                y = np.mean(boot['all chi2'])+_Y
            else:
                y = boot['best'][k2]+_Y

            plt.plot(amps[k1]*(x-offs[k1]),
                     amps[k2]*(y-offs[k2]), '-', color=_c,
                     alpha=0.5,
                    #label='c=%.2f'%boot['cord'][k1][k2]
                        )
            #plt.legend(fontsize=5)

            if showSingleFit and k1 in boot['fit to all data']['best'] and \
                    k2 in boot['fit to all data']['best']:
                plt.plot(amps[k1]*(boot['fit to all data']['best'][k1]-offs[k1]),
                         amps[k2]*(boot['fit to all data']['best'][k2]-offs[k2]),
                         'x', color='0.5')
                x, y = dpfit.errorEllipse(boot['fit to all data'], k1, k2)
                plt.plot(amps[k1]*(x-offs[k1]), amps[k2]*(y-offs[k2]), '-',
                            color=color1)#, label='c=%.2f'%boot['cord'][k1][k2])
                plt.plot(amps[k1]*(boot['best'][k1]-offs[k1]),
                         amps[k2]*(boot['best'][k2]-offs[k2]),
                         '+', color=_c)

            if i2==(len(showP)-1):
                plt.xlabel(T1, fontsize=fontsize)
                if offs[k1]<0:
                    plt.xlabel(T1+'\n+%f (%.0e)'%(np.abs(offs[k1]), 1/amps[k1]),
                                fontsize=fontsize)
                elif offs[k1]>0:
                    plt.xlabel(T1+'\n-%f (%.0e)'%(np.abs(offs[k1]), 1/amps[k1]),
                                fontsize=fontsize)
                ax.tick_params(axis='x', labelsize=fontsize*0.8)
            else:
                ax.xaxis.set_visible(False)
            if i1==0:
                plt.ylabel(T2, fontsize=fontsize)
                if offs[k2]<0:
                    plt.ylabel(T2+'\n+%f (%.0e)'%(np.abs(offs[k2]), 1/amps[k2]),
                                fontsize=fontsize)
                elif offs[k2]>0:
                    plt.ylabel(T2+'\n-%f (%.0e)'%(np.abs(offs[k2]), 1/amps[k2]),
                                fontsize=fontsize)
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

    where Gn is the nth order Hankel transform of g:

    Gn(rho) = int_0^infty[g(r)*Jn(rho*r) r*dr]

    see Also: https://en.wikipedia.org/wiki/Hankel_transform#Relation_to_the_Fourier_transform_(general_2D_case)
    """

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
            imPix = 1/(B.max()*_c*20)
            Nx = int(np.ptp(r)/imPix)+1
            Ny = int(np.ptp(r)/imPix)+1
            Nmax = 200 # RAM requirement balloons quickly
            if Nx>Nmax:
                print("WARNING: synthetic image size is too large >Nmax=%d"%Nmax)
                return
            x = np.linspace(-np.max(r), np.max(r), Nx)
            X, Y = np.meshgrid(x, x)
            retFull = True
        else:
            X,Y = XY
            imPix = np.diff(X).mean()
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
        #         for imX in np.linspace(-imPix/2, imPix/2, ns+2)[1:-1]:
        #             for imY in np.linspace(-imPix/2, imPix/2, ns+2)[1:-1]:
        #                 _r2 = (Xr[j,i]+imX)**2+(Yr[j,i]+imY)**2
        #                 Im[j,i] += 1./ns**2*np.interp(_r2,r2,I,right=0.0,left=0.0)
        Im = np.interp(Xr**2+Yr**2, r2, I, right=0, left=0)
        # -- azimutal variations in image
        PA = np.arctan2(Yr, Xr)

        PAvar = np.ones(PA.shape)
        for k in range(len(n)):
            # -- should be consistent with Visibility computation below!
            #PAvar += amp[k]*np.cos(n[k]*(PA + 3*np.pi/2 + phi[k]*np.pi/180))
            # -- changed on 2021-10-21
            PAvar += amp[k]*np.cos(n[k]*(PA + phi[k]*np.pi/180 - np.pi/2))

        Im *= PAvar
        # -- normalize image to total flux
        if np.sum(Im)>0:
            Im /= np.sum(Im)

        # -- numerical Visibility
        if numVis:
            if len(u.shape)==2:
                Vis = Im[:,:,None,None]*np.exp(-2j*_c*(u[None,None,:,:]*X[:,:,None,None] +
                                                      v[None,None,:,:]*Y[:,:,None,None]))
            elif len(u.shape)==1:
                Vis = Im[:,:,None]*np.exp(-2j*_c*(u[None,None,:,:]*X[:,:,None] +
                                                  v[None,None,:,:]*Y[:,:,None]))
            Vis = np.trapezoid(np.trapezoid(Vis, axis=0), axis=0)/np.trapezoid(np.trapezoid(Im, axis=0), axis=0)
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
    _N = np.trapezoid(I*r, r)
    if _N==0 or not np.isfinite(_N):
        _N = 1
    def Hankel(n):
        if not Bm is None:
            H = np.trapezoid(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*_c*Bm[None,:]*r[:,None]),
                         r, axis=0)/_N
            H[~np.isfinite(H)] = 0
            # if any(~np.isfinite(H)):
            #     print('H(%d), _N:'%n, H, _N)
            return np.interp(_B, Bm, H)
        else:
            H = np.trapezoid(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*_c*_B.flatten()[None,:]*r[:,None]),
                         r, axis=0)/_N
            return np.reshape(H, _B.shape)

    # -- visibility without PA variations -> Hankel0
    if V0 is None:
        Vis = Hankel(0)*(1.+0j) # -- force complex
    else:
        Vis = V0*(1.+0j) # -- force complex

    for i in range(len(n)):
        if np.abs(amp[i])>0:
            #Vis += amp[i]*(-1j)**n[i]*Hankel(n[i])*\
            #        np.cos(n[i]*(_PA+3*np.pi/2+phi[i]*np.pi/180))
            # -- changed on 2021-10-21
            Vis += amp[i]*(-1j)**n[i]*Hankel(n[i])*\
                    np.cos(n[i]*(_PA + phi[i]*np.pi/180 - np.pi/2))

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

        # -- truncated gaussian: r0 as FWHM = 2.35482*sigma
        #_i = np.exp(-(_r/r0*2.35482)**2)*(_r<=diam/2)*(_r>=(1-thick)*diam/2)

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
    plt.close(0)
    plt.figure(0, figsize=(11,4))
    plt.clf()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93,
                        bottom=0.12, hspace=0.25, wspace=0)
    ax = plt.subplot(1,4,1)
    ax.set_aspect('equal')
    plt.pcolormesh(X, Y, I, cmap='inferno', vmin=0, shading='auto', rasterized=True)
    # plt.imshow(I, cmap='gist_heat', vmin=0, origin='lower',
    #             extent=[_r[0], _r[-1], _r[0], _r[-1]])
    title = r'image %imX%d, $\theta$=%.2fmas'%(Nx, Nx, diam)
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
                          cmap='gist_stern', vmin=0, vmax=1,
                          shading='auto', rasterized=True)
    plt.colorbar(pvis)

    ax = plt.subplot(2,4,3, sharex=ax0, sharey=ax0)
    plt.title('|V| semi-analytical')
    ax.set_aspect('equal')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, np.abs(Visp),
                    cmap='gist_stern', vmin=0, vmax=1,
                    shading='auto', rasterized=True)

    ax = plt.subplot(2,4,4, sharex=ax0, sharey=ax0)
    imYn = 1. # in 1/100 visibility
    plt.title(r'$\Delta$|V| (1/100)')
    ax.set_aspect('equal')
    res = 100*(np.abs(Vis)-np.abs(Visp))
    pv = plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0,
                    res, cmap='RdBu', shading='auto',
                    vmin=-np.max(np.abs(res)),
                    vmax=np.max(np.abs(res)),
                    rasterized=True
                    )
    plt.colorbar(pv)
    print('median  |V|  residual (abs.) = %.3f'%np.median(np.abs(res)), '%')
    print('90perc  |V|  residual (abs.) = %.3f'%np.percentile(np.abs(res), 90), '%')

    ax = plt.subplot(2,4,6, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title(r'$\phi$ numerical')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, 180/np.pi*np.angle(Vis),
                    cmap='hsv', vmin=-180, vmax=180, shading='auto',
                    rasterized=True)
    plt.colorbar()

    ax = plt.subplot(2,4,7, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title(r'$\phi$ semi-analytical')
    plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, 180/np.pi*np.angle(Visp),
                    cmap='hsv', vmin=-180, vmax=180, shading='auto',
                    rasterized=True)

    ax = plt.subplot(2,4,8, sharex=ax0, sharey=ax0)
    imYn = 1
    plt.title(r'$\Delta\phi$ (deg)')
    ax.set_aspect('equal')
    res = 180/np.pi*((np.angle(Vis)-np.angle(Visp)+np.pi)%(2*np.pi)-np.pi)
    pp = plt.pcolormesh(_c*diam*U/wl0, _c*diam*V/wl0, res, shading='auto',
                        cmap='RdBu', vmin=-imYn, vmax=imYn, rasterized=True)
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
