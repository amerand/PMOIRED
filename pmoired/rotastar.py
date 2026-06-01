from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import numpy as np
#import scipy.interpolate
from scipy.special import assoc_legendre_p

from astropy import constants as CONST
from astropy import units as U
import urllib, os, glob, time, pickle


try:
    # Necessary while Python versions below 3.9 are supported.
    import importlib_resources as resources
except ImportError:
    from importlib import resources

if os.path.exists(resources.files("pmoired")):
    _dir_data = resources.files("pmoired")
    #print(_dir_data)

with open(os.path.join(_dir_data, 'imudata.pckl'), 'rb') as f:
    _imudata = pickle.load(f)
    _imuTeff = np.array([d['TEFF'] for d in _imudata])
    _imulogg = np.array([d['LOGG'] for d in _imudata])

def RGcola(colat, Rpole, Mass, w, verbose=False):
    """
    radius and effective gravity as function of colatitude (radians), 
    Rpole in Rsol, Mass in Msol, w as omega/omegacrit.
    -> R is in Rsol, logg log10(cgs), V in km/s, dR/dcolat in Rsol/rad
    return R, g/gpole, logg, velocity, dR/dcolat
    """

    # -- critical angular rotation rate 
    # https://arxiv.org/pdf/astro-ph/0603327, eq 13
    Omegacrit = np.sqrt(8/27*CONST.G*(Mass*U.Msun).to(U.kg)/(Rpole*U.Rsun).to(U.m)**3) # in 1/s
    Omega = w*Omegacrit

    if verbose:
        print(f"{Omegacrit=}")
    

    # -- https://arxiv.org/pdf/astro-ph/0603327, eq 12
    R = Rpole*np.ones(len(colat))
    dR = np.zeros(len(colat))
    dcolat = 1e-6

    if np.abs(w)>0:
        wsinc = np.abs(w)*np.sin(colat)
        _w = np.abs(np.sin(colat))>1e-6
        R[_w] = 3*Rpole/(wsinc[_w])*np.cos((np.pi + np.arccos(wsinc[_w]))/3)

        dR = 3*Rpole/(np.abs(w)*np.sin(colat+dcolat))*\
                np.cos((np.pi + np.arccos(np.abs(w)*np.sin(colat+dcolat)))/3) - R

    # -- based on eq 15
    V = (R*U.Rsun).to(U.km)*Omega # in km/s
    if verbose:
        print(f"{R.min()=}, {R.max()=}")
        print(f"Veq = {(R.max()*U.Rsun).to(U.km)*Omega:.1f}")    

    gpole = CONST.G*(Mass*U.Msun).to(U.kg)/(Rpole*U.Rsun).to(U.m)**2 
    if w==0:
        g = gpole + colat*0
    else:
        gr = -CONST.G*(Mass*U.Msun).to(U.kg)/(R*U.Rsun).to(U.m)**2 + \
             (R*U.Rsun).to(U.m)*(Omega*np.sin(colat))**2
        gc = (R*U.Rsun).to(U.m)*Omega**2*np.sin(colat)*np.cos(colat)
        if verbose:
            print(f"{gr.min()=}, {gr.max()=}")
            print(f"{gc.min()=}, {gc.max()=}")

        g = np.sqrt(gr**2 + gc**2)

    if verbose:
        print(f"{g.min()=}, {g.max()=}")

    logg = np.log10((g.to(U.cm/U.s**2)).value)
    
    if verbose:
        print(f"{logg.min()=}, {logg.max()=}")
    
    # R is in Rsol, logg log10(cgs), V in km/s, dR/dcolat in Rsol/rad
    return R, g/gpole, logg, V.value, dR/dcolat

def rotations(x, y, z, incl, pa):
    x, y, z = x, \
              y*np.cos(incl) + z*np.sin(incl), \
             -y*np.sin(incl) + z*np.cos(incl)
    return x*np.cos(pa) + y*np.sin(pa), \
          -x*np.sin(pa) + y*np.cos(pa), \
           z

def normaliseV(x, y, z):
    n = np.sqrt(x**2 + y**2 + z**2)
    n += n==0
    return x/n, y/n, z/n

def surface(N, Rpole, Mass, w, Tpole, incl=0, pa=0, beta=0.25, verbose=False, 
            vpuls=0, dist=None):
    """
    N: number of co-latitudes (better of odd)
    Rpole: polar radius (in Rsun)
    Mass: mass in Msun
    w: fractional rotational rate
    Tpole: polar temperature (K)
    incl: inclination in radians
    pa: position angle in radians
    beta: gravity darkening coef (0.25 canonical)
    vpuls: pulsation velocity in km/s
        for non radial, {(l,m):amplitude_km/s} or {(l,m):(amplitude_km/s, phase_in_lon_rad)}
        the l=0, m=0 is radial mode. Note that the phase is irrelevant for m=0!

    dist: dispance in pc
    """
    res = {'colat':[], 'lon':[], 'dS':[], }
    C = np.linspace(0, np.pi, N)
    #C = np.linspace(np.pi/(2*N), np.pi-np.pi/(2*N), N)
    dcolat = np.mean(np.diff(C))
    for c in C: # colatitude:
        r ,_ ,_ ,_ ,_ = RGcola(np.array([c]), Rpole, Mass, w)
        nl = max(int(2*N*np.sin(c)*r/Rpole), 1)
        dlon = 2*np.pi/nl
        ds = r**2*np.sin(c)*dlon*dcolat
        for l in np.linspace(-np.pi, np.pi, nl)[:-1]:
            res['colat'].append(c)
            res['lon'].append(l)
            res['dS'].append(ds) # surface element

    res['colat'] = np.array(res['colat'])
    res['lon'] = np.array(res['lon'])
    # -- fractional surface of the node (compared to the whole star)
    res['dS'] = np.array(res['dS'])[:,0]
    res['dS'] /= np.sum(res['dS'])
    
    res['R'], res['g/gpole'], res['logg'], res['V'], res['dR/dcolat'] = RGcola(res['colat'], Rpole, Mass, w, verbose=verbose)
    
    res['Teff'] = (Tpole*(res['g/gpole'])**beta).value
    if verbose:
        print(f"{res['Teff'].min()=}, {res['Teff'].max()=}")

    # -- star seen pole-on

    # -- position in Rsol
    res['x'] = res['R']*np.cos(res['lon'])*np.sin(res['colat'])
    res['y'] = res['R']*np.sin(res['lon'])*np.sin(res['colat']) 
    res['z'] = res['R']*np.cos(res['colat'])

    # -- rotational velocity in km/s
    res['vx'] = res['V']*np.sin(res['lon'])
    res['vy'] = res['V']*np.cos(res['lon'])
    res['vz'] = res['V']*0
    
    # -- tangantial vector along co-latitude
    res['dx1'] = res['R']*np.cos(res['colat']) + np.sin(res['colat'])*res['dR/dcolat']
    res['dx1'] *= np.cos(res['lon'])
    res['dy1'] = res['R']*np.cos(res['colat']) + np.sin(res['colat'])*res['dR/dcolat']
    res['dy1'] *= np.sin(res['lon'])
    res['dz1'] = -res['R']*np.sin(res['colat']) + np.cos(res['colat'])*res['dR/dcolat']
    res['dx1'], res['dy1'], res['dz1'] = normaliseV(res['dx1'], res['dy1'], res['dz1'])
    
    # -- tangential along longitude
    res['dx2'] = -np.sin(res['lon'])
    res['dy2'] =  np.cos(res['lon'])
    res['dz2'] =  0*res['R']
    res['dx2'], res['dy2'], res['dz2'] = normaliseV(res['dx2'], res['dy2'], res['dz2'])

    # -- rotations (inclination, PA):
    res['x'], res['y'], res['z'] = rotations(res['x'], res['y'], res['z'], incl, pa)
    res['vx'], res['vy'], res['vz'] = rotations(res['vx'], res['vy'], res['vz'], incl, pa)
    res['dx1'], res['dy1'], res['dz1'] = rotations(res['dx1'], res['dy1'], res['dz1'], incl, pa)
    res['dx2'], res['dy2'], res['dz2'] = rotations(res['dx2'], res['dy2'], res['dz2'], incl, pa)    

    # -- normal to the surface of the star (unit vector)
    v1 = np.array([res['dx1'], res['dy1'], res['dz1']])
    v2 = np.array([res['dx2'], res['dy2'], res['dz2']])
    res['nx'], res['ny'], res['nz'] = np.cross(v1, v2, axis=0)

    # -- convention: positive radial velocity means decreasing radius
    res['vpx'] = 0
    res['vpy'] = 0
    res['vpz'] = 0

    if type(vpuls)==dict:
        # non radial, {(l,m):amplitude_km/s} or {(l,m):(amplitude_km/s, phase_in_lon_rad)}
        
        for l,m in vpuls:
            if type(vpuls[(l,m)])==tuple or type(vpuls[(l,m)])==list:
                amp, phi = vpuls[(l,m)]
            else:
                amp = vpuls[(l,m)]
                phi = 0
            z = np.cos(m*(res['lon']-phi))*assoc_legendre_p(l, m, np.cos(res['colat']))
            z *= amp/np.max(z)
            #print(res['lon'].shape, res['colat'].shape, res['vx'].shape, res['nx'].shape, z.shape)
            res['vpx'] -= res['nx']*z[0]
            res['vpy'] -= res['ny']*z[0]
            res['vpz'] -= res['nz']*z[0]
    else:
        res['vpx'] -= res['nx']*vpuls
        res['vpy'] -= res['ny']*vpuls
        res['vpz'] -= res['nz']*vpuls

    res['vx'] += res['vpx']
    res['vy'] += res['vpy']
    res['vz'] += res['vpz']
    

    if not dist is None:
        c = (1*U.Rsun).to(U.m)/(dist*U.pc).to(U.m)*180*3600*1000/np.pi
        for k in ['x', 'y', 'z']:
            res[k+'_mas'] = res[k]*c.value

    if verbose:
        print(f"{res['vz'].min()=}, {res['vz'].max()=}")
    
    # -- for LD computation
    res['mu'] = np.arccos(res['nz'])
    # -- projected surface of the node towards observer 
    res['proj dS'] = res['dS']*res['nz']

    return res

def getAllAtlas9Files(directory='gridp00k2odfnew', table=None, TeffMax=12000, TeffMin=4000, overwrite=False):
    global _dir_data
    url='http://wwwuser.oats.inaf.it/castelli/grids/'+directory+'/'
    if table is None:
        table = directory.split('odfnew')[0].split('grid')[-1]
        table = 'f'+table+'tab.html'
        #print(f"{table=}")
    print(url+table)
    lines = urllib.request.urlopen(url+table).read().decode('utf-8').split('\n')
    lines = [x for x in lines if '.dat' in x]
    lines = [l.split('"')[1] for l in lines]
    savedir = url.split('/')[-1]
    if savedir=='':
        savedir = url.split('/')[-2]
    savedir = os.path.join(_dir_data, 'ATLAS9', savedir)
    print(f"{savedir=}")

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    found = 0
    for i,l in enumerate(lines):
        teff = int(l.split('t')[1].split('g')[0])
        if teff<=TeffMax and teff>=TeffMin:
            #print(savedir+'/'+l)
            if not os.path.exists(os.path.join(savedir, l)):
                tmp = urllib.request.urlopen(url+'/'+l, timeout=60).read().decode('utf-8')
                with open(os.path.join(savedir, l), 'w') as f:
                    f.write(tmp)
            else:
                found += 1
    if found >0:
        print(f'{found} files already on disk, not downloaded (you can use overwrite=True)')
    return savedir

def ReadOneAtlas9File(filename, plot=False):
    """
    returns a dictionnary with keys 'TEFF', 'LOGG', 'VTURB', 'LH' (all
    floats) and 'HNU', 'WAVEL', 'FLAMBDA' (np.array) units: TEFF in K,
    LOGG in log10(m/s2), VTURB in km/s, WAVEL in um, HNU in
    erg/s/cm/Hz/ster and FLAMBDA in erg/s/cm2/m/ster

    L/H is mixing length

    In each file: col.3=wavelength in nm
              col.4=frequency nu (s^-1)
              col.5=Hnu (erg/cm^2/s/Hz/ster)
              col.6=Hcont (erg/cm^2/s/Hz/ster)
              col.7=Hnu/Hcont
    Flambda=4*Hnu*c/wavelength^2  c=light velocity
    """
    f = open(filename, 'r')
    lines = f.readlines()

    WAVEL = []
    HNU = []
    FLAMBDA = []

    for l in lines:
        if 'TEFF' in l:
            tmp = l.split()
            TEFF = float(tmp[1])
            LOGG = float(tmp[3])
        if 'TITLE' in l:
            tmp = l.split()
            METAL = float(l.split('[')[1].split(']')[0])
            VTURB = float(tmp[2].split('=')[1])
            LH = float(tmp[3].split('=')[1])
        if 'FLUX' in l and len(l.strip()) > 4:
            l = l[9:] # remove beginning of the line, not used
            tmp = l.split()
            wl = float(tmp[0])
            if wl >50.0:
                ### there is a typo for short WL in files
                ### too lazy to correct it
                WAVEL.append(wl/1000.) # in um
                HNU.append(float(tmp[2]))

    FLAMBDA = 4*np.array(HNU)*299792458.0/\
              (np.array(WAVEL)*1e-6)**2 #erg/s/cm2/m/ster
    f.close()

    if plot:
        pyplot.figure(1)
        pyplot.clf()
        pyplot.plot(WAVEL, FLAMBDA, 'r', linewidth=2)
        pyplot.yscale('log')
        pyplot.xscale('log')

    return {'atlas9': filename,
            'TEFF':TEFF, 
            'LOGG': LOGG, 
            'VTURB':VTURB, 
            'LH':LH, # mixing length
            'METAL':METAL,
            'HNU':np.array(HNU), # Hnu (erg/cm^2/s/Hz/ster)
            'WAVEL':np.array(WAVEL), # wavelength in __um__
            'FLAMBDA':FLAMBDA, # 4*Hnu*c/wavelength^2  c=light velocity
           }
    
def Imudata(savedata=False):
    savedir = getAllAtlas9Files()
    files = glob.glob(os.path.join(savedir, '*odfnew.dat'))
    data = [ReadOneAtlas9File(f) for f in files]
    
    bands = {"B": 0.45, "V": 0.55, "R": 0.65, "I": 1.0, "H": 1.65, "K": 2.2}
    # -- approximate mass from Teff 
    teff2mass = lambda teff: np.interp(d['TEFF'], 
                        [4000, 6000, 8000, 10000, 15000], 
                        [0.5, 1, 1.5, 2., 4], right=4, left=0.5)
    powlaw = lambda mu,p: mu**p['alpha']
    
    for i,d in enumerate(data):
        if i%100 == 0:
            print(i+1, len(data))
        if d['LOGG']>2:
            #print(d['TEFF'], d['LOGG'], teff2mass(d['TEFF']))
            f = satlas.getClosestModel(d['TEFF'], d['LOGG'], teff2mass(d['TEFF']))
        else:
            #print(d['TEFF'], d['LOGG'])
            f = satlas.getClosestModel(d['TEFF'], d['LOGG'], 4)
        d['satlas'] = f
        alphas = {}
        for b in bands:
            tmp = satlas.readFile(f, band=b)
            mu0 = np.sqrt(1-float(tmp['diam'].split('/')[1])**2)
            newMu = (np.array(eval(tmp['_MUtab']))-mu0)/(1-mu0)
            w = newMu>=0
            if sum(w):
                fit = dpfit.leastsqFit(powlaw, newMu[w], {'alpha':0.1}, np.array(eval(tmp['_Itab']))[w], verbose=False)
            else:
                print(i, f, mu0)
            alphas[b] = float(fit['best']['alpha'])
        
        d['alpha'] = ([bands[b] for b in sorted(bands, key=lambda x: bands[x])],
                      [alphas[b] for b in sorted(bands, key=lambda x: bands[x])])
    if savedata:
        with open(os.path.join(_dir_data, 'imudata.pckl'), 'wb') as f:
            pickle.dump(data, f)
    return data 

def addFluxImu(star, wl, plot=False, verbose=False, plines=None):
    """
    add broad band flux to a model dict "star" for the wavelength vector "wl"

    plines = photospheric lines, affected by rotation and pulsation
    lines = line of sight line, not affected by rotation or pulsations

    """
    global _imudata, _imuTeff, _imulogg
    
    t0 = time.time()
    star['flux'] = np.zeros((len(star['x']), len(wl)) )
    star['ld'] = np.zeros((len(star['x']), len(wl)) )
    star['wl'] = wl
    # -- positive vz points towards observer -> blue shifted
    star['doppler wl'] = wl[None,:]*(1-star['vz'][:,None]/299792.4580)

    # take advantage of colatitude layering
    T = set(star['Teff'][star['nz']>=0])

    # == total flux over disk should be 1
    # alpha = np.linspace(0, 1, 41)
    # print('_a=', [round(float(a), 3) for a in alpha])
    # mu = np.linspace(0, 1, 1000)
    # res = []
    # for a in alpha:
    #     res.append(np.trapezoid(mu**a*np.sqrt(1-mu**2), mu))
    # res = np.array(res)
    # res /= res[0]
    # print('_c=', [round(float(a), 5) for a in res])

    _a= [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    _c= [1.0, 0.97042, 0.9429, 0.91671, 0.89175, 0.86795, 0.84522, 0.82351, 0.80275, 0.78288, 0.76384, 0.74559, 0.72808, 0.71128, 0.69513, 0.6796, 0.66466, 0.65028, 0.63642, 0.62307, 0.61019, 0.59777, 0.58577, 0.57418, 0.56298, 0.55215, 0.54167, 0.53153, 0.52171, 0.51219, 0.50297, 0.49403, 0.48536, 0.47695, 0.46878, 0.46085, 0.45314, 0.44565, 0.43837, 0.43129, 0.42441]
    norma = lambda a: np.interp(a, _a, _c)

    for t in T:
        w = (star['nz']>=0)*(star['Teff']==t)
        lg = np.mean(star['logg'][w])
        # 2 closest models
        d = ((_imuTeff-t)/10000.)**2 + (_imulogg-lg)**2
        i0 = np.argsort(d)[0]
        i1 = np.argsort(d)[1]
        f0 = 10**np.interp(np.log10(wl), 
                np.log10(_imudata[i0]['WAVEL']), 
                np.log10(_imudata[i0]['FLAMBDA']))

        f1 = 10**np.interp(np.log10(wl), 
                np.log10(_imudata[i1]['WAVEL']), 
                np.log10(_imudata[i1]['FLAMBDA']))

        # -- this ignores doppler wavelength shift!
        if _imudata[i0]['TEFF'] != _imudata[i1]['TEFF']:
            star['flux'][w] = f0 + (f1-f0)*(t-_imudata[i0]['TEFF'])/(_imudata[i1]['TEFF']-_imudata[i0]['TEFF'])
        elif _imudata[i0]['LOGG'] != _imudata[i1]['LOGG']:
            star['flux'][w] = f0 + (f1-f0)*(lg-_imudata[i0]['LOGG'])/(_imudata[i1]['LOGG']-_imudata[i0]['LOGG'])
        else:
            print('WARNING! I do not know how to interpolate')
        alpha = np.interp(wl, _imudata[i0]['alpha'][0], _imudata[i0]['alpha'][1])
        star['ld'][w] = star['nz'][w][:,None]**alpha[None,:]
        # -- flux normalisation
        star['ld'][w] /= norma(alpha)[None,:]

    spectrum = np.ones(len(wl))
    if not plines is None:
        """
        dict
        'wl0' : central wl in um
        'gaussian': full width half maximum (nm)
        'f': depth compared to 1. <0 for abs, >0 for emission
        opt: 'f XXXX': depth compared 1, at Teff=XXXX
        """
        w = star['nz']>=0
        #w = np.isfinite(star['nz'])

        for l in plines:
            #print(f'DBG> {l=}')
            # -- temp dependent depth
            D = {float(k.split(' ')[1]):l[k] for k in l if k.startswith('f') and ' ' in k}
            if len(D)==1:
                raise Exception('need at least 2 line depths at Teff!')
            if len(D)>1:
                Teff = np.array(sorted(D.keys()))
                F = np.array([D[k] for k in Teff])
                if 'gaussian' in l:
                    tmp = np.interp(star['Teff'], Teff, F)[w,None]*\
                            np.exp(-(star['doppler wl'][w,:] - l['wl0'])**2/
                                    (2*(l['gaussian']/1000/2.35482)**2))
                elif 'lorentzian' in l:
                    tmp = np.interp(star['Teff'], Teff, F)[w,None]*\
                     ((0.5*l['lorentzian']/1000)**2/
                        ((star['doppler wl'][w,:] - l['wl0'])**2+(0.5*l['lorentzian']/1000)**2))

            else:
                if 'gaussian' in l:
                    tmp = l['f']*np.exp(-(star['doppler wl'][w,:] - l['wl0'])**2/
                                     (2*(l['gaussian']/1000/2.35482)**2))
                elif 'lorentzian' in l:
                    tmp = l['f']*(0.5*l['lorentzian']/1000)**2/\
                        ((star['doppler wl'] - l['wl0'])**2+(0.5*l['lorentzian']/1000)**2)

            star['flux'][w,:] *= 1+tmp
            #star['flux'][w,:] *= star['dS'][w][:,None]*(1+tmp)
            spectrum += np.sum(star['proj dS'][w][:,None]*tmp, axis=0)/np.sum(star['proj dS'][w])

    star['spectrum'] = spectrum

    if verbose:
        print(f"flux computation in {1000*(time.time()-t0):.1f}ms for {len(wl)} wavelengths")

    if not plot:
        return star
        
    if type(plot)!=int:
        plot=0
    plt.close(plot)
    plt.figure(plot, figsize=(2.5*len(wl),8))
    
    for i, l in enumerate(wl):
        if i==0:
            ax0 = plt.subplot(3, len(wl), i+1, aspect='equal')
            ax1 = ax0        
        else:
            ax1 = plt.subplot(3, len(wl), i+1, aspect='equal', sharex=ax0, sharey=ax0)
        ax2 = plt.subplot( 3, len(wl), len(wl)+i+1, aspect='equal', sharex=ax0, sharey=ax0)
        ax3 = plt.subplot( 3, len(wl), 2*len(wl)+i+1, aspect='equal', sharex=ax0, sharey=ax0)
    
        if i==0:
            ax1.set_ylabel('Gravity darkening')
            ax2.set_ylabel('limb darkening')
            ax3.set_ylabel('total darkening')
            
        # if i==len(wl)-1:
        #     plt.triplot(star['x'][star['nz']>=0], star['y'][star['nz']>=0], '-k', alpha=0.1)
    
        N = len(set(star['Teff']))

        ax1.set_title(rf'$\lambda$={l}$\mu$m')
        ax1.tricontourf(star['x'][star['nz']>=0], 
                        star['y'][star['nz']>=0], 
                        star['flux'][star['nz']>=0,i],#*star['ld'][star['nz']>=0,i],
                        cmap='gist_heat', levels=2*N, zorder=0, vmin=0)
        ax2.tricontourf(star['x'][star['nz']>=0], 
                        star['y'][star['nz']>=0], 
                        star['ld'][star['nz']>=0,i],
                        cmap='gist_heat', levels=2*N, zorder=0, vmin=0)
        ax3.tricontourf(star['x'][star['nz']>=0], 
                        star['y'][star['nz']>=0], 
                        star['flux'][star['nz']>=0,i]*star['ld'][star['nz']>=0,i],
                        cmap='gist_heat', levels=2*N, zorder=0, vmin=0)
        
    plt.xlim(2.5,-2.5)
    plt.ylim(-2.5,2.5)
    plt.tight_layout()
    return star


Ncolat= 51
def Vrota(u, v, wl, param, plot=False, fullOutput=False,
          imFov=None, imPix=None, imX=0, imY=0, imN=None,):
    """
    u, v: baselines in m (1D np.array: dimension N)
    wl: wavelength in um (1D np.array: dimension M)

    parameters:
    - 'Rpole' in Rsun
    - 'Tpole' in K
    - 'Mass' in Msun
    - 'omega': fractional rotational rate (/OmegaCrit)
    - 'dist' in pc 
    - 'beta': optional (default 0.25)
    - 'vpuls': optional pulsation velocity, in km/s (default=0) 
        or {'vamp 2 1': ,'lon0 2 1':} for l=2,m=1 non radial mode 
            in km/s for and and degrees for lon0
    result: complex visibility (np.array of dimension N x M)

    """
    global star, Ncolat
    if 'beta' in param:
        beta = param['beta']
    else:
        beta = 0.25

    if 'vpuls' in param:
        vpuls = param['vpuls']
    else:
        vpuls = 0
    V = {k:param[k] for k in param if k.startswith('vamp')}
    if len(V)>0:
        vpuls = {}
        for k in V:
            l,m = int(k.split(' ')[1]), int(k.split(' ')[2])
            if k.replace('amp', 'lon0') in param:
                vpuls[(l,m)] = (param[k], param[k.replace('amp', 'lon0')]*np.pi/180)
            else:
                vpuls[(l,m)] = param[k]
        #print(vpuls)

    # -- mas to radians
    c =  np.pi/180/3600/1000*1e6

    if "x" in param and "y" in param:
        x0, y0 = param["x"], param["y"]
    else:
        x0, y0 = 0, 0


    star = surface(Ncolat, param['Rpole'], param['Mass'], param['omega'], param['Tpole'], 
                incl=param['incl']*np.pi/180, pa=param['projang']*np.pi/180, 
                beta=beta, vpuls=vpuls, verbose=False, dist=param['dist'])

    # -- only account for plines, i.e photospheric lines
    tmp = {k:param[k] for k in param if k.startswith('pline_')}
    if len(tmp)>0:
        L = set(['_'.join(k.split('_')[:2]) for k in tmp])
        plines = [{k.split('_')[2]:param[k] for k in param if k.startswith(l)} for l in L]
    else:
        plines = None
    tmp = {k:param[k] for k in param if k.startswith('line_')}
    #print(f'DBG> {plines=}')
    star = addFluxImu(star, wl, plines=plines)

    # -- visible points
    w = star['nz']>=0
    #print('DBG>', param['omega'], np.sum(w))

    if True:
        # -- with LD effects
        flx = star['flux'][w]*star['ld'][w]*star['proj dS'][w][:,None]
    else:
        # -- without LD effects
        flx = star['flux'][w]*star['proj dS'][w][:,None]
    
    # -- x,y phase offset is taken care somewhere else (in "oimodels.py")
    vis = np.exp(-2j*np.pi*c*(u[:,None,None]*(star['x_mas'][w][None,:,None]+x0) + 
                              v[:,None,None]*(star['y_mas'][w][None,:,None]+y0))/wl[None,None,:])

    # -- complex visibilities
    vis = np.sum(flx[None,:,:]*vis, axis=1)/np.sum(flx, axis=0)[None,:]
    # -- SED
    flx = np.sum(flx, axis=0)

    if not imFov is None:
        # -- find the shape of the star in polar coordinates
        _r2 = star['x_mas'][w]**2 + star['y_mas'][w]**2
        _pa = np.arctan2(star['x_mas'][w], star['y_mas'][w])
        palim = np.linspace(-np.pi, np.pi, int(np.sqrt(len(star['x_mas'][w])))+1)
        dpa = np.max(np.diff(palim))/2
        r2lim = np.zeros(len(palim))
        for i in range(len(palim)):
            r2lim[i] = np.max(_r2[np.abs((_pa-palim[i]+np.pi)%np.pi-np.pi)<=dpa])

        # -- compute cube
        if imN is None:
            imN = 2 * int(imFov / imPix / 2) + 1
        X = np.linspace(imX - imFov / 2, imX + imFov / 2, imN)
        Y = np.linspace(imY - imFov / 2, imY + imFov / 2, imN)
        _X, _Y = np.meshgrid(X, Y)

        cube = np.zeros((len(wl), len(X), len(Y)))

        # -- interpolations scale
        dr2 = 0.1*np.max(r2lim)/np.sqrt(len(star['x_mas'][w]))
        n = 3 # interpolations points
        for i,x in enumerate(X):
            for j,y in enumerate(Y):
                if (x-x0)**2+(y-y0)**2 <= np.interp(np.arctan2(x-x0, y-y0), palim, r2lim):
                    # -- some kind of inverse distance
                    d2 = 1/(dr2 + (x-x0-star['x_mas'][w])**2 + (y-y0-star['y_mas'][w])**2)
                    k = np.argsort(d2)[::-1]

                    # -- no need to take into account the surface of the node here
                    # -- the x,y grid provide the sampling

                    # -- closest neighbor        
                    #cube[:,j,i] = star['flux'][w][k[0],:]*star['ld'][w][k[0],:]
                
                    # -- interpolation
                    norm = 0
                    for l in range(n):
                        cube[:,j,i] += d2[k[l]]*star['flux'][w][k[l],:]*star['ld'][w][k[l],:] 
                        norm += d2[k[l]]
                    cube[:,j,i] /= norm

        # @@@@@@@ THIS IS EXTREMELY SLOW! @@@@@@@@@@@@@@@@@@@@
        # grP = [(float(star['x_mas'][w][i]+x0), float(star['y_mas'][w][i]+y0)) for i in range(len(star['x'][w]))]
        # print('DBG>', grP[:10])
        # gr = np.array([_X, _Y]).reshape(2, -1).T
        # for i in range(len(wl)):
        #     cube[i, :, :] = scipy.interpolate.RBFInterpolator(
        #         grP, star['flux'][w][:,i]*star['ld'][w][:,i],
        #         #kernel="gaussian", neighbors=2
        #         )(gr).reshape((imN, imN))
    else:
        _X, _Y, cube = None, None, None

    if fullOutput:
        # -- visibility, total spectrum (SED), x, y, cube, 
        #.      0.   1.   2.  3.  4,     5
        return vis, flx, _X, _Y, cube, star['spectrum']
    else:
        # -- only (complex) visibility
        return vis



