from matplotlib import pyplot as plt
import matplotlib.tri as mtri
import numpy as np
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
	print(_dir_data)

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
    R = Rpole + colat*0    
    if np.abs(w)>0:
        wsinc = np.abs(w)*np.sin(colat)
        _w = np.abs(np.sin(colat))>1e-6
        R[_w] = 3*Rpole/(wsinc[_w])*np.cos((np.pi + np.arccos(wsinc[_w]))/3)

    # -- based on eq 15
    V = (R*U.Rsun).to(U.km)*Omega # in km/s

    if verbose:
        print(f"{R.min()=}, {R.max()=}")
        print(f"Veq = {(R.max()*U.Rsun).to(U.km)*Omega:.1f}")
    
    dcolat = 1e-6
    dR = 3*Rpole/(np.abs(w)*np.sin(colat+dcolat))*\
            np.cos((np.pi + np.arccos(np.abs(w)*np.sin(colat+dcolat)))/3) -R

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
              y*np.cos(np.pi/2-incl) + z*np.sin(np.pi/2-incl), \
             -y*np.sin(np.pi/2-incl) + z*np.cos(np.pi/2-incl)
    return x*np.cos(pa) + y*np.sin(pa), \
          -x*np.sin(pa) + y*np.cos(pa), \
           z

def normaliseV(x, y, z):
    n = np.sqrt(x**2 + y**2 + z**2)
    n += n==0
    return x/n, y/n, z/n

def surface(N, Rpole, Mass, w, Tpole, incl=0, pa=0, beta=0.25, verbose=False, vpuls=0):
    res = {'colat':[], 'lon':[], 'dS':[], }
    dcolat = np.pi/N
    for c in np.linspace(0, np.pi, N): # colatitude:
        r ,_ ,_ ,_ , _ = RGcola(np.array([c]), Rpole, Mass, w)
        nl = max(int(2*N*np.sin(c)*r/Rpole), 1)
        dlon = 2*np.pi/nl
        ds = r**2*np.sin(c)*dlon*dcolat
        for l in np.linspace(-np.pi, np.pi, nl):
            res['colat'].append(c)
            res['lon'].append(l)
            res['dS'].append(ds) # surface 
            
    res['colat'] = np.array(res['colat'])
    res['lon'] = np.array(res['lon'])
    # -- fractional surface of the node (compared to the whole star)
    res['dS'] = np.array(res['dS'])
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

    res['vx']+= res['nx']*vpuls
    res['vy']+= res['ny']*vpuls
    res['vz']+= res['nz']*vpuls

    if verbose:
        print(f"{res['vz'].min()=}, {res['vz'].max()=}")
    
    # -- for LD computation
    res['mu'] = np.arccos(res['nz'])
    # -- projected surface of the node toward observer 
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

def addFluxImu(star, wl, plot=False, verbose=False):
    global _imudata, _imuTeff, _imulogg
    
    t0 = time.time()
    star['flux'] = np.zeros((len(star['x']), len(wl)) )
    star['ld'] = np.zeros((len(star['x']), len(wl)) )
    star['wl'] = wl

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


