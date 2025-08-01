import os
from urllib.request import urlopen
import numpy as np

this_dir, this_filename = os.path.split(__file__)

def createRossTable():
    # -- Load the LD/Ross sent by Hilding, where LD is the outer diameter
    cols = ['mass', 'L', 'Teff', 'logg', 'Ross', 'Outer', 'Ross/Outer']
    rossTable = {k:[] for k in cols}
    f = open(os.path.join(this_dir, 'Rosseland_SATLAS_giants.dat'))
    for l in f.readlines():
        if not l.strip().startswith('#'):
            for i,c in enumerate(cols):
                rossTable[c].append(float(l.split()[i]))
    f.close()
    f = open(os.path.join(this_dir,  'Rosseland_SATLAS_dwarfs.dat'))
    cols = ['mass', 'L', 'Teff', 'logg', 'Ross/Outer']
    for l in f.readlines():
        if not l.strip().startswith('#'):
            for i,c in enumerate(cols):
                rossTable[c].append(float(l.split()[i]))
    f.close()

    rossTable['file'] = []
    for i in range(len(rossTable['Teff'])):
        if rossTable["logg"][i]<4:
            d = 'https://cdsarc.u-strasbg.fr/ftp/J/A+A/554/A98/spheric/'
        else:
            d = 'https://cdsarc.u-strasbg.fr/ftp/J/A+A/556/A86/spheric/'
        rossTable['file'].append(d+f'ld_satlas_surface.2t{rossTable["Teff"][i]:.0f}'+
                                        f'g{100*rossTable["logg"][i]:.0f}'+
                                        f'm{10*rossTable["mass"][i]:.0f}.dat')
    return rossTable
# == init
rossTable = createRossTable()
rossFiles = [os.path.basename(f) for f in rossTable['file']]

def getClosestModel(Teff, logg, mass, localdir=None, verbose=False):
    """
    get closest SALTAS model based on Teff (K), logg (log cgs) and mass (Msun).

    downloaded on local directory
    """
    global rossTable
    #Teff, logg, mass = 7600, 1.9, 9.7

    dist = [((rossTable['Teff'][i]-Teff)/100)**2 +
            (10*(rossTable['logg'][i]-logg))**2 +
            (rossTable['mass'][i]-mass)**2 for i in range(len(rossTable['Teff']))]
    i = np.argmin(dist)
    if verbose:
        print('closest model:', i,
                f'Teff={rossTable['Teff'][i]}K',
                f'logg={rossTable['logg'][i]}',
                f'Mass={rossTable['mass'][i]}Msun')

    localfilename = os.path.basename(rossTable['file'][i])
    if not localdir is None:
        if os.path.isdir(localdir):
            localfilename = os.path.join(localdir, localfilename)
        else:
            print('WARNING:', localdir, 'does not exists!')

    if not os.path.exists(localfilename):
        if verbose:
            print('downloading and creating', localfilename)
        with open(localfilename, 'w') as f:
            for l in urlopen(rossTable['file'][i]).readlines():
                f.write(l.decode())
    else:
        if verbose:
            print(localfilename, 'already exists')
    return localfilename

def readFile(filename, band, component=None):
    global rossTable, rossFiles
    # -- read file:
    f = open(filename)
    cols = ['mu','B','V','R','I','H','K']
    wl_b = {'B':0.45, 'V':0.55, 'R':0.65, 'I':1.0, 'H':1.65, 'K':2.2}
    data = {c:[] for c in cols}
    for l in f.readlines():
        for k,c in enumerate(cols):
            data[c].append(float(l.split()[k]))
    for c in cols:
        data[c] = np.array(data[c])
    f.close()

    ic = 0
    if band in cols and band!='mu':
        ic = cols.index(band.upper())

    if type(band)==float or type(band)==int:
        _d = [np.abs(wl_b[k]-band) for k in wl_b.keys()]
        ic = cols.index(list(wl_b.keys())[np.argmin(_d)])

    i = rossFiles.index(os.path.basename(filename))

    print(rossTable['Ross/Outer'][i])
    if component is None:
        c = ''
    else:
        c = component+','
    P = {c+'ROSS':1.0,
         c+'diam': f'${c}ROSS/{rossTable['Ross/Outer'][i]:f}', # outer diam
         c+'profile':f'np.interp($R/(0.5*${c}diam), '+\
                        str([float(x) for x in list(np.sqrt(1-data['mu']**2)[::-1])])+\
                        ',          '+str([float(x) for x in list(data[cols[ic]][::-1])])+', left=1.0, right=0.0)'
        }

    return P
