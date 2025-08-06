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

    rossTable['spheric'] = []
    rossTable['planar'] = []

    for i in range(len(rossTable['Teff'])):
        if rossTable["logg"][i]<4:
            d = 'https://cdsarc.u-strasbg.fr/ftp/J/A+A/554/A98/'
        else:
            d = 'https://cdsarc.u-strasbg.fr/ftp/J/A+A/556/A86/'
        rossTable['spheric'].append(d+f'spheric/ld_satlas_surface.2t{rossTable["Teff"][i]:.0f}'+
                                        f'g{100*rossTable["logg"][i]:.0f}'+
                                        f'm{10*rossTable["mass"][i]:.0f}.dat')
        rossTable['planar'].append(d+f'planar/ld_t{rossTable["Teff"][i]:.0f}'+
                                        f'g{100*rossTable["logg"][i]:.0f}_surface.dat')


    return rossTable
# == init
rossTable = createRossTable()
rossFilesSph = [os.path.basename(f) for f in rossTable['spheric']]
rossFilesPla = [os.path.basename(f) for f in rossTable['spheric']]

def getClosestModel(Teff, logg, mass=None, localdir=None, verbose=False, modeltype='spheric'):
    """
    get closest SALTAS model based on Teff (K), logg (log cgs) and mass (Msun).

    downloaded on local directory.

    Models from Neilson 2013 and 2014:
        https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/554/A98
        https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/556/A86
    """
    global rossTable, _ross, _i

    if modeltype == 'spheric':
        dist = [((rossTable['Teff'][i]-Teff)/100)**2 +
                (10*(rossTable['logg'][i]-logg))**2 +
                (rossTable['mass'][i]-mass)**2 for i in range(len(rossTable['Teff']))]
    else:
        dist = [((rossTable['Teff'][i]-Teff)/100)**2 +
                (10*(rossTable['logg'][i]-logg))**2 for i in range(len(rossTable['Teff']))]

    # -- index of model in table
    _i = np.argmin(dist)

    if verbose:
        if modeltype=='spheric':
            print('closest model: #%d'%_i,
                    f'Teff={rossTable["Teff"][_i]:.0f}K',
                    f'logg={rossTable["logg"][_i]:.2f}',
                    f'Mass={rossTable["mass"][_i]:.1f}Msun')
        else:
            print('closest model: #%d'%_i,
                    f'Teff={rossTable["Teff"][_i]:.0f}K',
                    f'logg={rossTable["logg"][_i]:.2f}')

    localfilename = os.path.basename(rossTable[modeltype][_i])
    _ross = rossTable["Ross/Outer"][_i]
    if not localdir is None:
        if os.path.isdir(localdir):
            localfilename = os.path.join(localdir, localfilename)
        else:
            print('WARNING:', localdir, 'does not exists!')

    if not os.path.exists(localfilename):
        if verbose:
            print('downloading and creating', localfilename)
        with open(localfilename, 'w') as f:
            for l in urlopen(rossTable[modeltype][_i]).readlines():
                f.write(l.decode())
    else:
        if verbose:
            print(localfilename, 'already exists')
    return localfilename

def readFile(filename, band, component=None, verbose=False):
    """
    returns PMOIRED limbdarkened model (optionaly for a named component)
    based on SATLAS model stored in "filename", in a given band ("B", "V",
    "R", "I", "H" or "K").

    Models from Neilson & Lester 2013a and 2013b:
    https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/554/A98
    https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/556/A86
    """
    global rossTable, rossFilesSph, rossFilesPla, _data, _i
    # -- read file:
    if verbose:
        print('opening', filename)
    f = open(filename)
    cols = ['mu','B','V','R','I','H','K']
    wl_b = {'B':0.45, 'V':0.55, 'R':0.65, 'I':1.0, 'H':1.65, 'K':2.2}
    _data = {c:[] for c in cols}
    n=0
    for l in f.readlines():
        n+=1
        for k,c in enumerate(cols):
            _data[c].append(float(l.split()[k]))
    if verbose:
        print('read', n, 'lines')
    for c in cols:
        _data[c] = np.array(_data[c])
    f.close()

    ic = 0
    if band in cols and band!='mu':
        ic = cols.index(band.upper())

    if type(band)==float or type(band)==int:
        _d = [np.abs(wl_b[k]-band) for k in wl_b.keys()]
        ic = cols.index(list(wl_b.keys())[np.argmin(_d)])

    # if 'satlas' in filename:
    #     i = rossFilesSph.index(os.path.basename(filename))
    # else:
    #     i = rossFilesPla.index(os.path.basename(filename))

    if component is None:
        c = ''
    else:
        c = component+','
    P = {c+'ROSSDIAM':1.0,
        c+'_MUtab': str([float(x) for x in list(_data['mu'])]),
        c+'_Itab':str([float(x) for x in list(_data[cols[ic]])]),
        c+'profile':f'np.interp($MU, $'+c+'_MUtab, $'+c+'_Itab, left=0.0, right=1.0)'
        }
    if 'satlas' in filename:
        P.update({c+'diam': f'${c}ROSSDIAM/{rossTable['Ross/Outer'][_i]:f}'})
    else:
        # -- by definition
        P.update({c+'diam': f'${c}ROSSDIAM'})
    return P
