import os
import pickle
import time
import numpy as np
import multiprocessing

this_dir, this_filename = os.path.split(__file__)

directory = 'binlens_images2'
directory = os.path.join(this_dir, directory)

import binlen

debug = False
def clean():
    files = os.listdir(directory)
    for f in files:
        if f.startswith('_tmpPMOIRED') and f.endswith('.pickle'):
            os.system('rm -rf '+os.path.join(directory, f))
        if f in [txt]:
            os.system('rm '+os.path.join(directory, f))
    return

def computeSparseParam(MJD, param, parallel=False):
    images = computeImages(MJD, param, parallel=parallel)
    fluxes = []
    mjd = []
    for k in images:
        mjd.append(k)
        if type(images[k])==str:
            with open(images[k], 'rb') as f:
                tmp = pickle.load(f)
            fluxes.append(np.sum(tmp['I']))
        else:
            fluxes.append(np.sum(images[k]['I']))

    fluxes = np.array(fluxes)
    fluxes = fluxes[np.argsort(mjd)]
    mjd = sorted(mjd)
    s = 'np.interp($MJD, %s, %s)'%('['+','.join(['%.2f'%x for x in mjd])+']',
                                   '['+','.join(['%.3f'%x for x in fluxes])+']')
    if False:
        print('images = {',end='')
        for k in images:
            print(str(k)+':', end='')
            if type(images[k])==str:
                print('"'+images[k]+'"', end=',')
            else:
                print(','.join(['"'+x+'":len%d'%len(images[k][x]) for x in images[k]]), end=',')
            print()
        print('}')

    return {'sparse':images, 'spectrum(mjd)':s}

_lastParam = []
_lastMJD = {}
# files = list(filter(lambda x: x.startswith('_tmp') and
#                         x.endswith('.pickle'), os.listdir()))
# for f in files:
#     os.remove(f)
def computeImages(MJD, param, parallel=False, debug=False):
    """
    parallel is actually much slower!
    """
    global _lastParam, _lastMJD

    useFiles=True # use files to store sparse images -> SLOW!
    useFiles=False #

    if parallel:
        preptime = 0
        t0 = time.time()
        nproc = min(multiprocessing.cpu_count(), len(MJD))

        Pool = multiprocessing.Pool(nproc)
        res = []
        for i,mjd in enumerate(MJD):
            res.append(Pool.apply_async(computeImages, ([mjd], param, ),
                {'parallel':False, 'debug':False}))
        Pool.close()
        Pool.join()
        tmp = [r.get(timeout=1) for r in res]
        res = {}
        for t in tmp:
            res.update(t)
        return res

    t0 = time.time()
    cols = ['t0', 'u0', 'tE', 'b', 'q', 'theta', 'rhos']
    if not 't0' in param and 'mjd0' in param:
        param['t0'] = param['mjd0']- 49999.5 

    _p = [param[c] for c in cols]
    if _p!=_lastParam:
        # -- reset model computing, remove old images
        #files = list(filter(lambda x: x.startswith('img.') and
        #                        x.endswith('.pickle') and
        #                        len(x)==18, os.listdir()))
        files = list(filter(lambda x: os.path.exists(x),
                            [_lastMJD[k] for k in _lastMJD if type(k)==str]))
        for f in files:
            os.remove(f)
        #os.system('rm img.????.??.pickle')
        if debug:
            print('D> parameters have changed! removing %d old image(s)'%len(files))
        # -- reset recent parameters and MJDs
        _lastParam = _p
        _lastMJD = {}

    res = {}
    if debug:
        print('D> -- computeImage starts')
    for mjd in MJD:
        if mjd in _lastMJD:
            if debug:
                print('D> same parameters and known MJD: re-using image')
            res[round(mjd, 2)] = _lastMJD[mjd]
            continue

        if debug:
            print('D> computing for epoch MJD=%f:'%mjd)
        t1 = time.time()
        tmp = [param[k] for k in cols]
        tmp += [round(mjd - 49999.5, 2)]

        if debug:
            print('D> computing:', tmp)

        # -- call Fortran function
        ng, xg, yg = binlen.binlen_image(*tmp)
        xg = xg[:ng]
        yg = yg[:ng]
        # -- from Fortran code
        if param['b']<=1:
            offx = param['b']*(-0.5+1/(1+param['q']))
        else:
            offx = param['b']/2 -param['q']/(1+param['q'])/param['b']
        xg += offx

        t1 = time.time()
        if useFiles:
            imgp = '_tmpPMOIRED%d.pickle'%np.random.randint(1e16)
            while os.path.exists(imgp):
                imgp = '_tmpPMOIRED%d.pickle'%np.random.randint(1e16)
            if debug:
                print('D> image computed in %.2fs'%(time.time()-t1))
            imageTxt2Sparse( X=xg, Y=yg,rhos=param['rhos'], debug=debug,
                            savefile=True, filename=imgp)
            _lastMJD[mjd] = imgp
        else:
            _lastMJD[mjd] = imageTxt2Sparse(X=xg, Y=yg, rhos=param['rhos'],
                                    debug=debug, savefile=False)
        if debug:
            print('D> conversion in %.2fs'%(time.time()-t1))
        res[round(mjd, 2)] = _lastMJD[mjd]

    if debug:
        print('D> -- computeImage done in %.2fs'%(time.time()-t0))
    #print("computeImages:", res.keys())

    return res

def lightCurve(MJD, param, stepping=None):
    """
    param: dictionnary containing ['t0', 'u0', 'tE', 'b', 'q', 'theta', 'rhos']

    MJD: list of MJDs

    stepping: use a interpolated light curve instead of for every date
    """
    cols = ['t0', 'u0', 'tE', 'b', 'q', 'theta', 'rhos']
    if not 't0' in param and 'mjd0' in param:
        param['t0'] = param['mjd0']- 49999.5 
    res = []
    if stepping is None:
        X = MJD
    else:
        X = np.linspace(np.min(MJD), np.max(MJD), int(np.ptp(MJD)/stepping)+1)

    for mjd in X:
        tmp = [param[k] for k in cols]
        tmp += [mjd - 49999.5]
        ng, xg, yg = binlen.binlen_image(*tmp)
        # -- normalise fluxes to flux of source
        # -- "0.002" -> grid size in fortran code
        fscale = (0.002/param['rhos'])**2/np.pi
        res.append(ng*fscale)
        #xg = xg[:ng]
        #yg = yg[:ng]
        # -- from Fortran code
        #if param['b']<=1:
        #    offx = param['b']*(-0.5+1/(1+param['q']))
        #else:
        #    offx = param['b']/2 -param['q']/(1+param['q'])/param['b']
        #xg += offx
    res = np.array(res)
    if not stepping is None:
        res = np.interp(MJD, X, res)
    return res

def imageTxt2Sparse(X=None, Y=None, rhos=None, filename=None, debug=debug,savefile=True):
    """
    this is  slow!
    """
    tinit = time.time()
    if X is None and Y is None:
        data = []
        with open(filename) as f:
            for l in f.readlines():
                data.append((np.double(l.split()[0]), np.double(l.split()[1])))
        X, Y = np.array([d[0] for d in data]), np.array([d[1] for d in data])
    if debug:
        print('D> X,Y have length', len(X))
    t = time.time()
    x0 = np.mean(X)
    y0 = np.mean(Y)
    if debug:
        print('D> X,Y have length', len(X))
        print('D> X: %.3f..%.3f'%(min(X), max(X)))
        print('D> Y: %.3f..%.3f'%(min(Y), max(Y)))

    #dx = 0.005 # original file
    dx = 0.005*4 # binned

    nx, ny = int(X.ptp()/dx)+1, int(Y.ptp()/dx)+1

    x = np.linspace(X.min(), X.max(), nx)
    y = np.linspace(Y.min(), Y.max(), ny)

    _X, _Y = np.meshgrid(x, y)
    _I = np.zeros(_X.shape)
    t = time.time()

    _i = np.int_(nx/(nx+1)*(Y-y.min())/dx)
    _j = np.int_(nx/(nx+1)*(X-x.min())/dx)

    # -- weirdly enough, the np call is slower than a loop!
    #np.add.at(_I, list(zip(_i, _j)), 1)

    for k in range(len(_i)):
        _I[_i[k], _j[k]] += 1

    if debug:
        print('D> resampling in %.2fs'%(time.time()-t))
    # -- generate sparse image
    sparse = {'x':[], 'y':[], 'I':[]}
    for i in range(ny):
        for j in range(nx):
            if _I[i,j]>0:
                sparse['x'].append(_X[i,j])
                sparse['y'].append(_Y[i,j])
                sparse['I'].append(_I[i,j])
    for k in sparse:
        sparse[k] = np.array(sparse[k])

    if not rhos is None:
        # -- normalise fluxes to flux of source
        # -- not sure how to justify the "0.002"! -> Maybe the stepping in ray tracing?
        # "0.002" -> grid size in fortran code
        fscale = (0.002/rhos)**2/np.pi
        sparse['I'] *= fscale
    if savefile:
        if not filename.endswith('.pickle'):
            filename += '.pickle'
        if debug:
            print('D> saving in', filename)
        with open(filename, 'wb') as f:
            pickle.dump(sparse, f)
        return filename
    else:
        return sparse
