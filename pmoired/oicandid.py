# oicandid.py

import numpy as np
import matplotlib.pyplot as plt
import time

import multiprocessing

try:
    from moire import oimodels
except:
    import oimodels

def fitMap(oi, firstGuess=None, fitAlso=[], rmin=1, rmax=20, rstep=1.,
            multi=True):
    """
    firstguess: model. Star should be parametrized with '*,...' and companion 'c,...'.
        Other components can be used as well in the initial guess which can be
        passed in the variable "firstguess". Fit Only
    """
    default = {'*,ud':0.0, '*,f':1.0,
                'c,ud':0.0, 'c,f':0.1}

    if firstGuess is None:
        firstGuess = default
    default.update(firstGuess)
    firstGuess = default

    fitOnly = ['c,f', 'c,x', 'c,y']
    if type(fitAlso):
        fitOnly.extend(fitAlso)

    kwargs = {'maxfev':1000, 'ftol':1e-4, 'verbose':False,
            'fitOnly':fitOnly, }

    X, Y = [], []
    for r in np.linspace(rmin, rmax, int((rmax-rmin)/rstep+1)):
        t = np.linspace(0, 2*np.pi, max(int(2*np.pi*r/rstep+1), 5))[:-1]
        X.extend(list(r*np.cos(t)))
        Y.extend(list(r*np.sin(t)))

    plt.close(0)
    plt.figure(0)
    ax = plt.subplot(1,1,1, aspect='equal')
    plt.plot(X, Y, '+k', alpha=0.5)
    plt.plot(0, 0, '*g')
    N = len(X)

    res = []
    if multi:
        if type(multi)!=int:
            Np = min(multiprocessing.cpu_count(), N)
        else:
            Np = min(multi, N)
        print('running', N, 'fits...')
        # -- estimate fitting time by running 'Np' fit in parallel
        pool = multiprocessing.Pool(Np)
        t = time.time()
        for i in range(min(Np, N)):
            tmp = firstGuess.copy()
            tmp['c,x'] = X[i]
            tmp['c,y'] = Y[i]
            res.append(pool.apply_async(oimodels.fitOI, (oi, tmp, ), kwargs))
        pool.close()
        pool.join()
        print('initial estimate: %.1f fit per minute using %d threads (%.0fms per fit)'%(
               60/(time.time()-t)*min(Np, N), Np, 1000*(time.time()-t)/min(Np, N)))
        res = [r.get(timeout=1) for r in res]
        # -- run the remaining
        if N>Np:
            pool = multiprocessing.Pool(Np)
            for i in range(max(N-Np, 0)):
                tmp = firstGuess.copy()
                tmp['c,x'] = X[Np+i]
                tmp['c,y'] = Y[Np+i]
                res.append(pool.apply_async(oimodels.fitOI, (oi, tmp, ), kwargs))
            pool.close()
            pool.join()
            res = res[:Np]+[r.get(timeout=1) for r in res[Np:]]
    else:
        Np = 1
        tmp = firstGuess.copy()
        tmp['c,x'] = X[0]
        tmp['c,y'] = Y[0]
        t = time.time()
        res.append(oimodels.fitOI(oi, tmp, **kwargs))
        print('one fit takes ~%.2fs'%(time.time()-t))
        for i in range(N-1):
            tmp = firstGuess.copy()
            tmp['c,x'] = X[Np+i]
            tmp['c,y'] = Y[Np+i]
            res.append(oimodels.fitOI(oi, tmp, **kwargs))
    print('it took %.1fs, %.0f fit per minute on average (%.0fms per fit)'%(
            time.time()-t, 60/(time.time()-t)*N, 1000*(time.time()-t)/N))
    ncall = np.sum([r['info']['nfev'] for r in res])
    print('    %d calls, %.1f calls per fit, %.2fms per call'%(ncall,
            ncall/N, 1000*(time.time()-t)/ncall))

    # -- show displacement of fit:
    for r in res:
        r['dist'] = np.sqrt((r['firstGuess']['c,x']-r['best']['c,x'])**2+
                            (r['firstGuess']['c,y']-r['best']['c,y'])**2)
        plt.plot([r['firstGuess']['c,x'], r['best']['c,x']],
                 [r['firstGuess']['c,y'], r['best']['c,y']],
                 '-k', alpha=0.1)

    # -- filter to unique solutions:
    uni = [res[0]]
    for r in res[1:]:
        dupl = False
        for u in uni:
            d = (u['best']['c,x']-r['best']['c,x'])**2/\
                (u['uncer']['c,x']**2 + r['uncer']['c,x']**2) +\
                (u['best']['c,y']-r['best']['c,y'])**2/\
                (u['uncer']['c,y']**2 + r['uncer']['c,y']**2) +\
                (u['best']['c,f']-r['best']['c,f'])**2/\
                (u['uncer']['c,f']**2 + r['uncer']['c,f']**2)
            if d<3.:
                dupl = True
                continue
        if not dupl:
            uni.append(r)
    print('%d minima -> %d unique solutions'%(len(res), len(uni)))
    uni = sorted(uni, key=lambda x: x['chi2'])
    plt.scatter([f['best']['c,x'] for f in uni],
                [f['best']['c,y'] for f in uni],
                c = [np.log10(f['chi2']) for f in uni],
                cmap='viridis_r')
    # check if companion flux is less than central star flux
    for u in uni:
        if u['best']['c,f']>u['best']['*,f']:
            plt.plot(u['best']['c,x'], u['best']['c,y'],
                    'xr', markersize=12)

    #for u in uni:
    #    plt.text(u['best']['c,x'], u['best']['c,y'],
    #            '%.2f'%(u['best']['c,f']))
    plt.colorbar(label=r'log$_{10}$ $\chi^2$')
    ax.invert_xaxis()
    return uni
