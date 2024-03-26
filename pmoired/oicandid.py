# oicandid.py

import numpy as np
import matplotlib.pyplot as plt
import time
import scipy

from matplotlib.colors import LinearSegmentedColormap

import multiprocessing

from pmoired import oimodels

def fitMap(oi, firstGuess=None, companion='c', fitAlso=[],
            rmin=None, rmax=None, rstep=None,
            multi=True, fig=1, cmap=None, doNotFit=[], logchi2=False):
    """
    firstguess: model. Star should be parametrized with '*,...' and companion 'c,...'.
        Other components can be used as well in the initial guess which can be
        passed in the variable "firstguess". Fit Only
    """
    D = {'|V|':('OI_VIS', 'B/wl'),
         'V2':('OI_VIS2', 'B/wl'),
         'T3PHI':('OI_T3', 'Bmax/wl'),
        }
    if type(oi) == dict:
        oi = [oi]
    for o in oi:
        assert 'fit' in o, "'fit' should be defined!"
        # -- in m/um
        _bmaxR, _bmax = 0., 0.
        for e in o['fit']['obs']:
            R = np.mean(o['WL'])/np.diff(o['WL']).mean()
            if e in D:
                for b in o[D[e][0]].keys():
                    #print(o[D[e][0]][b].keys())
                    _bmax = max(_bmax, np.max(o[D[e][0]][b][D[e][1]]))
                    _bmaxR = max(_bmaxR, np.max(o[D[e][0]][b][D[e][1]])/R)

    print('max(B/wl), max(B/wl/R):', _bmax, _bmaxR)
    if rstep is None:
        rstep = 206.24/_bmax # one fringe resolution
        print('> rstep set to %.2f mas based on largest (B/wl)'%rstep)
    if rmin is None:
        rmin = rstep
        print('> rmin set to %.2f mas based on rstep'%rmin)
    if rmax is None:
        rmax = 40
        rmax = 0.5*206.265/_bmaxR # 1/2 full smearing
        print('> rmax set to %.1f mas based on largest (B/wl/R)'%rmax)

    if cmap is None:
        colors = [(0,1,1), 'yellow', 'orange', 'darkorange', 'red', 'darkred', 'black']
        nodes = [0.0, 0.2,0.4, 0.6, 0.8, 0.9, 1]
        nodes = np.linspace(0,1,len(colors))**1.5
        #colors = [(0,1,1), 'blue', 'black', 'darkred', 'red', 'darkorange', 'orange', 'yellow']
        #nodes = [0.0,  0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        cmap = LinearSegmentedColormap.from_list("candid", list(zip(nodes, colors)))

    # -- single star
    if type(firstGuess)==dict and '*,ud' in firstGuess:
        UD = oimodels.fitOI(oi, {'ud':firstGuess['*,ud']}, verbose=False)
    elif type(firstGuess)==dict and '*,diam' in firstGuess:
        UD = oimodels.fitOI(oi, {'ud':firstGuess['*,diam']}, verbose=False)
    else:
        UD = oimodels.fitOI(oi, {'ud':1.0}, verbose=False)
    print('uniform disk fit: chi2=%.3f'%UD['chi2'])
    default = {'*,ud':0.0, '*,f':1.0, companion+',ud':0.0, companion+',f':0.01}

    if firstGuess is None:
        firstGuess = default
    
    default.update(firstGuess)
    firstGuess = default

    if '*,diam' in firstGuess and '*,ud' in firstGuess:
        firstGuess.pop('*,ud')
    print('firstGuess:', firstGuess)


    fitOnly = ['c,f', 'c,x', 'c,y']
    if type(doNotFit) == list:
        for k in doNotFit:
            if k in fitOnly:
                fitOnly.remove(k)

    if type(fitAlso) == list:
        fitOnly.extend(fitAlso)

    kwargs = {'maxfev':10000, 'ftol':1e-5, 'verbose':False,
              'fitOnly':fitOnly, }

    X, Y = [], []
    for r in np.linspace(rmin, rmax, int((rmax-rmin)/rstep+1)):
        t = np.linspace(0, 2*np.pi, max(int(2*np.pi*r/rstep+1), 5))[:-1]
        X.extend(list(r*np.cos(t)))
        Y.extend(list(r*np.sin(t)))

    if type(fig) == int:
        plt.close(fig)
        plt.figure(fig, figsize=(9, 4))
        ax1 = plt.subplot(1,2,1, aspect='equal')
        ax2 = plt.subplot(2,2,2)
        ax3 = plt.subplot(2,2,4)
        # -- starting points on grid
        ax1.plot(X, Y, '+k', alpha=0.1)
        # -- central star
        ax1.plot(0, 0, '*c', label='central star')
        N = len(X)
        fig = True
    else:
        fig = False

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
            #print(tmp)
            res.append(pool.apply_async(oimodels.fitOI, (oi, tmp, ), kwargs))
        pool.close()
        pool.join()
        print('initial estimate: %.1f fit per minute using %d threads'%(3600/(time.time()-t), Np))
        res = [r.get(timeout=60) for r in res]
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

    # -- fit converged
    _n = len(res)
    res = [r for r in res if np.sum([r['uncer'][k] for k in r['uncer']])]
    if len(res)<_n:
        print('  - removing %d fits which did not converge properly'%(_n-len(res)))

    # -- solution within explored range
    _n = len(res)
    res = [r for r in res if
            np.sqrt(r['best']['c,x']**2+r['best']['c,y']**2)>=rmin and
            np.sqrt(r['best']['c,x']**2+r['best']['c,y']**2)<=rmax]
    if len(res)<_n:
        print('  - removing %d fits outside search range'%(_n-len(res)))

    # -- chi2 cannot be larger than 1.1*chi2 single UD
    _n = len(res)
    res = [r for r in res if r['chi2']<1.1*UD['chi2']]
    if len(res)<_n:
        print('  - removing %d fits chi2 > 1.1*chi2_single_star'%(_n-len(res)))

    n=0
    for r in res:
        if r['best']['c,f']>r['best']['*,f']:
            n +=1
            r['best']['c,x'] *= -1
            r['best']['c,y'] *= -1
            r['best']['c,f'] *= 1/r['best']['c,f']
    if n>0:
        print('  %d fits corrected for c,f>*,f'%n)

    # -- show displacement of fit:
    for r in res:
        r['displ'] = np.sqrt((r['firstGuess']['c,x']-r['best']['c,x'])**2+
                            (r['firstGuess']['c,y']-r['best']['c,y'])**2)
        if fig:
            ax1.plot([r['firstGuess']['c,x'], r['best']['c,x']],
                     [r['firstGuess']['c,y'], r['best']['c,y']],
                     '-k', alpha=0.1)

    # -- filter to unique solutions:
    uni = [res[0]]
    for r in res[1:]:
        dupl = False
        if not dupl:
            for u in uni:
                d = 0
                for k in u['fitOnly']:
                    d += (u['best'][k] - r['best'][k])**2/\
                         (u['uncer'][k]**2 + r['uncer'][k]**2)
                if d<len(u['fitOnly']):
                    dupl = True
                    continue
        if not dupl:
            uni.append(r)
    print('%d unique minima from %d fits'%(len(uni), len(res)))
    uni = sorted(uni, key=lambda x: x['chi2'])
    print('best fit chi2 = %.3f'%uni[0]['chi2'])
    print('{')
    for k in uni[0]['best'].keys():
        if uni[0]['uncer'][k]>0:
            n = max(int(2-np.log10(uni[0]['uncer'][k])), 0)
            f = '%.'+str(n)+'f'
            f = f+', # +- '+f
            print('  ',"'"+k+"':", f%(uni[0]['best'][k], uni[0]['uncer'][k]))
        else:
            print('  ',"'"+k+"':", uni[0]['best'][k], ',')
    print('}')
    r = np.sqrt(uni[0]['best']['c,x']**2+uni[0]['best']['c,y']**2)
    print('separation: %.4f mas'%r)
    pa = np.arctan2(uni[0]['best']['c,x'], uni[0]['best']['c,y'])
    pa *= 180/np.pi
    print('PA: %.2f degress'%pa)

    if fig:
        # -- best solution:
        #ax1.plot(uni[0]['best']['c,x'], uni[0]['best']['c,y'], '1r')
        #ax1.plot(uni[0]['best']['c,x'], uni[0]['best']['c,y'], '2r')
        #ax1.plot(uni[0]['best']['c,x'], uni[0]['best']['c,y'], '*y',
        #        label='best guess companion')
        c = [f['chi2'] for f in uni]
        if logchi2:
            c = np.log10(c)
        sca = ax1.scatter([f['best']['c,x'] for f in uni],
                          [f['best']['c,y'] for f in uni],
                         c = c,
                    cmap=cmap)
        # -- check that companion flux is less than central star flux
        if logchi2:
            plt.colorbar(sca, label=r'log$_{10}\chi^2$', ax=ax1)
        else:
            plt.colorbar(sca, label=r'$\chi^2$', ax=ax1)
        ax1.invert_xaxis()
        ax1.set_title('valid local minima')
        ax1.set_xlabel(r'x = RA offset (E$\leftarrow$ , mas)')
        ax1.set_ylabel(r'y = dec offset ($\rightarrow$N, mas)')

        X = [r['displ'] for r in res]
        bins = int(5*np.ptp(X)/np.std(X))

        ax2.hist(X, bins=bins, align='mid', color='k', histtype='step', alpha=0.9)
        ax2.hist(X, bins=bins, align='mid', color='k', histtype='stepfilled', alpha=0.05)
        ax2.vlines(rstep, 0, ax2.get_ylim()[1], label='grid step')
        ax2.set_xlabel('fit displacement (mas)')
        ax2.legend()

        X = [r['chi2'] for r in res]
        if logchi2:
            X = np.log10(X)
        bins = int(5*np.ptp(X)/np.std(X))
        ax3.hist(X, bins=bins, align='mid', color='k', histtype='step', alpha=0.9)
        ax3.hist(X, bins=bins, align='mid', color='k', histtype='stepfilled', alpha=0.05)
        if logchi2:
            ax3.vlines(np.log10(UD['chi2']), 0, ax3.get_ylim()[1], label='single star fit', color='r')
            ax3.set_xlabel('local minima log$_{10}\chi^2$')
        else:
            ax3.vlines(UD['chi2'], 0, ax3.get_ylim()[1], label='single star fit', color='r')
            ax3.set_xlabel('local minima $\chi^2$')

        ax3.legend(loc='upper left')
        plt.tight_layout()
    return uni

def _nSigmas(chi2r_TEST, chi2r_TRUE, NDOF):
    """
    - chi2r_TEST is the hypothesis we test
    - chi2r_TRUE is what we think is what described best the data
    - NDOF: numer of degres of freedom

    chi2r_TRUE <= chi2r_TEST

    returns the nSigma detection
    """
    p = scipy.stats.chi2.cdf(NDOF, NDOF*chi2r_TEST/chi2r_TRUE)
    log10p = np.log10(np.maximum(p, 1e-161)) ### Alex: 50 sigmas max
    #p = np.maximum(p, -100)
    res = np.sqrt(scipy.stats.chi2.ppf(1-p,1))
    # x = np.logspace(-15,-12,100)
    # c = np.polyfit(np.log10(x), np.sqrt(scipy.stats.chi2.ppf(1-x,1)), 1)
    c = np.array([-0.29842513,  3.55829518])
    if isinstance(res, np.ndarray):
        res[log10p<-15] = np.polyval(c, log10p[log10p<-15])
        res = np.nan_to_num(res)
        res += 90*(res==0)
    else:
        if log10p<-15:
            res =  np.polyval(c, log10p)
        if np.isnan(res):
            res = 90.
    return res
