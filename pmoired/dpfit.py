from astropy.utils.misc import coffee
import scipy.optimize
import numpy as np
from numpy import linalg
import time
from matplotlib import pyplot as plt
from functools import reduce

"""
IDEA: fit Y = F(X,A) where A is a dictionnary describing the
parameters of the function.

note that the items in the dictionnary should all be scalar!

author: amerand@eso.org

Tue 29 Jan 2013 17:03:21 CLST: working on adding correlations -> NOT WORKING!!!
Thu 28 Feb 2013 12:34:31 CLST: correcting leading to x2 for chi2 display
Mon  8 Apr 2013 10:51:03 BRT: alternate algorithms
Wed Aug 19 14:26:24 UTC 2015: updated randomParam
Wed Jan 24 10:20:06 CET 2018: Python 3 version
Tue 23 Jan 2024 16:33:35 CET: Adding "constant correlation by categories"

http://www.rhinocerus.net/forum/lang-idl-pvwave/355826-generalized-least-squares.html
"""

verboseTime=time.time()
def polyN(x, params):
    """
    Polynomial function. e.g. params={'A0':1.0, 'A2':2.0} returns
    x->1+2*x**2. The coefficients do not have to start at 0 and do not
    need to be continuous. Actually, the way the function is written,
    it will accept any '_x' where '_' is any character and x is a float.
    """
    res = 0
    for k in params.keys():
        res += params[k]*np.array(x)**float(k[1:])
    return res

def example():
    """
    very simple example
    """
    N = 50
    # -- generate data set
    X = np.linspace(-2,3,N)
    Y = -0.7 - 0.4*X + 0.1*X**2 + .1*X**3

    # -- error vector
    E = np.ones(N)*0.3
    np.random.seed(1234) # -- enure repeatibility
    Y += E*np.random.randn(N) # -- random errors

    # -- error covariance matrix
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i==j:
                C[i,j] = E[i]**2
            else:
                C[i,j] = 0.0*E[i]*E[j]

    p0 = {'A0':0.0, 'A1':0., 'A2':0, 'A3':0.0}
    m = ''
    for k in sorted(p0.keys()):
        i = int(k[1:])
        if len(m)>0:
            m+= ' + '
        if i==0:
            m+= k
        else:
            m+= k+'*X^%d'%i
    print('model: Y = '+m)
    # -- do the fit with simple error
    fit = leastsqFit(dpfunc.polyN, X, p0, Y, err=E, verbose=2,
                      doNotFit=[], ftol=1e-5, maxfev=500)

    # -- display data and best fit model
    plt.figure(1)
    plt.clf()
    plt.errorbar(X, Y, yerr=E, label='data', fmt='o')
    plt.plot(X, fit['model'], '.-g', linewidth=2, label='fit')

    # -- show uncertainties in the model, a bit oustside range
    x = np.linspace(X.min()-0.1*X.ptp(), X.max()+0.1*X.ptp(), 100)
    fit = randomParam(fit, N=100, x=x)
    plt.fill_between(x, fit['r_ym1s'], fit['r_yp1s'], color='g',
                     label='fit uncertainty', alpha=0.3)
    plt.legend(loc='upper center')

    # -- do the fit with covariance errors (uses curvefit)
    fitc = leastsqFit(dpfunc.polyN, X, p0, Y, err=C, verbose=0,
                      doNotFit=[], ftol=1e-5, maxfev=500)
    print('nfev=', fitc['info']['nfev'])
    test = 0.
    for k in fitc['best'].keys():
        test += (fit['best'][k]-fitc['best'][k])**2
        test += (fit['uncer'][k]-fitc['uncer'][k])**2
    print('difference leastsq / curve_fit:', test)
    return

def exampleBootstrap(centered=True):
    x = np.linspace(-0.02,1.73,36)
    if centered:
        x0 = x.mean() # reduces correlations
    else:
        x0 = 0 # lead to correlations and biases

    a = {'A0':9085., 'A1':-7736., 'A2':4781.0, 'A3':-1343.}
    y = dpfunc.polyN(x,a)
    a_ = leastsqFit(dpfunc.polyN, x-x0, a, y, verbose=0)['best']
    fits = []
    e = np.ones(len(x))*50
    y += np.random.randn(len(x))*e

    fits = bootstrap(dpfunc.polyN, x-x0, a_, y, err=e, verbose=1,
                     fitOnly=['A0','A1','A2','A3'])

    #-- all bootstraped fit, with average value and error bar:
    plotCovMatrix(fits[1:], fig=0)

    #-- first fit (with all data points), with error elipses:
    plotCovMatrix(fits[0], fig=None)

    #-- true value
    N = len(fits[0]['fitOnly'])
    for i,ki in enumerate(fits[0]['fitOnly']):
        for j,kj in enumerate(fits[0]['fitOnly']):
            plt.subplot(N,N,i+N*j+1)
            xl = plt.xlim()
            if i!=j:
                plt.plot(a_[ki], a_[kj], 'oc', markersize=10, alpha=0.5,
                         label='true')
            else:
                plt.vlines(a_[ki], 0, 0.99*plt.ylim()[1], color='c',
                       linewidth=3, alpha=0.5, label='true')
            plt.legend(loc='center right', prop={'size':7},
                       numpoints=1)

    plt.figure(1)
    plt.clf()
    label='bootstraping'
    for f in fits[1:]:
        plt.plot(x, dpfunc.polyN(x-x0,f['best']),'-', alpha=0.1, color='0.3',
                 label=label)
        label=''
    plt.errorbar(x,y,marker='o',color='k',yerr=e, linestyle='none',
                 label='data')
    plt.plot(x,fits[0]['model'], '-b', linewidth=3, label='all points fit')
    plt.plot(x,dpfunc.polyN(x-x0,a_), '-c', linewidth=3, label='true')

    plt.legend()
    return

def exampleCorrelation():
    """
    very simple example with correlated error bars

    """
    # -- generate fake data:
    N, noise, offset = 100, 0.2, 0.1
    X = np.linspace(0,10,N)
    Y = 0.0 + 1.0*np.sin(2*np.pi*X/1.2)
    Y += noise*np.random.randn(N)
    Y[:N//2] += offset
    Y[N//2:] -= offset

    # -- errors:
    E = np.ones(N)*np.sqrt(noise**2+offset**2)

    # -- covariance matric
    C = np.zeros((N, N))
    # -- diagonal = error
    C[range(N), range(N)] = noise**2
    rho = offset/np.sqrt(noise**2+offset**2)
    # -- non-diag: correlations
    for i in range(N):
        for j in range(N):
            if i!=j:
                if i<N//2 and j<N//2:
                    C[i,j] = rho*noise**2
                elif i>=N//2 and j>=N//2:
                    C[i,j] = rho*noise**2
                else:
                    C[i,j] = -0*rho*noise**2
    #print(np.round(C, 3))
    print('#'*12, 'without correlations', '#'*12)
    E = np.ones(len(X), )*noise
    fit=leastsqFit(dpfunc.fourier, X,
                     {'A0':0.1, 'A1':1.,'PHI1':0., 'WAV':1.2},
                     Y, err=E, verbose=1, normalizedUncer=0)
    plt.figure(0)
    plt.clf()
    plt.errorbar(X,Y,yerr=E,linestyle='none', fmt='.')
    plt.plot(X,fit['model'], color='r')
    plt.title('without correlations')

    print('#'*12, 'with correlations', '#'*12)

    #print np.round(E, 2)
    fit=leastsqFit(dpfunc.fourier, X,
                     {'A0':0.1, 'A1':1.1,'PHI1':0., 'WAV':1.2},
                     Y, err=linalg.inv(C), verbose=1, normalizedUncer=0)
    plt.figure(1)
    plt.clf()
    plt.errorbar(X,Y,yerr=np.sqrt(np.diag(C)),linestyle='none', fmt='.')
    plt.plot(X,fit['model'], color='r')
    plt.title('with correlations')
    return

def meta(x, params):
    """
    allows to call any combination of function defines inside dpfunc:

    params={'funcA;1:p1':, 'funcA;1:p2':,
            'funcA;2:p1':, 'funcA;2:p2':,
            'funcB:p1':, etc}

    funcA and funcB should be defined in dpfunc.py. Allows to call many
    instances of the same function (here funcA) and combine different functions.
    Outputs of the difference functions will be sumed usinf operator '+'. """

    # -- list of functions:
    funcs = set([k.strip().split(':')[0].strip() for k in params.keys()])
    #print funcs

    res = 0
    for f in funcs: # for each function
        # -- keep only relevant keywords
        kz = filter(lambda k: k.strip().split(':')[0].strip()==f, params.keys())
        tmp = {}
        for k in kz:
            # -- build temporary dict pf parameters
            tmp[k.split(':')[1].strip()]=params[k]
        ff = f.split(';')[0].strip() # actual function name
        if not ff in dpfunc.__dict__.keys():
            raise NameError(ff+' not defined in dpfunc')
        # -- add to result the function result
        res += dpfunc.__dict__[ff](x, tmp)
    return res

def invCOVconstRho(n,rho):
    """
    Inverse coveariance matrix nxn, with non diag==r,

    from wolfram alpha, n>=2
    https://www.wolframalpha.com/
    inv({{1,r},{r,1}})
    inv({{1,r,r},{r,1,r},{r,r,1}})
    inv({{1,r,r,r},{r,1,r,r},{r,r,1,r},{r,r,r,1}})
    inv({{1,r,r,r,r},{r,1,r,r,r},{r,r,1,r,r},{r,r,r,1,r},{r,r,r,r,1}})
    inv({{1,r,r,r,r,r},{r,1,r,r,r,r},{r,r,1,r,r,r},{r,r,r,1,r,r},{r,r,r,r,1,r},{r,r,r,r,r,1}})
    inv({{1,r,r,r,r,r,r},{r,1,r,r,r,r,r},{r,r,1,r,r,r,r},{r,r,r,1,r,r,r},{r,r,r,r,1,r,r},{r,r,r,r,r,1,r},{r,r,r,r,r,r,1}})
    etc...
    """
    return (rho*np.ones([n,n]) + (-(n-2)*rho - rho -1 )*np.identity(n))/((n-1)*rho**2 - (n-2)*rho - 1)


def iterable(obj):
    """
    https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

Ncalls=0
Tcalls=0
trackP={}

def leastsqFit(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-6,
               ftol=1e-5, fullOutput=True, normalizedUncer=True,
               follow=None, maxfev=5000, bounds={}, factor=100,
               correlations=None, addKwargs={}):
    """
    - params is a Dict containing the first guess.

    - fits 'y +- err = func(x,params)'. errors are optionnal. in case err is a
      ndarray of 2 dimensions, it is treated as the covariance of the
      errors.

      np.array([[err1**2, 0, .., 0],
                [0, err2**2, 0, .., 0],
                [0, .., 0, errN**2]]) is the equivalent of 1D errors

    - follow=[...] list of parameters to "follow" in the fit, i.e. to print in
      verbose mode

    - fitOnly is a LIST of keywords to fit. By default, it fits all
      parameters in 'params'. Alternatively, one can give a list of
      parameters not to be fitted, as 'doNotFit='

    - doNotFit has a similar purpose: for example if params={'a0':,
      'a1': 'b1':, 'b2':}, doNotFit=['a'] will result in fitting only
      'b1' and 'b2'. WARNING: if you name parameter 'A' and another one 'AA',
      you cannot use doNotFit to exclude only 'A' since 'AA' will be excluded as
      well...

    - normalizedUncer=True: the uncertainties are independent of the Chi2, in
      other words the uncertainties are scaled to the Chi2. If set to False, it
      will trust the values of the error bars: it means that if you grossely
      underestimate the data's error bars, the uncertainties of the parameters
      will also be underestimated (and vice versa).

    - bounds = dictionnary with lower/upper bounds. if bounds are not specified,
        (-inf/inf will be used)

    - verbose:
        True (or 1): show progress
        2: show progress and best fit with errors
        3: show progress, best fit with errors and correlations

    - correlations: dict describing correlations between subsets of data sets (categories).
        {'catg': list or np.array, same size as x. should contain values in [cat1, cat2] below

        'rho':{cat1:rho1, cat2:rho2...} # correlation factor between catg's (-1..1)
        'err':{cat1:err1, cat2:err2...} # uncertainties in catg's

        OR

        'invcov':{catg1:invcov1, catg2:invcov2, ...}
        }

    returns dictionary with:
    'best': bestparam,
    'uncer': uncertainties,
    'chi2': chi2_reduced,
    'model': func(x, bestparam)
    'cov': covariance matrix (normalized if normalizedUncer)
    'fitOnly': names of the columns of 'cov'
    """
    global Ncalls, pfitKeys, pfix, _func, trackP, _addKwargs
    # -- fit all parameters by default
    if fitOnly is None:
        if len(doNotFit)>0:
            fitOnly = filter(lambda x: x not in doNotFit, params.keys())
        else:
            fitOnly = params.keys()
        fitOnly = sorted(list(fitOnly)) # makes some display nicer

    # -- check that all parameters are numbers
    NaNs = []
    for k in fitOnly:
        #if not (type(params[k])==float or type(params[k])==int):
        if not (np.isscalar(params[k]) and type(params[k])!=str):
            NaNs.append(k)
    fitOnly = sorted(list(filter(lambda x: not x in NaNs, fitOnly)))

    # -- build fitted parameters vector:
    pfit = [params[k] for k in fitOnly]

    # -- built fixed parameters dict:
    pfix = {}
    for k in params.keys():
        if k not in fitOnly:
            pfix[k]=params[k]
    if verbose:
        print('[dpfit] %d FITTED parameters:'%len(fitOnly), end=' ')
        if len(fitOnly)<100 or (type(verbose)==int and verbose>1):
             print(fitOnly)
             print('[dpfit] epsfcn=', epsfcn, 'ftol=', ftol)
        else:
            print(' ')

    # -- actual fit
    Ncalls=0
    trackP={}
    t0=time.time()
    mesg=''
    if np.iterable(err) and len(np.array(err).shape)==2:
        if verbose:
            print('[dpfit] using scipy.optimize.curve_fit')
        # -- assumes err matrix is co-covariance
        _func = func
        _addKwargs = addKwargs
        pfitKeys = fitOnly
        plsq, cov = scipy.optimize.curve_fit(_fitFunc2, x, y, pfit,
                        sigma=err, epsfcn=epsfcn, ftol=ftol)
        info, mesg, ier = {'nfev':Ncalls, 'exec time':time.time()-t0}, 'curve_fit', None
    else:
        if (bounds is None or bounds=={}) and correlations is None:
            # ==== LEGACY! ===========================
            if verbose:
                print('[dpfit] using scipy.optimize.leastsq')
            plsq, cov, info, mesg, ier = \
                      scipy.optimize.leastsq(_fitFunc, pfit,
                            args=(fitOnly,x,y,err,func,pfix,verbose,follow),
                            full_output=True, epsfcn=epsfcn, ftol=ftol,
                            maxfev=maxfev, factor=factor)
            info['exec time'] = time.time() - t0
            mesg = mesg.replace('\n', '')
        else:
            #method = 'L-BFGS-B'
            method = 'BFGS'
            #method = 'Nelder-Mead'
            #method = 'Newton-CG'
            #method = 'dogleg'
            if verbose:
                print('[dpfit] using scipy.optimize.minimize (%s)'%method)
            Bounds = []
            if not bounds is None:
                for k in fitOnly:
                    if k in bounds.keys():
                        Bounds.append(bounds[k])
                    else:
                        Bounds.append((-np.inf, np.inf))
            if not correlations is None:
                C = set(correlations['catg'])
                if not 'invcov' in correlations:
                    correlations['invcov'] = {}
                for c in C:
                    if not c in correlations['invcov'] and not c is None:
                        if c in correlations['rho'] and c in correlations['err']:
                            _n = np.sum(np.array(correlations['catg'])==c)
                            correlations['invcov'][c] = invCOVconstRho(_n, correlations['rho'][c])
                            correlations['invcov'][c] /= correlations['err'][c]**2
                        else:
                            #print('warning: cannot compute covariance for "'+str(c)+'"')
                            #correlations['err'][c] = np.mean(err[correlations['catg']==c])
                            pass
            result = scipy.optimize.minimize(_fitFuncMin, pfit,
                            tol=ftol, options={'maxiter':maxfev, 'eps':epsfcn},
                            #bounds = Bounds, method=method,
                            args=(fitOnly,x,y,err,func,pfix,verbose,follow,correlations, addKwargs)
                            )
            plsq = result.x

            cov = result.hess_inv

            # # https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
            # # Do Moore-Penrose inverse discarding zero singular values.
            # _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
            # threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
            # if verbose:
            #     print('[dpfit] zeros in cov?', any(s<=threshold))
            # s = s[s > threshold]
            # VT = VT[:s.size]
            # cov = np.dot(VT.T / s**2, VT)
            info = {'nfev':result.nfev, 'exec time':time.time()-t0}
            mesg, ier = result.message, None
    # -- best fit -> agregate to pfix
    for i,k in enumerate(fitOnly):
        pfix[k] = plsq[i]
    model = func(x,pfix)

    if not correlations is None:
        cov = 2*cov/(len(model)-len(pfit)+1)

    #print('cov', cov)

    if verbose:
        print('[dpfit]', mesg)
        #print('[dpfit] info:', info)
        print('[dpfit]', info['nfev'], 'function calls', end=' ')
        t = 1000*info['exec time']/info['nfev']
        n=-int(np.log10(t))+3
        print('(',round(t, n), 'ms on average)')

    notsig = []
    if cov is None:
        if verbose:
            print('[dpfit] \033[31mWARNING: singular covariance matrix,', end=' ')
            print('uncertainties cannot be computed\033[0m')
        mesg += '; singular covariance matrix'
        #print('       ', info['fjac'].shape)
        # -- try to figure out what is going on!
        delta = np.array(pfit)-np.array(plsq)
        for i,k in enumerate(fitOnly):
            if 'fjac' in info:
                #print("info['ipvt']", info['ipvt'])
                if max(info['ipvt'])<len(info['ipvt']):
                    _i = list(info['ipvt']).index(i)
                else:
                    _i = list(info['ipvt']).index(i+1)
                test = max(np.abs(info['fjac'][_i,:]))==0
            else:
                test = np.abs(delta[i])<=epsfcn
            if test:
                if verbose:
                    print('[dpfit] \033[31m         parameter "'+k+'" does not change CHI2:', end=' ')
                    print('IT CANNOT BE FITTED\033[0m')
                mesg += '; parameter "'+k+'" does not change CHI2'
                notsig.append(k)
        cov = np.zeros((len(fitOnly), len(fitOnly)))

    # -- residuals
    if not correlations is None:
        reducedChi2 = result.fun
        ndof = max(len(model)-len(pfit)+1, 1)
        chi2 = reducedChi2*ndof
    elif np.iterable(err) and len(np.array(err).shape)==2:
        # -- assumes err matrix is co-covariance
        r = y - model
        chi2 = np.dot(np.dot(np.transpose(r), np.linalg.inv(err)), r)
        ndof = max(len(x)-len(pfit)+1, 1)
        reducedChi2 = chi2/ndof
    else:
        tmp = _fitFunc(plsq, fitOnly, x, y, err, func, pfix)
        try:
            chi2 = (np.array(tmp)**2).sum()
        except:
            chi2=0.0
            for x in tmp:
                chi2+=np.sum(x**2)
        try:
            ndof = max(len(model)-len(pfit)+1, 1)
        except:
            ndof = max(np.sum([1 if np.isscalar(i) else len(i) for i in tmp])-len(pfit)+1, 1)
        reducedChi2 = chi2/ndof
        if not np.isscalar(reducedChi2):
            reducedChi2 = np.mean(reducedChi2)

    if normalizedUncer:
        try:
            #print('\033[42m[normalising uncertainties]\033[0m')
            cov *= reducedChi2
        except:
            print('\033[41m[failed to normalise uncertainties]\033[0m')

    # -- uncertainties:
    uncer = {}
    for k in pfix.keys():
        if not k in fitOnly:
            uncer[k]=0 # not fitted, uncertatinties to 0
        else:
            i = fitOnly.index(k)
            if cov is None:
                uncer[k]= -1
            else:
                uncer[k]= np.sqrt(np.abs(np.diag(cov)[i]))

    # -- simple criteria to see if step is too large
    notconverg = []
    for k in filter(lambda x: x!='reduced chi2', trackP.keys()):
        n = len(trackP[k])
        std2 = np.std(trackP[k][(3*n)//4:])
        ptp2 = np.ptp(trackP[k][(3*n)//4:])
        if std2>2*uncer[k] and not k in notsig:
            notconverg.append(k)
    if len(notconverg) and verbose:
        print('[dpfit] \033[33mParameters', notconverg,
              'may not be converging properly\033[0m')
        print('[dpfit] \033[33mcheck with "showFit" '+
              '(too sensitive to relative variations?)\033[0m')

    if type(verbose)==int and verbose>1:
        #print('-'*30)
        print('# --     CHI2=', chi2)
        print('# -- red CHI2=', reducedChi2)
        print('# --     NDOF=', int(chi2/reducedChi2))
        #print('-'*30)
        dispBest({'best':pfix, 'uncer':uncer, 'fitOnly':fitOnly})

    # -- result:
    if fullOutput:
        cor = np.sqrt(np.diag(cov))
        cor = cor[:,None]*cor[None,:]
        cor[cor==0] = 1e-6
        cor = cov/cor
        for k in trackP.keys():
            trackP[k] = np.array(trackP[k])

        pfix= {'func':func,
               'best':pfix, 'uncer':uncer,
               'chi2':reducedChi2, 'model':model,
               'cov':cov, 'fitOnly':fitOnly,
               'epsfcn':epsfcn, 'ftol':ftol,
               'info':info, 'cor':cor, 'x':x, 'y':y, 'ndof':ndof,
               'doNotFit':doNotFit,
               'covd':{ki:{kj:cov[i,j] for j,kj in enumerate(fitOnly)}
                         for i,ki in enumerate(fitOnly)},
               'cord':{ki:{kj:cor[i,j] for j,kj in enumerate(fitOnly)}
                         for i,ki in enumerate(fitOnly)},
               'normalized uncertainties':normalizedUncer,
               'maxfev':maxfev, 'firstGuess':params,
               'track':trackP, 'mesg':mesg,
               'not significant':notsig,
               'not converging':notconverg,
               'factor':factor,
        }
        if type(verbose)==int and verbose>2 and np.size(cor)>1:
            dispCor(pfix)
    return pfix

def randomParam(fit, N=None, x='auto'):
    """
    get a set of randomized parameters (list of dictionnaries) around the best
    fited value, using a gaussian probability, taking into account the correlations
    from the covariance matrix.

    fit is the result of leastsqFit (dictionnary)

    returns a fit dictionnary with: 'ymin', 'ymax' and 'r_param' (a list of the
    randomized parameters)
    """
    if N is None:
        N = len(fit['x'])
    m = np.array([fit['best'][k] for k in fit['fitOnly']])
    res = [] # list of dictionnaries
    for k in range(N):
        p = dict(zip(fit['fitOnly'],np.random.multivariate_normal(m, fit['cov'])))
        p.update({k:fit['best'][k] for k in fit['best'].keys() if not k in
                 fit['fitOnly']})
        res.append(p)
    ymin, ymax = None, None
    tmp = []
    if type(x)==str and x == 'auto':
        x = fit['x']
    if not x is None:
        for r in res:
            tmp.append(fit['func'](x, r))
        tmp = np.array(tmp)
        fit['all_y'] = tmp
        fit['r_y'] = fit['func'](x, fit['best'])
        fit['r_ym1s'] = np.percentile(tmp, 16, axis=0)
        fit['r_yp1s'] = np.percentile(tmp, 84, axis=0)

    fit['r_param'] = res
    fit['r_x'] = x
    return fit

randomParam = randomParam # legacy

def bootstrap(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-6,
               ftol=1e-5, fullOutput=True, normalizedUncer=True,
               follow=None, Nboot=None):
    """
    bootstraping, called like leastsqFit. returns a list of fits: the first one
    is the 'normal' one, the Nboot following one are with ramdomization of data. If
    Nboot is not given, it is set to 10*len(x).
    """
    if Nboot==None:
        Nboot = 10*len(x)

    if 'fitOnly' in params and fitOnly is None:
        fitOnly = params['fitOnly']
    if 'best' in params:
        params = params['best']

    # first fit is the "normal" one
    fits = [leastsqFit(func, x, params, y,
                                     err=err, fitOnly=fitOnly, verbose=False,
                                     doNotFit=doNotFit, epsfcn=epsfcn,
                                     ftol=ftol, fullOutput=True,
                                     normalizedUncer=True)]
    for k in range(Nboot):
        s = np.int_(len(x)*np.random.rand(len(x)))
        fits.append(leastsqFit(func, x[s], params, y[s],
                                     err=err, fitOnly=fitOnly, verbose=False,
                                     doNotFit=doNotFit, epsfcn=epsfcn,
                                     ftol=ftol, fullOutput=True,
                                     normalizedUncer=True))
    return fits

def showBootstrap(boot, fig=1, fontsize=8):
    global _AX, _AY
    plt.close(fig); plt.figure(fig)
    n = len(boot[0]['fitOnly'])
    _AX, _AY = {}, {}
    for i1, k1 in enumerate(sorted(boot[0]['fitOnly'])):
        _AX[k1] = plt.subplot(n,n,i1*n+i1+1)
        _AX[k1].yaxis.set_visible(False)
        X = [b['best'][k1] for b in boot]
        bins = max(int(np.sqrt(len(X))), 10)
        # -- bootstraped fits
        h = plt.hist(X[1:], color='k', histtype='step', density=True, bins=bins)
        h = plt.hist(X[1:], color='0.9', density=True, bins=bins)
        # -- fit to all data
        plt.errorbar(X[0], 0.45*np.max(h[0]), xerr=boot[0]['uncer'][k1], color='orange',
                     marker='s', capsize=5, label='all data')
        # -- 1 sigma asymetric uncertainties
        X0, Xm, XM = np.percentile(X[1:], 50), np.percentile(X[1:], 16), np.percentile(X[1:], 100-16)
        Xm, XM = X0-Xm, XM-X0
        plt.errorbar(X0, 0.55*np.max(h[0]), xerr=([Xm], [XM]), color='blue',
                     marker='d', capsize=5, label='bootstrapped')
        d = int(2-np.round(.5*(np.log10(Xm)+np.log10(XM)), 0))
        fmt = '%s=\n$%.'+str(d)+'f^{+%.'+str(d)+'f}_{-%.'+str(d)+'f}$'
        plt.title(fmt%(k1, X0, Xm, XM), fontsize=1.2*fontsize)
        if i1!=len(sorted(boot[0]['fitOnly']))-1:
            _AX[k1].xaxis.set_visible(False)
        else:
            _AX[k1].tick_params(axis='x', labelsize=fontsize)
            #_AX[k1].callbacks.connect('xlim_changed', _callbackAxesBoot)

        plt.legend(loc='upper right', fontsize=0.6*fontsize)

    for i1, k1 in enumerate(sorted(boot[0]['fitOnly'])):
        for i2, k2 in enumerate(sorted(boot[0]['fitOnly'])):
            if i1<i2:
                if i1==0:
                    _AY[k2] = plt.subplot(n,n,i2*n+i1+1, sharex=_AX[k1])
                    _AY[k2].callbacks.connect('ylim_changed', _callbackAxesBoot)
                    plt.ylabel(k2, fontsize=1.2*fontsize)
                    ax = _AY[k2]
                    ax.tick_params(axis='y', labelsize=fontsize)
                else:
                    ax = plt.subplot(n,n,i2*n+i1+1, #sharex=_AX[k1],
                                     sharey=_AY[k2])
                    ax.yaxis.set_visible(False)

                X, Y = [b['best'][k1] for b in boot], [b['best'][k2] for b in boot]

                # -- bootstrap
                plt.plot(X[1:], Y[1:], '.k', alpha=0.1)
                plt.plot(np.median(X[1:]), np.median(Y[1:]), 'db', alpha=0.5)
                # -- error ellipse
                cov = np.cov([X[1:], Y[1:]])
                t = np.linspace(0,2*np.pi,100)
                sMa, sma, a = _ellParam(cov[0,0], cov[1,1], cov[0,1])
                Xe,Ye = sMa*np.cos(t), sma*np.sin(t)
                Xe,Ye = Xe*np.cos(a)+Ye*np.sin(a),-Xe*np.sin(a)+Ye*np.cos(a)
                plt.plot(np.median(X[1:])+Xe, np.median(Y[1:])+Ye, '-b', alpha=0.5)

                # -- fit to all data:
                plt.plot(X[0], Y[0], marker='s', color='orange', alpha=0.5)
                # -- err ellipse
                ell = errorEllipse(boot[0], k1, k2)
                plt.plot(ell[0], ell[1], '-', color='orange', alpha=0.5)

                if i2!=len(sorted(boot[0]['fitOnly']))-1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.tick_params(axis='x', labelsize=fontsize)

    plt.subplots_adjust(wspace=0, hspace=0)

def _callbackAxesBoot(ax):
    i = None
    for k in _AY.keys():
        if ax==_AY[k]:
            i = k
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


def randomize(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-6,
               ftol=1e-5, fullOutput=True, normalizedUncer=True,
               follow=None, Nboot=None):
    """
    bootstraping, called like leastsqFit. returns a list of fits: the first one
    is the 'normal' one, the Nboot following one are with ramdomization of data. If
    Nboot is not given, it is set to 10*len(x).
    """
    if Nboot==None:
        Nboot = 10*len(x)
    # first fit is the "normal" one
    fits = [leastsqFit(func, x, params, y,
                                     err=err, fitOnly=fitOnly, verbose=False,
                                     doNotFit=doNotFit, epsfcn=epsfcn,
                                     ftol=ftol, fullOutput=True,
                                     normalizedUncer=True)]
    for k in range(Nboot):
        s = err*np.random.randn(len(y))
        fits.append(leastsqFit(func, x, params, y+s,
                                     err=err, fitOnly=fitOnly, verbose=False,
                                     doNotFit=doNotFit, epsfcn=epsfcn,
                                     ftol=ftol, fullOutput=True,
                                     normalizedUncer=True))
    return fits

def _fitFunc(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None):
    """
    interface  scipy.optimize.leastsq:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors
    """
    global verboseTime, Ncalls
    Ncalls+=1

    params = {}
    # -- build dic from parameters to fit and their values:
    for i,k in enumerate(pfitKeys):
        params[k]=pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k]=pfix[k]
    if err is None:
        err = np.ones(np.array(y).shape)

    # -- compute residuals
    if (type(y)==np.ndarray and type(err)==np.ndarray) or \
            (np.isscalar(y) and type(err)==np.ndarray) or \
            (type(y)==np.ndarray and np.isscalar(err)) or \
            (np.isscalar(y) and np.isscalar(err)):
        model = func(x, params)
        res = ((np.array(y)-model)/err).flatten()
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x,params)
        if np.isscalar(err):
            err = 0*y + err
        #print 'DEBUG:', tmp.shape, y.shape, err.shape

        for k in range(len(y)):
            df = (np.array(tmp[k])-np.array(y[k]))/np.array(err[k])
            try:
                res.extend(list(df))
            except:
                res.append(df)

    try:
        chi2= np.sum(res**2)/max(len(res)-len(pfit)+1.0, 1)
    except:
        # list of elements
        chi2 = 0
        N = 0
        #res2 = []
        for r in res:
            if np.isscalar(r):
                chi2 += r**2
                N+=1
                r#es2.append(r)
            else:
                chi2 += np.sum(np.array(r)**2)
                N+=len(r)
                #res2.extend(list(r))
        chi2 /= max(N-len(pfit)+1.0, 1)

    if verbose and time.time()>(verboseTime+10):
        verboseTime = time.time()
        print('[dpfit]', time.asctime(), '%03d/%03d'%(Ncalls, int(Ncalls/len(pfit))), end=' ')
        print('CHI2: %6.4e'%chi2,end='|')
        if follow is None:
            print('')
        else:
            _follow = list(filter(lambda x: x in params.keys() and
                            type(params[x]) in [float, np.double], follow))
            print('|'.join([k+'='+'%5.2e'%params[k] for k in _follow]))
    for i,k in enumerate(pfitKeys):
        if not k in trackP:
            trackP[k] = [pfit[i]]
        else:
            trackP[k].append(pfit[i])
    if not 'reduced chi2' in trackP:
        trackP['reduced chi2'] = [chi2]
    else:
        trackP['reduced chi2'].append(chi2)

    return res

def _fitFuncMin(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False,
                follow=None, correlations=None, addKwargs={}, doTrackP=True):
    """
    interface  scipy.optimize.minimize:
    - x,y,err are the data to fit: f(x) = y +- err
    - pfit is a list of the paramters
    - pfitsKeys are the keys to build the dict
    pfit and pfix (optional) and combines the two
    in 'A', in order to call F(X,A)

    in case err is a ndarray of 2 dimensions, it is treated as the
    covariance of the errors.
    np.array([[err1**2, 0, .., 0],
             [ 0, err2**2, 0, .., 0],
             [0, .., 0, errN**2]]) is the equivalent of 1D errors

    correlations: dict describing correlations between subsets of data sets (categories). every
        categories should have either an inverse covariance or an error. function's parameter "err"
        will be ignored

        {'catg': list or np.array, same size as x. should contain values in [cat1, cat2] below
        'invcov':{catg1:invcov1, catg2:invcov2, ...},
        'err':{catg3:err3, ...}
        }

    """
    #print('_fitFuncMin', type(err))
    global verboseTime, Ncalls, trackP
    Ncalls+=1

    params = {}
    # -- build dic from parameters to fit and their values:
    for i,k in enumerate(pfitKeys):
        params[k]=pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k]=pfix[k]
    if err is None:
        #print('err is None')
        if not correlations is None:
            err = np.ones(len(correlations['catg']))
        else:
            err = np.ones(np.array(y).shape)

    # -- compute residuals
    if (type(y)==np.ndarray and type(err)==np.ndarray) or \
        np.isscalar(y):
        model = func(x, params, **addKwargs)
        if not err is None:
            # -- residuals
            res = ((np.array(y)-model)/err).flatten()
            # -- errors will be handled by the covariance (bellow)
            if not correlations is None and type(err)==np.ndarray and type(y)==np.ndarray:
                for k in correlations['err']:
                    w = np.where(correlations['catg']==k)
                    res[w] *= err[w]
        else:
            res = (np.array(y)-model).flatten()
    else:
        # much slower: this time assumes y (and the result from func) is
        # a list of things, each convertible in np.array
        res = []
        tmp = func(x, params, **addKwargs)
        if np.isscalar(err):
            err = 0*res + err

        for k in range(len(y)):
            df = (np.array(tmp[k])-np.array(y[k]))/np.array(err[k])
            try:
                res.extend(list(df))
            except:
                res.append(df)

    # -- compute chi2
    if correlations is None:
        try:
            chi2=(res**2).sum/(len(res)-len(pfit)+1.0)
        except:
            # list of elements
            chi2 = 0
            N = 0
            res2 = []
            for r in res:
                if np.isscalar(r):
                    chi2 += r**2
                    N+=1
                    res2.append(r)
                else:
                    chi2 += np.sum(np.array(r)**2)
                    N+=len(r)
                    res2.extend(list(r))
            res = res2
            chi2 /= float(N-len(pfit)+1)
    else:
        # -- take into correlations using "correlations" dict
        chi2 = 0
        res = np.array(res)
        for c in set(correlations['catg']):
            w = np.array(correlations['catg'])==c
            if c in correlations['invcov']:
                #print('invcov')
                chi2 += np.dot(np.dot(np.transpose(res[w]), correlations['invcov'][c]), res[w])
            elif c in correlations['err']:
                #print("correlations['err']")
                chi2 += np.sum(res[w]**2/correlations['err'][c]**2)
            else:
                # -- already in the residuals computation!
                chi2 += np.sum(res[w]**2)
        # -- reduced chi2
        chi2 /= float(len(res)-len(pfit)+1)

    if verbose and time.time()>(verboseTime+10):
        verboseTime = time.time()
        print('[dpfit]', time.asctime(), '%5d'%Ncalls,end='')
        print('CHI2: %6.4e'%chi2,end='|')
        if follow is None:
            print('')
        else:
            _follow = list(filter(lambda x: x in params.keys(), follow))
            print('|'.join([k+'='+'%5.2e'%params[k] for k in _follow]))
    if doTrackP:
        for i,k in enumerate(pfitKeys):
            if not k in trackP:
                trackP[k] = [pfit[i]]
            else:
                trackP[k].append(pfit[i])
        if not 'reduced chi2' in trackP:
            trackP['reduced chi2'] = [chi2]
        else:
            trackP['reduced chi2'].append(chi2)
    return chi2

def _fitFunc2(x, *pfit, verbose=True, follow=[], errs=None):
    """
    for curve_fit
    """
    global Ncalls, verboseTime
    Ncalls +=1
    params = {}
    # -- build dic from parameters to fit and their values:
    for i,k in enumerate(pfitKeys):
        params[k]=pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k]=pfix[k]

    res = _func(x, params)

    if verbose and time.time()>(verboseTime+10):
        verboseTime = time.time()
        print('[dpfit]', time.asctime(), '%5d'%Ncalls,end='')
        try:
            chi2=np.sum(res**2)/(len(res)-len(pfit)+1.0)
            print('CHI2: %6.4e'%chi2,end='')
        except:
            # list of elements
            chi2 = 0
            N = 0
            res2 = []
            for r in res:
                if np.isscalar(r):
                    chi2 += r**2
                    N+=1
                    res2.append(r)
                else:
                    chi2 += np.sum(np.array(r)**2)
                    N+=len(r)
                    res2.extend(list(r))

            res = res2
            print('CHI2: %6.4e'%(chi2/float(N-len(pfit)+1)), end=' ')
        if follow is None:
            print('')
        else:
            _follow = list(filter(lambda x: x in params.keys(), follow))
            print(' '.join([k+'='+'%5.2e'%params[k] for k in _follow]))
    return res

def errorEllipse(fit, p1, p2, n=100):
    """
    fit is a result from leastsqFit (dict)
    p1, p2 are parameters name (str)
    n number of point in ellipse (int, default 100)

    returns ellipse of errors (x1, x2), computed from the covariance. The n values are centered
    around fit['best']['p1'] and fit['best']['p2']
    """
    t = np.linspace(0,2*np.pi,n)
    if 'covd' in fit:
      sMa, sma, a = _ellParam(fit['covd'][p1][p1], fit['covd'][p2][p2], fit['covd'][p1][p2])
    else:
      i1 = fit['fitOnly'].index(p1)
      i2 = fit['fitOnly'].index(p2)
      sMa, sma, a = _ellParam(fit['cov'][i1,i1], fit['cov'][i2,i2], fit['cov'][i1,i2])

    X,Y = sMa*np.cos(t), sma*np.sin(t)
    X,Y = X*np.cos(a)+Y*np.sin(a),-X*np.sin(a)+Y*np.cos(a)
    return fit['best'][p1]+X, fit['best'][p2]+Y

def _ellParam(sA2, sB2, sAB):
    """
    sA2 is the variance of param A
    sB2 is the variance of param B
    sAB = rho*sA*sB the diagonal term (rho: correlation)

    returns the semi-major axis, semi-minor axis and orientation (in rad) of the
    ellipse.

    sMa, sma, a = ellParam(...)

    t = np.linspace(0,2*np.pi,100)
    X,Y = sMa*np.cos(t), sma*np.sin(t)
    X,Y = X*np.cos(a)+Y*np.sin(a), Y*np.cos(a)-X*np.sin(a)

    ref: http://www.scribd.com/doc/50336914/Error-Ellipse-2nd
    """
    a = np.arctan2(2*sAB, (sB2-sA2))/2
    sMa = np.sqrt(1/2.*(sA2+sB2-np.sqrt((sA2-sB2)**2+4*sAB**2)))
    sma = np.sqrt(1/2.*(sA2+sB2+np.sqrt((sA2-sB2)**2+4*sAB**2)))

    return sMa, sma, a

def dispBest(fit, pre='', asStr=False, asDict=True, color=True, showOnly=None):
    #tmp = sorted(fit['best'].keys())
    # -- fitted param:
    tmp = sorted(fit['fitOnly'])
    # -- unfitted:
    tmp += sorted(list(filter(lambda x: x not in fit['fitOnly'], fit['best'].keys())))

    if not showOnly is None:
        tmp = showOnly

    uncer = fit['uncer']
    if 'uncer+' in fit and 'uncer-' in fit:
        uncerp = fit['uncer+']
        uncerm = fit['uncer-']
    else:
        uncerp, uncerm = None, None

    pfix = fit['best']

    res = ''

    maxLength = np.max(np.array([len(k) for k in tmp]))
    if asDict:
        format_ = "'%s':"
    else:
        format_ = "%s"
    # -- write each parameter and its best fit, as well as error
    # -- writes directly a dictionary
    for ik,k in enumerate(tmp):
        padding = ' '*(maxLength-len(k))
        formatS = format_+padding
        if ik==0 and asDict:
            formatS = pre+'{'+formatS
        else:
            formatS = pre+formatS
        if uncer[k]>0:
            if uncerp is None:
                if uncer[k]>0 and np.isfinite(uncer[k]):
                    ndigit = max(-int(np.log10(uncer[k]))+2, 0)
                else:
                    ndigit = 1

                if asDict:
                    fmt = '%.'+str(ndigit)+'f, # +/- %.'+str(ndigit)+'f'
                else:
                    fmt = '%.'+str(ndigit)+'f +/- %.'+str(ndigit)+'f'
                if color:
                    col = ('\033[94m', '\033[0m')
                else:
                    col = ('', '')
                res += col[0]+formatS%k+fmt%(pfix[k], uncer[k])+col[1]+'\n'
            else:
                ndigit = max(-int(np.log10(uncerp[k]))+2, 0)
                ndigit = max(-int(np.log10(uncerm[k]))+2, ndigit)
                check = 2*np.abs(uncerp[k]-uncerm[k])/(uncerp[k]+uncerm[k]) < 0.2
                if check:
                    if asDict:
                        fmt = '%.'+str(ndigit)+'f, # +/- %.'+str(ndigit)+'f'
                    else:
                        fmt = '%.'+str(ndigit)+'f +/- %.'+str(ndigit)+'f'
                    if color:
                        col = ('\033[94m', '\033[0m')
                    else:
                        col = ('', '')
                    res += col[0]+formatS%k+fmt%(pfix[k], uncerp[k])+col[1]+'\n'
                else:
                    if asDict:
                        fmt = '%.'+str(ndigit)+'f, # +%.'+str(ndigit)+'f -%.'+str(ndigit)+'f'
                    else:
                        fmt = '%.'+str(ndigit)+'f +%.'+str(ndigit)+'f -%.'+str(ndigit)+'f'
                    if color:
                        col = ('\033[94m', '\033[0m')
                    else:
                        col = ('', '')
                    res += col[0]+formatS%k+fmt%(pfix[k], uncerp[k], uncerm[k])+col[1]+'\n'

            #print(formatS%k, fmt%(pfix[k], uncer[k]))
        elif uncer[k]==0:
            if color:
                if k in fit['fitOnly']:
                    # fit did not converge...
                    col = ('\033[91m', '\033[0m')
                else:
                    col = ('\033[97m', '\033[0m')
            else:
                col = ('', '')
            if isinstance(pfix[k], str):
                #print(formatS%k , "'"+pfix[k]+"'", ',')
                res += col[0]+formatS%k+"'"+pfix[k]+"',"+col[1]+'\n'
            else:
                #print(formatS%k , pfix[k], ',')
                res += col[0]+formatS%k+str(pfix[k])+','+col[1]+'\n'
        else:
            #print(formatS%k , pfix[k], end='')
            res += formatS%k+pfix[k]
            if asDict:
                #print(', # +/-', uncer[k])
                res += '# +/- '+str(uncer[k])+'\n'
            else:
                #print('+/-', uncer[k])
                res += '+/- '+str(uncer[k])+'\n'
    if asDict:
        #print(pre+'}') # end of the dictionnary
        res += pre+'}\n'
    if asStr:
        return res
    else:
        print(res)
        return

def dispCor(fit, ndigit=2, pre='', asStr=False, html=False, maxlen=140):
    # -- parameters names:
    nmax = np.max([len(x) for x in fit['fitOnly']])
    # -- compact display if too wide
    if not maxlen is None and nmax+(ndigit+2)*len(fit['fitOnly'])+4>maxlen:
        ndigit=1
    if not maxlen is None and nmax+(ndigit+2)*len(fit['fitOnly'])+4>maxlen:
        ndigit=0

    fmt = '%%%ds'%nmax
    #fmt = '%2d:'+fmt
    fmtd = '%'+'%d'%(ndigit+3)+'.'+'%d'%ndigit+'f'

    if len(fit['fitOnly'])>100:
        # -- hexadecimal
        count = lambda i: '%2s'%hex(i)[2:]
    else:
        # -- decimal
        count = lambda i: '%2d'%i

    if not asStr:
        print(pre+'Correlations (%) ', end=' ')
        print('\033[45m>=95\033[0m', end=' ')
        print('\033[41m>=90\033[0m', end=' ')
        print('\033[43m>=80\033[0m', end=' ')
        print('\033[46m>=60\033[0m', end=' ')
        #print('\033[0m>=20\033[0m', end=' ')
        print('\033[37m<60%\033[0m')
        print(pre+' '*(3+nmax), end=' ')
        for i in range(len(fit['fitOnly'])):
            #print('%2d'%i+' '*(ndigit-1), end=' ')
            if i%2 and ndigit<2:
                c = '\033[47m'
            else:
                c = '\033[0m'
            print(c+count(i)+'\033[0m'+' '*(ndigit), end='')
        print(pre+'')
    else:
        if html:
            res = '<table style="width:100%" border="1">\n'
            res += '<tr>'
            res += '<th>'+pre+' '*(2+ndigit+nmax)+' </th>'
            for i in range(len(fit['fitOnly'])):
                res += '<th>'+'%2d'%i+' '*(ndigit)+'</th>'
            res += '</tr>\n'
        else:
            res = pre+' '*(3+ndigit//2+nmax)+' '
            for i in range(len(fit['fitOnly'])):
                #res += '%2d'%i+' '*(ndigit)
                res += count(i)+' '*(ndigit)
            res += '\n'

    for i,p in enumerate(fit['fitOnly']):
        if i%2 and ndigit<2:
            c = '\033[47m'
        else:
            c = '\033[0m'
        if not asStr:
            print(pre+c+count(i)+':'+fmt%(p)+'\033[0m', end=' ')
        elif html:
            res += '<tr>\n<td>'+pre+fmt%(i,p)+'</td >\n'
        else:
            res += pre+fmt%(i,p)+' '

        for j, x in enumerate(fit['cor'][i,:]):
            if i==j:
                c = '\033[2m'
            else:
                c = '\033[0m'
            hcol = '#FFFFFF'
            if i!=j:
                if abs(x)>=0.95:
                    col = '\033[45m'
                    hcol= '#FF66FF'
                elif abs(x)>=0.9:
                    col = '\033[41m'
                    hcol = '#FF6666'
                elif abs(x)>=0.8:
                    col = '\033[43m'
                    hcol = '#FFEE66'
                elif abs(x)>=0.6:
                    col = '\033[46m'
                    hcol = '#CCCCCC'
                elif abs(x)<0.6:
                    col = '\033[37m'
                    hcol = '#FFFFFF'
                else:
                    col = ''
            elif i%2 and ndigit<2:
                col = '\033[47m'
            else:
                col = '\033[0m'
            # tmp = fmtd%x
            # tmp = tmp.replace('0.', '.')
            # tmp = tmp.replace('1.'+'0'*ndigit, '1.')
            # if i==j:
            #     tmp = '#'*(2+ndigit)
            # print(c+col+tmp+'\033[0m', end=' ')
            if i==j:
                tmp = '#'*(1+ndigit)
            else:
                if ndigit==0:
                    #tmp = '%2d'%int(round(10*x, 0))
                    if x<0:
                        tmp = '-'
                    else:
                        tmp = '+'
                if ndigit==1:
                    #tmp = '%2d'%int(round(10*x, 0))
                    if x<0:
                        tmp = '--'
                    else:
                        tmp = '++'
                elif ndigit==2:
                    tmp = '%3d'%int(round(100*x, 0))

            if not asStr:
                print(c+col+tmp+'\033[0m', end=' ')
            elif html:
                res += '<td bgcolor="%s">'%hcol+tmp+'</td>\n'
            else:
                res += tmp+' '
        if not asStr:
            print('')
        elif html:
            res += '</tr>\n'
        else:
            res += '\n'
    if html:
        res += '</table>\n'
    if asStr:
        return res

def plotCovMatrix(fit, fig=0):
    if not fig is None:
        plt.figure(fig)
        plt.clf()
    else:
        # overplot
        pass

    t = np.linspace(0,2*np.pi,100)
    if isinstance(fit , dict):
        fitOnly = fit['fitOnly']
        N = len(fit['fitOnly'])
    else:
        fitOnly = fit[0]['fitOnly']
        N = len(fit[0]['fitOnly'])

    for i in range(N):
        for j in range(N):
            if i!=j:
                ax = plt.subplot(N, N, i+j*N+1)
                if isinstance(fit , dict):
                    sMa, sma, a = _ellParam(fit['cov'][i,i], fit['cov'][j,j], fit['cov'][i,j])
                    X,Y = sMa*np.cos(t), sma*np.sin(t)
                    X,Y = X*np.cos(a)+Y*np.sin(a),-X*np.sin(a)+Y*np.cos(a)
                    plt.errorbar(fit['best'][fitOnly[i]],
                                 fit['best'][fitOnly[j]],
                                 xerr=np.sqrt(fit['cov'][i,i]),
                                 yerr=np.sqrt(fit['cov'][j,j]), color='b',
                                 linewidth=1, alpha=0.5, label='single fit')
                    plt.plot(fit['best'][fitOnly[i]]+X,
                                 fit['best'][fitOnly[j]]+Y,'-b',
                                 label='cov. ellipse')
                else: ## assumes case of bootstraping
                    plt.plot([f['best'][fitOnly[i]] for f in fit],
                             [f['best'][fitOnly[j]] for f in fit],
                             '.', color='0.5', alpha=0.4, label='bootstrap')
                    plt.errorbar(np.mean([f['best'][fitOnly[i]] for f in fit]),
                                 np.mean([f['best'][fitOnly[j]] for f in fit]),
                                 xerr=np.mean([f['uncer'][fitOnly[i]] for f in fit]),
                                 yerr=np.mean([f['uncer'][fitOnly[j]] for f in fit]),
                                 color='k', linewidth=1, alpha=0.5,
                                 label='boot. avg')
                #plt.legend(loc='upper right', prop={'size':7}, numpoints=1)
                if not fig is None:
                    if isinstance(fit , dict):
                        if j==N-1 or j+1==i:
                            plt.xlabel(fitOnly[i])
                        if i==0 or j+1==i:
                            plt.ylabel(fitOnly[j])
                    else:
                        if j==N-1:
                            plt.xlabel(fitOnly[i])
                        if i==0:
                            plt.ylabel(fitOnly[j])

            if i==j and not isinstance(fit , dict):
                ax = plt.subplot(N, N, i+j*N+1)
                X = [f['best'][fitOnly[i]] for f in fit]
                h = plt.hist(X, color='0.8',bins=max(len(fit)/30, 3))
                a = {'MU':np.median(X), 'SIGMA':np.std(X), 'AMP':len(X)/10.}
                g = leastsqFit(dpfunc.gaussian, 0.5*(h[1][1:]+h[1][:-1]), a, h[0])
                plt.plot(0.5*(h[1][1:]+h[1][:-1]), g['model'], 'r')
                plt.errorbar(g['best']['MU'], g['best']['AMP']/2,
                             xerr=g['best']['SIGMA'], color='r',
                             marker='o', label='gauss fit')
                plt.text(g['best']['MU'], 1.1*g['best']['AMP'],
                         r'%s = %4.2e $\pm$ %4.2e'%(fitOnly[i],
                                                g['best']['MU'],
                                                g['best']['SIGMA']),
                         color='r', va='center', ha='center')
                print('%s = %4.2e  +/-  %4.2e'%(fitOnly[i],
                                                g['best']['MU'],
                                                g['best']['SIGMA']))
                plt.ylim(0,max(plt.ylim()[1], 1.2*g['best']['AMP']))
                if not fig is None:
                    if j==N-1:
                        plt.xlabel(fitOnly[i])
                    if i==0:
                        plt.ylabel(fitOnly[j])
                plt.legend(loc='lower center', prop={'size':7},
                           numpoints=1)
            #--
            try:
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(8)
            except:
                pass
    return

def factors(n):
    """
    returns the factirs for integer n
    """
    return list(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

def subp(N, imax=None):
    """
    gives the dimensions of a gird of plots for N plots, allow up to imax empty
    plots to have a squarish grid
    """
    if N==2:
        return (1,2)
    if imax is None:
        imax = max(2, N//5)
    S = {}
    for i in range(imax+1):
        F = np.array(factors(N+i))
        S[N+i] = sorted(F[np.argsort(np.abs(F-np.sqrt(N)))][:2])
        if N+i==int(np.sqrt(N+i))**2:
            return [int(np.sqrt(N+i)), int(np.sqrt(N+i))]
    # -- get the one with most squarish aspect ratio
    K = list(S.keys())
    R = [S[k][1]/S[k][0] for k in K]
    k = K[np.argmin(R)]
    return S[K[np.argmin(R)]]

def _callbackAxes(ax):
    """
    make sure y ranges follows the data, after x range has been adjusted
    """
    xlim = ax.get_xlim()
    x = np.arange(len(T['reduced chi2']))
    w = (x>=xlim[0])*(x<=xlim[1])
    for k in AX.keys():
        if k=='reduced chi2' and np.max(T[k][w])/np.min(T[k][w])>100:
            AX[k].set_yscale('log')
            AX[k].set_ylim(0.9*np.min(T[k][w]),
                           1.1*np.max(T[k][w]))
        else:
            AX[k].set_yscale('linear')
            AX[k].set_ylim(np.min(T[k][w])-0.1*np.ptp(T[k][w]),
                           np.max(T[k][w])+0.1*np.ptp(T[k][w]))
    return

def showFit(fit, fig=99):
    """
    plot the evolution of the fitted parameters as function of iteration,
    as well as chi2
    """
    global AX, T
    plt.close(fig)
    plt.figure(fig, figsize=(8, 5))
    S = subp(len(fit['track']))
    #print(len(fit['track']), S)
    fontsize = min(max(12/np.sqrt(S[1]), 5), 10)

    # -- plot chi2
    k = 'reduced chi2'
    AX = {k:plt.subplot(S[1], S[0], 1)}
    T = fit['track']
    plt.plot(fit['track'][k], '.-g')
    #plt.ylabel(k, fontsize=fontsize)
    plt.title(k, fontsize=fontsize, x=0.02, y=0.9, ha='left', va='top',
            bbox={'color':'w', 'alpha':0.1})
    if fit['track'][k][0]/fit['track'][k][1] >10:
        plt.yscale('log')
    if S[0]>1:
        AX[k].xaxis.set_visible(False)
    plt.yticks(fontsize=fontsize)
    plt.hlines(fit['chi2'], 0, len(fit['track'][k])-1, alpha=0.5, color='orange')
    if fit['track'][k][0]/fit['track'][k][-1]>100:
        AX[k].set_yscale('log')

    # -- plot all parameters:
    for i, k in enumerate(sorted(filter(lambda x: x!='reduced chi2', fit['track'].keys()))):

        r = np.arange(len(fit['track'][k]))

        AX[k] = plt.subplot(S[1], S[0], i+2, sharex=AX['reduced chi2'])

        # -- evolution of parameters
        if k in fit['not significant'] or k in fit['not converging']:
            color = (0.8, 0.3, 0)
        else:
            color=(0, 0.4, 0.8)
        plt.plot(r, fit['track'][k], '-', color=color)
        # -- when parameter converged within uncertainty
        w = np.abs(fit['track'][k] - fit['best'][k])<=fit['uncer'][k]
        # -- from the end of the sequence, when parameter was within error
        r0 = np.argmax([all(w[x:]) for x in range(len(w))])

        #r0 = r[w][::-1][np.argmax((np.diff(r[w])!=1)[::-1])]
        #plt.plot(r[w], fit['track'][k][w], 'o', color=color)

        plt.plot(r[r>=r0], fit['track'][k][r>=r0], '.', color=color)
        plt.title(k, fontsize=fontsize, x=0.05, y=0.9, ha='left', va='top')

        plt.yticks(fontsize=fontsize)
        if i+2<len(fit['track'])-S[0]+1:
            AX[k].xaxis.set_visible(False)
        else:
            plt.xticks(fontsize=fontsize)
        plt.fill_between([0, len(fit['track'][k])-1],
                         fit['best'][k]-fit['uncer'][k],
                         fit['best'][k]+fit['uncer'][k],
                        color='orange', alpha=0.2)
        plt.hlines(fit['best'][k], 0, len(fit['track'][k])-1, alpha=0.5, color='orange')
        plt.vlines(r0, plt.ylim()[0], plt.ylim()[1], linestyle=':',
                    color='k', alpha=0.5)
    for k in AX.keys():
        AX[k].callbacks.connect('xlim_changed', _callbackAxes)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    return
