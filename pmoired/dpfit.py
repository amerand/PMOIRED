import scipy.optimize
import numpy as np
from numpy import linalg
import time
from matplotlib import pyplot as plt

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

http://www.rhinocerus.net/forum/lang-idl-pvwave/355826-generalized-least-squares.html
"""

verboseTime=time.time()
Ncalls=0

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

Ncalls=0
Tcalls=0
def leastsqFit(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-7,
               ftol=1e-5, fullOutput=True, normalizedUncer=True,
               follow=None, maxfev=200, bounds={}):
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

    returns dictionary with:
    'best': bestparam,
    'uncer': uncertainties,
    'chi2': chi2_reduced,
    'model': func(x, bestparam)
    'cov': covariance matrix (normalized if normalizedUncer)
    'fitOnly': names of the columns of 'cov'
    """
    global Ncalls, pfitKeys, pfix, _func, data_errors
    # -- fit all parameters by default
    if fitOnly is None:
        if len(doNotFit)>0:
            fitOnly = filter(lambda x: x not in doNotFit, params.keys())
        else:
            fitOnly = params.keys()
        fitOnly = list(fitOnly)
        fitOnly.sort() # makes some display nicer

    # -- check that all parameters are numbers
    NaNs = []
    for k in fitOnly:
        #if not (type(params[k])==float or type(params[k])==int):
        if not (np.isscalar(params[k]) and type(params[k])!=str):
            NaNs.append(k)
    fitOnly = list(filter(lambda x: not x in NaNs, fitOnly))

    # -- build fitted parameters vector:
    pfit = [params[k] for k in fitOnly]

    # -- built fixed parameters dict:
    pfix = {}
    for k in params.keys():
        if k not in fitOnly:
            pfix[k]=params[k]
    if verbose:
        print('[dpfit] %d FITTED parameters:'%(len(fitOnly)), fitOnly)
    # -- actual fit
    Ncalls=0
    t0=time.time()

    if np.iterable(err) and len(np.array(err).shape)==2:
        # -- assumes err matrix is co-covariance
        _func = func
        pfitKeys = fitOnly
        plsq, cov = scipy.optimize.curve_fit(_fitFunc2, x, y, pfit,
                        sigma=err, epsfcn=epsfcn, ftol=ftol)
        info, mesg, ier = {'nfev':Ncalls, 'exec time':time.time()-t0}, 'curve_fit', None
    else:
        if bounds is None or bounds == {}:
            # ==== LEGACY! ===========================
            if verbose:
                print('[dpfit] using scipy.optimize.leastsq')
            plsq, cov, info, mesg, ier = \
                      scipy.optimize.leastsq(_fitFunc, pfit,
                            args=(fitOnly,x,y,err,func,pfix,verbose,follow,),
                            full_output=True, epsfcn=epsfcn, ftol=ftol,
                            maxfev=maxfev, )
            info['exec time'] = time.time() - t0
        else:
            method = 'L-BFGS-B'
            #method = 'SLSQP'
            #method = 'TNC'
            #method = 'trust-constr'
            if verbose:
                print('[dpfit] using scipy.optimize.minimize (%s)'%method)
            Bounds = []
            for k in fitOnly:
                if k in bounds.keys():
                    Bounds.append(bounds[k])
                else:
                    Bounds.append((-np.inf, np.inf))

            result = scipy.optimize.minimize(_fitFuncMin, pfit,
                            tol=ftol, options={'maxiter':maxfev},
                            bounds = Bounds, method=method,
                            args=(fitOnly,x,y,err,func,pfix,verbose,follow,)
                            )
            plsq = result.x
            display(result)
            try:
                # https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
                # Do Moore-Penrose inverse discarding zero singular values.
                _, s, VT = np.linalg.svd(result.jac, full_matrices=False)
                threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
                if verbose:
                    print('[dpfit] zeros in cov?', any(s<=threshold))
                s = s[s > threshold]
                VT = VT[:s.size]
                cov = np.dot(VT.T / s**2, VT)
            except:
                cov = np.zeros((len(fitOnly), len(fitOnly)))
            # ------------------------------------------------------
            info, mesg, ier = {'nfev':Ncalls, 'exec time':time.time()-t0}, result.message, None

    if verbose:
        print('[dpfit]', mesg)
        print('[dpfit] number of function call:', info['nfev'])
        t = 1000*info['exec time']/info['nfev']
        n=-int(np.log10(t))+3
        print('[dpfit] time per function call:', round(t, n), '(ms)')

        #print('[dpfit]', info)

    if cov is None:
        cov = np.zeros((len(fitOnly), len(fitOnly)))

    # -- best fit -> agregate to pfix
    for i,k in enumerate(fitOnly):
        pfix[k] = plsq[i]

    # -- reduced chi2
    model = func(x,pfix)
    # -- residuals
    if np.iterable(err) and len(np.array(err).shape)==2:
        # -- assumes err matrix is co-covariance
        r = y - model
        chi2 = np.dot(np.dot(np.transpose(r), np.linalg.inv(err)), r)
        reducedChi2 = chi2/(len(y)-len(pfit))
    else:
        tmp = _fitFunc(plsq, fitOnly, x, y, err, func, pfix)
        try:
            chi2 = (np.array(tmp)**2).sum()
        except:
            chi2=0.0
            for x in tmp:
                chi2+=np.sum(x**2)

        reducedChi2 = chi2/float(np.sum([1 if np.isscalar(i) else
                                         len(i) for i in tmp])-len(pfit))
        if not np.isscalar(reducedChi2):
            reducedChi2 = np.mean(reducedChi2)

    if normalizedUncer:
        try:
            cov *= reducedChi2
        except:
            pass

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

    if verbose:
        print('-'*30)
        print('        CHI2=', chi2)
        print('REDUCED CHI2=', reducedChi2)
        print('-'*30)
        if normalizedUncer:
            print('(uncertainty normalized to data dispersion)')
        else:
            print('(uncertainty assuming error bars are correct)')
        tmp = list(pfix.keys()); tmp.sort()
        maxLength = np.max(np.array([len(k) for k in tmp]))
        format_ = "'%s':"
        # -- write each parameter and its best fit, as well as error
        # -- writes directly a dictionnary
        print('') # leave some space to the eye
        for ik,k in enumerate(tmp):
            padding = ' '*(maxLength-len(k))
            formatS = format_+padding
            if ik==0:
                formatS = '{'+formatS
            if uncer[k]>0:
                ndigit = max(-int(np.log10(uncer[k]))+2, 0)
                fmt = '%.'+str(ndigit)+'f, # +/- %.'+str(ndigit)+'f'
                #print(formatS%k , round(pfix[k], ndigit), ',', end='')
                #print('# +/-', round(uncer[k], ndigit))
                print(formatS%k, fmt%(pfix[k], uncer[k]))
            elif uncer[k]==0:
                if isinstance(pfix[k], str):
                    print(formatS%k , "'"+pfix[k]+"'", ',')
                else:
                    print(formatS%k , pfix[k], ',')
            else:
                print(formatS%k , pfix[k], ',', end='')
                print('# +/-', uncer[k])
        print('}') # end of the dictionnary

    # -- result:
    if fullOutput:
        cor = np.sqrt(np.diag(cov))
        cor = cor[:,None]*cor[None,:]
        cor[cor==0] = 1e-6
        cor = cov/cor
        pfix= {#'func':func,
                'best':pfix, 'uncer':uncer,
               'chi2':reducedChi2, 'model':model,
               'cov':cov, 'fitOnly':fitOnly,
               'epsfcn':epsfcn, 'ftol':ftol,
               'info':info, 'cor':cor, 'x':x, 'y':y,
               'doNotFit':doNotFit,
               'covd':{ki:{kj:cov[i,j] for j,kj in enumerate(fitOnly)}
                         for i,ki in enumerate(fitOnly)},
               'cord':{ki:{kj:cor[i,j] for j,kj in enumerate(fitOnly)}
                         for i,ki in enumerate(fitOnly)},
               'normalized uncertainties':normalizedUncer,
               'maxfev':maxfev,
               'firstGuess':params,
        }
        if type(verbose)==int and verbose>1 and np.size(cor)>1:
            dispCor(pfix)
    return pfix

def randomParam(fit, N=None, x=None):
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
    if x is None:
        x = fit['x']
    for r in res:
        tmp.append(fit['func'](x, r))
    tmp = np.array(tmp)
    fit['r_param'] = res
    fit['r_ym1s'] = np.percentile(tmp, 16, axis=0)
    fit['r_yp1s'] = np.percentile(tmp, 84, axis=0)
    fit['r_x'] = x
    fit['r_y'] = fit['func'](x, fit['best'])
    fit['all_y'] = tmp
    return fit

randomParam = randomParam # legacy

def bootstrap(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-7,
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

def randomize(func, x, params, y, err=None, fitOnly=None,
               verbose=False, doNotFit=[], epsfcn=1e-7,
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

    if verbose and time.time()>(verboseTime+5):
        verboseTime = time.time()
        print('[dpfit]', time.asctime(), '%5d'%Ncalls,end=' ')
        try:
            chi2=(res**2).sum/(len(res)-len(pfit)+1.0)
            print('CHI2: %6.4e'%chi2,end=' ')
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
            print('CHI2: %6.4e'%(chi2/float(N-len(pfit)+1)), end='|')
        if follow is None:
            print('')
        else:
            _follow = list(filter(lambda x: x in params.keys(), follow))
            print('|'.join([k+'='+'%5.2e'%params[k] for k in _follow]))
    return res

def _fitFuncMin(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None):
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
    if type(y)==np.ndarray and type(err)==np.ndarray:
        model = func(x,params)
        res= ((np.array(y)-model)/err).flatten()
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

    if verbose and time.time()>(verboseTime+5):
        verboseTime = time.time()
        print('[dpfit]', time.asctime(), '%5d'%Ncalls,end='')
        print('CHI2: %6.4e'%chi2,end='|')
        if follow is None:
            print('')
        else:
            _follow = list(filter(lambda x: x in params.keys(), follow))
            print('|'.join([k+'='+'%5.2e'%params[k] for k in _follow]))
    return chi2


def _fitFunc2(x, *pfit, verbose=True, follow=[], errs=None):
    """
    for curve_fit
    """
    global pfitKeys, pfix, _func, Ncalls, verboseTime
    Ncalls +=1
    params = {}
    # -- build dic from parameters to fit and their values:
    for i,k in enumerate(pfitKeys):
        params[k]=pfit[i]
    # -- complete with the non fitted parameters:
    for k in pfix:
        params[k]=pfix[k]

    res = _func(x, params)

    if verbose and time.time()>(verboseTime+5):
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
    i1 = fit['fitOnly'].index(p1)
    i2 = fit['fitOnly'].index(p2)
    t = np.linspace(0,2*np.pi,n)
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

def dispCor(fit, ndigit=2):
    # -- parameters names:
    nmax = np.max([len(x) for x in fit['fitOnly']])
    fmt = '%%%ds'%nmax
    fmt = '%3d:'+fmt
    fmtd = '%'+'%d'%(ndigit+3)+'.'+'%d'%ndigit+'f'
    print('Correlations ', end=' ')
    print('\033[45m>=.9\033[0m', end=' ')
    print('\033[41m>=.8\033[0m', end=' ')
    print('\033[43m>=.7\033[0m', end=' ')
    print('\033[46m>=.5\033[0m', end=' ')
    print('\033[0m>=.2\033[0m', end=' ')
    print('\033[37m<.2\033[0m')

    print(' '*(2+ndigit+nmax), end=' ')
    for i in range(len(fit['fitOnly'])):
        print('%3d'%i+' '*(ndigit-1), end=' ')
    print('')
    for i,p in enumerate(fit['fitOnly']):
        print(fmt%(i,p), end=' ')
        for j, x in enumerate(fit['cor'][i,:]):
            if i==j:
                c = '\033[2m'
            else:
                c = '\033[0m'
            if i!=j:
                if abs(x)>=0.9:
                    col = '\033[45m'
                elif abs(x)>=0.8:
                    col = '\033[41m'
                elif abs(x)>=0.7:
                    col = '\033[43m'
                elif abs(x)>=0.5:
                    col = '\033[46m'
                elif abs(x)<0.2:
                    col = '\033[37m'
                else:
                    col = ''
            else:
                col = ''
            tmp = fmtd%x
            tmp = tmp.replace('0.', '.')
            tmp = tmp.replace('1.'+'0'*ndigit, '1.')
            if i==j:
                tmp = '#'*(2+ndigit)
            print(c+col+tmp+'\033[0m', end=' ')
        print('')

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
