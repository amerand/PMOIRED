from pmoired import oimodels, dpfit
import numpy as np
import matplotlib.pyplot as plt

def varVsErr(y, e, x=None, n=2, verbose=0, fig=None, normalised=True):
    """
    compare the variance in a data vector 'y' (as function of optional 'x') with error 'e':
    perform a polynomial fit of order 'n' (default 2) and compare the variance of the residuals
    to the errors and deduce the correlations

    fig="int" will show a plot

    returns {'rho':correlation, 'err':median error} -> can be used for building 'correlations'
    dictionnary for dpfit.leastsqFit

    """
    if x is None:
        x = np.linspace(-1, 1, len(y))
    p = {'A%d'%i:0.1*np.ptp(y)*np.random.rand()/np.ptp(x)**i for i in range(n+1)}
    p['A0'] = np.mean(y)

    fit = dpfit.leastsqFit(dpfit.polyN, x-np.mean(x), p, y, e, verbose=0)
    rho = 1 - np.std(y-fit['model'])**2/np.median(e)**2
    rho = min(rho,1)
    if rho<0:
        if normalised:
            res = {'rho':0, 'err': np.std(y-fit['model'])/np.median(e)}
        else:
            res = {'rho':0, 'err': np.std(y-fit['model'])}
    else:
        if normalised:
            res = {'rho':rho, 'err':1}
        else:
            res = {'rho':rho, 'err':np.median(e)}

    # -- check
    if not fig is None:
        verbose=True
    if verbose:
        corr = {'catg':np.ones(len(y)),
                'rho':{1:res['rho']},
                'err':{1:res['err']}
               }
        fitc = dpfit.leastsqFit(dpfit.polyN, x-np.mean(x), p, y, e, verbose=0,
                                        correlations=corr)
        if fig:
            fit = dpfit.randomParam(fit, N=100)
            fitc = dpfit.randomParam(fitc, N=100)
            s = r'$\chi^2$ %.3f -> %.3f (with $\rho$=%.3f and [e]=%.4f)'%(fit['chi2'], fitc['chi2'],
                                                                          res['rho'], res['err'])
            dx = 0.2*np.mean(np.gradient(x))
            plt.close(fig)
            plt.figure(fig)
            plt.suptitle(s)
            ax1 = plt.subplot(211)
            res['ax1'] = ax1

            plt.plot(x, y, '.k')
            plt.errorbar(x, y, yerr=e, linestyle='none', color='k', capsize=2)
            plt.errorbar(x+dx, y, yerr=res['err'], linestyle='none', color='b', capsize=2, alpha=0.2)

            plt.fill_between(x, fit['r_ym1s'], fit['r_yp1s'], color='r', alpha=0.2)
            plt.fill_between(x, fitc['r_ym1s'], fitc['r_yp1s'], color='c', alpha=0.2)

            ax2 = plt.subplot(212, sharex=ax1)
            res['ax2'] = ax2

            plt.plot(x, y-fitc['model'], '.k')
            plt.errorbar(x, y-fitc['model'], yerr=e, linestyle='none', color='k', capsize=2)
            plt.errorbar(x+dx, y-fitc['model'], yerr=res['err'], linestyle='none', color='b', capsize=2, alpha=0.2)

            plt.fill_between(x, fit['r_ym1s']-fitc['model'], fit['r_yp1s']-fitc['model'], color='r', alpha=0.2)
            plt.fill_between(x, fitc['r_ym1s']-fitc['model'], fitc['r_yp1s']-fitc['model'], color='c', alpha=0.2)
            plt.tight_layout()
        else:
            s = 'chi2 %.3f -> %.3f (with rho=%.3f and [e]=%.4f)'%(fit['chi2'], fitc['chi2'],
                                                                  res['rho'], res['err'])
            print(s)

    return res

def corrSpectra(res):
    """
    takes results from oimodels.residualsOI(..., fullOutput=True) and compute correlations per spectra.

    return a
    """
    resi, wh, wl, data, err, models = res
    _wh = np.array(wh)
    corr = {'catg':_wh, 'rho':{}, 'err':{}}
    for T in ['V2', '|V|']:
        tags = set(filter(lambda x: x.startswith(T), wh))
        #print(T, tags)
        for t in tags:
            w = np.where(_wh==t)
            tmp = varVsErr(data[w], err[w], wl[w], normalised=True)
            corr['rho'][t] = float(tmp['rho'])
            corr['err'][t] = float(tmp['err'])

    return corr
