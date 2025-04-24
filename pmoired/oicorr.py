from pmoired import oimodels, dpfit
import numpy as np
import matplotlib.pyplot as plt

def varVsErr(y, e, x=None, n='auto', verbose=0, fig=None, normalised=False):
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
    ptpx = np.ptp(x)
    if ptpx==0:
        ptpx = 1
    if type(n)==int:
        p = {'A%d'%i:0.1*np.ptp(y)*np.random.rand()/ptpx**i for i in range(n+1)}
        p['A0'] = np.mean(y)
        fit = dpfit.leastsqFit(dpfit.polyN, x-np.mean(x), p, y, e, verbose=0)
    elif n=='auto':
        j=1
        test = True
        chi2 = 0
        while test:
            p = {'A%d'%i:0.1*np.ptp(y)*np.random.rand()/ptpx**i for i in range(j+1)}
            p['A0'] = np.mean(y)
            fit = dpfit.leastsqFit(dpfit.polyN, x-np.mean(x), p, y, e, verbose=0)
            if chi2==0:
                chi2 = fit['chi2']
                test = j<len(y)//2-1
            else:
                test = (chi2-fit['chi2'])/chi2 > 0.01 and j<(len(y)//2-1)
                chi2 = fit['chi2']
            j+=1
        n = j-1

    # -- how to justify this? was done numerically...
    rho = 1 - np.std(y-fit['model'])**2/np.median(e)**2
    rho = min(rho,1)

    if rho<=0: # -> means data variance is same as uncertainty
        if normalised:
            res = {'rho':0, 'err':np.std(y-fit['model'])/np.median(e),
                   'poly':fit['best'], 'x0':np.mean(x)}
        else:
            res = {'rho':0, 'err':np.std(y-fit['model']),
                   'poly':fit['best'], 'x0':np.mean(x)}
    else:
        if normalised:
            res = {'rho':rho, 'err':1,
                   'poly':fit['best'], 'x0':np.mean(x)}
        else:
            res = {'rho':rho, 'err': np.median(e),
                   'poly':fit['best'], 'x0':np.mean(x)}

    # -- check
    if not fig is None:
        verbose=True
    if verbose:
        corr = {'catg':np.ones(len(y)),
                'rho':{1:res['rho']},
                'err':{1:res['err']},
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
    resi, wh, wl, data, err, models, ins = res
    _wh = np.array(wh)
    corr = {'catg':_wh, 'rho':{}, 'err':{}, 'poly':{}, 'x0':{}}
    for T in ['V2', '|V|', 'T3PHI']:
        tags = set(filter(lambda x: x.startswith(T), wh))
        #print(T, tags)
        for t in tags:
            w = np.where(_wh==t)
            tmp = varVsErr(data[w], err[w], wl[w], normalised=False)
            if tmp['rho']>0:
                corr['rho'][t] = float(tmp['rho'])
                corr['err'][t] = float(tmp['err'])
                corr['poly'][t] = tmp['poly']
                corr['x0'][t] = tmp['x0']
    return corr
