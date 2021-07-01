try:
    from pmoired import oimodels, oifits, oicandid, oifake
except:
    import oimodels, oifits, oicandid , oifake


import time

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
try:
    # -- see https://stackoverflow.com/questions/64174552
    multiprocessing.set_start_method('spawn')
except:
    pass

print('[P]arametric [M]odeling of [O]ptical [I]nte[r]ferom[e]tric [D]ata', end=' ')
print('https://github.com/amerand/PMOIRED')

import scipy
import astropy
import astroquery
import matplotlib
import sys

__versions__={'python':sys.version,
              'numpy':np.__version__,
              'scipy':scipy.__version__,
              'astropy': astropy.__version__,
              'astroquery': astroquery.__version__,
              'matplotlib':matplotlib.__version__
              }

class OI:
    def __init__(self, filenames=None, insname=None, targname=None, ,
                 withHeader=True, medFilt=None, binning=None,
                 tellurics=None, debug=False, verbose=True):
        """
        filenames: is either a single file (str) or a list of OIFITS files (list
            of str).

        insname: which instrument to select. Not needed if only one instrument
            per file. If multi instruments in files, all will be loaded.

        targname: which target. Not needed if only one target in files

        with_header: will load full header (default=True)

        medFilt: apply median filter of width 'medFilt'. Default no filter

        binning: bin data by this factor (integer). default no binning

        tellurics: pass a telluric correction vector, or a list of vectors,
            one per file. If nothing given, will use the tellurics in the OIFITS
            file. Works with results from 'pmoired.tellcorr'

        verbose: default is True

        See Also: addData
        """
        # -- load data
        self.data = []
        self.debug = debug
        if not filenames is None:
            self.addData(filenames, insname=insname, targname=targname,
                            verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                            tellurics=tellurics, binning=binning)
        else:
            self.data = []
        # -- last best fit to the data
        self.bestfit = None
        # -- bootstrap results:
        self.boot = None
        # -- grid / random fits:
        self.grid = None
        # -- detection limit grid / random fits:
        self.limgrid = None
        # -- CANDID results:
        self.candidFits = None
        # -- current figure
        self.fig = 0
        # -- modeled quantities:
        self.fluxes = {}
        self.spectra = {}
        self.images = {}
        self._model = []

    def addData(self, filenames, insname=None, targname=None, withHeader=True,
                medFilt=None, tellurics=None, binning=None, verbose=True):
        """
        add data to the existing ones:

        filenames: is either a single file (str) or a list of OIFITS files (list
            of str).

        insname: which instrument to select. Not needed if only one instrument
            per file. If multi instruments in files, all will be loaded.

        targname: which target. Not needed if only one target in files

        with_header: will load full header (default=True)

        medFilt: apply median filter of width 'medFilt'. Default no filter

        binning: bin data by this factor (integer). default no binning

        tellurics: pass a telluric correction vector, or a list of vectors,
            one per file. If nothing given, will use the tellurics in the OIFITS
            file. Works with results from 'pmoired.tellcorr'

        verbose: default is True

        """
        if not type(filenames)==list or not type(filenames)==tuple:
            filenames = [filenames]
        self.data.extend(oifits.loadOI(filenames, insname=insname, targname=targname,
                        verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                        tellurics=tellurics, debug=self.debug, binning=binning))
        return
    def setSED(self, wl, sed, err=0.01):
        """
        force the SED in data to "sed" as function of "wl" (in um)
        err is the relative error (default 0.01 == 1%)

        will update/create OI_FLUX and interpolate the SED in "FLUX" for
        each telescope.
        """
        for d in self.data:
            if 'OI_FLUX' in d:
                # -- replace flux
                tmp = np.interp(d['WL'], wl, sed)
                for k in d['OI_FLUX'].keys():
                    s = tmp[None,:] + 0*d['OI_FLUX'][k]['MJD'][:,None]
                    d['OI_FLUX'][k]['FLUX'] = s
                    d['OI_FLUX'][k]['RFLUX'] = s
                    d['OI_FLUX'][k]['EFLUX'] = err*s
                    d['OI_FLUX'][k]['FLAG'] *= False
            else:
                # -- add oi_flux
                flux = {}
                for mjd in d['configurations per MJD']:
                    d['configurations per MJD'][mjd].extend(d['telescopes'])

                mjd = np.array(sorted(list(d['configurations per MJD'])))
                s = np.interp(d['WL'], wl, sed)[None,:] + 0*mjd[:,None]
                for t in d['telescopes']:
                    flux[t] = {'FLUX':s, 'RFLUX':s, 'EFLUX':err*s, 'FLAG':s==0, 'MJD':mjd}
                d['OI_FLUX'] = flux
        return
    def setupFit(self, fit, update=False, debug=False):
        """
        set fit parameters by giving a dictionnary (or a list of dict, same length
        as 'data'):

        'obs': list of observables in
            'FLUX': Flux
            'NFLUX': Flux normalized to continuum
            'V2': sqared Visibility
            '|V|': visibility modulus
            'DPHI': differential phase (wrt continuum)
            'T3PHI': closure phases
            'T3AMP': closure amplitude

        'wl ranges': gives a list of wavelength ranges (in um) where to fit.
            e.g. [(1.5, 1.6), (1.65, 1.75)]
            it will not override flagged data
        -> by default, the full defined range is fitted.

        'min error': forcing errors to have a minimum value. Keyed by the same
            values as 'obs'. e.g. {'V2':0.04, 'T3PHI':1.5} sets the minimum error
            to 0.04 in V2 (absolute) and 1.5 degrees for T3PHI

        'min relative error': same as 'min error', but for relative values. Useful
            for FLUX, V2, |V| or T3AMP

        'max error': similar to 'min error' but will ignore (flag) data above

        'max relative error': similar to 'min relative error' but will ignore
            (flag) data above

        'Nr':int, number of points to compute radial profiles (default=100)

        'spec res pix':float, spectral resolution spread in pixel
            (e.g. for GRAVITY, use 1.4)

        'continuum ranges':list of ranges where continuum need to be computed,
            same format as 'wl ranges'. Not needed if you use Gaussian or
            Lorentzian in model with defined syntax ("line_...")

        'ignore negative flux':bool. Default is False
        """
        correctType = type(fit)==dict
        correctType = correctType or (type(fit)==list and
                                       len(fit)==len(self.data) and
                                        all([type(f)==dict for f in fit]))
        assert correctType, "parameter 'fit' must be a dictionnary or a list of dict"

        if type(fit)==dict:
            for d in self.data:
                assert _checkSetupFit(fit), 'setup dictionnary is incorrect'
                if 'fit' in d and update:
                    d['fit'].update(fit)
                else:
                    d['fit'] = fit.copy()

        if type(fit)==list:
            for i,d in enumerate(self.data):
                assert _checkSetupFit(fit[i]), 'setup dictionnary is incorrect'
                if 'fit' in d and update:
                    d['fit'].update(fit[i])
                else:
                    d['fit'] = fit[i].copy()
        if debug:
            print([d['fit']['obs'] for d in self.data])

        for d in self.data:
            if 'obs' in d['fit']:
                if debug:
                    print(d['filename'],
                        list(filter(lambda x: x.startswith('OI_'), d.keys())))
                d['fit']['obs'] = _checkObs(d, d['fit']['obs']).copy()
                if debug:
                    print(d['fit']['obs'])
        return

    def _setPrior(self, model, prior=None, autoPrior=True):
        if autoPrior:
            if prior is None:
                prior = []
            tmp = oimodels.autoPrior(model)
            # -- avoid overridding user provided priors
            for t in tmp:
                test = True
                for p in prior:
                    if t[0]==p[0] and t[1]==p[1]:
                        test = False
                if test:
                    prior.append(t)

        if not prior is None:
            for d in self._merged:
                if not 'fit' in d:
                    d['fit'] = {'prior':prior}
                else:
                    d['fit']['prior'] = prior
        return prior

    def doFit(self, model=None, fitOnly=None, doNotFit='auto', useMerged=True,
              verbose=2, maxfev=10000, ftol=1e-5, epsfcn=1e-8, follow=None,
              prior=None, autoPrior=True):
        """
        model: a dictionnary describing the model
        fitOnly: list of parameters to fit (default: all)
        doNotFit: list of parameters not to fit (default: none)
        maxfev: maximum number of iterations
        ftol: chi2 stopping criteria
        follow: list of parameters to display as fit is going on

        prior: list of priors as tuples. For example
            [('a+b', '=', 1.2, 0.1)] if a and b are parameters. a+b = 1.2 +- 0.1
            [('a', '<', 'sqrt(b)', 0.1)] if a and b are parameters. a<sqrt(b), and
                the penalty is 0 for a==sqrt(b) but rises significantly
                for a>sqrt(b)+0.1

        autoPrior: (default: True). Set automaticaly priors such as "diam>0" or
            "diamout>diamin".
        """
        if model is None:
            try:
                model = self.bestfit['best']
                if doNotFit=='auto':
                    doNotFit = self.bestfit['doNotFit']
                    fitOnly = self.bestfit['fitOnly']
            except:
                assert True, ' first guess as "model={...}" should be provided'

        if doNotFit=='auto':
            doNotFit = []
        # -- merge data to accelerate computations
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        prior = self._setPrior(model, prior, autoPrior)
        self.bestfit = oimodels.fitOI(self._merged, model, fitOnly=fitOnly,
                                      doNotFit=doNotFit, verbose=verbose,
                                      maxfev=maxfev, ftol=ftol, epsfcn=epsfcn,
                                      follow=follow)
        self._model = oimodels.VmodelOI(self._merged, self.bestfit['best'])
        self.computeModelSpectra()
        self.bestfit['prior'] = prior
        return
    def showFit(self):
        """
        show how chi2 / fitted parameters changed with each iteration of the fit.
        """
        if not self.bestfit is None:
            self.fig += 1
            oimodels.dpfit.exploreFit(self.bestfit, fig=self.fig)
        return

    def candidFitMap(self, rmin=None, rmax=None, rstep=None, cmap=None,
                    firstGuess=None, fitAlso=[], fig=None, doNotFit=[],
                    logchi2=False, multi=True):
        """
        not to be used! still under development
        """
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        if fig is None:
            self.fig += 1
            fig = self.fig
        self.candidFits = oicandid.fitMap(self._merged, rmin=rmin, rmax=rmax,
                                          rstep=rstep, firstGuess=firstGuess,
                                          fitAlso=fitAlso, fig=fig, cmap=cmap,
                                          doNotFit=doNotFit, logchi2=logchi2,
                                          multi=multi)
        self.bestfit = self.candidFits[0]
        self.computeModelSpectrum
        return

    def detectionLimit(self, expl, param, Nfits=None, nsigma=3, model=None, multi=True):
        """
        check the detection limit for parameter "param" from "model" (default
        bestfit), using an exploration grid "expl" (see gridFit for a description),
        assuming that "param"=0 is non-detection. The detection limit is computed
        for a number of sigma "nsigma" (default=3).

        See Also: gridFit, showLimGrid
        """
        if model is None and not self.bestfit is None:
            model = self.bestfit['best']
        assert not model is None, 'first guess should be provided: model={...}'
        assert param in model, '"param" should be one of the key in dict "model"'
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        self.limgrid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       multi=multi, dLimParam=param,
                                       dLimSigma=nsigma)
        #self.limgrid = [{'best':g} for g in self.limgrid]
        self._limexpl = expl
        self._limexpl['param'] = param
        self._limexpl['nsigma'] = nsigma
        return

    def showLimGrid(self, px=None, py=None, aspect=None,
                    vmin=None, vmax=None, mag=False, cmap='inferno'):
        """
        show the results from `detectionLimit` as 2D coloured map.

        px, py: parameters to show in x and y axis (default the first 2 parameters
            from the grid, in alphabetical order)
        aspect: default is None, but can be set to "equal"
        vmin, vmax: apply cuts to the colours (default: None)
        logV: show the colour as log scale
        cmap: valid matplotlib colour map (default="spring")

        The crosses are the starting points of the fits, and the coloured dots
        show the value of the limit for nsigma.

        See Also: detectionLimit
        """
        assert not self.limgrid is None, 'You should run detectionLimit first!'
        self.fig += 1
        params = []
        for k in ['grid', 'rand', 'randn']:
            if k in self._limexpl:
                for g in self._limexpl[k]:
                    params.append(g)
        params = sorted(params)
        if px is None:
            px = params[0]
        if py is None:
            py = params[1]
        xy = False
        if aspect is None and \
                (px=='x' or px.endswith(',x')) and \
                (py=='y' or py.endswith(',y')):
            aspect = 'equal'
            xy = True

        self.fig+=1
        plt.close(self.fig)
        plt.figure(self.fig, figsize=(8,4))
        ax1 = plt.subplot(121, aspect=aspect)
        if xy:
            ax1.invert_xaxis()
        if mag:
            c =[-2.5*np.log10(r[self._limexpl['param']]) for r in self.limgrid]
        else:
            c =[r[self._limexpl['param']] for r in self.limgrid]

        print('median', self._limexpl['param'], ':', np.median(c), ' (mag)' if mag else '')
        print('1sigma (68%)', np.percentile(c, 16), '->',
                np.percentile(c, 100-16))
        if len(self.limgrid)>40:
            print('2sigma (95%)', np.percentile(c, 2.5), '->',
                    np.percentile(c, 100-2.5))
        if len(self.limgrid)>1600:
            print('3sigma (99.7%)', np.percentile(c, 0.15), '->',
                    np.percentile(c, 100-0.15))

        plt.scatter([r[px] for r in self.limgrid],
                    [r[py] for r in self.limgrid],
                    c=c, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title('%.1f$\sigma$ detection'%self._limexpl['nsigma'])
        plt.colorbar(label=self._limexpl['param']+(' (mag)' if mag else ''))
        plt.xlabel(px)
        plt.ylabel(py)

        plt.subplot(122)
        plt.hist(c, bins=max(int(np.sqrt(len(self.limgrid))), 5))
        plt.xlabel(self._limexpl['param']+(' (mag)' if mag else ''))
        plt.tight_layout()
        return

    def gridFit(self, expl, Nfits=None, model=None, fitOnly=None, doNotFit=None,
                maxfev=5000, ftol=1e-5, multi=True, epsfcn=1e-8, prior=None,
                autoPrior=True):
        """
        perform "Nfits" fit on data, starting from "model" (default last best fit),
        with grid / randomised parameters. Nfits can be determined from "expl" if
        "grid" param are defined.

        expl = {'grid':{'p1':(0, 1, 0.1), 'p2':(-1, 1, 0.5), ...},
                'rand':{'p3':(0, 1), 'p4':(-np.pi, np.pi), ...},
                'randn':{'p5':(0, 1), 'p6':(np.pi/2, np.pi), ...}}

        grid=(min, max, step): explore all values for "min" to "max" with "step"
        rand=(min, max): uniform randomized parameter
        randn=(mean, std): normaly distributed parameter

        parameters should only appear once in either grid, rand or randn

        if "grid" are defined, they will define N as:
        Nfits = prod_i (max_i-min_i)/step_i + 1

        See Also: showGrid, detectionLimit
        """
        if model is None and not self.bestfit is None:
            model = self.bestfit['best']
            if doNotFit is None:
                doNotFit = self.bestfit['doNotFit']
            if fitOnly is None:
                fitOnly = self.bestfit['fitOnly']
            if prior is None and 'prior' in self.bestfit:
                prior = self.bestfit['prior']
                print('prior:', prior)

        assert not model is None, 'first guess should be provided: model={...}'
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        prior = self._setPrior(model, prior, autoPrior)
        self.grid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       fitOnly=fitOnly, doNotFit=doNotFit,
                                       maxfev=maxfev, ftol=ftol, multi=multi,
                                       epsfcn=epsfcn)
        self.bestfit = self.grid[0]
        self.bestfit['prior'] = prior
        self.computeModelSpectrum()
        self._expl = expl
        return

    def showGrid(self, px=None, py=None, color='chi2', aspect=None,
                vmin=None, vmax=None, logV=False, cmap='spring', fig=None):
        """
        show the results from `gridFit` as 2D coloured map.

        px, py: parameters to show in x and y axis (default the first 2 parameters
            from the grid, in alphabetical order)
        color: which parameter to show as colour (default is 'chi2')
        aspect: default is None, but can be set to "equal"
        vmin, vmax: apply cuts to the colours (default: None)
        logV: show the colour as log scale
        cmap: valid matplotlib colour map (default="spring")

        The crosses are the starting points of the fits, and the circled dot
        is the global minimum.

        See Also: gridFit
        """
        assert not self.grid is None, 'You should run gridFit first!'
        params = []
        for k in self._expl:
            for g in self._expl[k]:
                params.append(g)
        params = sorted(params)
        if px is None:
            px = params[0]
        if py is None:
            py = params[1]
        xy = False
        if aspect is None and \
                (px=='x' or px.endswith(',x')) and \
                (py=='y' or py.endswith(',y')):
            aspect = 'equal'
            xy = True
        if fig is None:
            self.fig += 1
            fig=self.fig

        oimodels.showGrid(self.grid, px, py, color=color, fig=self.fig,
                          vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap,
                          logV=logV)
        if xy:
            # -- usual x axis inversion when showing coordinates on sky
            plt.gca().invert_xaxis()
        return

    def bootstrapFit(self, Nfits=None, multi=True):
        """
        perform 'Nfits' bootstrapped fits around dictionnary parameters found
        by a previously ran fit. By default Nfits is set to the number of
        "independent" data. 'multi' sets the number of threads
        (default==all available).

        See Also: showBootstrap
        """
        if self._merged is None:
            self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)

        assert not self.bestfit is None, 'you should run a fit first (using "doFit")'
        model = self.bestfit

        self.boot = oimodels.bootstrapFitOI(self._merged, model, Nfits, multi=multi)
        return

    def showBootstrap(self, sigmaClipping=4.5, combParam={}, showChi2=False, fig=None):
        """
        example:
        combParam={'SEP':'np.sqrt($c,x**2+$c,y**2)',
                   'PA':'np.arctan2($c,x, $c,y)*180/np.pi'}

        See Also: bootstrapFit
        """
        if combParam=={}:
            self.boot = oimodels.analyseBootstrap(self.boot,
                                sigmaClipping=sigmaClipping, verbose=0)

        if not fig is None:
            self.fig = fig
        else:
            self.fig += 1
        oimodels.showBootstrap(self.boot, showRejected=0, fig=self.fig,
                               combParam=combParam, sigmaClipping=sigmaClipping,
                               showChi2=showChi2)
        return

    def show(self, model='best', fig=None, obs=None, logV=False, logB=False, logS=False,
             showFlagged=False, spectro=None, showUV=True, perSetup=False,
             allInOne=False, imFov=None, imPix=None, imPow=1., imMax=None,
             checkImVis=False, vLambda0=None, imWl0=None, cmap='inferno',
             imX=0, imY=0, showChi2=False):
        """
        - model: dict defining a model to be overplotted. if a fit was performed,
            the best fit models will be displayed by default. Set to None for no
            models
        - fig: figure number (int)
        - obs: list of pbservables to show (in ['|V|', 'V2', 'T3PHI', 'DPHI',
            'FLUX']). Defautl will show all data. If fit was performed, fitted
            observables will be shown.
        - logV, logB: show visibilities, baselines in log scale (boolean)
        - showFlagged: show data flagged in the file (boolean)
        - showUV: show u,v coordinated (boolean)
        - spectro: force spectroscopic mode
        - vLambda0: show sepctroscopic data with velocity scale,
            around this central wavelength (in microns)
        - perSetup: each instrument/spectroscopic setup in a differn plot (boolean)
        - allInOne: all data in a single figure

        show image and sepctrum of model: set imFov to a value to show image
        - imFov: field of view in mas
        - imPix: imPixel size in mas
        - imMax: cutoff for image display (0..1) or in percentile ('0'..'100')
        - imPow: power law applied to image for display. use 0<imPow<1 to show low
            surface brightness features
        - imX, imY: center of image (in mas)
        - imWl0: list of wavelength (um) to show the image default (min, max)
        - cmap: color map (default 'bone')
        - checkImVis: compute visibility from image to check (can be wrong if
            fov is too small)
        """
        t0 = time.time()

        if not imFov is None and imPix is None:
            imPix = imFov/100.

        if not imFov is None:
            assert imPix>imFov/500, "the pixel of the synthetic image is too small!"

        if spectro is None:
            N = [len(d['WL']) for d in self.data]
            spectro = max(N)>20

        if allInOne or perSetup:
            data = oifits.mergeOI(self.data, collapse=False, verbose=False)
        else:
            data = self.data

        if not fig is None:
            self.fig = fig
        else:
            self.fig += 1
            fig = self.fig

        if model=='best' and type(self.bestfit) is dict and \
                    'best' in self.bestfit:
            #print('showing best fit model')
            model = self.bestfit['best']
        elif not type(model) is dict:
            #print('no model to show...')
            model = None

        if perSetup:
            # # -- try to be clever about grouping
            # R = []
            # for d in data:
            #     r = (np.mean(d['WL'])/np.abs(np.mean(np.diff(d['WL'])))//5)*5.0
            #     if d['insname'].startswith('PIONIER'):
            #         R.append('%s R%.0f'%(d['insname'].split('_')[0], r))
            #     else:
            #         R.append('%s R%.0f'%(d['insname'], r))
            # group = []
            # _obs = []
            # for r in sorted(list(set(R))):
            #     group.append(oifits.mergeOI([data[i] for i in range(len(data)) if R[i]==r], verbose=False))
            #     tmp = []
            #     for i,d in enumerate(data):
            #         if R[i]==r and 'fit' in d and 'obs' in d['fit']:
            #             tmp.extend(d['fit']['obs'])
            #     _obs.append(list(set(tmp)))
            # # -- group
            # print(len(group))

            for j,g in enumerate(data):
                if 'fit' in g and 'obs' in g['fit']:
                    #print(j, g['fit'])
                    _obs = g['fit']['obs']
                else:
                    _obs = obs
                oimodels.showOI(g,
                        param=model, fig=self.fig, obs=_obs, logV=logV, logS=logS,
                        logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV, allInOne=False,
                        imFov=imFov, imPix=imPix, imPow=imPow, imMax=imMax,
                        checkImVis=checkImVis, vLambda0=vLambda0, imWl0=imWl0,
                        cmap=cmap, imX=imX, imY=imY, showChi2=showChi2)
                self.fig+=1
                if imFov is None:
                    #plt.suptitle(sorted(set(R))[j])
                    pass
                else:
                    #plt.figure(self.fig)
                    #plt.suptitle(sorted(set(R))[j])
                    self.fig+=1
            return
        elif allInOne:
            # -- figure out the list of obs, could be heteregenous
            if not obs is None:
                _obs = list(obs)
            else:
                for d in data:
                    _obs = []
                    if not 'fit' in d or not 'obs' in d['fit']:
                        if 'OI_T3' in d:
                            _obs.append('T3PHI')
                        if 'OI_VIS2' in d:
                            _obs.append('V2')
                        if 'OI_VIS' in d:
                            _obs.append('|V|')
                        if 'OI_FLUX' in d:
                            _obs.append('FLUX')
                        if not 'fit' in d:
                            d['fit'] = {}
                        d['fit']['obs'] = _obs

            #print('_obs', _obs)
            self._model = oimodels.showOI(self.data, param=model, fig=self.fig,
                    obs=None, logV=logV, logB=logB, logS=logS, showFlagged=showFlagged,
                    spectro=spectro, showUV=showUV, allInOne=allInOne,
                    imFov=imFov, imPix=imPix, imPow=imPow, imMax=imMax,
                    checkImVis=checkImVis, vLambda0=vLambda0, imWl0=imWl0,
                    cmap=cmap, imX=imX, imY=imY, showChi2=showChi2)
            if allInOne:
                self.fig += 1
            else:
                self.fig += len(self.data)
        else:
            self._model = []
            for i,d in enumerate(data):
                if 'fit' in d and 'obs' in d['fit']:
                    #print(j, g['fit'])
                    _obs = d['fit']['obs']
                else:
                    _obs = None
                self._model.append(oimodels.showOI([d], param=model, fig=self.fig,
                        obs=_obs, logV=logV, logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV, logS=logS,
                        imFov=imFov if i==(len(data)-1) else None,
                        imPix=imPix, imPow=imPow, imMax=imMax,
                        checkImVis=checkImVis, vLambda0=vLambda0,
                        imWl0=imWl0, cmap=cmap, imX=imX, imY=imY,
                        showChi2=showChi2))
                self.fig += 1
        if not imFov is None:
            self.fig += 1
        #print('done in %.2fs'%(time.time()-t0))
        return

    def computeHalfLightRadiiFromParam(self):
        """
        result stored in self.halfradP
        """
        if not self.bestfit is None:
            self.halfradP = oimodels.halfLightRadiusFromParam(self.bestfit,
                                                             verbose=verbose)
        else:
            print('no best fit model to compute half light radii!')
        return

    def computeHalfLightRadiiFromImages(self, incl=0, projang=0, x0=0, y0=0,
                                        excludeCentralPix=True):
        """
        use models' synthetic images (self.images) to compute half-ligh radii (HLR)
        as function of wavelength.
        incl, projang: inclination and projection angle (in degrees) to
            de-project images (default 0,0, i.e. assume face-on)
        x0, y0: center of image to compute HLR (default: 0,0)
        excludeCentralPix: exclude central pixel for the computation of the HLR
            (default: True)

        result is stored in self.halfradI as a dictionary:
            'WL': wavelength (in um, sorted)
            'HLR': half-light radii, (in mas, same length as WL)
            'I': the radial cumulative intensity profiles
            'R': the radial dimension (im mas)
        """
        # if len(self._model)==0:
        #     print('no best images model to compute half light radii!')
        assert self.images!={}, 'run ".computeModelImages" first!'

        res = {'WL':self.images['WL'], 'HLR':[],  'I':[]}
        scale = self.images['scale']
        Xp = (self.images['X']-x0)*np.cos(-projang*np.pi/180+np.pi/2) + \
             (self.images['Y']-y0)*np.sin(-projang*np.pi/180+np.pi/2)
        Yp = -(self.images['X']-x0)*np.sin(-projang*np.pi/180+np.pi/2) + \
              (self.images['Y']-y0)*np.cos(-projang*np.pi/180+np.pi/2)
        Yp /= np.cos(incl*np.pi/180)
        R = np.sqrt((Xp-x0)**2+(Yp-y0)**2).flatten()

        if excludeCentralPix:
            r = np.linspace(scale, np.max(R), int(np.max(R)/scale))
        else:
            r = np.linspace(0, np.max(R), int(np.max(R)/scale))
        res['R'] = r

        for i, wl in enumerate(res['WL']):
            tmp = self.images['cube'][i].flatten()
            if excludeCentralPix:
                I = [tmp[(R<=x)*(R>scale)].sum() for x in r]
            else:
                I = [tmp[R<=x].sum() for x in r]
            res['I'].append(np.array(I))
            res['HLR'].append(np.interp(I[-1]/2, I, r))
        res['HLR'] = np.array(res['HLR'])
        self.halfradI = res
        return

    def computeModelImages(self, imFov, model='best', imPix=None,
                           imX=0, imY=0):
        """
        Compute an image cube of the synthetic model, for each wavelength in
        the data. By default, the model used is the best fit model
        (self.bestfit['best']), but a dictionnary can be provided using "model=".
        a field of view ("imFov", in mas) must be given as first parameter.

        Parameters for the images are:
        imFov: mandatory Field of View , (in mas)
        imPix: pixel size (in mas). Default: imFov/101
        imX, imY: center of the field (in mas). Default is 0,0

        result is stored in the dictionnary self.images:
        'WL': wavelength (in um), sorted
        'cube': image cube, cube[i] is the i-th wavelength
        'X', 'Y': 2D arrays of coordinates of pixels
        'scale': actual pixel scale (in mas)

        See Also: computeModelSpectra

        """
        if model=='best' and not self.bestfit is None:
            model = self.bestfit['best']
        assert type(model) is dict, "model must be a dictionnary"
        if imPix is None:
            imPix = imFov/101
        tmp = [oimodels.VmodelOI(d, model, imFov=imFov, imPix=imPix,
                                imX=imX, imY=imY) for d in self.data]

        # -- image coordinates in mas
        X, Y = np.meshgrid(np.linspace(-imFov/2, imFov/2, 2*int(imFov/imPix/2)+1),
                           np.linspace(-imFov/2, imFov/2, 2*int(imFov/imPix/2)+1))
        X += imX
        Y += imY
        scale = np.diff(X).max()
        res = {'WL':[], 'cube':[], 'X':X, 'Y':Y, 'scale':scale}

        for t in tmp:# for each OIFITS object
            for i, wl in enumerate(t['WL']):
                if not wl in res['WL']:
                    res['WL'].append(wl)
                    if 'cube' in t['MODEL']:
                        res['cube'].append(t['MODEL']['cube'][i])
                    else:
                        res['cube'].append(t['MODEL']['image'])
        res['cube'] = np.array([res['cube'][i] for i in np.argsort(res['WL'])])
        res['WL'] = sorted(np.array(res['WL']))
        self.images = res
        return

    def computeModelSpectra(self, model='best', uncer=True, Niter=100):
        """
        Compute the fluxes (i.e. SED) and/or spectra for each component of given
        model (default model is the self.bestfit['best']). If "uncer=True"
        (default) and "model='best'", uncertainties of the spectra will be computed
        using the covariance matrix of best fit, using "Niter" (default=100).

        Fluxes and/or spectra will be computed depending if "FLUX" and/or "NFLUX"
        were fitted (i.e. defined in the fit setup dictionnary).

        result is a dictionnary stored in self.spectra:
        for fluxes (i.e. SED):
            'flux WL': wavelength vector
            'flux COMP': fluxes for each components of the model
            'err flux COMP': uncertainties of fluxes for each component
            'flux TOTAL': total flux
            'err flux TOTAL': uncertainty on total flux
        for normalised spectra:
            replace 'flux' by 'normalised spectrum'

        See Also: computeModelImages
        """
        if model=='best' and not self.bestfit is None:
            model = self.bestfit['best']
            # -- randomise parameters, using covariance
            models = oimodels.dpfit.randomParam(self.bestfit, N=Niter,
                                                x=None)['r_param']
        elif uncer:
            print('I do not know how to compute uncertainties!')
            models = None

        assert type(model) is dict, "model must be a dictionnary"

        allWLc = [] # -- continuum -> absolute flux
        allWLs = [] # -- with spectral lines -> normalized flux

        for i,o in enumerate(self.data):
            # -- user-defined wavelength range
            fit = {'wl ranges':[(min(o['WL']), max(o['WL']))]}
            if not 'fit' in o:
                o['fit'] = fit.copy()
            elif not 'wl ranges' in o['fit']:
                # -- weird but necessary to avoid a global 'fit'
                fit.update(o['fit'])
                o['fit'] = fit.copy()

            w = np.zeros(o['WL'].shape)
            closest = []
            for WR in o['fit']['wl ranges']:
                w += (o['WL']>=WR[0])*(o['WL']<=WR[1])
                closest.append(np.argmin(np.abs(o['WL']-0.5*(WR[0]+WR[1]))))
            o['WL mask'] = np.bool_(w)
            if not any(o['WL mask']):
                for clo in closest:
                    o['WL mask'][clo] = True
            if 'NFLUX' in o['fit']['obs']:
                allWLs.extend(list(o['WL'][o['WL mask']]))
            else:
                allWLc.extend(list(o['WL'][o['WL mask']]))

        allWLc = np.array(sorted(list(set(allWLc))))
        allWLs = np.array(sorted(list(set(allWLs))))
        M = {}
        if len(allWLc):
            allWL = {'WL':allWLc, 'fit':{'obs':[]}} # minimum required
            tmp = oimodels.VmodelOI(allWL, model)
            try:
                fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                          tmp['MODEL'].keys() if k.endswith(',flux')}
            except:
                fluxes = {'total': tmp['MODEL']['totalflux']}

            if not models is None:
                tmps = [oimodels.VmodelOI(allWL, m) for m in models]
                try:
                    efluxes = ({k.split(',')[0]:np.std([t['MODEL'][k] for t in tmps], axis=0) for k in
                            tmp['MODEL'].keys() if k.endswith(',flux')}
                            )
                except:
                    efluxes = {'total':np.std([t['MODEL']['totalflux'] for t in tmps], axis=0)}
            else:
                efluxes={}
            M['flux WL'] = allWLc
            M['flux COMP'] = fluxes
            M['err flux COMP'] = efluxes
            M['flux TOTAL'] = tmp['MODEL']['totalflux']
            if not models is None:
                M['err flux TOTAL'] = {'total':np.std([t['MODEL']['totalflux'] for t in tmps], axis=0)}

        else:
            M['flux WL'] = np.array([])
            M['flux COMP'] = {}
            M['flux TOTAL'] = np.array([])

        if len(allWLs):
            allWL = {'WL':allWLs, 'fit':{'obs':[]}} # minimum required
            tmp = oimodels.VmodelOI(allWL, model)
            fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                      tmp['MODEL'].keys() if k.endswith(',flux')}
            if not models is None:
                tmps = [oimodels.VmodelOI(allWL, m) for m in models]
                efluxes = ({k.split(',')[0]:np.std([t['MODEL'][k] for t in tmps], axis=0) for k in
                          tmp['MODEL'].keys() if k.endswith(',flux')}
                          )
            else:
                efluxes={}
            M['normalised spectrum WL'] = allWLs
            M['normalised spectrum COMP'] = fluxes
            M['err normalised spectrum COMP'] = efluxes
            M['normalised spectrum TOTAL'] = tmp['MODEL']['totalflux']
            if not models is None:
                M['err normalised spectrum TOTAL'] = np.std([t['MODEL']['totalflux'] for t in tmps], axis=0)

        else:
            M['normalised spectrum WL'] = np.array([])
            M['normalised spectrum COMP'] = {}
            M['normalised spectrum TOTAL'] = np.array([])

        self.spectra = M
        return

def _checkObs(data, obs):
    """
    data: OI dict
    obs: list of observable in ['|V|', 'V2', 'DPHI', 'T3PHI', 'FLUX', 'NFLUX']

    returns list of obs actually in data
    """
    ext = {'|V|':'OI_VIS', 'DPHI':'OI_VIS', 'PHI':'OI_VIS',
           'V2':'OI_VIS2',
           'T3PHI':'OI_T3', 'T3AMP':'OI_T3',
           'FLUX':'OI_FLUX',
           'NFLUX':'OI_FLUX',
           }
    return [o for o in obs if o in ext and ext[o] in data]

def _checkSetupFit(fit):
    """
    check for setupFit
    """
    keys = {'min error':dict, 'min relative error':dict,
            'max error':dict, 'max relative error':dict,
            'mult error':dict,
            'obs':list, 'wl ranges':list,
            'Nr':int, 'spec res pix':float,
            'continuum ranges':list,
            'ignore negative flux':bool,
            'prior':list}
    ok = True
    for k in fit.keys():
        if not k in keys.keys():
            print('!WARNING! unknown fit setup "'+k+'"')
            ok = False
        elif type(fit[k]) != keys[k]:
            print('!WARNING! fit setup "'+k+'" should be of type', keys[k])
            ok = False
    return ok
