try:
    from pmoired import oimodels, oifits, oifake #, oicandid
except:
    import oimodels, oifits, oifake #, oicandid


import multiprocessing
try:
    # -- see https://stackoverflow.com/questions/64174552
    multiprocessing.set_start_method('spawn')
except:
    pass

import numpy as np
import matplotlib.pyplot as plt
import scipy
import astropy
import astroquery
import matplotlib
import sys
import os
import pickle
import time

print('[P]arametric [M]odeling of [O]ptical [I]nte[r]ferom[e]tric [D]ata', end=' ')
print('https://github.com/amerand/PMOIRED')

__versions__={'python':sys.version,
              'numpy':np.__version__,
              'scipy':scipy.__version__,
              'astropy': astropy.__version__,
              'astroquery': astroquery.__version__,
              'matplotlib':matplotlib.__version__
              }

try:
    jup = os.popen('jupyter --version').readlines()
    for j in jup:
        __versions__[j.split(':')[0].strip()] = j.split(':')[1].split('\n')[0].strip()
except:
    # -- cannot get versions of jupyter tools
    pass

class OI:
    def __init__(self, filenames=None, insname=None, targname=None,
                 withHeader=True, medFilt=None, binning=None,
                 tellurics=None, debug=False, verbose=True):
        """
        filenames: is either a single file (str) or a list of OIFITS files (list
            of str). Can also be the name of a ".pmrd" binary file from another
            PMOIRED session.

        insname: which instrument to select. Not needed if only one instrument
            per file. If multi instruments in files, all will be loaded.

        targname: which target. Not needed if only one target in files

        with_header: will load full header (default=True)

        medFilt: apply median filter of width 'medFilt'. Default no filter

        binning: bin spectral data by integer factor. default no binning (1)

        tellurics: pass a telluric correction vector, or a list of vectors,
            one per file. If nothing given, will use the tellurics in the OIFITS
            file (from 'pmoired.tellcorr')

        verbose: default is True

        See Also: addData, load, save
        """
        self.debug = debug
        # -- last best fit to the data
        self.bestfit = None
        # -- bootstrap results:
        self.boot = None
        # -- grid / random fits:
        self.grid = None
        # -- detection limit grid / random fits:
        self.limgrid = None
        # -- CANDID results:
        #self.candidFits = None
        # -- current figure
        self.fig = 0
        # -- modeled quantities:
        self.spectra = {}
        self.images = {}
        self._model = []
        self.data = []
        if type(filenames)==str and filenames.endswith('.pmrd'):
            self.load(filenames)
        elif not filenames is None:
            self.addData(filenames, insname=insname, targname=targname,
                            verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                            tellurics=tellurics, binning=binning)
        else:
            self.data = []

    def save(self, name=None, override=False):
        """
        save session as binary file (not OIFITS :()

        See Also: load
        """
        if name is None:
            name = time.strftime("PMOIRED_save_%Y%m%dT%H%M%S")
        if not name.endswith('.pmrd'):
            name += '.pmrd'
        assert not (os.path.exists(name) and not override), 'file "'+name+'" already exists'
        ext = ['data', 'bestfit', 'boot', 'grid', 'limgrid', 'fig', 'spectra',
                'images']
        with open(name, 'wb') as f:
            data = {k:self.__dict__[k] for k in ext}
            pickle.dump(data, f)
        print('object saved as "'+name+'"', end=' ')
        print('[size %.1fM]'%(os.stat('HD163296_all_star+rim+disk.pmrd').st_size/2**20))
        return

    def load(self, name, debug=False):
        """
        Load session from a binary file (not OIFITS :()

        See Also: save
        """
        if not os.path.exists(name) and os.path.exists(name+'.pmrd'):
            name += '.pmrd'
        assert os.path.exists(name), 'file "'+name+'" does not exist'
        with open(name, 'rb') as f:
            data = pickle.load(f)
        if type(data)==tuple and len(data)==8:
            # --
            self.data, self.bestfit, self.boot, \
                self.grid, self.limgrid, self.fig, \
                self.spectra, self._model = data
            return
        assert type(data)==dict, 'unvalid data format?'
        loaded = []
        for k in data:
            try:
                self.__dict__[k] = data[k]
                loaded.append(k)
            except:
                pass
        if debug:
            print('loaded:', loaded)
        return

    def info(self):
        """
        Print out information about the current session
        """
        # -- data
        print('== DATA', '='*40)
        for i,d in enumerate(self.data):
            print(' >', i, 'file="'+d['filename']+'"', 'ins="'+d['insname']+'"',
                '-'.join(d['telescopes']),
                '%dxWL=%.2f..%.2fum [R~%.0f]'%(len(d['WL']), d['WL'].min(), d['WL'].max(),
                                            np.mean(d['WL']/np.gradient(d['WL']))))
        if not self.bestfit is None:
            print('== FIT', '='*40)
            self.showBestfit()

        if not self.boot is None:
            print("== BOOTSTRAPPING == ")
            oimodels.analyseBootstrap(self.boot)
        return

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
        self.computeModelSpectra(uncer=False)
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

    # def candidFitMap(self, rmin=None, rmax=None, rstep=None, cmap=None,
    #                 firstGuess=None, fitAlso=[], fig=None, doNotFit=[],
    #                 logchi2=False, multi=True):
    #     """
    #     not to be used! still under development
    #     """
    #     self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
    #     if fig is None:
    #         self.fig += 1
    #         fig = self.fig
    #     self.candidFits = oicandid.fitMap(self._merged, rmin=rmin, rmax=rmax,
    #                                       rstep=rstep, firstGuess=firstGuess,
    #                                       fitAlso=fitAlso, fig=fig, cmap=cmap,
    #                                       doNotFit=doNotFit, logchi2=logchi2,
    #                                       multi=multi)
    #     self.bestfit = self.candidFits[0]
    #     self.computeModelSpectra
    #     return

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
                autoPrior=True, constrain=None):
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

        constain: set of conditions on the grid search. same as prior's syntax,
        but wille exclude some initial guesses.

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
        self._grid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       fitOnly=fitOnly, doNotFit=doNotFit,
                                       maxfev=maxfev, ftol=ftol, multi=multi,
                                       epsfcn=epsfcn, constrain=constrain)
        self._expl = expl
        self.grid = oimodels.analyseGrid(self._grid, self._expl)
        self.bestfit = self.grid[0]
        self.bestfit['prior'] = prior
        #self.computeModelSpectra()
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
            # -- find other components:
            tmp = oimodels.computeLambdaParams(self.grid[0]['best'])
            C = [k.split(',')[0] for k in tmp if ',' in k]
            C = list(set(C))
            leg = False
            for c in C:
                if c+',x' in tmp and c+',y' in tmp:
                    if c+',x' != px and c+',y' != py:
                        try:
                            plt.plot(tmp[c+',x'], tmp[c+',y'], '*', label=c,
                                    markersize=10, alpha=0.5)
                            leg = True
                        except:
                            print('error:', c, tmp[c+',x'], tmp[c+',y'])
                else:
                    plt.plot(0,0,'*', label=c,
                            markersize=10, alpha=0.5)
                    leg = True

            if leg:
                plt.legend(fontsize=7)
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
        if self.boot==None or len(self.boot)==0:
            print('run bootstrapping first!')
            return

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
             showFlagged=False, spectro=None, showUV=True, perSetup=True,
             allInOne=False, imFov=None, imPix=None, imPow=1., imMax=1, imPlx=None,
             checkImVis=False, vLambda0=None, imWl0=None, cmap='inferno',
             imX=0, imY=0, showChi2=False, cColors={}, cMarkers={},
             showSED=None, showPhotCent=False):
        """
        - model: dict defining a model to be overplotted. if a fit was performed,
            the best fit models will be displayed by default. Set to None for no
            models
        - fig: figure number (int)
        - obs: list of pbservables to show (in ['|V|', 'V2', 'T3PHI', 'DPHI',
            'FLUX']). Default will show all data. If fit was performed, fitted
            observables will be shown.
        - logV, logB: show visibilities, baselines in log scale (boolean)
        - showFlagged: show data flagged in the file (boolean)
        - showUV: show u,v coordinated (boolean)
        - spectro: force spectroscopic mode
        - vLambda0: show sepctroscopic data with velocity scale,
            around this central wavelength (in microns)
        - perSetup: each instrument/spectroscopic setup in a differn plot (boolean)
        - allInOne: all data in a single figure

        show image and spectrum of model: set imFov to a value to show image
        - imFov: field of view in mas
        - imPix: imPixel size in mas
        - imMax: cutoff for image display (0..1) or in percentile ('0'..'100')
        - imPow: power law applied to image for display. use 0<imPow<1 to show low
            surface brightness features
        - imX, imY: center of image (in mas)
        - imWl0: list of wavelength (um) to show the image default (min, max)
        - cmap: color map (default 'inferno')
        - checkImVis: compute visibility from image to check (can be wrong if
            fov is too small)
        - cColors: an optional dictionary to set the color of each components in
            the SED plot
        - showSED: True
        """
        t0 = time.time()

        if not imFov is None and imPix is None:
            imPix = imFov/100.

        if showSED:
            # -- show *only* SED
            showIM = False
            if imFov is None:
                imFov = 1.0
                imPix = 0.1
        else:
            showIM = True

        if not imFov is None:
            if showSED is None:
                showSED = True
            assert imPix>imFov/500, "the pixel of the synthetic image is too small!"
        else:
            if showSED is None:
                showSED = False
        #print('debug:', 'imFov', imFov, 'imPix', imPix,
        #        'showIM', showIM, 'showSED', showSED)
        if spectro is None:
            N = [len(d['WL']) for d in self.data]
            spectro = max(N)>20

        if allInOne and perSetup:
            perSetup = False

        if allInOne:
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
            showSED = False
            showIM = False
            imFov = None

        if perSetup:
            if perSetup == 'insname':
                perSetup = list(set([d['insname'] for d in self.data]))
            elif not type(perSetup)==list:
                perSetup = list(set([d['insname'].split(' ')[0].split('_')[0] for d in self.data]))

            if type(perSetup)==list:
                data = []
                for s in perSetup:
                    data.append(oifits.mergeOI([self.data[i] for i in range(len(self.data))
                                    if s in self.data[i]['insname']], verbose=False))

            for j,g in enumerate(data):
                if 'fit' in g and 'obs' in g['fit']:
                    _obs = g['fit']['obs']
                else:
                    _obs = obs
                oimodels.showOI(g,
                        param=model, fig=self.fig, obs=_obs, logV=logV,
                        logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV, allInOne=True,
                        imFov=None,  checkImVis=checkImVis, vLambda0=vLambda0,
                        showChi2=showChi2)
                self.fig+=1
                if type(perSetup)==list:
                    plt.suptitle(perSetup[j])
            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix,imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent)
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
                    obs=None, logV=logV, logB=logB, showFlagged=showFlagged,
                    spectro=spectro, showUV=showUV, allInOne=allInOne,
                    imFov=None, #imPix=imPix, imPow=imPow, imMax=imMax,
                    #imWl0=imWl0, cmap=cmap, imX=imX, imY=imY,
                    #cColors=cColors, cMarkers=cMarkers
                    checkImVis=checkImVis, vLambda0=vLambda0, showChi2=showChi2,
                    )
            if allInOne:
                self.fig += 1
            else:
                self.fig += len(self.data)
            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix, imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent)

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
                        spectro=spectro, showUV=showUV,
                        checkImVis=checkImVis, vLambda0=vLambda0,
                        showChi2=showChi2))
                self.fig += 1
            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix, imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent)
        return

    def showModel(self, model='best', imFov=None, imPix=None, imX=0, imY=0,
                  imPow=1, imMax=None, imWl0=None, cColors={}, cMarkers={},
                  showSED=True, showIM=True, fig=None, cmap='inferno',
                  logS=False, imPlx=None, imPhotCent=False):
        """
        oi: result from loadOI for mergeOI,
            or a wavelength vector in um (must be a np.ndarray)
        param: parameter dictionnary, describing the model
        m: result from Vmodel, if none given, computed from 'oi' and 'param'
        fig: which figure to plot on (default 1)
        figHeight: height of the figure, in inch

        imFov: field of view in mas
        imPix: imPixel size in mas
        imMax: cutoff for image display (0..1) or in percentile ('0'..'100')
        imPow: power law applied to image for display. use 0<imPow<1 to show low
            surface brightness features
        imX, imY: center of image (in mas)
        imWl0: list of wavelength (um) to show the image default (min, max)
        cmap: color map (default 'bone')
        imPlx: parallax (in mas) optional, will add sec axis in AU
        """
        if model=='best' and type(self.bestfit) is dict and \
                    'best' in self.bestfit:
            #print('showing best fit model')
            model = self.bestfit['best']
        assert type(model)==dict, 'model should be a dictionnary!'
        if not imFov is None:
            self.computeModelImages(imFov, model=model, imPix=imPix,
                                    imX=imX, imY=imY)
        self.computeModelSpectra(model=model, uncer=False)

        if imWl0 is None and showIM:
            #imWl0 = self.images['WL'].min(), self.images['WL'].max()
            imWl0 = [np.mean(self.images['WL'])]

        if not type(imWl0)==list and not type(imWl0)==tuple and \
            not type(imWl0)==set and not type(imWl0)==np.ndarray and \
            not imWl0 is None:
            imWl0 = [imWl0]
        if imWl0 is None:
            imWl0 = []
        if showSED:
            nplot = len(imWl0)+1
        else:
            nplot = len(imWl0)

        if fig is None:
            fig = self.fig
            self.fig+=1
        figWidth, figHeight = None, 4
        if figWidth is None and figHeight is None:
            figHeight =  min(max(nplot, 8), 5)
            figWidth = min(figHeight*nplot, 9.5)
        if figWidth is None and not figHeight is None:
            figWidth = min(figHeight*nplot, 9.5)
        if not figWidth is None and figHeight is None:
            figHeight =  max(figWidth/nplot, 6)
        plt.close(fig)
        plt.figure(fig, figsize=(figWidth, figHeight))
        i = -1 # default, in case only SED
        for i,wl0 in enumerate(imWl0):
            ax = plt.subplot(1, nplot, i+1, aspect='equal')
            if not imPlx is None:
                mas2au = lambda x: x/imPlx
                au2mas = lambda x: x*imPlx
                axx = ax.secondary_xaxis('top', functions=(mas2au, au2mas))
                axy = ax.secondary_yaxis('right', functions=(mas2au, au2mas))
                axx.tick_params(axis='x', labelsize=6, labelcolor=(0.7,0.5,0.2),
                                pad=0)
                axy.tick_params(axis='y', labelsize=6, labelcolor=(0.7,0.5,0.2),
                                pad=0, labelrotation=-90)
                axx.set_xlabel('(AU)\nplx=\n%.2fmas'%imPlx,
                                color=(0.7,0.5,0.2), x=0, fontsize=5)
            # -- index of image in cube, closest wavelength
            i0 = np.argmin(np.abs(self.images['WL']-wl0))
            # -- normalised image
            im = np.abs(self.images['cube'][i0]/np.max(self.images['cube'][i0]))
            # -- photocenter:
            xphot = np.sum(im*self.images['X'])/np.sum(im)
            yphot = np.sum(im*self.images['Y'])/np.sum(im)

            if not imMax is None:
                if type(imMax)==str:
                    _imMax = np.percentile(im**imPow, float(imMax))
                else:
                    _imMax = imMax**imPow
            else:
                _imMax = np.max(imMax**imPow)

            pc = plt.pcolormesh(self.images['X'], self.images['Y'],
                                im**imPow, cmap=cmap, vmax=_imMax, vmin=0,
                                shading='auto')
            cb = plt.colorbar(pc, ax=ax,
                              orientation='horizontal' if len(imWl0)>1 else
                              'vertical')
            if imPhotCent:
                plt.plot(xphot, yphot, color='w', marker=r'$\bigotimes$',
                        alpha=0.5, label='photo centre', linestyle='none')
                plt.legend(fontsize=6)

            Xcb = np.linspace(0,1,5)*_imMax
            XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
            XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
            cb.set_ticks(Xcb)
            cb.set_ticklabels(XcbL)
            cb.ax.tick_params(labelsize=6)
            ax.invert_xaxis()
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.set_xlabel(r'$\leftarrow$ E (mas)')

            if i==0:
                ax.set_ylabel(r'N $\rightarrow$ (mas)')
            if not imPow == 1:
                title = 'Image$^{%.2f}$ '%imPow
            else:
                title ='Image '
            n = 2-np.log10(np.median(np.abs(np.diff(self.images['WL']))))

            title += '$\lambda$=%.'+str(int(n))+'f$\mu$m'
            title = title%self.images['WL'][i0]
            plt.title(title, fontsize=9, y=1.05 if imPlx else None)
        symbols = {}
        a, b, c = 0.9, 0.6, 0.1
        colors = [(c,a,b), (a,b,c), (b,c,a),
                  (a,c,b), (c,b,a), (b,a,c),
                  (a,c,c), (c,a,c), (c,c,a),
                  (b,b,c), (b,c,b), (c,b,b),
                  (b,b,b)]
        markers = ['1', '2', '3', '4'] # 3 branches crosses
        _ic = 0
        _im =  0
        #print('debug', 'i', i, 'nplot', nplot)
        if showSED:
            ax = plt.subplot(1, nplot, i+2)
            if len(self.spectra['normalised spectrum WL']):
                key = 'normalised spectrum '
            else:
                key = 'flux '
            for c in sorted(self.spectra[key+'COMP']):
                if c in cColors:
                    col = cColors[c]
                else:
                    col = colors[_ic%len(colors)]
                    _ic+=1
                w = self.spectra[key+'COMP'][c]>0
                plt.plot(self.spectra[key+'WL'][w], self.spectra[key+'COMP'][c][w],
                        '.-', label=c, color=col)
            plt.plot(self.spectra[key+'WL'], self.spectra[key+'TOTAL'],
                    '.-k', label='TOTAL')
            if logS:
                plt.yscale('log')
            else:
                plt.ylim(0)
            plt.xlabel('wavelength ($\mu$m)')
            if key=='flux ':
                plt.title('SED', fontsize=9)
            else:
                plt.title('normalised spectra', fontsize=9)
                # -- show continuum
                #w = self.spectra[key+'CMASK']
                #plt.plt.plot(self.spectra[key+'WL'][w],
                #             self.spectra[key+'TOTAL'][w],
                #             'sc', label='cont.', alpha=0.5)
            plt.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
            plt.legend(fontsize=6)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

        plt.tight_layout()
        return

    def showBestfit(self):
        if not self.bestfit is None:
            print('chi2 = %f'%self.bestfit['chi2'])
            oimodels.dpfit.dispBest(self.bestfit)
            oimodels.dpfit.dispCor(self.bestfit)
        else:
            print('no fit to show')
        return

    def computeHalfLightRadiiFromParam(self):
        """
        result stored in self.halfradP
        """
        if not self.bestfit is None:
            self.halfradP = oimodels.halfLightRadiusFromParam(self.bestfit)
        else:
            print('no best fit model to compute half light radii!')
        return

    def deprojectImages(self, incl=0, projang=0, x0=0, y0=0):
        """
        deproject coordinates of synthetic images with inc, projang (in degrees),
        and center x0, y0 (in mas, default 0,0)

        Adds 'Xp' and 'Yp' deprojected coordinates of the pixels in "self.images"
        """
        assert self.images!={}, 'run ".computeModelImages" first!'
        if type(incl)==str and incl in self.images['model']:
            incl = self.images['model'][incl]
        if type(projang)==str and projang in self.images['model']:
            projang = self.images['model'][projang]

        Xp = (self.images['X']-x0)*np.cos(-projang*np.pi/180) + \
             (self.images['Y']-y0)*np.sin(-projang*np.pi/180)
        Yp = -(self.images['X']-x0)*np.sin(-projang*np.pi/180) + \
              (self.images['Y']-y0)*np.cos(-projang*np.pi/180)
        Xp /= np.cos(incl*np.pi/180)
        self.images['Xp'] = Xp
        self.images['Yp'] = Yp
        return

    def computeHalfLightRadiiFromImages(self, incl=0, projang=0, x0=0, y0=0,
                                        excludeCentralPix=True):
        """
        use models' synthetic images (self.images) to compute half-ligh radii (HLR)
        as function of wavelength.

        incl, projang: inclination and projection angle (in degrees) to
            de-project images (default 0,0, i.e. assume face-on). These also can
            by parameters' keyword from the model used to compute images.
        x0, y0: center of image to compute HLR (default: 0,0)
        excludeCentralPix: exclude central pixel for the computation of the HLR
            (default: True)

        result is stored in "self.halfradI" as a dictionary:
            'WL': wavelength (in um, sorted)
            'HLR': half-light radii, (in mas, same length as WL)
            'I': the radial cumulative intensity profiles
            'R': the radial dimension (im mas)
        """
        # if len(self._model)==0:
        #     print('no best images model to compute half light radii!')
        assert self.images!={}, 'run ".computeModelImages" first!'
        self.deprojectImages(incl, projang, x0, y0)

        res = {'WL':self.images['WL'], 'HLR':[],  'I':[]}
        scale = self.images['scale']
        Xp, Yp = self.images['Xp'], self.images['Yp']
        R = np.sqrt((Xp-x0)**2+(Yp-y0)**2).flatten()

        if excludeCentralPix:
            r = np.linspace(scale, np.max(R), 2*int(np.max(R)/scale))
        else:
            r = np.linspace(0, np.max(R), 2*int(np.max(R)/scale))
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
        res = {'WL':[], 'cube':[], 'X':X, 'Y':Y, 'scale':scale,
                'model':model}

        for t in tmp:# for each OIFITS object
            for i, wl in enumerate(t['WL']):
                if not wl in res['WL']:
                    res['WL'].append(wl)
                    if 'cube' in t['MODEL']:
                        res['cube'].append(t['MODEL']['cube'][i])
                    else:
                        res['cube'].append(t['MODEL']['image'])
        res['cube'] = np.array([res['cube'][i] for i in np.argsort(res['WL'])])
        res['WL'] = np.array(sorted(res['WL']))

        # -- photocenter
        res['photcent x'] = np.sum(res['X'][None,:,:]*res['cube'], axis=(1,2))/\
                        np.sum(res['cube'], axis=(1,2))

        res['photcent y'] = np.sum(res['Y'][None,:,:]*res['cube'], axis=(1,2))/\
                        np.sum(res['cube'], axis=(1,2))

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
        models = None
        if model=='best' and not self.bestfit is None:
            model = self.bestfit['best']
            # -- randomise parameters, using covariance
            models = oimodels.dpfit.randomParam(self.bestfit, N=Niter,
                                                x=None)['r_param']

        assert type(model) is dict, "model must be a dictionnary"

        allWLc = [] # -- continuum -> absolute flux
        allWLs = [] # -- with spectral lines -> normalized flux

        allCont = []

        for i,o in enumerate(self.data):
            # -- user-defined wavelength range
            fit = {'wl ranges':[(min(o['WL']), max(o['WL']))]}
            if 'fit' in o and 'continuum ranges' in o['fit']:
                fit['continuum ranges'] = o['fit']['continuum ranges']
                for c in o['fit']['continuum ranges']:
                    if not c in allCont:
                        allCont.append(c)

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
        #print('SED:', allWLc)
        #print('spe:', allWLs)
        M = {'model':model}
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
            allWL = {'WL':allWLs, 'fit':{'obs':['NFLUX'],
                                    'continuum ranges':allCont}} # minimum required
            tmp = oimodels.VmodelOI(allWL, model, timeit=False)
            fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                      tmp['MODEL'].keys() if k.endswith(',flux')}
            if not models is None:
                tmps = [oimodels.VmodelOI(allWL, m) for m in models]
                efluxes = ({k.split(',')[0]:np.std([t['MODEL'][k] for t in tmps], axis=0) for k in
                          tmp['MODEL'].keys() if k.endswith(',flux')}
                          )
            else:
                efluxes={}
            M['normalised spectrum CMASK'] = tmp['WL cont']
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
