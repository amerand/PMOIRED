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

class OI:
    def __init__(self, filenames, insname=None, targname=None, verbose=True,
               withHeader=True, medFilt=None, tellurics=None, debug=False,
               binning=None):
        """
        filenames: is either a single file (str) or a list of OIFITS files (list
            of str).

        insname: which instrument to select. Not needed if only one instrument
            per file. If multi instruments in files, all will be loaded.

        targname: which target. Not needed if only one target in files

        with_header: will load full header (default=False)

        medFilt: apply median filter of width 'medFilt'. Default no filter

        binning: bin data by this factor (integer). default no binning

        tellurics: pass a telluric correction vector, or a list of vectors,
            one per file. If nothing given, will use the tellurics in the OIFITS
            file. Works with results from 'pmoired.tellcorr'
        """
        # -- load data
        self.data = []
        self.debug = debug
        self.addData(filenames, insname=insname, targname=targname,
                        verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                        tellurics=tellurics, binning=binning)

        # -- last best fit to the data
        self.bestfit = None
        # -- bootstrap results:
        self.boot = None
        # -- grid / random fits:
        self.grid = None
        # -- CANDID results:
        self.candidFits = None
        # -- current figure
        self.fig = 0
        # -- modeled quantities:
        self.fluxes = {}
        self.spectra = {}

    def addData(self, filenames, insname=None, targname=None, verbose=True,
                withHeader=True, medFilt=None, tellurics=None, binning=None):
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
        -> by default, all possible observables are fitted

        'wl ranges': gives a list of wavelength ranges (in um) where to fit.
            e.g. [(1.5, 1.6), (1.65, 1.75)]
            it will not override flagged data
        -> by default, the full defined range is fitted.

        'min error': forcing errors to have a minimum value. Keyed by the same
            values as 'obs'. e.g. {'V2':0.04, 'T3PHI':1.5} sets the minimum error
            to 0.04 in V2 (absolute) and 1.5 degrees for T3PHI

        'min relative error': same as 'min error', but for relative values. Useful
            for FLUX, V2, |V| or T3AMP

        'max error': similar to 'min error' but will ignore (flag) data above a
            certain error
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

    def doFit(self, model=None, fitOnly=None, doNotFit='auto', useMerged=True, verbose=2,
              maxfev=10000, ftol=1e-5, epsfcn=1e-8, follow=None):
        """
        model: a dictionnary describing the model
        fitOnly: list of parameters to fit (default: all)
        doNotFit: list of parameters not to fit (default: none)
        maxfev: maximum number of iterations
        ftol: chi2 stopping criteria
        follow: list of parameters to display as fit is going on
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
        self.bestfit = oimodels.fitOI(self._merged, model, fitOnly=fitOnly,
                                      doNotFit=doNotFit, verbose=verbose,
                                      maxfev=maxfev, ftol=ftol, epsfcn=epsfcn,
                                      follow=follow)
        self._model = oimodels.VmodelOI(self._merged, self.bestfit['best'])
        self.computeModelSpectrum()
        return
    def showFit(self):
        if not self.bestfit is None:
            self.fig += 1
            oimodels.dpfit.exploreFit(self.bestfit, fig=self.fig)
        return

    def candidFitMap(self, rmin=None, rmax=None, rstep=None, cmap=None,
                    firstGuess=None, fitAlso=[], fig=None, doNotFit=[],
                    logchi2=False, multi=True):
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

    def gridFit(self, expl, Nfits=None, model=None, fitOnly=None, doNotFit=None,
                     maxfev=5000, ftol=1e-5, multi=True, epsfcn=1e-8):
        """
        perform "Nfits" fit on data, starting from "model" (default last best fit),
        with grid / randomised parameters. Nfits can be determined from "expl" if
        "grid" param are defined.

        expl = {'grid':{'p1':(0,1,0.1), 'p2':(-1,1,0.5), ...},
                'rand':{'p3':(0,1), 'p4':(-np.pi, np.pi), ...},
                'randn':{'p5':(0, 1), 'p6':(np.pi/2, np.pi), ...}}

        grid=(min, max, step): explore all values for "min" to "max" with "step"
        rand=(min, max): uniform randomized parameter
        randn=(mean, std): normaly distributed parameter

        parameters should only appear once in either grid, rand or randn

        if "grid" are defined, they will define N as:
        Nfits = prod_i (max_i-min_i)/step_i + 1
        """
        if model is None and not self.bestfit is None:
            model = self.bestfit['best']
            if doNotFit is None:
                doNotFit = self.bestfit['doNotFit']
            if fitOnly is None:
                fiitOnly = self.bestfit['fitOnly']
        assert not model is None, 'first guess should be provided: model={...}'
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        self.grid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       fitOnly=fitOnly, doNotFit=doNotFit,
                                       maxfev=maxfev, ftol=ftol, multi=multi,
                                       epsfcn=epsfcn)
        self.bestfit = self.grid[0]
        self.computeModelSpectrum()
        self._expl = expl
        return

    def showGrid(self, px=None, py=None, color='chi2', aspect=None,
                vmin=None, vmax=None, cmap='spring', logV=False):
        assert not self.grid is None, 'You should run gridFit first!'
        self.fig += 1
        params = []
        for k in self._expl:
            for g in self._expl[k]:
                params.append(g)
        params = sorted(params)
        if px is None:
            px = params[0]
        if py is None:
            py = params[1]
        if aspect is None and \
                (px=='x' or px.endswith(',x')) and \
                (py=='y' or px.endswith(',y')):
            aspect='equal'
        oimodels.showGrid(self.grid, px, py, color=color, fig=self.fig,
                          vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap,
                          logV=logV)
        return

    def bootstrapFit(self, Nfits=None, model=None, multi=True):
        """
        perform 'Nfits' bootstrapped fits around dictionnary parameters 'model'.
        by default Nfits is set to the number of data, and model to the last best
        fit. 'multi' sets the number of threads (default==all available).
        """
        if self._merged is None:
            self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        if model is None:
            assert not self.bestfit is None, 'you should run a fit first or give an explicit model'
            model = self.bestfit
        self.boot = oimodels.bootstrapFitOI(self._merged, model, Nfits, multi=multi)
        return

    def showBootstrap(self, sigmaClipping=4.5, combParam={}, showChi2=False):
        """
        example:
        combParam={'SEP':'np.sqrt($c,x**2+$c,y**2)',
                   'PA':'np.arctan2($c,x, $c,y)*180/np.pi'}
        """
        if combParam=={}:
            self.boot = oimodels.analyseBootstrap(self.boot,
                                sigmaClipping=sigmaClipping, verbose=0)
        self.fig += 1
        oimodels.showBootstrap(self.boot, showRejected=0, fig=self.fig,
                               combParam=combParam, sigmaClipping=sigmaClipping,
                               showChi2=showChi2)
        self.fig += 1
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
            for i,d in enumerate(data):
                if 'fit' in d and 'obs' in d['fit']:
                    #print(j, g['fit'])
                    _obs = d['fit']['obs']
                else:
                    _obs = None
                self._model = oimodels.showOI([d], param=model, fig=self.fig,
                        obs=_obs, logV=logV, logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV, logS=logS,
                        imFov=imFov if i==(len(data)-1) else None,
                        imPix=imPix, imPow=imPow, imMax=imMax,
                        checkImVis=checkImVis, vLambda0=vLambda0,
                        imWl0=imWl0, cmap=cmap, imX=imX, imY=imY, showChi2=showChi2)
                self.fig += 1
        if not imFov is None:
            self.fig += 1
        #print('done in %.2fs'%(time.time()-t0))
        return

    def halfLightRadii(self, verbose=True):
        if not self.bestfit is None:
            self.halfrad = oimodels.halfLightRadiusFromParam(self.bestfit,
                                                            verbose=verbose)
        else:
            print('no best fit model to compute half light radii!')
        return

    def computeModelSpectrum(self, model='best'):
        if model=='best' and not self.bestfit is None:
            model = self.bestfit['best']
        assert type(model) is dict, "model must be a dictionnary"

        allWLc = [] # -- continuum -> absolute flux
        allWLs = [] # -- with spectral lines -> normalized flux

        for i,o in enumerate(self.data):
            if 'fit' in o and 'obs' in o['fit'] and 'NFLUX' in o['fit']['obs']:
                if 'WL mask' in o:
                    allWLs.extend(list(o['WL'][o['WL mask']]))
                else:
                    allWLs.extend(list(o['WL']))
            else:
                if 'WL mask' in o:
                    allWLc.extend(list(o['WL'][o['WL mask']]))
                else:
                    allWLc.extend(list(o['WL']))
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
            M['flux WL'] = allWLc
            M['flux COMP'] = fluxes
            M['flux TOTAL'] = tmp['MODEL']['totalflux']
        else:
            M['flux WL'] = np.array([])
            M['flux COMP'] = {}
            M['flux TOTAL'] = np.array([])

        if len(allWLs):
            allWL = {'WL':allWLs, 'fit':{'obs':[]}} # minimum required
            tmp = oimodels.VmodelOI(allWL, model)
            fluxes = {k.split(',')[0]:tmp['MODEL'][k] for k in
                      tmp['MODEL'].keys() if k.endswith(',flux')}
            M['normalised spectrum WL'] = allWLs
            M['normalised spectrum COMP'] = fluxes
            M['normalised spectrum TOTAL'] = tmp['MODEL']['totalflux']
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
    check for setupFit:
    """
    keys = {'min error':dict, 'min relative error':dict,
            'max error':dict, 'max relative error':dict,
            'mult error':dict,
            'obs':list, 'wl ranges':list,
            'Nr':int, 'spec res pix':float,
            'continuum ranges':list,
            'ignore negative flux':bool}
    ok = True
    for k in fit.keys():
        if not k in keys.keys():
            print('!WARNING! unknown fit setup "'+k+'"')
            ok = False
        elif type(fit[k]) != keys[k]:
            print('!WARNING! fit setup "'+k+'" should be of type', keys[k])
            ok = False
    return ok
