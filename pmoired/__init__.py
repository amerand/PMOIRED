try:
    from pmoired import oimodels, oifits, oicandid, oifake
except:
    import oimodels, oifits, oicandid, oifake
import time

print('[P]arametric [M]odeling of [O]ptical [I]nte[r]ferom[e]tric [D]ata', end=' ')
print('https://github.com/amerand/PMOIRED')

class OI:
    def __init__(self, filenames, insname=None, targname=None, verbose=True,
               withHeader=True, medFilt=False, tellurics=None):
        """
        filenames: is either a single file (str) or a list of OIFITS files (list
            of str).

        insname: which instrument to select. Not needed if only one instrument
            per file

        targname: which target. Not needed if only one target in files

        with_header: will load full header (default=False)

        medfilt: apply median filter of width 'medfilt'. Default no filter

        tellurics: pass a telluric correction vector, or a list of vectors,
            one per file. If nothing given, will use the tellurics in the oifits
            file
        """
        # -- load data
        self.data = []
        self.addData(filenames, insname=insname, targname=targname,
                        verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                        tellurics=tellurics)

        # -- last fit to the data
        self.bestfit = None
        # -- bootstrap results:
        self.boot = None
        # -- CANDID results:
        self.candidFits = None
        # -- current figure
        self.fig = 0

    def addData(self, filenames, insname=None, targname=None, verbose=True,
                withHeader=False, medFilt=False, tellurics=None):
        if not type(filenames)==list:
            filenames = [filenames]
        self.data.extend(oifits.loadOI(filenames, insname=insname, targname=targname,
                        verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                        tellurics=tellurics))
        return

    def setupFit(self, fit, update=False):
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
                    d['fit'] = fit

        if type(fit)==list:
            for i,d in enumerate(self.data):
                assert _checkSetupFit(fit[i]), 'setup dictionnary is incorrect'
                if 'fit' in d and update:
                    d['fit'].update(fit[i])
                else:
                    d['fit'] = fit[i]
        return

    def doFit(self, model=None, fitOnly=None, doNotFit='auto', useMerged=True, verbose=2,
              maxfev=10000, ftol=1e-4):
        """
        model: a dictionnary describing the model
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
                                      maxfev=maxfev, ftol=ftol)
        self._model = oimodels.VmodelOI(self._merged, self.bestfit['best'])
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
        return

    def bootstrapFit(self, Nfits=None, model=None, multi=True):
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        if model is None:
            assert not self.bestfit is None, 'you should run a fit first'
            model = self.bestfit
        self.boot = oimodels.bootstrapFitOI(self._merged, model, Nfits, multi=multi)
        return

    def showBootstrap(self, sigmaClipping=4.5, fig=None, combParam={},
                        showChi2=False):
        """
        example:
        combParam={'SEP':'np.sqrt($c,x**2+$c,y**2)',
                   'PA':'np.arctan2($c,x, $c,y)*180/np.pi'}
        """
        if combParam=={}:
            self.boot = oimodels.analyseBootstrap(self.boot,
                                sigmaClipping=sigmaClipping, verbose=0)
        if not fig is None:
            self.fig += 1
            self.fig = fig
        oimodels.showBootstrap(self.boot, showRejected=0, fig=self.fig,
                               combParam=combParam, sigmaClipping=sigmaClipping,
                               showChi2=showChi2)
        self.fig += 1
        return

    def show(self, model='best', fig=None, obs=None, logV=False, logB=False,
             showFlagged=False, spectro=None, showUV=True, perSetup=True,
             allInOne=False, fov=None, pix=None, imPow=1., imMax=None,
             checkImVis=False, vLambda0=None, imWl0=None, cmap='magma',
             dx=0, dy=0):
        t0 = time.time()

        if not fov is None and pix is None:
            pix = fov/100.

        if spectro is None:
            N = [len(d['WL']) for d in self.data]
            spectro = max(N)>20

        if perSetup:
            data = oifits.mergeOI(self.data, collapse=False, verbose=False)
        else:
            data = self.data

        if not fig is None:
            self.fig = fig
        else:
            self.fig += 1
            fig = self.fig

        if model=='best':
            #print('showing best fit model')
            model=self.bestfit
        else:
            #print('showing model:', model)
            pass

        if not perSetup or allInOne:
            self._model = oimodels.showOI(self.data, param=model, fig=self.fig, obs=obs,
                    logV=logV, logB=logB, showFlagged=showFlagged,
                    spectro=spectro, showUV=showUV, allInOne=allInOne,
                    fov=fov, pix=pix, imPow=imPow, imMax=imMax,
                    checkImVis=checkImVis, vLambda0=vLambda0, imWl0=imWl0,
                    cmap=cmap, dx=dx, dy=dy)
            if allInOne:
                self.fig += 1
            else:
                self.fig += len(self.data)
        else:
            for i,d in enumerate(data):
                self._model = oimodels.showOI([d], param=model, fig=self.fig, obs=obs,
                        logV=logV, logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV,
                        fov=fov if i==(len(data)-1) else None,
                        pix=pix, imPow=imPow, imMax=imMax,
                        checkImVis=checkImVis, vLambda0=vLambda0,
                        imWl0=imWl0, cmap=cmap, dx=dx, dy=dy)
                self.fig += 1
        if not fov is None:
            self.fig += 1
        print('done in %.2fs'%(time.time()-t0))
        return
    def getSpectrum(self, comp, model='best'):
        if model=='best' and not self.bestfit is None:
            model = self.bestfit['best']
        assert type(model) is dict, "model must be a dictionnary"
        kz = filter(lambda k: k.startswith(comp+','), model.keys())
        #return {m['insname']:(m['WL'], oimodels[])}
        pass

def _checkSetupFit(fit):
    """
    check for setupFit:
    """
    keys = {'min error':dict, 'min relative error':dict,
            'max error':dict, 'max relative error':dict,
            'mult error':dict,
            'obs':list, 'wl ranges':list,
            'Nr':int, 'spec res pix':float,
            'cont ranges':list}
    ok = True
    for k in fit.keys():
        if not k in keys.keys():
            print('!WARNING! unknown fit setup "'+k+'"')
            ok = False
        elif type(fit[k]) != keys[k]:
            print('!WARNING! fit setup "'+k+'" should be of type', keys[k])
            ok = False
    return ok
