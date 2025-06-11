from pmoired import oimodels, oifits, oifake, oicorr #, chi2map

import multiprocessing
try:
    # -- see https://stackoverflow.com/questions/64174552
    multiprocessing.set_start_method('spawn')
except:
    pass

import sys
import os
import pickle
import time
import requests
from inspect import signature

import numpy as np
#import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib.pyplot as plt
import scipy
import astropy
import astroquery
import matplotlib

__version__= '1.3.2'

FIG_MAX_WIDTH = 9.5
FIG_MAX_HEIGHT = 6
MAX_THREADS = multiprocessing.cpu_count()
US_SPELLING = True
oimodels.US_SPELLING = US_SPELLING

def setUSspelling(b):
    US_SPELLING = b
    oimodels.US_SPELLING = US_SPELLING

#print('[P]arametric [M]odeling of [O]ptical [I]nte[r]ferom[e]tric [D]ata', end=' ')
#print('https://github.com/amerand/PMOIRED')

__versions__={'pmoired':__version__,
              'python':sys.version,
              'numpy':np.__version__,
              'scipy':scipy.__version__,
              'astropy': astropy.__version__,
              'astroquery': astroquery.__version__,
              'matplotlib':matplotlib.__version__
              }

# not required for the core module
# try:
#     jup = os.popen('jupyter --version').readlines()
#     for j in jup:
#         __versions__[j.split(':')[0].strip()] = j.split(':')[1].split('\n')[0].strip()
# except:
#     # -- cannot get versions of jupyter tools
#     pass

def _isiterable(x):
    res = True
    try:
        iter(x)
    except:
        res = False
    return res

class OI:
    def __init__(self, filenames=None, insname=None, targname=None,
                 withHeader=True, medFilt=None, binning=None,
                 tellurics=None, useTelluricsWl=False, wlOffset=0.0,
                 debug=False, verbose=True, dMJD=None):
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

        useTelluricsWL: use telluric calibrated wavelength (False)

        wlOffset: add an offset (in um) to WL table

        dMJD: split data in chuncks of dMJD -> necessary if using $MJD in parameters
            *other than 'x', 'y'*. all data within chuncks dMJD will have same model

        verbose: default is True

        See Also: addData, load, save
        """
        self.debug = debug
        # -- last best fit to the data
        self.bestfit = {}
        # -- bootstrap results:
        self.boot = None
        # -- grid / random fits:
        self.grid = None
        self._expl = None
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
        self.dMJD = dMJD
        self._correlations = None
        if type(filenames)==str and filenames.endswith('.pmrd'):
            print('loading session saved in', filenames)
            self.load(filenames)
        elif not filenames is None:
            self.addData(filenames, insname=insname, targname=targname,
                            verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                            tellurics=tellurics, binning=binning,
                            useTelluricsWl=useTelluricsWl, wlOffset=wlOffset)
        else:
            self.data = []
        self._merged = []
    def __del__(self):
        """not doing anything"""
        self.data = []
    def save(self, name=None, overwrite=False):
        """
        save session as binary file (not OIFITS :()

        See Also: load
        """
        if name is None:
            name = time.strftime("PMOIRED_save_%Y%m%dT%H%M%S")
        if not name.endswith('.pmrd'):
            name += '.pmrd'
        #assert not (os.path.exists(name) and not overwrite), 'file "'+name+'" already exists (use "overwrite=True")'
        if os.path.exists(name) and not overwrite:
            raise Exception('file "'+name+'" already exists (use "overwrite=True")')

        ext = ['data', 'bestfit', 'boot', 'grid', 'limgrid', 'fig', 'spectra',
                'images', '_expl']
        with open(name, 'wb') as f:
            data = {k:self.__dict__[k] for k in ext}
            if type(data['bestfit'])==dict and 'func' in data['bestfit']:
                data['bestfit']['func'] = None # avoid potential problems...
                data['bestfit']['x'] = None # takes too much space
                data['bestfit']['y'] = None # takes too much space
            if type(data['grid'])==list:
                for g in data['grid']:
                    if 'func' in g:
                        g['func'] = None # avoid potential problems...
                        g['x'] = None # takes too much space
                        g['y'] = None # takes too much space

            pickle.dump(data, f)
        print('object saved as "'+name+'"', end=' ')
        print('[size %.1fM]'%(os.stat(name).st_size/2**20))
        return

    def load(self, name, debug=False):
        """
        Load session from a binary file (not OIFITS :()

        See Also: save
        """
        if not os.path.exists(name) and os.path.exists(name+'.pmrd'):
            name += '.pmrd'
        #assert os.path.exists(name), 'file "'+name+'" does not exist'
        if not os.path.exists(name):
            raise Exception('file "'+name+'" does not exist')
        with open(name, 'rb') as f:
            data = pickle.load(f)
        if type(data)==tuple and len(data)==8:
            # --
            self.data, self.bestfit, self.boot, \
                self.grid, self.limgrid, self.fig, \
                self.spectra, self._model = data
            return
        #assert type(data)==dict, 'unvalid data format?'
        if type(data)!=dict:
            raise Exception('unvalid data format (should be dict)')
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

    def sizeof(self):
        print('.data    : %.3fMB'%(_recsizeof(self.data)/1024**2))
        print('._merged : %.3fMB'%(_recsizeof(self.data)/1024**2))
        print('.bestfit : %.3fMB'%(_recsizeof(self.bestfit)/1024**2))
        print('.spectra : %.3fMB'%(_recsizeof(self.spectra)/1024**2))
        print('.images  : %.3fMB'%(_recsizeof(self.images)/1024**2))
        print('.boot    : %.3fMB'%(_recsizeof(self.boot)/1024**2))
        print('.grid    : %.3fMB'%(_recsizeof(self.grid)/1024**2))
        print('.limgrid : %.3fMB'%(_recsizeof(self.limgrid)/1024**2))

    def info(self):
        """
        Print out information about the current session
        """
        # -- data
        print('== DATA', '='*5)
        for i,d in enumerate(self.data):
            print(' >', i, 'file="'+d['filename']+'"', 'ins="'+d['insname']+'"',
                '-'.join(d['telescopes']),
                '%dxWL=%.2f..%.2fum [R~%.0f]'%(len(d['WL']), d['WL'].min(), d['WL'].max(),
                                            np.mean(d['WL']/np.gradient(d['WL']))))
        if not self.bestfit == {}:
            print('== FIT', '='*40)
            self.showBestfit()

        if not self.boot is None:
            print("== BOOTSTRAPPING == ")
            oimodels.analyseBootstrap(self.boot)
        return

    def addData(self, filenames, insname=None, targname=None, withHeader=True,
                medFilt=None, tellurics=None, binning=None, verbose=True,
                useTelluricsWl=False, wlOffset=0.0):
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

        useTelluricsWL: use telluric calibrated wavelength (False)

        verbose: default is True

        """
        filenames = oifits._globlist(filenames)
        if not type(filenames)==list or not type(filenames)==tuple:
            filenames = [filenames]
        self.data.extend(oifits.loadOI(filenames, insname=insname, targname=targname,
                        verbose=verbose, withHeader=withHeader, medFilt=medFilt,
                        tellurics=tellurics, debug=self.debug, binning=binning,
                        useTelluricsWl=useTelluricsWl, wlOffset=wlOffset))
        # -- force recomputation of images
        self.images = {}
        self.spectra = {}
        self._merged = []
        return

    def getESOPipelineParams(self, verbose=True):
        for i,d in enumerate(self.data):
            if 'header' in d:
                print('\033[5m=', d['filename'], '='*20, '\033[0m')
                self.data[i]['recipes'] = oifits.getESOPipelineParams(d['header'],
                                                                    verbose=verbose)
        return
    def setSED(self, wl, sed, err=0.01, unit=None):
        """
        force the SED in data to "sed" as function of "wl" (in um)
        err is the relative error (default 0.01 == 1%)

        will update/create OI_FLUX and interpolate the SED in "FLUX" for
        each telescope.
        """
        for i,d in enumerate(self.data):
            tmp = []
            for j in range(len(d['WL'])):
                w = np.abs(wl-d['WL'][j])<3*d['dWL'][j]
                if np.sum(w):
                    # -- kernel
                    k = np.exp(-(wl[w]-d['WL'][j])**2/(2*(d['dWL'][j]/2.35482)**2))
                    tmp.append(np.mean(sed[w]*k)/np.mean(k))
                else:
                    tmp.append(np.interp(d['WL'][j], wl, sed))
            tmp = np.array(tmp)

            if 'OI_FLUX' in d:
                # -- replace flux
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
                s = tmp[None,:] + 0*mjd[:,None]
                for t in d['telescopes']:
                    flux[t] = {'FLUX':s, 'RFLUX':s, 'EFLUX':err*s, 'FLAG':s==0, 'MJD':mjd}
                self.data[i]['OI_FLUX'] = flux
            if not unit is None:
                if 'units' in self.data[i]:
                    self.data[i]['units']['FLUX'] = unit
                else:
                    self.data[i]['units'] = {'FLUX': unit}
        return

    def setupFit(self, fit, update=False, debug=False, insname=None):
        """
        set fit parameters by giving a dictionnary (or a list of dict, same length
        as 'data'):

        insname: only apply to this insname (default: None -> apply to all).
            Can be a string (single insname) or list of insnames.

        "fit" contains the following keys (only "obs" is mandatory):

        'obs': list of observables in
            'FLUX': Flux
            'NFLUX': Flux normalised to continuum
            'V2': sqared Visibility
            '|V|': visibility modulus
            'DPHI': differential phase (wrt continuum)
            'N|V|': differential visibility (wrt continuum)
            'T3PHI': closure phases
            'T3AMP': closure amplitude
            'CF': correlated flux

        'wl ranges': gives a list of wavelength ranges (in um) where to fit.
            e.g. [(1.5, 1.6), (1.65, 1.75)]
            it will not override flagged data
        -> by default, the full defined range is fitted.

        'baseline ranges': gives a list of baselines ranges (in m) where to fit.
            e.g. [(10, 20), (40, 100)] it will not override flagged data
            it will affect bose baselines AND triangles if any baseline if out of range(s)
        -> by default, the full defined range is fitted.

        'MJD ranges': a list of ranges in MJD where to fit.

        'min error': forcing errors to have a minimum value. Keyed by the same
            values as 'obs'. e.g. {'V2':0.04, 'T3PHI':1.5} sets the minimum error
            to 0.04 in V2 (absolute) and 1.5 degrees for T3PHI

        'min relative error': same as 'min error', but for relative values. Useful
            for FLUX, V2, |V| or T3AMP

        'max error': similar to 'min error' but will ignore (flag) data above

        'max relative error': similar to 'min relative error' but will ignore
            (flag) data above

        'mult error': syntax similar to 'min error': multiply all errors by a
            value

        'Nr':int, number of points to compute radial profiles (default=100)

        'wl kernel':float, spectral resolution spread in pixel
            (e.g. for GRAVITY, use 1.4)

        'continuum ranges':list of ranges where continuum need to be computed,
            same format as 'wl ranges'. Not needed if you use Gaussian or
            Lorentzian in model with defined syntax ("line_...")

        'ignore negative flux':bool. Default is False

        'correlations':True takes into account spectral correlations, can be a dict
            with correlations per observables

        'spatial kernel': multiply vis / convolve images with spatial kernel of FWHM (in mas)

        example: oi.setupFit({'obs':['V2', 'T3PHI'], 'max error':{'T3PHI':10}})
          to fit V2 and T3PHI data, reject data with errors in T3PHI>10 degrees
        """
        correctType = type(fit)==dict
        correctType = correctType or (type(fit)==list and
                                       len(fit)==len(self.data) and
                                        all([type(f)==dict for f in fit]))

        #assert correctType, "parameter 'fit' must be a dictionnary or a list of dict "+\
        #    "with same length as data (%d)"%len(self.data)
        if not correctType:
            raise Exception("parameter 'fit' must be a dictionnary or a list of dict "+\
                            "with same length as data (%d)"%len(self.data))

        if insname is None:
            insname = list(set([d['insname'] for d in self.data]))
        if type(insname) == str:
            insname = [insname]
        test = list(filter(lambda i: i not in [d['insname'] for d in self.data], insname))
        #assert len(test)==0, str(test)+' not an actual insname in data: '+\
        #        str(set([d['insname'] for d in self.data]))
        if len(test)!=0:
            raise Exception(str(test)+' not an actual insname in data: '+\
                            str(set([d['insname'] for d in self.data])))

        if type(fit)==dict:
            for d in self.data:
                #assert _checkSetupFit(fit), 'setup dictionnary is incorrect'
                if not _checkSetupFit(fit):
                    raise Exception('setup dictionnary is incorrect')
                if d['insname'] in insname:
                    if 'fit' in d and update:
                        d['fit'].update(fit)
                    else:
                        d['fit'] = fit.copy()
        if type(fit)==list:
            for i,d in enumerate(self.data):
                #assert _checkSetupFit(fit[i]), 'setup dictionnary is incorrect'
                if not _checkSetupFit(fit):
                    raise Exception('setup dictionnary is incorrect')

                if d['insname'] in insname:
                    if 'fit' in d and update:
                        d['fit'].update(fit[i])
                    else:
                        d['fit'] = fit[i].copy()
        if debug:
            print('fit>obs:' ,[d['fit']['obs'] for d in self.data])

        for d in self.data:
            if 'fit' in d and 'obs' in d['fit']:
                if debug:
                    print(d['filename'],
                        list(filter(lambda x: x.startswith('OI_'), d.keys())))
                d['fit']['obs'] = _checkObs(d, d['fit']['obs']).copy()
                if debug:
                    print('fit>obs:', d['fit']['obs'])
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
        if prior is None:
            prior = []
        return prior

    def doFit(self, model=None, fitOnly=None, doNotFit='auto',
              verbose=2, maxfev=10000, ftol=1e-5, epsfcn=1e-6, follow=None,
              prior=None, autoPrior=True, factor=100, _zeroCorrelations=False,
              _maxRho=1):
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
        if not all(['fit' in d for d in self.data]):
            raise Exception('define fit context with "setupFit" first!')

        # -- warning: "correlations"" is global (i.e. applies to all data)!
        if any(['fit' in d and 'correlations' in d['fit'] and
            ((type(d['fit']['correlations'])==bool and d['fit']['correlations']) or
            (type(d['fit']['correlations'])==dict and d['fit']['correlations']!={})) for d in self.data]):
            #print('\033[41musing correlations\033[0m')
            correlations = True
        else:
            correlations = False

        if not prior is None:
            #assert _checkPrior(prior), 'ill formed "prior"'
            if not _checkPrior(prior):
                raise Exception('ill formed "prior"')

        if model is None:
            try:
                model = self.bestfit['best']
                if doNotFit=='auto':
                    doNotFit = self.bestfit['doNotFit']
                    fitOnly =  self.bestfit['fitOnly']
            except:
                #assert True, ' first guess as "model={...}" should be provided'
                raise Exception(' first guess as "model={...}" should be provided')

        if doNotFit=='auto':
            doNotFit = []

        # -- merge data to accelerate computations
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)
        prior = self._setPrior(model, prior, autoPrior)
        if correlations:
            # -- this is going to be very slow for bootstrapping :(
            _tmp = oimodels.residualsOI(self._merged, model, fullOutput=True, what=True)
            self._correlations = oicorr.corrSpectra(_tmp)

            # -- check if correlations levels were given by user
            for _m in self._merged:
                if 'correlations' in _m['fit'] and type(_m['fit']['correlations'])==dict:
                    for k in _m['fit']['correlations']:
                        _w = np.where((_tmp[6]==_m['insname'])*np.array([x.startswith(k) for x in _tmp[1]]))
                        for o in set(np.array(_tmp[1])[_w]):
                            if o in self._correlations['rho']:
                                self._correlations['rho'][o] = _m['fit']['correlations'][k]
                            if o in self._correlations['poly']:
                                self._correlations['poly'][o] = None

            if _zeroCorrelations:
                # force to use the minimizing algo from correlations, without correlations
                # should give the same results as no correlations!
                print('\033[41mDEBUG: setting all correlations to 0\033[0m')
                for k in list(self._correlations['rho'].keys()):
                   self._correlations['rho'].pop(k)
                   self._correlations['err'].pop(k)

            for k in self._correlations['rho']:
                self._correlations['rho'][k] = min(self._correlations['rho'][k], _maxRho)
        else:
            self._correlations = None

        self.bestfit = oimodels.fitOI(self._merged, model, fitOnly=fitOnly,
                                      doNotFit=doNotFit, verbose=verbose,
                                      maxfev=maxfev, ftol=ftol, epsfcn=epsfcn,
                                      follow=follow, factor=factor, prior=prior,
                                      correlations=self._correlations)
        if verbose:
            if len(self.bestfit['not significant']):
                print('\033[31mWARNING: thiese parameters do not change the chi2!:', end=' ')
                print(self.bestfit['not significant'], '\033[0m')
                print('\033[34m-> Try checking the syntax of your model\033[0m')
            if 'not converging' in self.bestfit and len(self.bestfit['not converging']):
                print('\033[33mCAUTION: these parameters may not be converging properly:', end=' ')
                print(self.bestfit['not converging'], '\033[0m')
                print('\033[34m-> Try inspecting the convergence by running ".showFit()"')
                # -- this is not robust!
                # print('-> Try redefining parameters to be less sensitive to *relative* variations')
                # for k in self.bestfit['not converging']:
                #     try:
                #         n = int(-np.round(np.log10(self.bestfit['uncer'][k])+2, 0))
                #         V0 = round(self.bestfit['best'][k], n)
                #         fmt = '%.'+str(max(n, 0))+'f'
                #         _k = k+' DELTA'
                #         print('{"%s":"'%k+fmt%V0+'+$%s", "%s":%f}'%(_k, _k, self.bestfit['best'][k]-V0))
                #     except:
                #         pass
                print('\033[0m')

        # -- priors are added as data
        self.bestfit['ndof'] -= len(prior)
        self.bestfit['prior'] = prior
        # -- compute final model
        self._model = oimodels.VmodelOI(self._merged, self.bestfit['best'])
        self.computeModelSpectra(uncer=False)
        return

    def _chi2FromModel(self, model=None, prior=None, autoPrior=True, reduced=True,
                       ndof=None, nfit=None, debug=False):
        """
        WARNING: does not take into correlations!
        """
        if not prior is None:
            #assert _checkPrior(prior), 'ill formed "prior"'
            if not _checkPrior(prior):
                raise Exception('ill formed "prior"')
        fitOnly = []
        if model is None:
            try:
                model = self.bestfit['best']
                doNotFit = self.bestfit['doNotFit']
                fitOnly = self.bestfit['fitOnly']
                ndof = self.bestfit['ndof']
                prior = self.bestfit['prior']
                ndof += len(prior) # this should not be the case!?
            except:
                #assert True, ' first guess as "model={...}" should be provided'
                raise Exception(' first guess as "model={...}" should be provided')
        if prior is None:
            prior = []

        # -- merge data to accelerate computations
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)
        prior += self._setPrior(model, prior, autoPrior)

        # -- warning: "correlations"" is global (i.e. applies to all data)!
        if any(['fit' in d and 'correlations' in d['fit'] and
            ((type(d['fit']['correlations'])==bool and d['fit']['correlations']) or
            (type(d['fit']['correlations'])==dict and d['fit']['correlations']!={})) for d in self.data]):
            #print('\033[41musing correlations\033[0m')
            correlations = True
        else:
            correlations = False

        if correlations:
            # -- this is going to be very slow for bootstrapping :(
            _tmp = oimodels.residualsOI(self._merged, model, fullOutput=True, what=True)
            self._correlations = oicorr.corrSpectra(_tmp)
            ignoreErr = np.zeros(len(_tmp[0]))
            for k in self._correlations['rho']:
                if self._correlations['rho'][k]>0:
                    w = np.where(np.array(_tmp[1])==k)
                    ignoreErr[w] = 1.0

            # -- check if correlations levels were given by user
            for _m in self._merged:
                if 'correlations' in _m['fit'] and type(_m['fit']['correlations'])==dict:
                    for k in _m['fit']['correlations']:
                        _w = np.where((_tmp[6]==_m['insname'])*np.array([x.startswith(k) for x in _tmp[1]]))
                        for o in set(np.array(_tmp[1])[_w]):
                            if o in self._correlations['rho']:
                                self._correlations['rho'][o] = _m['fit']['correlations'][k]
                            if o in self._correlations['poly']:
                                self._correlations['poly'][o] = None
            C = set(self._correlations['catg'])
            if not 'invcov' in self._correlations:
                self._correlations['invcov'] = {}
            for c in C:
                if not c in self._correlations['invcov'] and not c is None:
                    if c in self._correlations['rho'] and c in self._correlations['err']:
                        _n = np.sum(np.array(self._correlations['catg'])==c)
                        self._correlations['invcov'][c] = dpfit.invCOVconstRho(_n, self._correlations['rho'][c])
                        self._correlations['invcov'][c] /= self._correlations['err'][c]**2
                    else:
                        #print('warning: cannot compute covariance for "'+str(c)+'"')
                        #correlations['err'][c] = np.mean(err[correlations['catg']==c])
                        pass
        else:
            self._correlations = None
            ignoreErr = None

        for m in self._merged:
            if 'fit' in m:
                m['fit']['prior'] = prior
            else:
                m['fit'] = {'prior':prior}

        tmp = oimodels.residualsOI(self._merged, model, debug=debug)
        if not self._correlations is None:
            pfit = [model[k] for k in fitOnly]
            pfix = {k:model[k] for k in model if not k in fitOnly}

            return dpfit._fitFuncMin(pfit, fitOnly, self._merged, 0.0,
                    func=oimodels.residualsOI, doTrackP=False,
                    pfix=pfix, correlations=self._correlations,
                    addKwargs={'ignoreErr':ignoreErr})

        if reduced:
            if ndof is None and not nfit is None:
                ndof = len(tmp)-nfit+1

            if ndof is None:
                #print("warning: can't figure out ndof")
                return np.mean(tmp**2)
            else:
                #print("ndof:", ndof)
                return np.sum(tmp**2)/ndof
        else:
            return np.sum(tmp**2)

    def showFit(self):
        """
        show how chi2 / fitted parameters changed with each iteration of the fit.
        """
        if not self.bestfit=={}:
            self.fig += 1
            oimodels.dpfit.showFit(self.bestfit, fig=self.fig)
        return

    def detectionLimit(self, expl, param, Nfits=None, nsigma=3, model=None, multi=True,
                        prior=None, constrain=None):
        """
        check the detection limit for parameter "param" from "model" (default
        bestfit), using an exploration grid "expl" (see gridFit for a description),
        assuming that "param"=0 is non-detection. The detection limit is computed
        for a number of sigma "nsigma" (default=3).

        See Also: gridFit, showLimGrid
        """
        if not prior is None:
            #assert _checkPrior(prior), 'ill formed "prior"'
            if not _checkPrior(prior):
                raise Exception('ill formed "prior"')

        if model is None and not self.bestfit=={}:
            model = self.bestfit['best']
        #assert not model is None, 'first guess should be provided: model={...}'
        if model is None:
            raise Exception('first guess should be provided: model={...}')
        #assert param in model, '"param" should be one of the key in dict "model"'
        if not param in model:
            raise Exception('"param" should be one of the key in dict "model"')

        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False)
        oimodels.MAX_THREADS = MAX_THREADS
        self.limgrid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       multi=multi, dLimParam=param,
                                       dLimSigma=nsigma, prior=prior,
                                       constrain=constrain)
        #self.limgrid = [{'best':g} for g in self.limgrid]
        self._limexpl = expl
        self._limexpl['param'] = param
        self._limexpl['nsigma'] = nsigma
        return

    def showLimGrid(self, px=None, py=None, aspect=None, logV=False,
                    vmin=None, vmax=None, mag=False, cmap='inferno',
                    x0=None, y0=None, radProfile=True):
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
        #assert not self.limgrid is None, 'You should run detectionLimit first!'
        if self.limgrid is None:
            raise Exception('You should run detectionLimit first!')

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
        if xy and radProfile:
            plt.figure(self.fig, figsize=(FIG_MAX_WIDTH,FIG_MAX_WIDTH/3))
            ax1 = plt.subplot(131, aspect=aspect)
        else:
            plt.figure(self.fig, figsize=(FIG_MAX_WIDTH,FIG_MAX_WIDTH/2))
            ax1 = plt.subplot(121, aspect=aspect)

        if xy:
            ax1.invert_xaxis()
        if mag:
            c = np.array([-2.5*np.log10(r[self._limexpl['param']]) for r in self.limgrid])
            _unit = 'mag'
        elif logV:
            c = np.array([np.log10(r[self._limexpl['param']]) for r in self.limgrid])
            _unit = 'log10'
        else:
            c = np.array([r[self._limexpl['param']] for r in self.limgrid])
            _unit = ''
        cx = np.array([r[px] for r in self.limgrid])
        cy = np.array([r[py] for r in self.limgrid])

        cx, cy, c = cx[np.isfinite(c)], cy[np.isfinite(c)], c[np.isfinite(c)]

        print('distribution of %.1fsigma detections:'%self._limexpl['nsigma'])
        print(' median', self._limexpl['param'], ':', round(np.median(c),4), _unit)
        if len(self.limgrid)>13:
            print(' 1sigma (68%%) %.4f -> %.4f'%(np.percentile(c, 16),
                                                 np.percentile(c, 100-16)))
        if len(self.limgrid)>40:
            print('        (90%%) %.4f -> %.4f'%(np.percentile(c, 5),
                                                 np.percentile(c, 100-5)))
        if len(self.limgrid)>80:
            print(' 2sigma (95%%) %.4f -> %.4f'%(np.percentile(c, 2.5),
                                                 np.percentile(c, 100-2.5)))
        if len(self.limgrid)>200:
            print('        (99%%) %.4f -> %.4f'%(np.percentile(c, 0.5),
                                                 np.percentile(c, 100-0.5)))
        if len(self.limgrid)>1600:
            print(' 3sigma (99.7%%) %.4f -> %.4f'%(np.percentile(c, .15),
                                                   np.percentile(c, 100-.15)))

        plt.scatter(cx, cy, c=c, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(r'%.1f$\sigma$ detection'%self._limexpl['nsigma'])
        plt.colorbar(label=self._limexpl['param']+' '+_unit)
        plt.xlabel(px)
        plt.ylabel(py)

        if xy and radProfile:
            plt.subplot(132)
        else:
            plt.subplot(122)

        plt.hist(c, bins=max(int(np.sqrt(len(self.limgrid))), 5))
        plt.xlabel(self._limexpl['param']+' '+_unit)

        if xy  and radProfile:
            # -- radial detection limit
            ax3 = plt.subplot(133)
            if x0 is None:
                _x0 = lambda z: 0
            elif type(x0)==str:
                if not x0 in self.limgrid[0]:
                    print('warning: unknown parameter', x0)
                    _x0 = lambda z: 0
                else:
                    _x0 = lambda z: z[x0]
            else: # assume number
                _x0 = lambda z: x0

            if y0 is None:
                _y0 = lambda z: 0
            elif type(y0)==str:
                if not y0 in self.limgrid[0]:
                    print('warning: unknown parameter', y0)
                    _y0 = lambda z: 0
                else:
                    _y0 = lambda z: z[y0]
            else: # assume number
                _y0 = lambda z: y0

            R = np.array([np.sqrt((x[px]-_x0(x))**2+(x[py]-_y0(x))**2) for x in self.limgrid])
            r = np.linspace(min(R), max(R), int(np.sqrt(len(self.limgrid))))
            plt.plot(R, c, '.k', alpha=0.2)
            _r, _f = [], []
            for i in range(len(r)-1):
                w = (R>=r[i])*(R<=r[i+1])
                if sum(w):
                    _r.append(0.5*(r[i]+r[i+1]))
                    _f.append(np.median(c[w]))
            plt.plot(_r, _f, '-r', linewidth=5, alpha=0.5)
            plt.xlabel('radial distance (mas)')
            plt.ylabel(self._limexpl['param']+' '+_unit)
            if mag:
                ax3.invert_yaxis()
            plt.title('radial detection limit')

        try:
            plt.tight_layout()
        except:
            pass
        return

    def gridFit(self, expl, Nfits=None, model=None, fitOnly=None, doNotFit=None,
                maxfev=None, ftol=None, multi=True, epsfcn=None, prior=None,
                autoPrior=True, constrain=None, verbose=2, deltaChi2=None):
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

        constrain: set of conditions on the grid search. same as prior's syntax,
        but will exclude some initial guesses.
            e.g. constrain = [('np.sqrt(p1**2+p2**2)', '<', 1)]
            will restrict the search for (p1,p2) satisfying the condition
            however, best fit can be found outside the grid (unless 'prior' is
            specified)

        See Also: showGrid, detectionLimit
        """
        if not prior is None:
            #assert _checkPrior(prior), 'ill formed "prior"'
            if not _checkPrior(prior):
                raise Exception('ill formed "prior"')

        # -- warning: "correlations"" is global (i.e. applies to all data)!
        if any(['fit' in d and 'correlations' in d['fit'] and
            ((type(d['fit']['correlations'])==bool and d['fit']['correlations']) or
            (d['fit']['correlations']!={})) for d in self.data]):

            correlations = True
        else:
            correlations = False

        if model is None and not self.bestfit=={}:
            model = self.bestfit['best']
            if doNotFit is None:
                doNotFit = self.bestfit['doNotFit']
            if fitOnly is None:
                fitOnly = self.bestfit['fitOnly']
            if prior is None and 'prior' in self.bestfit:
                prior = self.bestfit['prior']
            if ftol is None:
                ftol = self.bestfit['ftol']
            if maxfev is None:
                maxfev = self.bestfit['maxfev']
            if epsfcn is None:
                epsfcn = self.bestfit['epsfcn']
        if maxfev is None:
            maxfev=5000
        if ftol is None:
            ftol=1e-5
        if epsfcn is None:
            epsfcn=1e-6

        #assert not model is None, 'first guess should be provided: model={...}'
        if model is None:
            raise Exception('first guess should be provided: model={...}')

        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)
        prior = self._setPrior(model, prior, autoPrior)
        if correlations:
            # -- this is going to be very slow for bootstrapping :(
            _tmp = oimodels.residualsOI(self._merged, model, fullOutput=True, what=True)
            #print('len(tmp)', len(tmp))
            self._correlations = oicorr.corrSpectra(_tmp)

            # -- check if correlations levels were given by user
            for _m in self._merged:
                if 'correlations' in _m['fit'] and type(_m['fit']['correlations'])==dict:
                    for k in _m['fit']['correlations']:
                        _w = np.where((_tmp[6]==_m['insname'])*np.array([x.startswith(k) for x in _tmp[1]]))
                        for o in set(np.array(_tmp[1])[_w]):
                            if o in self._correlations['rho']:
                                self._correlations['rho'][o] = _m['fit']['correlations'][k]
                            if o in self._correlations['poly']:
                                self._correlations['poly'][o] = None
        else:
            self._correlations = None

        self.grid = oimodels.gridFitOI(self._merged, model, expl, Nfits,
                                       fitOnly=fitOnly, doNotFit=doNotFit,
                                       maxfev=maxfev, ftol=ftol, multi=multi,
                                       epsfcn=epsfcn, constrain=constrain,
                                       prior=prior, verbose=verbose,
                                       correlations=self._correlations)
        self._expl = expl
        self.grid = oimodels.analyseGrid(self.grid, self._expl, verbose=verbose,
                                         deltaChi2=deltaChi2)
        self.bestfit = self.grid[0]
        self.bestfit['prior'] = prior
        self.bestfit['constrain'] = constrain
        #self.computeModelSpectra()
        return

    def showGrid(self, px=None, py=None, color='chi2', aspect=None,
                vmin=None, vmax=None, logV=False, cmap='magma', fig=None,
                interpolate=False, legend=True, tight=False,
                significance=False):
        """
        show the results from `gridFit` as 2D coloured map.

        px, py: parameters to show in x and y axis (default the first 2 parameters
            from the grid, in alphabetical order)

        color: which parameter to show as colour (default is 'chi2')

        aspect: default is None, but can be set to "equal"

        vmin, vmax: apply cuts to the colours (default: None)

        logV: show the colour as log scale

        cmap: valid matplotlib colour map (default="spring")

        interpolate: show a continuous interpolation of the minimum chi2 map

        significance: shows the detection in sigmas

        The crosses are the starting points of the fits, and the circled dot
        is the global minimum.

        See Also: gridFit
        """
        #assert not self.grid is None, 'You should run gridFit first!'
        if self.grid is None:
            raise Exception('You should run gridFit first!')

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
        if (px=='x' or px.endswith(',x')) and \
            (py=='y' or py.endswith(',y')):
            aspect = 'equal'
            xy = True
        if fig is None:
            self.fig += 1
            fig=self.fig

        if not significance is False:
            #print('significance:', significance)
            m = self.grid[0]['best'].copy()
            if type(significance)==str:
                # -- parameters to 0 to get non-detection chi2
                if not significance in m:
                    raise Exception('"significance" should be the parameters which give the chi2 ref when ==0')
                m[significance] = 0
                significance = self._chi2FromModel(m)
            elif type(significance)==float:
                pass
            elif type(significance)==dict:
                significance = self._chi2FromModel(significance)
            elif xy:
                m[px.replace(',x', ',f')]=0
                significance = self._chi2FromModel(m)
            #print('significance:', significance)
        oimodels.showGrid(self.grid, px, py, color=color, fig=self.fig,
                          vmin=vmin, vmax=vmax, aspect=aspect, cmap=cmap,
                          logV=logV, interpolate=interpolate,
                          expl=self._expl, tight=tight, significance=significance,
                          #constrain=self.bestfit['constrain']
                          )
        if xy:
            # -- usual x axis inversion when showing coordinates on sky
            plt.gca().invert_xaxis()
            # -- find other components:
            tmp = oimodels.computeLambdaParams(self.grid[0]['best'], MJD=None)
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
            plt.xlabel(r'E $\leftarrow$ '+px)
            plt.ylabel(py+r'$\rightarrow$ N')
            if leg and legend:
                plt.legend(fontsize=7)
        return

    def bootstrapFit(self, Nfits=None, multi=True, keepFlux=False, verbose=2,
                     strongMJD=False, randomiseParam=True, additionalRandomise=None):
        """
        perform 'Nfits' bootstrapped fits around dictionnary parameters found
        by a previously ran fit. By default Nfits is set to the number of
        "independent" data. 'multi' sets the number of threads
        (default==all available).

        keepFlux: do not randomize fluxes (False by default)

        additionalRandomise: optional function to randomise the data in "additional residuals"
            additionalRandomise(True) will randomise the data
            additionalRandomise(False) will reset the data to its original order and weights

        See Also: showBootstrap
        """
        if self._merged is None:
            self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)

        #assert not self.bestfit=={}, 'you should run a fit first (using "doFit")'
        if self.bestfit=={}:
            raise Exception('you should run a fit first (using "doFit")')

        model = self.bestfit
        oimodels.MAX_THREADS = MAX_THREADS
        self.boot = oimodels.bootstrapFitOI(self._merged, model, Nfits, multi=multi,
                                            keepFlux=keepFlux, verbose=verbose,
                                            strongMJD=strongMJD, randomiseParam=randomiseParam,
                                            additionalRandomise=additionalRandomise,
                                            sigmaClipping=None,
                                            correlations=self._correlations)
        if not additionalRandomise is None:
            additionalRandomise(False)
        return

    def showBootstrap(self, sigmaClipping=None, combParam={}, showChi2=False, fig=None,
                      alternateParameterNames=None, showSingleFit=True, chi2MaxClipping=None,
                      ignore=None):
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
                                sigmaClipping=sigmaClipping, verbose=0,
                                chi2MaxClipping=chi2MaxClipping)

        if not fig is None:
            self.fig = fig
        else:
            self.fig += 1
        oimodels.showBootstrap(self.boot, showRejected=0, fig=self.fig, showChi2=showChi2,
                               combParam=combParam, sigmaClipping=sigmaClipping,
                               alternateParameterNames=alternateParameterNames,
                               showSingleFit=showSingleFit, chi2MaxClipping=chi2MaxClipping,
                               ignore=ignore)
        return

    def showTellurics(self, fig=None):
        """
        show telluric corrections
        """
        if not fig is None:
            self.fig = fig
        else:
            self.fig += 1
            fig = self.fig
        showAny = 0
        for d in self.data:
            if 'TELLURICS' in d:
                showAny += 1
        if not showAny:
            print('nothing to show!')
            return
        plt.close(self.fig)
        plt.figure(self.fig)
        p = 1
        for d in self.data:
            if 'TELLURICS' in d:
                plt.subplot(showAny, 1, p)
                p+=1
                plt.plot(d['WL'], d['TELLURICS'])
                plt.xlabel(r'wavelength ($\mu$m)')
                plt.ylabel("flux (arb. unit)")
        return

    def show(self, model='best', fig=None, obs=None, logV=False, logB=False, logS=False,
             showFlagged=False, spectro=None, showUV=True, perSetup=True,
             allInOne=False, imFov=None, imPix=None, imPow=1., imMax=1, imPlx=None,
             checkImVis=False, vWl0=None, imWl0=None, cmap='inferno',
             imX=0, imY=0, imTight=False, showChi2=False, cColors={}, cMarkers={},
             showSED=None, showPhotCent=False, imLegend=True, bckgGrid=True,
             barycentric=False, autoLimV=False, t3B='max'):
        """
        - model: dict defining a model to be overplotted. if a fit was performed,
            the best fit models will be displayed by default. Set to None for no
            models
        - fig: figure number (int)
        - obs: list of pbservables to show (in ['|V|', 'V2', 'T3PHI', 'DPHI',
            'FLUX', 'N|V|']). Default will show all data. If fit was performed,
            fitted observables will be shown.
        - logV, logB: show visibilities, baselines in log scale (boolean)
        - showFlagged: show data flagged in the file (boolean)
        - showUV: show u,v coordinated (default=True)
        - bckgGrid: shows background grid (default=True)
        - spectro: force spectroscopic mode
        - vWl0: show spectroscopic data with velocity scale,around this central
            wavelength (in microns)
        - perSetup: each instrument/spectroscopic setup in a differn plot (boolean)
        - allInOne: all data in a single figure
        - barycentric: use barycentric velocities, if possible
        - autoLimV: automatic limits for visibility (|V|, V2) plots (default=False).

        show image(s) and spectrum of model: set imFov to a value to show image
        - imFov: field of view in mas
        - imPix: imPixel size in mas
        - imMax: cutoff for image display (0..1) or in percentile ('0'..'100')
        - imPow: power law applied to image for display. use 0<imPow<1 to show low
            surface brightness features
        - imX, imY: center of image (in mas)
        - imWl0: list of wavelength (um) to show the image default (min, max)
        - imPlx: parallax in mas, to show secondary scales in AU
        - imTight: force image to be limited to given FoV
        - imLegend: show names and position of components (default=True)
        - cmap: color map (default 'inferno')
        - checkImVis: compute visibility from image to check (can be wrong if
            fov is too small)
        - cColors: an optional dictionary to set the color of each components in
            the SED plot
        - showSED: show SED of components (default=True)
        - t3B: baseline for displaying T3 'min', 'max' or 'avg'
        """
        oimodels.FIG_MAX_WIDTH = FIG_MAX_WIDTH
        oimodels.FIG_MAX_HEIGHT = FIG_MAX_HEIGHT

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
            #assert imPix>imFov/500, "the pixel of the synthetic image is too small!"
            if imPix<imFov/500:
                raise Exception("the pixel of the synthetic image is too small!")

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

        if allInOne or perSetup:
            data = oifits.mergeOI(self.data, collapse=False, verbose=False, dMJD=self.dMJD)
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

        if not model is None and not imFov is None and checkImVis:
            allInOne = False
            perSetUp = False
            # -- prepare computing model's images
            print('computing model images and corresponding visibilities:')
            t0 = time.time()
            self.computeModelImages(model=model, imFov=imFov, imPix=imPix,
                                    imX=imX, imY=imY, visibilities=True)
            print('in %.1fs'%(time.time()-t0))
        self._dataAxes = {}
        if perSetup:
            if self.debug:
                print('DEBUG show: perSetup')
            def betterinsname(d):
                n = -int(np.log10(np.ptp(d['WL'])/len(d['WL']))-1)
                f = '%.'+str(n)+'fum'
                return d['insname']+'_'+f%min(d['WL'])+'_'+f%max(d['WL'])+'_'+str(len(d['WL']))

            if perSetup == 'insname':
                insnames = [d['insname'] for d in self.data]
                perSetup = list(set(insnames))
            elif perSetup == 'strict':
                insnames = [betterinsname(d) for d in self.data]
                perSetup = list(set(insnames))
            else:
                # -- defaut -> use simplified names
                insnames = [d['insname'].split(' ')[0].split('_')[0] for d in self.data]
                perSetup = list(set(insnames))

            if type(perSetup)==list:
                data = []
                for s in perSetup:
                    data.append(oifits.mergeOI([self.data[i] for i in range(len(self.data))
                                    if s in insnames[i]],
                                    collapse=False, verbose=False, dMJD=self.dMJD))
                if spectro:
                    useStrict = False
                    for D in data: # for each setup
                        if len(D)>1:
                            if all(['OI_VIS' in d for d in D]):
                                if len(set(['_'.join(sorted(d['OI_VIS'].keys())) for d in D]))>1:
                                    useStrict = True
                    if useStrict:
                        insnames = [betterinsname(d) for d in self.data]
                        perSetup = list(set(insnames))
                        data = []
                        for s in perSetup:
                            data.append(oifits.mergeOI([self.data[i] for i in range(len(self.data))
                                            if s in insnames[i]],
                                            collapse=False, verbose=False, dMJD=self.dMJD))

            for j,g in enumerate(data):
                if self.debug:
                    print('DEBUG show: data', j+1,'/', len(data))

                if 'fit' in g and 'obs' in g['fit']:
                    _obs = g['fit']['obs']
                else:
                    _obs = obs
                if checkImVis:
                    _imoi = self.vfromim
                else:
                    _imoi = None

                oimodels.showOI(g,
                        param=model, fig=self.fig, obs=_obs, logV=logV,
                        logB=logB, showFlagged=showFlagged, showIm=False,
                        spectro=spectro, showUV=showUV, allInOne=True,
                        imFov=None, checkImVis=checkImVis, vWl0=vWl0, imoi=_imoi,
                        showChi2=showChi2, debug=self.debug, bckgGrid=bckgGrid,
                        barycentric=barycentric, autoLimV=autoLimV, t3B=t3B)
                self._dataAxes[perSetup[j]] = oimodels.ai1ax
                self._dataFig = oimodels.ai1fig

                self.fig+=1
                if type(perSetup)==list:
                    plt.suptitle(perSetup[j])
            if not model is None and 'additional residuals' in model:
                if 'plot' in signature(model['additional residuals']).parameters:
                    model['additional residuals'](model, plot=self.fig)
                    self.fig+=1
                else:
                    pass

            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix,imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax, imTight=imTight,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent, imLegend=imLegend,
                               debug=self.debug, vWl0=vWl0, bckgGrid=bckgGrid,
                               barycentric=barycentric)
            return
        elif allInOne:
            if self.debug:
                print('DEBUG show: allInOne')

            if checkImVis:
                print('cannot check visibilites from images with "allInOne=True"')
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
                        if 'OI_CF' in d:
                            _obs.append('CF')
                        if 'OI_FLUX' in d:
                            _obs.append('FLUX')
                        if not 'fit' in d:
                            d['fit'] = {}
                        d['fit']['obs'] = _obs
            if checkImVis:
                imoi = self.vfromim
            else:
                imoi = None

            self._model = oimodels.showOI(self.data, param=model, fig=self.fig,
                    obs=None, logV=logV, logB=logB, showFlagged=showFlagged,
                    spectro=spectro, showUV=showUV, allInOne=allInOne,
                    imFov=None, showIm=False, #imPix=imPix, imPow=imPow, imMax=imMax,
                    #imWl0=imWl0, cmap=cmap, imX=imX, imY=imY,
                    #cColors=cColors, cMarkers=cMarkers
                    checkImVis=checkImVis, imoi=imoi, vWl0=vWl0, showChi2=showChi2,
                    debug=self.debug, bckgGrid=bckgGrid,
                    barycentric=barycentric, autoLimV=autoLimV, t3B=t3B)
            self._dataAxes['ALL'] = oimodels.ai1ax
            self._dataFig = oimodels.ai1fig

            if allInOne:
                self.fig += 1
            else:
                self.fig += len(self.data)
            if not model is None and 'additional residuals' in model:
                if 'plot' in signature(model['additional residuals']).parameters:
                    model['additional residuals'](model, plot=self.fig)
                    self.fig+=1
                else:
                    pass

            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix, imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax, imTight=imTight,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent, imLegend=imLegend,
                               debug=self.debug, vWl0=vWl0, bckgGrid=bckgGrid,
                               barycentric=barycentric)
        else:
            self._model = []
            for i,d in enumerate(data):
                _obs = None
                if 'fit' in d and 'obs' in d['fit']:
                    #print(j, g['fit'])
                    _obs = d['fit']['obs']

                if checkImVis:
                    imoi = [self.vfromim[i]]
                else:
                    imoi = None

                self._model.append(oimodels.showOI([d], param=model, fig=self.fig,
                        obs=_obs, logV=logV, logB=logB, showFlagged=showFlagged,
                        spectro=spectro, showUV=showUV, imFov=None, showIm=False,
                        checkImVis=checkImVis, vWl0=vWl0, bckgGrid=bckgGrid,
                        showChi2=showChi2, debug=self.debug, imoi=imoi,
                        barycentric=barycentric, autoLimV=autoLimV, t3B=t3B))
                self.fig += 1
                self._dataAxes[i] = oimodels.ai1ax
                self._dataFig = oimodels.ai1fig

            if not imFov is None or showSED:
                self.showModel(model=model, imFov=imFov, imPix=imPix, imPlx=imPlx,
                               imX=imX, imY=imY, imPow=imPow, imMax=imMax, imTight=imTight,
                               imWl0=imWl0, cColors=cColors, cMarkers=cMarkers,
                               cmap=cmap, logS=logS, showSED=showSED, showIM=showIM,
                               imPhotCent=showPhotCent, imLegend=imLegend,
                               debug=self.debug, vWl0=vWl0, bckgGrid=bckgGrid,
                               barycentric=barycentric)
        return

    def showModel(self, model='best', imFov=None, imPix=None, imX=0, imY=0,
                  imPow=1, imMax=None, imWl0=None, cColors={}, cMarkers={},
                  showSED=True, showIM=True, fig=None, cmap='inferno',
                  logS=False, imPlx=None, imPhotCent=False, debug=False,
                  imLegend=True, vWl0=None, WL=None, bckgGrid=True,
                  barycentric=False, imTight=False):
        """
        model: parameter dictionnary, describing the model

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

        # -- just if the function is used to show models
        if len(self.data)==0 or (len(self.data)==1 and not WL is None):
            #assert not WL is None, 'specify wavelength vector "WL="'
            if WL is None:
                raise Exception('specify wavelength vector "WL="')
            # -- create fake data
            self.data = [{'WL':np.array(WL), 'fit':{'obs':'|V|'}, 'insname':'',
                          'MJD':[60000.]}]
        if model=='best' and type(self.bestfit) is dict and \
                    'best' in self.bestfit:
            #model = self.bestfit['best']
            model = oimodels.computeLambdaParams(self.bestfit['best'])

        # -- this is actually not needed
        #model = oimodels.computeLambdaParams(model,)

        if not type(model)==dict:
            raise Exception('model should be a dictionnary!')

        if not imFov is None:
            self.computeModelImages(imFov, model=model, imPix=imPix,
                                    imX=imX, imY=imY)
        self.computeModelSpectra(model=model, uncer=False)

        # -- components
        comps = set([k.split(',')[0].strip() for k in model.keys() if ',' in k and not '&dwl' in k])
        symbols = {}
        a, b, c = 0.9, 0.6, 0.1
        colors = [(c,a,b), (a,b,c), (b,c,a),
                  (a,c,b), (c,b,a), (b,a,c),
                  (a,c,c), (c,a,c), (c,c,a),
                  (b,b,c), (b,c,b), (c,b,b),
                  (b,b,b)]
        markers = ['1', '2', '3', '4', '+', 'x'] # 3 branches crosses
        msizes  = [ 8,   8,   8,   8,   6,   6]
        for i, c in enumerate(sorted(comps)):
            symbols[c] = {'m':markers[i%len(markers)],
                          'c':colors[i%len(colors)],
                          's':msizes[i%len(markers)]}
            if c in cColors:
                symbols[c]['c'] = cColors[c]

        # assumes first data set has barycentric correction
        if barycentric and 'barycorr_km/s' in self.data[0]:
            bcorr = (1+self.data[0]['barycorr_km/s']/2.998e5)
        else:
            bcorr = 1.0

        if imWl0 is None and showIM and not imFov is None:
            #imWl0 = self.images['WL'].min(), self.images['WL'].max()
            imWl0 = [np.mean(self.images['WL'])*bcorr]

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
            figHeight =  min(max(nplot, 8), FIG_MAX_HEIGHT)
            figWidth = min(figHeight*nplot, FIG_MAX_WIDTH)
        if figWidth is None and not figHeight is None:
            figWidth = min(figHeight*nplot, FIG_MAX_WIDTH)
        if not figWidth is None and figHeight is None:
            figHeight =  max(figWidth/nplot, FIG_MAX_HEIGHT)
        plt.close(fig)
        plt.figure(fig, figsize=(figWidth, figHeight))
        i = -1 # default, in case only SED
        self._modelAxes = {}
        for i,wl0 in enumerate(imWl0):
            ax = plt.subplot(1, nplot, i+1, aspect='equal')
            if not 'images' in self._modelAxes:
                self._modelAxes['images'] = [ax]
            else:
                self._modelAxes['images'].append(ax)
            if not imPlx is None:
                mas2au = lambda x: x/imPlx
                au2mas = lambda x: x*imPlx
                axx = ax.secondary_xaxis('top', functions=(mas2au, au2mas))
                axy = ax.secondary_yaxis('right', functions=(mas2au, au2mas))
                axx.tick_params(axis='x', labelsize=6, labelcolor=(0.7,0.5,0.2),
                                pad=0)
                axy.tick_params(axis='y', labelsize=6, labelcolor=(0.7,0.5,0.2),
                                pad=0, labelrotation=-60)
                if i==0:
                    axx.set_xlabel('AU at\n'+r'$\varpi$=%.2fmas'%imPlx,
                                    color=(0.7,0.5,0.2), x=0, fontsize=5)
            # -- index of image in cube, closest wavelength
            i0 = np.argmin(np.abs(self.images['WL']*bcorr-wl0))
            # -- normalised image
            im = np.abs(self.images['cube'][i0]/np.max(self.images['cube'][i0]))
            # -- photocenter:
            xphot = np.sum(im*self.images['X'])/np.sum(im)
            yphot = np.sum(im*self.images['Y'])/np.sum(im)

            if not imMax is None:
                if type(imMax)==str:
                    _imMax = np.percentile((im**imPow)[(im**imPow)>0], float(imMax))
                    #_imMax = np.percentile(im**imPow, float(imMax))
                else:
                    _imMax = imMax**imPow
            else:
                _imMax = np.max(im**imPow)

            pc = plt.pcolormesh(self.images['X'], self.images['Y'],
                                im**imPow, cmap=cmap, vmax=_imMax, vmin=0,
                                shading='auto', rasterized=True)
            cb = plt.colorbar(pc, ax=ax,
                              orientation='horizontal' if len(imWl0)>1 else
                              'vertical')
            if imPhotCent:
                plt.plot(xphot, yphot, color='w', marker=r'$\bigotimes$',
                        alpha=0.5, label='photo centre', linestyle='none')
                plt.legend(fontsize=6)
            if '#spatial kernel' in model:
                xk = np.max(self.images['X'])-model['#spatial kernel']
                yk = np.min(self.images['Y'])+model['#spatial kernel']
                c = plt.Circle((xk, yk), model['#spatial kernel']/2,
                                color=(0.8, 0.6, 0.3), linewidth=2, alpha=0.5, label='kernel FWHM')
                plt.gca().add_patch(c)

            Xcb = np.linspace(0,1,5)*_imMax
            XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
            XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
            cb.set_ticks(Xcb)
            cb.set_ticklabels(XcbL)
            cb.ax.tick_params(labelsize=6)

            if imTight:
                ax.set_xlim(np.min(self.images['X']), np.max(self.images['X']))
                ax.set_ylim(np.min(self.images['Y']), np.max(self.images['Y']))

            ax.invert_xaxis()
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            # ax.set_xlabel(r'E $\leftarrow$ x (mas)')
            # if i==0:
            #     ax.set_ylabel(r'y $\rightarrow$ N (mas)')
            ax.set_xlabel(r'$\delta$RA $\leftarrow$ E (mas)')
            if i==0:
               ax.set_ylabel(r'$\delta$Dec N $\rightarrow$ (mas)')

            title = ''
            if not imPow == 1:
                if imPow==0.5:
                    #title = r'$\sqrt{\mathrm{I}} at $'
                    cb.set_label(r'$\sqrt{Flux}$')
                elif imPow<=1 and np.abs(imPow-1/int(1/imPow))<1e-3 and int(1/imPow)<6:
                    #title = 'Image$^{1/%d}$ '%int(1/imPow)
                    #title = r'$\sqrt[%d]{\mathrm{I}}$ at '%int(1/imPow)
                    cb.set_label(r'$\sqrt[%d]{Flux}$'%int(1/imPow))
                else:
                    #title = '$\mathrm{I}^{%.2f}$ at '%imPow
                    cb.set_label('$Flux^{%.2f}$'%imPow)
            else:
                cb.set_label('linear scale')
                title =''
            if len(self.images['WL'])>1:
                n = 2-np.log10(np.median(np.abs(np.diff(self.images['WL']))))
            else:
                n = 3

            title += r'$\lambda$=%.'+str(int(n))+r'f$\mu$m'
            title = title%self.images['WL'][i0]
            if not vWl0 is None:
                title+= '\n v= %.0fkm/s'%((self.images['WL'][i0]*bcorr-vWl0)/self.images['WL'][i0]*299792)
            plt.title(title, fontsize=9, y=1.05 if imPlx else None)

            cmodel = oimodels.computeLambdaParams(model)
            if imLegend:
                # -- show position of each components
                for c in sorted(comps):
                    if c+',x' in model.keys():
                        x = cmodel[c+',x']
                    else:
                        x = 0.0
                    if c+',y' in model.keys():
                        y = cmodel[c+',y']
                    else:
                        y = 0.0
                    if np.isreal(x) and np.isreal(y):
                        plt.plot(x, y, symbols[c]['m'],
                                color=symbols[c]['c'], label=c,
                                markersize=symbols[c]['s'])
                    #plt.plot(x, y, '.w', markersize=8, alpha=0.5)

                if i==0 and len(comps)>0:
                    maxc = 0
                    if len(comps)>0:
                        maxc = max([len(c) for c in comps])
                    if len(imWl0)<4 and maxc<10:
                        ncol=3
                    else:
                        ncol=2
                    plt.legend(fontsize=6, ncol=ncol)

        if showSED:
            ax = plt.subplot(1, nplot, i+2)
            if not 'SED' in self._modelAxes:
                self._modelAxes['SED'] = [ax]
            if not vWl0 is None:
                um2kms = lambda um: (um-vWl0)/um*299792
                kms2um = lambda kms: vWl0*(1 + kms/299792)
                axv = ax.secondary_xaxis('top', functions=(um2kms, kms2um),
                                         color='0.6')
                axv.tick_params(axis='x', labelsize=5,
                                labelbottom=True,
                                labeltop=False,)
                axv.set_xlabel('velocity (km/s)', fontsize=6)

            if len(self.spectra['normalised spectrum WL']):
                key = 'normalised spectrum '
            else:
                key = 'flux '
            for c in sorted(self.spectra[key+'COMP']):
                col = symbols[c]['c']
                w = self.spectra[key+'COMP'][c]>0
                plt.plot(self.spectra[key+'WL'][w]*bcorr, self.spectra[key+'COMP'][c][w],
                        '-', label=c, color=col, linewidth=1.5)
            plt.plot(self.spectra[key+'WL']*bcorr, self.spectra[key+'TOTAL'],
                    '-', label='TOTAL', linewidth=2, color='0.4')
            # -- show imWl0
            #plt.scatter(imWl0, np.interp(imWl0, self.spectra[key+'WL'], self.spectra[key+'TOTAL']),
            #            c=imWl0, cmap='jet', marker='d')
            #plt.plot(imWl0, np.interp(imWl0, self.spectra[key+'WL'], self.spectra[key+'TOTAL']),
            #         'dk')

            if logS:
                plt.yscale('log')
            else:
                plt.ylim(0)
            if bcorr==1:
                plt.xlabel(r'wavelength ($\mu$m)')
            else:
                plt.xlabel(r'barycentric wavelength ($\mu$m)')

            if key=='flux ':
                plt.title('SED', fontsize=9)
            else:
                if US_SPELLING:
                    plt.title('normalized spectra', fontsize=9)
                else:
                    plt.title('normalised spectra', fontsize=9)
                # -- show continuum
                #w = self.spectra[key+'CMASK']
                #plt.plt.plot(self.spectra[key+'WL'][w],
                #             self.spectra[key+'TOTAL'][w],
                #             'sc', label='cont.', alpha=0.5)
            if bckgGrid:
                plt.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
            if not imLegend or len(imWl0)==0:
                plt.legend(fontsize=6)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
        try:
            plt.tight_layout()
        except:
            pass
        return

    def showBestfit(self):
        if not self.bestfit=={}:
            print('chi2 = %f'%self.bestfit['chi2'])
            oimodels.dpfit.dispBest(self.bestfit)
            oimodels.dpfit.dispCor(self.bestfit)
        else:
            print('no fit to show')
        return

    def sparseFitFluxes(self, firstGuess, N={}, initFlux={}, refFlux=None,
                    significance=3.5, fitOnly=None, doNotFit=None,
                    maxfev=5000, ftol=1e-5, epsfcn=1e-6, prior=[]):
        """
        Sparse Discrete Wavelet Transform to model spectrum of components.

        firstGuess: initial model, do not include DWT parameters!
        N: interger or dictionnary. The number of DWT coefficients. These should
            be a power of 2, and no more than half the number of spectral
            channels fitted. The keys of the dictionnary are the components we
            wish to model this way. It is advised to keep both components with
            the same numbers of coefficients, since we do not know a priori which
            component is responsible for the spectral features.
        initFlux: dictionary keyed by components to pass an initial value for
            the average flux. Default is 1 for every component
        refFlux: name of reference component (string) to fix on the wavelet
            spectrum. Its average will be fixed to the given value (1 of value
            given by initFlux). By default, it assumes there is no reference flux
            in the DWT spectra.
        significance: (in sigma) each fit will trim from the fit the DWT
            coefficient with less than that (default=4)

        other parameters: see 'doFit'
        """
        self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)
        self.bestfit = oimodels.sparseFitFluxes(self._merged, firstGuess, N=N,
                                        initFlux=initFlux, refFlux=refFlux,
                                        significance=significance, fitOnly=fitOnly,
                                        doNotFit=doNotFit, maxfev=maxfev, ftol=ftol,
                                        epsfcn=epsfcn, prior=prior)
        self.computeModelSpectra(uncer=False)
        self.bestfit['prior'] = prior
        return

    def computeHalfLightRadiiFromParam(self):
        """
        result stored in self.halfradP
        """
        if not self.bestfit=={}:
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
        #assert self.images!={}, 'run ".computeModelImages" first!'
        if self.images=={}:
            raise Exception('run ".computeModelImages" first!')

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
                                        excludeCentralPix=True, maxRadius=None):
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
        #assert self.images!={}, 'run ".computeModelImages" first!'
        if self.images=={}:
            raise Exception('run ".computeModelImages" first!')

        self.deprojectImages(incl, projang, x0, y0)

        res = {'WL':self.images['WL'], 'HLR':[],  'I':[]}
        scale = self.images['scale']
        Xp, Yp = self.images['Xp'], self.images['Yp']
        R = np.sqrt((Xp-x0)**2+(Yp-y0)**2).flatten()
        PA = np.arctan2(Yp-y0, Xp-x0)

        if maxRadius is None:
            maxRadius = np.max(R)

        if excludeCentralPix:
            r = np.linspace(scale, maxRadius, 2*int(maxRadius/scale))
        else:
            r = np.linspace(0, maxRadius, 2*int(maxRadius/scale))
        res['R'] = r

        for i, wl in enumerate(res['WL']):
            tmp = self.images['cube'][i].flatten()
            if excludeCentralPix:
                I = [tmp[(R<=x)*(R>scale)*(R<=maxRadius)].sum() for x in r]
            else:
                I = [tmp[(R<=x)*(R<=maxRadius)].sum() for x in r]
            res['I'].append(np.array(I))
            res['HLR'].append(np.interp(I[-1]/2, I, r))
        res['HLR'] = np.array(res['HLR'])
        self.halfradI = res
        return

    def computeModelImages(self, imFov, model='best', imPix=None,
                           imX=0, imY=0, visibilities=False, debug=False):
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
        if debug:
            print('D> -- computeModelImages')
        if model=='best' and not self.bestfit=={}:
            model = self.bestfit['best']

        #assert type(model) is dict, "model must be a dictionnary"
        if not type(model) is dict:
            raise Exception("model must be a dictionnary")

        if imPix is None:
            imPix = imFov/101

        if 'model' in self.images and str(self.images['model'])==str(model) and\
            np.abs(imFov-np.ptp(self.images['X']))<1e-6 and\
            np.abs(imPix-np.diff(self.images['X'])[0][0])<1e-6 and\
            np.abs(imX-np.mean(self.images['X']))<1e-6 and\
            np.abs(imY-np.mean(self.images['Y']))<1e-6:
            #print('-- nothing to be done')
            pass
            return

        if len(self._merged)==0:
            # -- merge data to accelerate computations
            self._merged = oifits.mergeOI(self.data, collapse=True, verbose=False, dMJD=self.dMJD)

        if not visibilities:
            # -- fast:
            if debug:
                print('D> on "merged"')
            tmp = [oimodels.VmodelOI(d, model, imFov=imFov, imPix=imPix,
                                     imX=imX, imY=imY) for d in self._merged]
        else:
            # -- slow:
            if debug:
                print('D> on "data"')
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

        # == convolve by a spatial kernel -> done per component
        # if '#spatial kernel' in model:
        #     kfwhm = model['#spatial kernel']
        # else:
        #     kfwhm = None
        # if not kfwhm is None:
        #     # -- kernel
        #     s = kfwhm/(2*np.sqrt(2*np.log(2)))
        #     ker = np.exp(-((res['X']-imX)**2+(res['Y']-imY)**2)/(2*s**2))/(s*np.sqrt(2*np.pi))
        #     # -- for each wavelength in the cube
        #     for i,c in enumerate(res['cube']):
        #         # print('c', c.shape)
        #         #res['cube'][i] = scipy.signal.convolve2d(c, ker, mode='same')
        #         res['cube'][i] = scipy.signal.fftconvolve(c, ker, mode='same')

        res['WL'] = np.array(sorted(res['WL']))

        # -- photocenter
        res['photcent x'] = np.sum(res['X'][None,:,:]*res['cube'], axis=(1,2))/\
                        np.sum(res['cube'], axis=(1,2))

        res['photcent y'] = np.sum(res['Y'][None,:,:]*res['cube'], axis=(1,2))/\
                        np.sum(res['cube'], axis=(1,2))
        self.images = res

        # -- add synthetic visibilities
        if not visibilities:
            return

        syn = []

        for d in self.data:
            # -- assumes OI_VIS and OI_VIS2 have same u,v,wl,mjd structures
            # -- this is ensured while loading data
            if 'OI_VIS' in d:
                key = 'OI_VIS'
            else:
                key = 'OI_VIS2'
            tmp = {'OI_VIS':{}, 'OI_VIS2':{}, 'WL':d['WL'], 'OI_FLUX':{},
                'MJD':d['MJD']}
            wl = np.array([np.argmin(np.abs(x-res['WL'])) for x in d['WL']])
            norm = np.sum(res['cube'][wl,:,:], axis=(1,2))
            _c = np.pi**2/180/3600/1000*1e6
            for k in d[key]:
                # -- dims are uv, wl, x, y
                phi = -2j*_c*(res['X'][None,None,:,:]*d[key][k]['u/wl'][:,:,None,None] +
                              res['Y'][None,None,:,:]*d[key][k]['v/wl'][:,:,None,None])
                # -- complex visibility
                vis = np.sum(res['cube'][wl,:,:][None,:,:,:]*np.exp(phi), axis=(2,3))/norm
                tmp['OI_VIS'][k] = {'|V|':np.abs(vis),
                                    'PHI':np.angle(vis)*180/np.pi,
                                    'u/wl': d[key][k]['u/wl'],
                                    'v/wl': d[key][k]['v/wl'],
                                    'B/wl': d[key][k]['B/wl'],
                                    'MJD':d[key][k]['MJD'],
                                    'MJD2':d[key][k]['MJD2'],
                                    }
                if 'OI_VIS' in d:
                    tmp['OI_VIS'][k]['FLAG'] = d['OI_VIS'][k]['FLAG']
                else:
                    tmp['OI_VIS'][k]['FLAG'] = d[key][k]['FLAG']

                tmp['OI_VIS2'][k] = {'V2':np.abs(vis)**2,
                                    'u/wl': d[key][k]['u/wl'],
                                    'v/wl': d[key][k]['v/wl'],
                                    'B/wl': d[key][k]['B/wl'],
                                    'MJD':d[key][k]['MJD'],
                                    'MJD2':d[key][k]['MJD2'],
                                    }
                if 'OI_VIS2' in d:
                    tmp['OI_VIS2'][k]['FLAG'] = d['OI_VIS2'][k]['FLAG']
                else:
                    tmp['OI_VIS2'][k]['FLAG'] = d[key][k]['FLAG']
            tmp = oimodels.computeDiffPhiOI(tmp, param=model)

            if 'OI_T3' in d:
                tmp['OI_T3'] = {}
                for k in d['OI_T3']:
                    tmp['OI_T3'][k] = { x: d['OI_T3'][k][x].copy() for x in
                        ['formula', 'MJD', 'Bmin/wl', 'Bmax/wl', 'Bavg/wl', 'FLAG', 'MJD2']
                        if x in d['OI_T3'][k]}
            tmp = oimodels.computeT3fromVisOI(tmp)

            if 'OI_FLUX' in d:
                for k in d['OI_FLUX'].keys():
                    tmp['OI_FLUX'][k] = {'FLUX':d['OI_FLUX'][k]['FLAG']*0 +
                                        np.sum(res['cube'], axis=(1,2))[None,:],
                                        'EFLUX':d['OI_FLUX'][k]['FLAG']*0 + 1,
                                        'RFLUX':d['OI_FLUX'][k]['FLAG']*0 +
                                        np.sum(res['cube'], axis=(1,2))[None,:],
                                        'MJD':d['OI_FLUX'][k]['MJD'],
                                        'FLAG':d['OI_FLUX'][k]['FLAG'],
                                        #'MJD2':d['OI_FLUX'][k]['MJD2'],
                                        }
            tmp = oimodels.computeNormFluxOI(tmp, param=model)
            syn.append(tmp)
        if debug:
            print('D> saving synth obs')
        self.vfromim = syn
        return

    def computeModelSpectra(self, model='best', uncer=False, Niter=100):
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
        if model=='best' and not self.bestfit=={}:
            model = self.bestfit['best']
            # -- randomise parameters, using covariance
            if uncer:
                models = oimodels.dpfit.randomParam(self.bestfit, N=Niter,
                                                    x=None)['r_param']

        #assert type(model) is dict, "model must be a dictionnary"
        if not type(model) is dict:
            raise Exception("model must be a dictionnary")

        if len(self._merged)>0:
            self.spectra = _computeSpectra(model, self._merged, models=models)
        else:
            self.spectra = _computeSpectra(model, self.data, models=models)

def _computeSpectra(model, data, models):
    """
    model: dictionnary
    data: oi.data or oi._merged
    models: dict containing the various computations:
        'flux TOTAL': total flux (1D, same as 'flux WL')
        'flux WL': wavelength vector (in um)
        'flux COMP': dict per components
        'err flux COMP': error on flux
        ...

    """
    allWLc  = [] # -- continuum -> absolute flux
    allWLs  = [] # -- with spectral lines -> normalised flux
    allMJD  = []
    allCont = []
    Nr = None
    for i,o in enumerate(data):
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

        if 'fit' in o and 'Nr' in o['fit']:
            if Nr is None:
                Nr = o['fit']['Nr']
            else:
                Nr = max(Nr, o['fit']['Nr'])

        w = np.zeros(o['WL'].shape)
        closest = []
        for WR in o['fit']['wl ranges']:
            w += (o['WL']>=WR[0])*(o['WL']<=WR[1])
            closest.append(np.argmin(np.abs(o['WL']-0.5*(WR[0]+WR[1]))))
        o['WL mask'] = np.bool_(w)
        if not any(o['WL mask']):
            for clo in closest:
                o['WL mask'][clo] = True
        if 'obs' in o['fit'] and 'NFLUX' in o['fit']['obs']:
            allWLs.extend(list(o['WL'][o['WL mask']]))
        else:
            allWLc.extend(list(o['WL'][o['WL mask']]))
        try:
            allMJD += list(o['MJD'])
        except:
            pass

    allWLc = np.array(sorted(list(set(allWLc))))
    allWLs = np.array(sorted(list(set(allWLs))))
    allMJD = np.array(sorted(list(set(allMJD))))

    #print('SED:', allWLc)
    #print('spe:', allWLs)
    M = {'model':model}
    if len(allWLc):
        allWL = {'WL':allWLc, 'fit':{'obs':[]}, 'MJD':allMJD} # minimum required
        if not Nr is None:
            allWL['fit']['Nr'] = Nr
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
        allWL = {'WL':allWLs,
                 'MJD':allMJD,
                 'fit':{'obs':['NFLUX'],
                 'continuum ranges':allCont}} # minimum required
        if not Nr is None:
            allWL['fit']['Nr'] = Nr

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
        if 'WL cont' in tmp:
            M['normalised spectrum CMASK'] = tmp['WL cont']
        else:
            M['normalised spectrum CMASK'] = np.ones(len(allWLs), dtype=bool)

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
    return M

def _checkObs(data, obs):
    """
    data: OI dict
    obs: list of observable in ['|V|', 'V2', 'DPHI', 'N|V|', 'T3PHI', 'FLUX',
                                'NFLUX', 'CF']

    returns list of obs actually in data
    """
    ext = {'|V|':'OI_VIS', 'N|V|':'OI_VIS', 'DPHI':'OI_VIS', 'PHI':'OI_VIS',
           'V2':'OI_VIS2',
           'T3PHI':'OI_T3', 'T3AMP':'OI_T3',
           'FLUX':'OI_FLUX',
           'NFLUX':'OI_FLUX',
           'CF': 'OI_CF'
           }
    return [o for o in obs if o in ext and ext[o] in data]

def _checkSetupFit(fit):
    """
    check for setupFit
    """
    keys = {'min error':dict, 'min relative error':dict,
            'max error':dict, 'max relative error':dict,
            'mult error':dict,
            'obs':list,
            'wl ranges':list,
            'baseline ranges':list,
            'MJD ranges':list,
            'Nr':int, 'wl kernel':(float,int),
            'continuum ranges':list,
            'ignore negative flux':bool,
            'prior':list,
            'DPHI order':int,
            'N|V| order':int,
            'NFLUX order': int,
            'correlations': (bool, dict),
            'spatial kernel': float,
            'smear':(int),
            }

    ok = True
    if not 'obs' in fit:
        raise Exception('list of observables should be defined (see method setupFit)')
    knownObs =  ['V2', '|V|', 'N|V|', 'PHI', 'DPHI', 'T3PHI', 'T3AMP', 'FLUX', 'NFLUX', 'CF']
    for k in fit['obs']:
        if k not in knownObs:
            raise Exception("Unknown observable '"+k+"', not in "+str(knownObs) )

    for k in fit.keys():
        if not k in keys.keys():
            print('!WARNING! unknown fit setup "'+k+'"')
            ok = False
        elif type(keys[k]) == tuple:
            if not type(fit[k]) in keys[k]:
                print('!WARNING! fit setup "'+k+'" should be one of types', keys[k])
                ok = False
        elif type(fit[k]) != keys[k]:
            print('!WARNING! fit setup "'+k+'" should be of type', keys[k])
            ok = False
    return ok

def _checkPrior(prior):
    if not type(prior)==list:
        print('\033[31mERROR\033[0m: "prior" should be a list of tuples')
        return False
    if not all([type(p)==tuple for p in prior]):
        print('\033[31mERROR\033[0m: "prior" should be a list of tuples')
        return False
    if not all([len(p) in [3,4] for p in prior]):
        print('\033[31mERROR\033[0m: priors should be tuples of length 3 or 4')
        return False
    test = [type(p[0])==str and p[1] in ['=', '<', '>', '<=', '>='] for p in prior]
    if not all(test):
        print('\033[31mERROR\033[0m: -ed tuple(s) in the "prior" list')
        for i,p in enumerate(prior):
            if not test[i]:
                print(' ->', p)

        return False
    return True

def _recsizeof(s):
    """
    recursive size (in Bytes) of s (useful for debugging!)

    divide by 1024**2 for megabytes
    """
    if type(s)==dict:
        tmp = sys.getsizeof(s)
        for k in s.keys():
            tmp += _recsizeof(s[k])
        return tmp
    elif type(s)==list or type(s)==tuple or type(s)==np.ndarray:
        tmp = sys.getsizeof(s)
        for x in s:
            tmp += _recsizeof(x)
        return tmp
    else:
        return sys.getsizeof(s)


if __name__ == "__main__":
    pass
