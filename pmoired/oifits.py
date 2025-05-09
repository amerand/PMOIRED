import copy
import glob
import os
from collections import OrderedDict

import numpy as np
from astropy.io import fits
import scipy.signal
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as aU

def _isiterable(x):
    res = True
    try:
        iter(x)
    except:
        res = False
    return res

def _globlist(filenames, strict=False):
    if type(filenames)!=str and _isiterable(filenames):
        res = []
        for f in filenames:
            res.extend(_globlist(f, strict=strict))
        return res
    assert type(filenames)==str, 'cannot find file(s) for '+str(filenames)

    if os.path.exists(filenames):
        return [filenames]
    else:
        g = sorted(glob.glob(filenames))
        if not strict and len(g)==0:
            print('\033[31mWARNING\033[0m no file(s) found for '+str(filenames))
        else:
            assert len(g)>0, 'no file(s) found for '+str(filenames)
        return g

def loadOI(filename, insname=None, targname=None, verbose=True,
           withHeader=False, medFilt=None, tellurics=None, debug=False,
           binning=None, useTelluricsWl=True, barycentric=False, ignoreCF=False,
           wlOffset=0.0, orderedWl=True):
    """
    load OIFITS "filename" and return a dict:

    'WL': wavelength (1D array)
    'OI_VIS2': dict index by baseline. each baseline is a dict as well:
        ['V2', 'EV2', 'u', 'v', 'MJD'] v2 and ev2 1d array like wl, u and v are scalar
    'OI_VIS': dict index by baseline. each baseline is a dict as well:
        ['|V|', 'E|V|', 'PHI', 'EPHI', 'u', 'v', 'MJD'] (phases in degrees)
    'OI_T3': dict index by baseline. each baseline is a dict as well:
        [AMP', 'EAMP', 'PHI', 'EPHI', 'u1', 'v1', 'u2', 'v2', 'MJD'] (phases in degrees)

    can contains other things, *but None can start with 'OI_'*:
    'filename': name of the file
    'insname': name of the instrument (default: all of them)
    'targname': name of the instrument (default: single one if only one)
    'header': full header of the file if "withHeader" is True (default False)

    All variables are keyed by the name of the baseline (or triplet). e.g.:

    oi['OI_VIS2']['V2'][baseline] is a 2D array: shape (len(u), len(wl))

    One can specify "insname" for instrument name. If None provided, returns as
    many dictionnary as they are instruments (either single dict for single
    instrument, of list of dictionnary).

    binning: binning factor (integer)
    useTelluricsWL: use wavelength calibration provided by the tellurics (default==True)
    barycentric: compute barycentric velocities (default=False)

    """
    if debug:
        print('DEBUG: loadOI', type(filename), type(filename)==np.str_)#filename)
    if tellurics is True:
        tellurics = None

    if type(filename)!=str and type(filename)!=np.str_: # for Guillaume B
        res = []
        for f in filename:
            tmp = loadOI(f, insname=insname, withHeader=withHeader,
                         medFilt=medFilt, tellurics=tellurics, targname=targname,
                         verbose=verbose, debug=debug, binning=binning,
                         useTelluricsWl=useTelluricsWl, barycentric=barycentric,
                         wlOffset=wlOffset)
            if type(tmp)==list:
                res.extend(tmp)
            elif type(tmp)==dict:
                res.append(tmp)
        return res

    res = {}
    # -- memmap=False assumes files are small so all is leaded at once
    # -- memmap=True (default) is bad if one reopen the file multiple times!
    # see https://docs.astropy.org/en/stable/io/fits/index.html#working-with-large-files
    h = fits.open(filename, memmap=False)

    # -- how many instruments?
    instruments = []
    ins2targ = {}
    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_WAVELENGTH':
            instruments.append(hdu.header['INSNAME'])
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_TARGET':
            targets = {hdu.data['TARGET'][i].strip():hdu.data['TARGET_ID'][i] for
                        i in range(len(hdu.data['TARGET']))}
            # -- weird case when targets is defined multiple times
            targets = {}
            for i in range(len(hdu.data['TARGET'])):
                k = hdu.data['TARGET'][i].strip()
                if not k in targets:
                    targets[k] = hdu.data['TARGET_ID'][i]
                else:
                    if type(targets[k])!=list:
                        targets[k] = [targets[k], hdu.data['TARGET_ID'][i]]
                    else:
                        targets[k].append(hdu.data['TARGET_ID'][i])
        if 'INSNAME' in hdu.header and 'XTENSION' in hdu.header and\
            hdu.header['XTENSION'].strip()=='BINTABLE':
            if 'TARGET_ID' in hdu.data.columns.names:
                if not hdu.header['INSNAME'] in ins2targ:
                    ins2targ[hdu.header['INSNAME']] = set(hdu.data['TARGET_ID'])
                else:
                    ins2targ[hdu.header['INSNAME']] = set(list(ins2targ[hdu.header['INSNAME']])+
                                                          list(hdu.data['TARGET_ID']))
    # -- keep only targets for the instrument, if specified
    if not insname is None and insname in ins2targ:
        try:
            targets = {t:targets[t] for t in targets if targets[t] in ins2targ[insname]}
        except:
            print(targets, ins2targ)

    if targname is None and len(targets)==1:
        targname = list(targets.keys())[0]
    assert targname in targets.keys(), 'unknown targname "'+str(targname)+'", '+\
        'should be within ['+', '.join(['"'+t+'"' for t in list(targets.keys())])+']'

    if insname is None:
        if len(instruments)==1:
            insname = instruments[0]
        else:
            h.close()
            print('insname not specified, loading %s'%str(instruments))
            # -- return list: one dict for each insname
            return [loadOI(filename, insname=ins, withHeader=withHeader,
                           verbose=verbose, medFilt=medFilt,
                           useTelluricsWl=useTelluricsWl,
                           barycentric=barycentric) for ins in instruments]

    assert insname in instruments, 'unknown instrument "'+insname+'", '+\
        'should be in ['+', '.join(['"'+t+'"' for t in instruments])+']'

    res['insname'] = insname
    # -- spectral resolution
    res['filename'] = filename
    res['targname'] = targname

    # -- for now, only catching ESO pipelines, in the future we can add more
    if 'PROCSOFT' in h[0].header:
        res['pipeline'] = h[0].header['PROCSOFT']
    elif 'ESO PRO REC1 PIPE ID' in h[0].header:
        res['pipeline'] = h[0].header['ESO PRO REC1 PIPE ID']
    else:
        res['pipeline'] = ''

    if 'LST' in h[0].header:
        res['LST'] = h[0].header['LST']/3600.
    else:
        res['LST'] = None

    if withHeader:
        res['header'] = h[0].header

    if verbose:
        print('loadOI: loading', res['filename'])
        print('  > insname:', '"'+insname+'"','targname:', '"'+targname+'"',
                'pipeline:', '"'+res['pipeline']+'"')

    # -- wavelength
    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_WAVELENGTH' and\
            hdu.header['INSNAME']==insname:
            # -- OIFITS in m, here we want um
            res['WL']  = np.array(hdu.data['EFF_WAVE'], dtype=np.float64)*1e6 + wlOffset
            res['dWL'] = np.array(hdu.data['EFF_BAND'], dtype=np.float64)*1e6

            # -- make sure data are sorted by increasing wavelength
            #wls = np.arange(len(res['WL']))
            #wls = np.argsort(res['WL'])
            #res['WL'] = res['WL'][wls]
            #res['dWL'] = res['dWL'][wls]

    assert 'WL' in res, 'OIFITS is inconsistent: no wavelength table for insname="%s"'%(insname)

    if 'GRAVITY' in insname:
        try:
            # -- VLTI GRAVITY 4T specific
            OPL = {}
            for i in range(4):
                Tel = h[0].header['ESO ISS CONF STATION%d'%(i+1)]
                opl = 0.5*(h[0].header['ESO DEL DLT%d OPL START'%(i+1)] +
                           h[0].header['ESO DEL DLT%d OPL END'%(i+1)])
                opl += h[0].header['ESO ISS CONF A%dL'%(i+1)]
                OPL[Tel] = opl
            res['OPL'] = OPL
            T = np.mean([h[0].header['ESO ISS TEMP TUN%d'%i] for i in [1,2,3,4]]) # T in C
            P = h[0].header['ESO ISS AMBI PRES'] # pressure in mbar
            H = h[0].header['ESO ISS AMBI RHUM'] # relative humidity: TODO outside == inside probably no ;(
            #print('T(C), P(mbar), H(%)', T, P, H)
            res['n_lab'] = n_JHK(res['WL'].astype(np.float64), 273.15+T, P, H)
        except:
            #print('warning: could not read DL positions')
            pass

    oiarrays = {}
    # -- build OI_ARRAY dictionnary to name the baselines
    for ih, hdu in enumerate(h):
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_ARRAY':
            arrname = hdu.header['ARRNAME'].strip()
            oiarrays[arrname] = dict(zip(hdu.data['STA_INDEX'],
                                         np.char.strip(hdu.data['STA_NAME'])))
    if oiarrays=={} and 'TELESCOP' in h[0].header and h[0].header['TELESCOP']=='VLTI':
        print('  > \033[33mWarning: no OI_ARRAY extension, guessing from header (VLTI)\033[0m')
        tmp = {}
        for i in range(8):
            k = 'ESO ISS CONF STATION%d'%(i+1)
            if k in h[0].header:
                tmp[i+1] = h[0].header[k].strip()
        oiarrays['VLTI'] = tmp

    if oiarrays=={}:
        print('  > \033[33mWarning: no OI_ARRAY extension, telescopes will have default names\033[0m')
    #print('oiarrays:', oiarrays)
    # -- assumes there is only one array!
    #oiarray = dict(zip(h['OI_ARRAY'].data['STA_INDEX'],
    #               np.char.strip(h['OI_ARRAY'].data['STA_NAME'])))

    # -- V2 baselines == telescopes pairs
    res['OI_VIS2'] = {}
    res['OI_VIS'] = {}
    res['OI_T3'] = {}
    res['OI_FLUX'] = {}
    res['OI_CF'] = {}
    ignoredTellurics = False
    for ih, hdu in enumerate(h):
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='TELLURICS':
            if not tellurics is False:
                if len(hdu.data['TELL_TRANS'])==len(res['WL']):
                    res['TELLURICS'] = hdu.data['TELL_TRANS']
                    if useTelluricsWl and 'CORR_WAVE' in [c.name for c in hdu.data.columns]:
                        # -- corrected wavelength
                        res['WL'] = hdu.data['CORR_WAVE']*1e6
                    res['PWV'] = hdu.header['PWV']
            else:
                ignoredTellurics = True

        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_FLUX' and\
                    hdu.header['INSNAME']==insname:
            w = wTarg(hdu, targname, targets)
            if not any(w):
                print('  > \033[33mWARNING\033[0m: no data in OI_FLUX [HDU #%d]  for target="%s"/target_id='%(
                      ih, targname), targets[targname])
                continue
            try:
                oiarray = oiarrays[hdu.header['ARRNAME'].strip()]
                sta1 = [oiarray[s] for s in hdu.data['STA_INDEX']]
            except:
                sta1 = ['STA'+str(s) for s in hdu.data['STA_INDEX']]
            for k in set(sta1):
                # --
                w = (np.array(sta1)==k)*wTarg(hdu, targname, targets)
                try:
                    # some GRAVITY Data have non-standard naming :(
                    res['OI_FLUX'][k] = {'FLUX':hdu.data['FLUX'][w].reshape(w.sum(), -1),
                                         'EFLUX':hdu.data['FLUXERR'][w].reshape(w.sum(), -1),
                                         'FLAG':hdu.data['FLAG'][w].reshape(w.sum(), -1),
                                         'MJD':hdu.data['MJD'][w],
                                         'MJD2':hdu.data['MJD'][w][:,None]+0*res['WL'][None,:],
                                         }
                except:
                    res['OI_FLUX'][k] = {'FLUX':hdu.data['FLUXDATA'][w].reshape(w.sum(), -1),
                                         'EFLUX':hdu.data['FLUXERR'][w].reshape(w.sum(), -1),
                                         'FLAG':hdu.data['FLAG'][w].reshape(w.sum(), -1),
                                         'MJD':hdu.data['MJD'][w],
                                         'MJD2':hdu.data['MJD'][w][:,None]+0*res['WL'][None,:],
                                         }
                if any(w):
                    res['OI_FLUX'][k]['FLAG'] = np.logical_or(res['OI_FLUX'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_FLUX'][k]['FLUX']))
                    res['OI_FLUX'][k]['FLAG'] = np.logical_or(res['OI_FLUX'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_FLUX'][k]['EFLUX']))

        elif 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_VIS2' and\
                    hdu.header['INSNAME']==insname:
            #w = hdu.data['TARGET_ID']==targets[targname]
            w = wTarg(hdu, targname, targets)

            if not any(w):
                print('  > \033[33mWARNING\033[0m: no data in OI_VIS2 [HDU #%d]  for target="%s"/target_id='%(
                            ih, targname), targets[targname])
                continue
            try:
                oiarray = oiarrays[hdu.header['ARRNAME'].strip()]
                sta2 = [oiarray[s[0]]+oiarray[s[1]] for s in hdu.data['STA_INDEX']]
            except:
                sta2 = ['STA'+str(s[0])+'STA'+str(s[1]) for s in hdu.data['STA_INDEX']]

            #print('     sta2:', sta2)
            if debug:
                print('DEBUG: loading OI_VIS2', set(sta2))
            for k in set(sta2):
                w = (np.array(sta2)==k)*wTarg(hdu, targname, targets)
                if k in res['OI_VIS2'] and any(w):
                    for k1, k2 in [('V2', 'VIS2DATA'), ('EV2', 'VIS2ERR'), ('FLAG', 'FLAG')]:
                        res['OI_VIS2'][k][k1] = np.append(res['OI_VIS2'][k][k1],
                                                          hdu.data[k2][w].reshape(w.sum(), -1), axis=0)
                    for k1, k2 in [('u', 'UCOORD'), ('v', 'VCOORD'), ('MJD','MJD')]:
                        res['OI_VIS2'][k][k1] = np.append(res['OI_VIS2'][k][k1],
                                                          hdu.data[k2][w])
                    tmp = hdu.data['UCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS2'][k]['u/wl'] = np.append(res['OI_VIS2'][k]['u/wl'],
                                                          tmp, axis=0)
                    tmp = hdu.data['VCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS2'][k]['v/wl'] = np.append(res['OI_VIS2'][k]['v/wl'], tmp, axis=0)
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['V2']))
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['EV2']))
                    res['OI_VIS2'][k]['MJD2'] = res['OI_VIS2'][k]['MJD'][:,None]+0*res['WL'][None,:]
                elif any(w):
                    res['OI_VIS2'][k] = {'V2':hdu.data['VIS2DATA'][w].reshape(w.sum(), -1),
                                         'EV2':hdu.data['VIS2ERR'][w].reshape(w.sum(), -1),
                                         'u':hdu.data['UCOORD'][w],
                                         'v':hdu.data['VCOORD'][w],
                                         'MJD':hdu.data['MJD'][w],
                                         'MJD2':hdu.data['MJD'][w][:,None]+0*res['WL'][None,:],
                                         'u/wl': hdu.data['UCOORD'][w][:,None]/
                                                res['WL'][None,:],
                                         'v/wl': hdu.data['VCOORD'][w][:,None]/
                                                res['WL'][None,:],
                                         'FLAG':hdu.data['FLAG'][w].reshape(w.sum(), -1)
                                        }
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['V2']))
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['EV2']))
                if any(w):
                    res['OI_VIS2'][k]['B/wl'] = np.sqrt(res['OI_VIS2'][k]['u/wl']**2+
                                                        res['OI_VIS2'][k]['v/wl']**2)
                    res['OI_VIS2'][k]['PA'] = np.angle(res['OI_VIS2'][k]['v/wl']+
                                                    1j*res['OI_VIS2'][k]['u/wl'], deg=True)

        # -- V baselines == telescopes pairs
        elif 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_VIS' and\
                    hdu.header['INSNAME']==insname:
            #w = hdu.data['TARGET_ID']==targets[targname]
            w = wTarg(hdu, targname, targets)
            if not any(w):
                print('  > \033[33mWARNING\033[0m: no data in OI_VIS [HDU #%d]  for target="%s"/target_id='%(
                            ih, targname), targets[targname])
                continue
            try:
                oiarray = oiarrays[hdu.header['ARRNAME'].strip()]
                sta2 = [oiarray[s[0]]+oiarray[s[1]] for s in hdu.data['STA_INDEX']]
            except:
                sta2 = ['STA'+str(s[0])+'STA'+str(s[1]) for s in hdu.data['STA_INDEX']]


            if 'AMPTYP' in hdu.header and hdu.header['AMPTYP'] == 'correlated flux':
                ext = 'OI_CF'
                vis = 'CF'
                for k in hdu.header:
                    if k.startswith('TTYPE') and hdu.header[k].strip()=='VISAMP':
                        _unit = hdu.header[k.replace('TYPE','UNIT')].strip()
                if 'units' in res:
                    res['units']['CF'] = _unit
                else:
                    res['units']={'CF':_unit}
                if debug:
                    print('DEBUG: loading OI_CF', set(sta2))

            else:
                ext = 'OI_VIS'
                vis = '|V|'
                if debug:
                    print('DEBUG: loading OI_VIS', set(sta2))

            for k in hdu.header:
                if k.startswith('TTYPE') and hdu.header[k].strip()=='VISPHI':
                    try:
                        _unit = hdu.header[k.replace('TYPE','UNIT')].strip()
                    except:
                        _unit = 'None'
            _unit = 'deg'
            if 'units' in res:
                res['units']['PHI'] = _unit
            else:
                res['units']={'PHI':_unit}

            for k in set(sta2):
                w = (np.array(sta2)==k)*wTarg(hdu, targname, targets)
                # if debug:
                #    print(' | ', k, any(w))
                if k in res[ext] and any(w):
                    for k1, k2 in [(vis, 'VISAMP'), ('E'+vis, 'VISAMPERR'),
                                    ('PHI', 'VISPHI'), ('EPHI', 'VISPHIERR'),
                                    ('FLAG', 'FLAG')]:
                        res[ext][k][k1] = np.append(res[ext][k][k1],
                                                         hdu.data[k2][w].reshape(w.sum(), -1), axis=0)
                    for k1, k2 in [('u', 'UCOORD'), ('v', 'VCOORD'), ('MJD', 'MJD')]:
                        res[ext][k][k1] = np.append(res[ext][k][k1],
                                                          hdu.data[k2][w])
                    tmp = hdu.data['UCOORD'][w][:,None]/res['WL'][None,:]
                    res[ext][k]['u/wl'] = np.append(res[ext][k]['u/wl'], tmp, axis=0)
                    tmp = hdu.data['VCOORD'][w][:,None]/res['WL'][None,:]
                    res[ext][k]['v/wl'] = np.append(res[ext][k]['v/wl'], tmp, axis=0)
                    res[ext][k]['FLAG'] = np.logical_or(res[ext][k]['FLAG'],
                                                             ~np.isfinite(res[ext][k][vis]))
                    res[ext][k]['FLAG'] = np.logical_or(res[ext][k]['FLAG'],
                                                             ~np.isfinite(res[ext][k]['E'+vis]))

                elif any(w):
                    res[ext][k] = {vis:hdu.data['VISAMP'][w].reshape(w.sum(), -1),
                                   'E'+vis:hdu.data['VISAMPERR'][w].reshape(w.sum(), -1),
                                   'MJD':hdu.data['MJD'][w],
                                   'MJD2':hdu.data['MJD'][w][:,None]+0*res['WL'][None,:],
                                   'u':hdu.data['UCOORD'][w],
                                   'v':hdu.data['VCOORD'][w],
                                   'u/wl': hdu.data['UCOORD'][w][:,None]/
                                            res['WL'][None,:],
                                   'v/wl': hdu.data['VCOORD'][w][:,None]/
                                            res['WL'][None,:],
                                   'FLAG':hdu.data['FLAG'][w].reshape(w.sum(), -1)
                                    }
                    try:
                        # -- weird bug in some files from Alex
                        res[ext][k]['PHI'] = hdu.data['VISPHI'][w].reshape(w.sum(), -1)
                        res[ext][k]['EPHI'] = hdu.data['VISPHIERR'][w].reshape(w.sum(), -1)
                    except:
                        res[ext][k]['PHI'] = 0.0*res[ext][k][vis]
                        res[ext][k]['EPHI'] = 0.0*res[ext][k][vis]+1.0

                if any(w):
                    res[ext][k]['B/wl'] = np.sqrt(res[ext][k]['u/wl']**2+res[ext][k]['v/wl']**2)
                    res[ext][k]['PA'] = np.angle(res[ext][k]['v/wl']+1j*res[ext][k]['u/wl'], deg=True)

                    res[ext][k]['FLAG'] = np.logical_or(res[ext][k]['FLAG'],
                                                             ~np.isfinite(res[ext][k][vis]))
                    res[ext][k]['FLAG'] = np.logical_or(res[ext][k]['FLAG'],
                                                              ~np.isfinite(res[ext][k]['E'+vis]))


    if res['OI_CF'] == {}:
        if debug:
            print('no sed flux found')
        res.pop('OI_CF')

    # -- recollect baselines, in case multiple HDUs were read
    sta2 = []
    for k in ['OI_VIS', 'OI_VIS2', 'OI_CF']:
        if k in res:
            sta2 += list(res[k].keys())
            if debug:
                print('baselines for:', k, sorted(res[k].keys()))
    sta2 = list(set(sta2))

    res = match_VIS_VIS2_CF(res, debug=debug, ignoreCF=ignoreCF)

    if debug:
        print('checking for missing baselines for OI_T3')
        print('known baselines are', sta2)
    M = [] # missing baselines in T3
    for ih, hdu in enumerate(h):
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_T3' and\
                    hdu.header['INSNAME']==insname:
            #w = hdu.data['TARGET_ID']==targets[targname]
            w = wTarg(hdu, targname, targets)
            if not any(w):
                print('  > \033[33mWARNING\033[0m: no data in OI_T3 [HDU #%d]  for target="%s"/target_id='%(
                            ih, targname), targets[targname])
                continue
            # -- T3 baselines == telescopes pairs
            try:
                oiarray = oiarrays[hdu.header['ARRNAME'].strip()]
                sta3 = [oiarray[s[0]]+oiarray[s[1]]+oiarray[s[2]] for s in hdu.data['STA_INDEX']]
            except:
                sta3 = ['STA'+str(s[0])+'STA'+str(s[1])+'STA'+str(s[2])
                        for s in hdu.data['STA_INDEX']]

            # -- limitation: assumes all telescope have same number of char!
            n = len(sta3[0])//3 # number of char per telescope
            _unit = 'deg'
            for k in hdu.header:
                if k.startswith('TTYPE') and hdu.header[k].strip()=='T3PHI' and\
                        k.replace('TYPE','UNIT') in hdu.header:
                    _unit = hdu.header[k.replace('TYPE','UNIT')].strip()
            if 'units' in res:
                res['units']['T3PHI'] = _unit
            else:
                res['units']={'T3PHI':_unit}

            for k in set(sta3):
                w = (np.array(sta3)==k)*wTarg(hdu, targname, targets)
                # -- find triangles
                t, s, m = [], [], []
                # -- first baseline
                if k[:2*n] in sta2 or k[:2*n] in M:
                    t.append(k[:2*n])
                    s.append(1)
                elif k[n:2*n]+k[:n] in sta2 or k[:2*n] in M:
                    t.append(k[n:2*n]+k[:n])
                    s.append(-1)
                else:
                    t.append(k[:2*n])
                    s.append(1)
                    M.append(k[:2*n])

                # -- second baseline
                if k[n:] in sta2 or k[n:] in M:
                    t.append(k[n:])
                    s.append(1)
                elif k[2*n:3*n]+k[n:2*n] in sta2 or k[2*n:3*n]+k[n:2*n] in M:
                    t.append(k[2*n:3*n]+k[n:2*n])
                    s.append(-1)
                else:
                    t.append(k[n:])
                    s.append(1)
                    M.append(k[n:])

                # -- third baseline
                if k[2*n:3*n]+k[:n] in sta2 or k[2*n:3*n]+k[:n] in M:
                    t.append(k[2*n:3*n]+k[:n])
                    s.append(1)
                elif k[:n]+k[2*n:3*n] in sta2 or k[:n]+k[2*n:3*n] in M:
                    t.append(k[:n]+k[2*n:3*n])
                    s.append(-1)
                else:
                    t.append(k[2*n:3*n]+k[:n])
                    s.append(1)
                    M.append(k[2*n:3*n]+k[:n])

                if k in res['OI_T3'] and any(w):
                    for k1, k2 in [('T3AMP', 'T3AMP'), ('ET3AMP', 'T3AMPERR'),
                                    ('T3PHI', 'T3PHI'), ('ET3PHI', 'T3PHIERR'),
                                    ('FLAG', 'FLAG')]:
                        res['OI_T3'][k][k1] = np.append(res['OI_T3'][k][k1],
                                                        hdu.data[k2][w].reshape(w.sum(), -1), axis=0)
                    for k1, k2 in [('u1', 'U1COORD'), ('u2', 'U2COORD'),
                                   ('v1', 'V1COORD'), ('v2', 'V2COORD'),
                                   ('MJD', 'MJD')]:
                        res['OI_T3'][k][k1] = np.append(res['OI_T3'][k][k1],
                                                         hdu.data[k2][w])
                    # -- commented for MATISSE, otherwise things get screwed?
                    # res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                    #                                         ~np.isfinite(res['OI_T3'][k]['T3AMP']))
                    # res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                    #                                         ~np.isfinite(res['OI_T3'][k]['ET3AMP']))
                    res['OI_T3'][k]['MJD2'] = res['OI_T3'][k]['MJD'][:,None] + 0*res['WL'][None,:]

                elif any(w):
                    res['OI_T3'][k] = {'T3AMP':hdu.data['T3AMP'][w].reshape(w.sum(), -1),
                                       'ET3AMP':hdu.data['T3AMPERR'][w].reshape(w.sum(), -1),
                                       'T3PHI':hdu.data['T3PHI'][w].reshape(w.sum(), -1),
                                       'ET3PHI':hdu.data['T3PHIERR'][w].reshape(w.sum(), -1),
                                       'MJD':hdu.data['MJD'][w],
                                       'u1':hdu.data['U1COORD'][w],
                                       'v1':hdu.data['V1COORD'][w],
                                       'u2':hdu.data['U2COORD'][w],
                                       'v2':hdu.data['V2COORD'][w],
                                       'formula': (s, t),
                                       'FLAG':hdu.data['FLAG'][w].reshape(w.sum(), -1)
                                        }
                if any(w):
                    res['OI_T3'][k]['B1'] = np.sqrt(res['OI_T3'][k]['u1']**2+res['OI_T3'][k]['v1']**2)
                    res['OI_T3'][k]['B2'] = np.sqrt(res['OI_T3'][k]['u2']**2+res['OI_T3'][k]['v2']**2)
                    res['OI_T3'][k]['B3'] = np.sqrt((res['OI_T3'][k]['u1']+res['OI_T3'][k]['u2'])**2+
                                                    (res['OI_T3'][k]['v1']+res['OI_T3'][k]['v2'])**2)
                    bmax = np.maximum(res['OI_T3'][k]['B1'], res['OI_T3'][k]['B2'])
                    bmax = np.maximum(res['OI_T3'][k]['B3'], bmax)
                    bmin = np.minimum(res['OI_T3'][k]['B1'], res['OI_T3'][k]['B2'])
                    bmin = np.minimum(res['OI_T3'][k]['B3'], bmin)

                    bavg = (res['OI_T3'][k]['B1'] +
                            res['OI_T3'][k]['B2'] +
                            res['OI_T3'][k]['B3'])/3

                    res['OI_T3'][k]['Bmax/wl'] = bmax[:,None]/res['WL'][None,:]
                    res['OI_T3'][k]['Bmin/wl'] = bmin[:,None]/res['WL'][None,:]
                    res['OI_T3'][k]['Bavg/wl'] = bavg[:,None]/res['WL'][None,:]

                    # -- commented for MATISSE, otherwise things get screwed
                    # res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                    #                                        ~np.isfinite(res['OI_T3'][k]['T3AMP']))
                    # res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                    #                                         ~np.isfinite(res['OI_T3'][k]['ET3AMP']))
                res['OI_T3'][k]['MJD2'] = res['OI_T3'][k]['MJD'][:,None] + 0*res['WL'][None,:]

    if not binning is None:
        res = _binOI(res, binning=binning, medFilt=medFilt)

    key = 'OI_VIS'
    if res['OI_VIS']=={} and res['OI_T3']=={}:
        res.pop('OI_VIS')
        key = 'OI_VIS2'
    if res['OI_VIS2']=={}:
        res.pop('OI_VIS2')
        if key == 'OI_VIS2':
            key = 'OI_CF'
    if 'OI_CF' in res and (res['OI_CF']=={} or ignoreCF):
        res.pop('OI_CF')

    if res['OI_FLUX']=={}:
        res.pop('OI_FLUX')

    if res['OI_T3']=={}:
        res.pop('OI_T3')
    else:
        if len(M) and debug:
            print('    warning: \033[43mmissing baselines\033[0m', M, 'to define T3')
        # -- match MJDs for T3 computations:
        for k in res['OI_T3'].keys():
            s, t = res['OI_T3'][k]['formula']
            w0, w1, w2 = [], [], []
            if debug:
                print('DEBUG: OI_T3', k, t, res['OI_T3'][k]['MJD'])
                #for i in range(3):
                #    print(' VIS :', t[i], res['OI_VIS'][t[i]]['MJD'])
                #    print(' VIS2:', t[i], res['OI_VIS'][t[i]]['MJD'])

            # -- add missing baselines
            A = {}
            for m in M:
                if m in t:
                    # -- added sign on 2021/09/22
                    if t.index(m)==0:
                        u = s[0]*res['OI_T3'][k]['u1']
                        v = s[0]*res['OI_T3'][k]['v1']
                    elif t.index(m)==1:
                        u = s[1]*res['OI_T3'][k]['u2']
                        v = s[1]*res['OI_T3'][k]['v2']
                    elif t.index(m)==2:
                        u = -s[2]*(res['OI_T3'][k]['u1']+res['OI_T3'][k]['u2'])
                        v = -s[2]*(res['OI_T3'][k]['v1']+res['OI_T3'][k]['v2'])
                    # -- add data
                    tmp = {}
                    #if key=='OI_VIS':
                    #    _K = ['|V|', 'PHI']
                    #else:
                    #    _K = ['V2']
                    _K = ['V2', '|V|', 'PHI']
                    for _k in _K:
                        tmp[_k] = np.zeros((len(res['OI_T3'][k]['MJD']),
                                              len(res['WL'])))
                        tmp['E'+_k] = np.ones((len(res['OI_T3'][k]['MJD']),
                                               len(res['WL'])))
                    tmp['u'], tmp['v'] = u, v
                    tmp['u/wl'] = u[:,None]/res['WL'][None,:]
                    tmp['v/wl'] = v[:,None]/res['WL'][None,:]
                    tmp['B/wl'] = np.sqrt(tmp['u/wl']**2 + tmp['v/wl']**2)
                    tmp['FLAG'] = np.ones((len(res['OI_T3'][k]['MJD']),
                                           len(res['WL'])), dtype=bool)
                    tmp['MJD'] = 1.0*res['OI_T3'][k]['MJD']
                    tmp['MJD2'] = tmp['MJD'][:,None] + 0*res['WL'][None,:]
                    tmp['PA'] = np.angle(tmp['v/wl']+1j*tmp['u/wl'], deg=True)
                    A[m] = tmp

            if len(A):
                for m in A:
                    if debug:
                        print('    adding fake', m, 'to', key, end=' ')
                        print(A[m].keys())
                    #res[key][m] = A[m]

                    res['OI_VIS'][m] = copy.deepcopy(A[m])
                    for _k in ['V2', 'EV2']:
                        res['OI_VIS'][m].pop(_k)
                    res['OI_VIS2'][m] = copy.deepcopy(A[m])
                    for _k in ['|V|', 'E|V|', 'PHI', 'EPHI']:
                        res['OI_VIS2'][m].pop(_k)
                    #print(m, 'VIS ', res['OI_VIS'][m].keys())
                    #print(m, 'VIS2', res['OI_VIS2'][m].keys())
                    M.remove(m)

            for i, mjd in enumerate(res['OI_T3'][k]['MJD']):
                # check data are within ~10s
                if min(np.abs(res[key][t[0]]['MJD']-mjd))<1e-4:
                    w0.append(np.argmin((res[key][t[0]]['MJD']-mjd)**2+
                                        (res[key][t[0]]['u']-s[0]*res['OI_T3'][k]['u1'][i])**2+
                                        (res[key][t[0]]['v']-s[0]*res['OI_T3'][k]['v1'][i])**2))

                else:
                    w0.append(len(res[key][t[0]]['MJD']))
                    if debug:
                        print('\033[43mWARNING\033[0m: missing MJD [0]', mjd, k, key, t[0])
                    # -- add fake data for MJD
                    for _key in ['OI_VIS', 'OI_VIS2']:
                        res[_key][t[0]]['MJD'] = np.append(res[_key][t[0]]['MJD'], mjd)
                        res[_key][t[0]]['MJD2'] = res[_key][t[0]]['MJD'][:,None] + 0*res['WL'][None,:]
                        res[_key][t[0]]['u'] = np.append(res[_key][t[0]]['u'],
                                                       s[0]*res['OI_T3'][k]['u1'][i])
                        res[_key][t[0]]['v'] = np.append(res[_key][t[0]]['v'],
                                                       s[0]*res['OI_T3'][k]['v1'][i])
                        res[_key][t[0]]['u/wl'] = np.append(res[_key][t[0]]['u/wl'],
                                                       s[0]*np.array([res['OI_T3'][k]['u1'][i]/
                                                       res['WL']]), axis=0)
                        res[_key][t[0]]['v/wl'] = np.append(res[_key][t[0]]['v/wl'],
                                                       s[0]*np.array([res['OI_T3'][k]['v1'][i]/
                                                       res['WL']]), axis=0)
                        res[_key][t[0]]['PA'] = np.angle(res[_key][t[0]]['v/wl']+
                                                     1j*res[_key][t[0]]['u/wl'], deg=True)

                        res[_key][t[0]]['B/wl'] = np.append(res[_key][t[0]]['B/wl'],
                                                     np.array([np.sqrt(res['OI_T3'][k]['u1'][i]**2+
                                                                       res['OI_T3'][k]['v1'][i]**2)/
                                                     res['WL']]), axis=0)
                        res[_key][t[0]]['FLAG'] = np.append(res[_key][t[0]]['FLAG'],
                                                    np.array([np.ones(len(res['WL']), dtype=bool)]), axis=0)

                    #if key=='OI_VIS':
                    res['OI_VIS'][t[0]]['|V|'] = np.append(res['OI_VIS'][t[0]]['|V|'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[0]]['E|V|'] = np.append(res['OI_VIS'][t[0]]['E|V|'],
                                             np.array([res['WL']*0+1]), axis=0)
                    res['OI_VIS'][t[0]]['PHI'] = np.append(res['OI_VIS'][t[0]]['PHI'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[0]]['EPHI'] = np.append(res['OI_VIS'][t[0]]['EPHI'],
                                            np.array([res['WL']*0+1]), axis=0)
                    #else:
                    res['OI_VIS2'][t[0]]['V2'] = np.append(res['OI_VIS2'][t[0]]['V2'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS2'][t[0]]['EV2'] = np.append(res['OI_VIS2'][t[0]]['EV2'],
                                            np.array([res['WL']*0+1]), axis=0)
                    if debug:
                        print(' test:', mjd, t[0], res['OI_VIS'][t[0]]['MJD'], w0)

                if min(np.abs(res[key][t[1]]['MJD']-mjd))<1e-4:
                    #w1.append(np.argmin(np.abs(res[key][t[1]]['MJD']-mjd)))
                    w1.append(np.argmin((res[key][t[1]]['MJD']-mjd)**2+
                                        (res[key][t[1]]['u']-s[1]*res['OI_T3'][k]['u2'][i])**2+
                                        (res[key][t[1]]['v']-s[1]*res['OI_T3'][k]['v2'][i])**2))
                else:
                    w1.append(len(res[key][t[1]]['MJD']))
                    if debug:
                        print('\033[43mWARNING\033[0m: missing MJD [1]', mjd, k, key, t[1])
                    # -- add fake data for MJD
                    for _key in ['OI_VIS', 'OI_VIS2']:
                        res[_key][t[1]]['MJD'] = np.append(res[_key][t[1]]['MJD'], mjd)
                        res[_key][t[1]]['MJD2'] = res[_key][t[1]]['MJD'][:,None] + 0*res['WL'][None,:]

                        res[_key][t[1]]['u'] = np.append(res[_key][t[1]]['u'],
                                                       s[1]*res['OI_T3'][k]['u2'][i])
                        res[_key][t[1]]['v'] = np.append(res[_key][t[1]]['v'],
                                                       s[1]*res['OI_T3'][k]['v2'][i])
                        res[_key][t[1]]['u/wl'] = np.append(res[_key][t[1]]['u/wl'],
                                                       s[1]*np.array([res['OI_T3'][k]['u2'][i]/
                                                       res['WL']]), axis=0)
                        res[_key][t[1]]['v/wl'] = np.append(res[_key][t[1]]['v/wl'],
                                                       s[1]*np.array([res['OI_T3'][k]['v2'][i]/
                                                       res['WL']]), axis=0)
                        res[_key][t[1]]['PA'] = np.angle(res[_key][t[1]]['v/wl']+
                                                     1j*res[_key][t[1]]['u/wl'], deg=True)

                        res[_key][t[1]]['B/wl'] = np.append(res[_key][t[1]]['B/wl'],
                                                     np.array([np.sqrt(res['OI_T3'][k]['u2'][i]**2+
                                                                       res['OI_T3'][k]['v2'][i]**2)/
                                                     res['WL']]), axis=0)
                        res[_key][t[1]]['FLAG'] = np.append(res[_key][t[1]]['FLAG'],
                                                    np.array([np.ones(len(res['WL']), dtype=bool)]), axis=0)
                    #if key=='OI_VIS':
                    res['OI_VIS'][t[1]]['|V|'] = np.append(res['OI_VIS'][t[1]]['|V|'],
                                                           np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[1]]['E|V|'] = np.append(res['OI_VIS'][t[1]]['E|V|'],
                                                           np.array([res['WL']*0+1]), axis=0)
                    res['OI_VIS'][t[1]]['PHI'] = np.append(res['OI_VIS'][t[1]]['PHI'],
                                                           np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[1]]['EPHI'] = np.append(res['OI_VIS'][t[1]]['EPHI'],
                                                           np.array([res['WL']*0+1]), axis=0)
                    #else:
                    res['OI_VIS2'][t[1]]['V2'] = np.append(res['OI_VIS2'][t[1]]['V2'],
                                                           np.array([res['WL']*0]), axis=0)
                    res['OI_VIS2'][t[1]]['EV2'] = np.append(res['OI_VIS2'][t[1]]['EV2'],
                                                           np.array([res['WL']*0+1]), axis=0)
                    if debug:
                        print(' test:', mjd, t[1], res['OI_VIS'][t[1]]['MJD'], w1)

                if min(np.abs(res[key][t[2]]['MJD']-mjd))<1e-4:
                    #w2.append(np.argmin(np.abs(res[key][t[2]]['MJD']-mjd)))
                    w2.append(np.argmin((res[key][t[2]]['MJD']-mjd)**2+
                                        (res[key][t[2]]['u']+s[2]*res['OI_T3'][k]['u1'][i]+s[2]*res['OI_T3'][k]['u2'][i])**2+
                                        (res[key][t[2]]['v']+s[2]*res['OI_T3'][k]['v1'][i]+s[2]*res['OI_T3'][k]['v2'][i])**2))

                else:
                    # -- will be added at the end
                    w2.append(len(res[key][t[2]]['MJD']))
                    if debug:
                        print('\033[43mWARNING\033[0m: missing MJD [2]', mjd, k, key, t[2])
                    # -- add fake data for MJD
                    for _key in ['OI_VIS', 'OI_VIS2']:
                        kldg = 1
                        res[_key][t[2]]['MJD'] = np.append(res[_key][t[2]]['MJD'], mjd)
                        res[_key][t[2]]['MJD2'] = res[_key][t[2]]['MJD'][:,None] + 0*res['WL'][None,:]

                        res[_key][t[2]]['u'] = np.append(res[_key][t[2]]['u'],
                                                       -kldg*s[2]*res['OI_T3'][k]['u1'][i]
                                                       -kldg*s[2]*res['OI_T3'][k]['u2'][i])
                        res[_key][t[2]]['v'] = np.append(res[_key][t[2]]['v'],
                                                       -kldg*s[2]*res['OI_T3'][k]['v1'][i]
                                                       -kldg*s[2]*res['OI_T3'][k]['v2'][i])
                        res[_key][t[2]]['u/wl'] = np.append(res[_key][t[2]]['u/wl'],
                                                       kldg*s[2]*np.array([(-res['OI_T3'][k]['u1'][i]
                                                                       -res['OI_T3'][k]['u2'][i])/
                                                       res['WL']]), axis=0)
                        res[_key][t[2]]['v/wl'] = np.append(res[_key][t[2]]['v/wl'],
                                                       kldg*s[2]*np.array([(-res['OI_T3'][k]['v1'][i]
                                                                       -res['OI_T3'][k]['v2'][i])/
                                                       res['WL']]), axis=0)
                        res[_key][t[2]]['PA'] = np.angle(res[_key][t[2]]['v/wl']+
                                                         1j*res[_key][t[2]]['u/wl'], deg=True)

                        res[_key][t[2]]['B/wl'] = np.append(res[_key][t[2]]['B/wl'],
                                                     np.array([np.sqrt((res['OI_T3'][k]['u1'][i]+res['OI_T3'][k]['u2'][i])**2+
                                                                       (res['OI_T3'][k]['v1'][i]+res['OI_T3'][k]['v2'][i])**2)/
                                                     res['WL']]), axis=0)
                        res[_key][t[2]]['FLAG'] = np.append(res[_key][t[2]]['FLAG'],
                                                    np.array([np.ones(len(res['WL']), dtype=bool)]), axis=0)
                    res['OI_VIS'][t[2]]['|V|'] = np.append(res['OI_VIS'][t[2]]['|V|'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[2]]['E|V|'] = np.append(res['OI_VIS'][t[2]]['E|V|'],
                                            np.array([res['WL']*0+1]), axis=0)
                    res['OI_VIS'][t[2]]['PHI'] = np.append(res['OI_VIS'][t[2]]['PHI'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS'][t[2]]['EPHI'] = np.append(res['OI_VIS'][t[2]]['EPHI'],
                                            np.array([res['WL']*0+1]), axis=0)
                    res['OI_VIS2'][t[2]]['V2'] = np.append(res['OI_VIS2'][t[2]]['V2'],
                                            np.array([res['WL']*0]), axis=0)
                    res['OI_VIS2'][t[2]]['EV2'] = np.append(res['OI_VIS2'][t[2]]['EV2'],
                                            np.array([res['WL']*0+1]), axis=0)
                    if debug:
                        print(' test:', mjd, t[2], res['OI_VIS'][t[2]]['MJD'], w2)

            res['OI_T3'][k]['formula'] = [s, t, w0, w1, w2]
            # except:
            #     print('warning! triplet', k,
            #           'has no formula in', res[key].keys())
            #     res['OI_T3'][k]['formula'] = None

    # -- not sure why had to be ran twice?
    #res = match_VIS_VIS2_CF(res, debug=debug)

    if 'OI_FLUX' in res:
        res['telescopes'] = sorted(list(res['OI_FLUX'].keys()))
    else:
        res['telescopes'] = []
        for k in res[key].keys():
            res['telescopes'].append(k[:len(k)//2])
            res['telescopes'].append(k[len(k)//2:])
        res['telescopes'] = sorted(list(set(res['telescopes'])))
    res['baselines'] = sorted(list(res[key].keys()))
    if 'OI_T3' in res.keys():
        res['triangles'] = sorted(list(res['OI_T3'].keys()))
    else:
        res['triangles'] = []

    if not 'TELLURICS' in res.keys():
        res['TELLURICS'] = np.ones(res['WL'].shape)

    if not tellurics is None and not tellurics is False:
        # -- forcing tellurics to given vector
        res['TELLURICS'] = tellurics
        if not binning is None and len(tellurics)==len(_WL):
            res['TELLURICS'] = _binVec(res['WL'], _WL, tellurics)

    if 'OI_FLUX' in res.keys():
        for k in res['OI_FLUX'].keys():
            # -- raw flux
            res['OI_FLUX'][k]['RFLUX'] = res['OI_FLUX'][k]['FLUX'].copy()
            # -- corrected from tellurics
            res['OI_FLUX'][k]['FLUX'] /= res['TELLURICS'][None,:]

    if not medFilt is None:
        if type(medFilt) == int:
            kernel_size = 2*(medFilt//2)+1
        else:
            kernel_size = None
        res = medianFilt(res, kernel_size)

    confperMJD = {}
    for l in filter(lambda x: x in ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX'], res.keys()):
        for k in res[l].keys():
            for mjd in res[l][k]['MJD']:
                if not mjd in confperMJD.keys():
                    confperMJD[mjd] = [k]
                else:
                    if not k in confperMJD[mjd]:
                        confperMJD[mjd].append(k)
    res['configurations per MJD'] = confperMJD
    # -- all MJDs in the file
    res['MJD'] = np.array(sorted(set(res['configurations per MJD'].keys())))

    # -- mean barycentric correction for the dataset
    if 'ORIGIN' in h[0].header and h[0].header['ORIGIN']=='ESO-PARANAL':
        Paranal = EarthLocation.of_site('Paranal')
        if 'RA' in h[0].header and 'DEC'  in h[0].header:
            sc = SkyCoord(ra=h[0].header['RA']*aU.deg,
                          dec=h[0].header['DEC']*aU.deg)
            # -- correction in m/s
            barycorr = sc.radial_velocity_correction(location=Paranal,
                        obstime=Time(h[0].header['DATE-OBS']))
            res['barycorr_km/s'] = barycorr.value/1000

    if verbose:
        mjd = []
        for e in ['OI_VIS2', 'OI_VIS', 'OI_T3', 'OI_FLUX']:
            if e in res.keys():
                for k in res[e].keys():
                    mjd.extend(list(res[e][k]['MJD']))
        mjd = np.array(sorted(set(mjd)))
        #print('  > MJD:', sorted(set(mjd)))
        print('  > MJD:', mjd.shape, '[', min(mjd), '..', max(mjd), ']')
        print('  >', '-'.join(res['telescopes']), end=' | ')
        _R = np.mean(res['WL']/res['dWL'])
        _Rp = np.abs(np.mean(res['WL']/np.gradient(res['WL'])))
        # -- resolution very different from spectral chanel pitch
        if np.abs((_Rp-_R)/_R)>0.5:
            _cr = '\033[31m'
        else:
            _cr = '\033[0m'

        print('WL:', res['WL'].shape, '[', round(np.min(res['WL']), 3), '..',
              round(np.max(res['WL']), 3),
              '] um (R~%.0f %sP~%.0f\033[0m)'%(_R, _cr, _Rp),end=' ')

        if not binning is None:
            print('(binx%d)'%binning, end=' ')
        #print(sorted(list(filter(lambda x: x.startswith('OI_'), res.keys()))),
        #            end=' | ')
        Kz = sorted(list(filter(lambda x: x.startswith('OI_'), res.keys())))
        _Kz = [_k.split('OI_')[1] for _k in Kz]
        print(dict(zip(_Kz, [len(res[k].keys()) for k in Kz])), end=' | ')

        print('TELL:', res['TELLURICS'].min()<1
                        if not ignoredTellurics else 'IGNORED!', end=' ')
        if 'PWV' in res:
            print('pwv=%.2fmm'%res['PWV'])
        else:
            print()
        # print('  >', 'telescopes:', res['telescopes'],
        #       'baselines:', res['baselines'],
        #       'triangles:', res['triangles'])
    # -- done!
    h.close()
    try:
        res['recipes'] = getESOPipelineParams(res['header'], verbose=False)
    except:
        if verbose:
            print('WARNING: error while reading ESO pipelines parameters')

    if orderedWl:
        # == order by increasing WL:
        wls = np.argsort(res['WL'])
        res['WL'] = res['WL'][wls]
        res['dWL'] = res['dWL'][wls]
        res['TELLURICS'] = res['TELLURICS'][wls]
        O = {'OI_VIS2':['V2', 'EV2', 'u/wl', 'v/wl', 'B/wl', 'FLAG', 'MJD2', 'PA'],
            'OI_VIS': ['|V|', 'E|V|', 'PHI', 'EPHI', 'u/wl', 'v/wl', 'B/wl', 'FLAG', 'MJD2', 'PA'],
            'OI_CF':  ['CF', 'ECF', 'PHI', 'EPHI', 'u/wl', 'v/wl', 'B/wl', 'FLAG', 'MJD2', 'PA'],
            'OI_T3':  ['T3AMP', 'ET3AMP', 'T3PHI', 'ET3PHI', 'Bmin/wl', 'Bmax/wl', 'Bavg/wl', 'FLAG', 'MJD2'],
            }
        for o in O:
            if not o in res:
                continue
            for k in res[o]:
                for e in O[o]:
                    if e in res[o][k]:
                        res[o][k][e] = res[o][k][e][:,wls]
    return res

def _binOI(res, binning=None, medFilt=None, noError=False):
    """
    binning: number of spectral channels to bin (>2, None or 1 does nothing).
    """
    if binning is None and not 'binWL' in res:
        return res

    # -- keep track of true wavelength
    _WL = res['WL']*1.0
    if 'dWL' in res:
        _dWL = res['dWL']*1.0
    else:
        _dWL = np.gradient(res['WL'])
    if 'binWL' in res:
        res['WL'] = res['binWL']*1.0
        res['dWL'] = np.gradient(res['WL'])
    else:
        res['WL'] = np.linspace(res['WL'].min(),
                                res['WL'].max(),
                                len(res['WL'])//binning)
        res['dWL'] = binning*np.interp(res['WL'], _WL, _dWL)

    if 'TELLURICS' in res:
        res['TELLURICS'] =_binVec_flag(res['WL'], _WL,
                             np.array([res['TELLURICS']]),
                             np.array([res['TELLURICS']<0]),
                             medFilt=medFilt,
                             retFlag=False)[0]
    if 'OI_FLUX' in res:
        # -- Note the binning of flux is not weighted, because
        # -- it would make the tellurics correction incorrect
        # -- overwise
        for k in res['OI_FLUX']:
            #print(res['OI_FLUX'][k].keys())
            res['OI_FLUX'][k]['FLUX'], flag =_binVec_flag(res['WL'], _WL,
                                            res['OI_FLUX'][k]['FLUX'],
                                            res['OI_FLUX'][k]['FLAG'],
                                            medFilt=medFilt, retFlag=True)
            res['OI_FLUX'][k]['EFLUX'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_FLUX'][k]['EFLUX'],
                                                res['OI_FLUX'][k]['FLAG'],
                                                medFilt=medFilt)
            res['OI_FLUX'][k]['FLAG'] = flag
            res['OI_FLUX'][k]['MJD2'] = res['OI_FLUX'][k]['MJD'][:,None] + 0*res['WL'][None,:]

    if 'OI_VIS2' in res:
        for k in res['OI_VIS2']:
            #print('OI_VIS', k, res['WL'].shape, _WL.shape, res['OI_VIS2'][k]['V2'].shape,
            #    res['OI_VIS2'][k]['FLAG'].shape)
            res['OI_VIS2'][k]['V2'], flag =\
                                   _binVec_flag(res['WL'], _WL,
                                            res['OI_VIS2'][k]['V2'],
                                            res['OI_VIS2'][k]['FLAG'],
                                            None if noError else res['OI_VIS2'][k]['EV2'] ,
                                            medFilt=medFilt,
                                            retFlag=True)
            if not noError:
                res['OI_VIS2'][k]['EV2'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_VIS2'][k]['EV2'],
                                                res['OI_VIS2'][k]['FLAG'],
                                                res['OI_VIS2'][k]['EV2'],
                                                medFilt=medFilt)
            res['OI_VIS2'][k]['FLAG'] = flag
            res['OI_VIS2'][k]['MJD2'] = res['OI_VIS2'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_VIS2'][k]['u/wl'] = res['OI_VIS2'][k]['u'][:,None]/res['WL'][None,:]
            res['OI_VIS2'][k]['v/wl'] = res['OI_VIS2'][k]['v'][:,None]/res['WL'][None,:]
            res['OI_VIS2'][k]['B/wl'] = np.sqrt(res['OI_VIS2'][k]['u'][:,None]**2+
                res['OI_VIS2'][k]['v'][:,None]**2)/res['WL'][None,:]

    if 'OI_VIS' in res:
        for k in res['OI_VIS']:
            res['OI_VIS'][k]['|V|'],  flag =_binVec_flag(res['WL'], _WL,
                                            res['OI_VIS'][k]['|V|'],
                                            res['OI_VIS'][k]['FLAG'],
                                            None if noError else res['OI_VIS'][k]['E|V|'],
                                            medFilt=medFilt,
                                            retFlag=True)
            res['OI_VIS'][k]['PHI'] =_binVec_flag(res['WL'], _WL,
                                            res['OI_VIS'][k]['PHI'],
                                            res['OI_VIS'][k]['FLAG'],
                                            None if noError else res['OI_VIS'][k]['EPHI'],
                                            medFilt=medFilt)
            if not noError:
                res['OI_VIS'][k]['E|V|'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_VIS'][k]['E|V|'],
                                                res['OI_VIS'][k]['FLAG'],
                                                res['OI_VIS'][k]['E|V|'],
                                                medFilt=medFilt)
                res['OI_VIS'][k]['EPHI'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_VIS'][k]['EPHI'],
                                                res['OI_VIS'][k]['FLAG'],
                                                res['OI_VIS'][k]['EPHI'],
                                                medFilt=medFilt)
            res['OI_VIS'][k]['FLAG'] = flag
            res['OI_VIS'][k]['MJD2'] = res['OI_VIS'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_VIS'][k]['u/wl'] = res['OI_VIS'][k]['u'][:,None]/res['WL'][None,:]
            res['OI_VIS'][k]['v/wl'] = res['OI_VIS'][k]['v'][:,None]/res['WL'][None,:]
            res['OI_VIS'][k]['B/wl'] = np.sqrt(res['OI_VIS'][k]['u'][:,None]**2+
                res['OI_VIS'][k]['v'][:,None]**2)/res['WL'][None,:]

    if 'OI_CF' in res:
        for k in res['OI_CF']:
            res['OI_CF'][k]['CF'],  flag =_binVec_flag(res['WL'], _WL,
                                            res['OI_CF'][k]['CF'],
                                            res['OI_CF'][k]['FLAG'],
                                            None if noError else res['OI_CF'][k]['ECF'],
                                            medFilt=medFilt,
                                            retFlag=True)
            res['OI_CF'][k]['PHI'] =_binVec_flag(res['WL'], _WL,
                                            res['OI_CF'][k]['PHI'],
                                            res['OI_CF'][k]['FLAG'],
                                            None if noError else res['OI_CF'][k]['EPHI'],
                                            medFilt=medFilt)
            if not noError:
                res['OI_CF'][k]['ECF'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_CF'][k]['ECF'],
                                                res['OI_CF'][k]['FLAG'],
                                                res['OI_CF'][k]['ECF'],
                                                medFilt=medFilt)
                res['OI_CF'][k]['EPHI'] = 1/_binVec_flag(res['WL'], _WL,
                                                1/res['OI_CF'][k]['EPHI'],
                                                res['OI_CF'][k]['FLAG'],
                                                res['OI_CF'][k]['EPHI'],
                                                medFilt=medFilt)
            res['OI_CF'][k]['FLAG'] = flag
            res['OI_CF'][k]['MJD2'] = res['OI_CF'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_VIS'][k]['u/wl'] = res['OI_VIS'][k]['u'][:,None]/res['WL'][None,:]
            res['OI_VIS'][k]['v/wl'] = res['OI_VIS'][k]['v'][:,None]/res['WL'][None,:]
            res['OI_VIS'][k]['B/wl'] = np.sqrt(res['OI_VIS'][k]['u'][:,None]**2+
                                               res['OI_VIS'][k]['v'][:,None]**2)/res['WL'][None,:]

    if 'OI_T3' in res:
        for k in res['OI_T3']:
            if 'T3AMP' in res['OI_T3'][k]:
                res['OI_T3'][k]['T3AMP'] =_binVec_flag(res['WL'], _WL,
                                                res['OI_T3'][k]['T3AMP'],
                                                res['OI_T3'][k]['FLAG'],
                                                None if noError else res['OI_T3'][k]['ET3AMP'],
                                                medFilt=medFilt)
            if 'T3PHI' in res['OI_T3'][k]:
                res['OI_T3'][k]['T3PHI'], flag =_binVec_flag(res['WL'], _WL,
                                                res['OI_T3'][k]['T3PHI'],
                                                res['OI_T3'][k]['FLAG'],
                                                None if noError else res['OI_T3'][k]['ET3PHI'],
                                                medFilt=medFilt, retFlag=True, phase=True)
            if not noError:
                if 'T3AMP' in res['OI_T3'][k]:
                    res['OI_T3'][k]['ET3AMP'] = 1/_binVec_flag(res['WL'], _WL,
                                                    1/res['OI_T3'][k]['ET3AMP'],
                                                    res['OI_T3'][k]['FLAG'],
                                                    res['OI_T3'][k]['ET3AMP'],
                                                    medFilt=medFilt)
                if 'T3PHI' in res['OI_T3'][k]:
                    res['OI_T3'][k]['ET3PHI'] = 1/_binVec_flag(res['WL'], _WL,
                                                    1/res['OI_T3'][k]['ET3PHI'],
                                                    res['OI_T3'][k]['FLAG'],
                                                    res['OI_T3'][k]['ET3PHI'],
                                                    medFilt=medFilt)
            res['OI_T3'][k]['FLAG'] = flag
            res['OI_T3'][k]['MJD2'] = res['OI_T3'][k]['MJD'][:,None] + 0*res['WL'][None,:]
            res['OI_T3'][k]['u1/wl'] = res['OI_T3'][k]['u1'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['v1/wl'] = res['OI_T3'][k]['v1'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['u2/wl'] = res['OI_T3'][k]['u2'][:,None]/res['WL'][None,:]
            res['OI_T3'][k]['v2/wl'] = res['OI_T3'][k]['v2'][:,None]/res['WL'][None,:]
            bmax = np.maximum(res['OI_T3'][k]['B1'], res['OI_T3'][k]['B2'])
            bmax = np.maximum(res['OI_T3'][k]['B3'], bmax)
            bmin = np.minimum(res['OI_T3'][k]['B1'], res['OI_T3'][k]['B2'])
            bmin = np.minimum(res['OI_T3'][k]['B3'], bmin)
            res['OI_T3'][k]['Bmax/wl'] = bmax[:,None]/res['WL'][None,:]
            res['OI_T3'][k]['Bmin/wl'] = bmin[:,None]/res['WL'][None,:]
            res['OI_T3'][k]['Bavg/wl'] = (res['OI_T3'][k]['B1'][:,None]+
                                          res['OI_T3'][k]['B2'][:,None]+
                                          res['OI_T3'][k]['B3'][:,None])/res['WL'][None,:]/3
    if 'MODEL' in res:
        if 'totalflux' in res['MODEL']:
            res['MODEL']['totalflux'] =_binVec_flag(res['WL'], _WL,
                                            np.array([res['MODEL']['totalflux']]),
                                            np.bool([0*res['MODEL']['totalflux']]),
                                            0.0*np.array([res['MODEL']['totalflux']])+1,
                                            medFilt=medFilt)[0]
        if 'totalnflux' in res['MODEL']:
            res['MODEL']['totalnflux'] =_binVec_flag(res['WL'], _WL,
                                            res['MODEL']['totalnflux'],
                                            np.bool(0*res['MODEL']['totalnflux']),
                                            0.0*res['MODEL']['totalflux']+1,
                                            medFilt=medFilt)
        #print('MODEL:', res['MODEL'].keys())

    return res


def splitOIbyMJD(oi, dMJD=0):
    """
    ois: list of oi results from loadOI
    dMJD: step in MJD for grouping

    will return a list of OIs which data are within dMJD (in day).
    This function will only split OIs, and not merge them.
    function mergeOI should be run first!
    """
    MJDs = sorted(set(list(oi['MJD'])))
    #print('all MJDs:', MJDs)
    if len(MJDs)==1:
        return [oi]

    # -- group MJDs
    gMJDs=[]
    tmp = []
    for mjd in MJDs:
        if len(tmp)==0 or mjd<tmp[0]+dMJD:
            tmp.append(mjd)
        else:
            gMJDs.append(tmp)
            tmp = [mjd]
    gMJDs.append(tmp.copy())
    #print('groups (%d: %s):'%(len(gMJDs), ','.join([str(len(x)) for x in gMJDs])), gMJDs)
    # -- need to implement the actual split!
    res = []
    for mjds in gMJDs:
        #print('**FOR MJDS:**', mjds)
        tmp = copy.deepcopy(oi)
        E = [e for e in ['OI_VIS', 'OI_CVIS', 'OI_CF', 'OI_VIS2', 'OI_FLUX', 'OI_T3'] if e in tmp]
        for e in E:
            for k in tmp[e]:
                if 'MJD' in tmp[e][k]:
                    w = np.array([x in mjds for x in tmp[e][k]['MJD']])
                    _mjd = True
                    if e=='OI_VIS':
                        ivis = list(np.arange(len(w))[w])
                else:
                    _mjd = False
                if 'MJD2' in tmp[e][k]:
                    w2 = np.array([ [x in mjds for x in X] for X in tmp[e][k]['MJD2'] ])
                    _mjd2 = True
                else:
                    _mjd2 = False
                for l in tmp[e][k]:
                    if type(tmp[e][k][l])==np.ndarray:
                        if _mjd and w.shape==tmp[e][k][l].shape:
                            #print('  ',l, '-> MJD')
                            tmp[e][k][l] = tmp[e][k][l][w]
                            pass
                        elif _mjd2 and w2.shape==tmp[e][k][l].shape:
                            #print('  ',l, '-> MJD2')
                            tmp[e][k][l] = tmp[e][k][l][w2]
                            pass
                        else:
                            print('WARNING!!!',e,k,l, tmp[e][k][l].shape, w.shape, w2.shape)
                    elif l=='formula':
                        #print('formula', tmp[e][k][l])
                        s0, s1, s2 = tmp[e][k][l][0][0], tmp[e][k][l][0][1], tmp[e][k][l][0][2]
                        t0, t1, t2 = tmp[e][k][l][1]
                        i0, i1, i2 = np.array(tmp[e][k][l][2]), \
                                     np.array(tmp[e][k][l][3]), \
                                     np.array(tmp[e][k][l][4])

                        i0 = [ivis.index(x) for x in i0[w]]
                        i1 = [ivis.index(x) for x in i1[w]]
                        i2 = [ivis.index(x) for x in i2[w]]
                        try:
                            tmp[e][k][l] = [(s0[w], s1[w], s2[w]), (t0, t1, t2), i0, i1, i2]
                        except:
                            tmp[e][k][l] = [(s0, s1, s2), (t0, t1, t2), i0, i1, i2]
                    else:
                        print('WARNING!!!', e,k,l,)
        tmp['configurations per MJD'] = {k:tmp['configurations per MJD'][k]
                                         for k in tmp['configurations per MJD'] if k in mjds}
        tmp['MJD'] = [x for x in tmp['MJD'] if x in mjds]
        res.append(tmp)

    return res

def match_VIS_VIS2_CF(res, debug=False, ignoreCF=False):
    """
    make sure there is a 1-to-1 correspondance between VIS and CF:
    (this is purely due to a limitation of PMOIRED)
    """
    if debug:
        print('** match_VIS_VIS2_CF **')
    if (not ignoreCF) and 'OI_VIS' in res and 'OI_CF' in res:
        if debug:
            print('checking OI_CF <-> OI_VIS')
        kz = list(set(list(res['OI_VIS'].keys())+list(res['OI_CF'].keys())))
        for k in kz:
            if not k in res['OI_VIS']:
                if debug:
                    print(k, 'is in OI_CF but \033[43mmissing from OI_VIS!\033[0m')
                res['OI_VIS'][k] = {}
                for x in res['OI_CF'][k].keys():
                    if not 'CF' in x and not 'PHI' in x:
                        res['OI_VIS'][k][x] = res['OI_VIS2'][k][x].copy()
                # -- all data are invalid
                res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_CF'][k]['FLAG'], True)
                res['OI_VIS'][k]['|V|'] = 0*res['OI_CF'][k]['CF']
                res['OI_VIS'][k]['E|V|'] = 1 + 0*res['OI_CF'][k]['ECF']
                res['OI_VIS'][k]['PHI'] = 0*res['OI_CF'][k]['CF']
                res['OI_VIS'][k]['EPHI'] = 360 + 0*res['OI_CF'][k]['ECF']
                #print('missing VIS', k, res['OI_VIS'][k].keys())
            elif not k in res['OI_CF']:
                if debug:
                    print(k, 'is in OI_VIS but \033[43mmissing from OI_CF!\033[0m')
                res['OI_CF'][k] = {}
                for x in res['OI_VIS'][k].keys():
                    if not '|V|' in x and not 'PHI' in x:
                        res['OI_CF'][k][x] = res['OI_VIS'][k][x].copy()
                # -- all data are invalid
                res['OI_CF'][k]['FLAG'] = np.logical_or(res['OI_VIS'][k]['FLAG'], True)
                res['OI_CF'][k]['CF'] = 0*res['OI_VIS'][k]['|V|']
                res['OI_CF'][k]['ECF'] = 1+0*res['OI_VIS'][k]['E|V|']
                res['OI_CF'][k]['PHI'] = 0*res['OI_VIS'][k]['|V|']
                res['OI_CF'][k]['EPHI'] = 360+0*res['OI_VIS'][k]['E|V|']

            if sorted(res['OI_VIS'][k]['MJD']) != sorted(res['OI_CF'][k]['MJD']):
                if debug:
                    print(k, '\033[43mmismatched coverage VIS/CF!\033[0m')
                    print(' VIS:',  res['OI_VIS'][k]['MJD'])
                    print(' CF :',  res['OI_CF'][k]['MJD'])
                # -- all encountered MJDs:
                allMJD = sorted(list(res['OI_VIS'][k]['MJD'])+list(res['OI_CF'][k]['MJD']))
                # -- ones from VIS
                wv = [allMJD.index(mjd) for mjd in res['OI_VIS'][k]['MJD']]
                # -- ones from VIS2
                wv2 = [allMJD.index(mjd) for mjd in res['OI_CF'][k]['MJD']]

                #print(' ', wv, wv2)
                # -- update VIS
                res['OI_VIS'][k]['MJD'] = np.array(allMJD)
                for x in ['u', 'v']:
                    tmp = np.zeros(len(allMJD))
                    tmp[wv] = res['OI_VIS'][k][x]
                    if x in res['OI_CF'][k]:
                        tmp[wv2] = res['OI_CF'][k][x]
                    res['OI_VIS'][k][x] = tmp
                for x in ['|V|', 'E|V|', 'PHI', 'EPHI', 'u/wl', 'v/wl', 'FLAG', 'B/wl', 'PA', 'MJD2']:
                    if x=='FLAG':
                        tmp = np.ones((len(allMJD), len(res['WL'])), dtype='bool')
                    else:
                        tmp = np.zeros((len(allMJD), len(res['WL'])))
                    tmp[wv,:] = res['OI_VIS'][k][x]
                    if x in res['OI_CF'][k] and x!='FLAG':
                        tmp[wv2,:] = res['OI_CF'][k][x]
                    res['OI_VIS'][k][x] = tmp

                # -- VIS2
                res['OI_CF'][k]['MJD'] = np.array(allMJD)
                for x in ['u', 'v']:
                    if x in res['OI_VIS'][k]:
                        tmp = 1.*res['OI_VIS'][k][x]
                    else:
                        tmp = np.zeros(len(allMJD))
                    tmp[wv2] = res['OI_CF'][k][x]
                    res['OI_CF'][k][x] = tmp
                for x in ['CF', 'ECF', 'PHI', 'EPHI', 'u/wl', 'v/wl', 'FLAG', 'B/wl', 'PA', 'MJD2']:
                    if x in res['OI_VIS'][k] and x!='FLAG':
                        tmp = res['OI_VIS'][k][x].copy()
                    elif x=='FLAG':
                        tmp = np.ones((len(allMJD), len(res['WL'])), dtype='bool')
                    else:
                        tmp = np.zeros((len(allMJD), len(res['WL'])))
                    tmp[wv2,:] = res['OI_CF'][k][x]
                    res['OI_CF'][k][x] = tmp
                # -- make sure it worked
                #print(' VIS :',  res['OI_VIS'][k]['MJD'])
                #print(' VIS2:',  res['OI_VIS2'][k]['MJD'])

    # -- make sure there is a 1-to-1 correspondance between VIS and VIS2:
    # (this is purely due to a limitation of PMOIRED)
    if 'OI_VIS' in res and 'OI_VIS2' in res:
        if debug:
            print('checking OI_VIS <-> OI_VIS2')

        kz = list(set(list(res['OI_VIS'].keys())+list(res['OI_VIS2'].keys())))
        for k in kz:
            if not k in res['OI_VIS']:
                if debug:
                    print('baseline', k, 'is in OI_VIS2 but \033[43mmissing from OI_VIS!\033[0m')
                res['OI_VIS'][k] = {}
                for x in res['OI_VIS2'][k].keys():
                    if not 'V2' in x:
                        res['OI_VIS'][k][x] = res['OI_VIS2'][k][x].copy()
                # -- all data are invalid
                res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'], True)
                res['OI_VIS'][k]['|V|'] = 0*res['OI_VIS2'][k]['V2']
                res['OI_VIS'][k]['E|V|'] = 1 + 0*res['OI_VIS2'][k]['EV2']
                res['OI_VIS'][k]['PHI'] = 0*res['OI_VIS2'][k]['V2']
                res['OI_VIS'][k]['EPHI'] = 360 + 0*res['OI_VIS2'][k]['EV2']
                #print('missing VIS', k, res['OI_VIS'][k].keys())
            elif not k in res['OI_VIS2']:
                if debug:
                    print(k, 'is in OI_VIS but \033[43mmissing from OI_VIS2!\033[0m')
                res['OI_VIS2'][k] = {}
                for x in res['OI_VIS'][k].keys():
                    if not '|V|' in x and not 'PHI' in x:
                        res['OI_VIS2'][k][x] = res['OI_VIS'][k][x].copy()
                # -- all data are invalid
                res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'], True)
                res['OI_VIS2'][k]['V2'] = 0*res['OI_VIS'][k]['|V|']
                res['OI_VIS2'][k]['EV2'] = 1+0*res['OI_VIS'][k]['E|V|']

            if sorted(res['OI_VIS'][k]['MJD']) != sorted(res['OI_VIS2'][k]['MJD']):
                if debug:
                    print(k, '\033[43mmismatched coverage VIS/VIS2!\033[0m')
                    print(' VIS :',  res['OI_VIS'][k]['MJD'])
                    print(' VIS2:',  res['OI_VIS2'][k]['MJD'])
                # -- all encountered MJDs:
                allMJD = sorted(list(res['OI_VIS'][k]['MJD'])+list(res['OI_VIS2'][k]['MJD']))
                # -- ones from VIS
                wv = [allMJD.index(mjd) for mjd in res['OI_VIS'][k]['MJD']]
                # -- ones from VIS2
                wv2 = [allMJD.index(mjd) for mjd in res['OI_VIS2'][k]['MJD']]
                if debug:
                    print('DEBUD: wv2=', wv2)
                #print(' ', wv, wv2)
                # -- update VIS
                res['OI_VIS'][k]['MJD'] = np.array(allMJD)
                for x in ['u', 'v']:
                    tmp = np.zeros(len(allMJD))
                    tmp[wv] = res['OI_VIS'][k][x]
                    if x in res['OI_VIS2'][k]:
                        tmp[wv2] = res['OI_VIS2'][k][x]
                    res['OI_VIS'][k][x] = tmp
                for x in ['|V|', 'E|V|', 'PHI', 'EPHI', 'u/wl', 'v/wl', 'FLAG', 'B/wl', 'PA', 'MJD2']:
                    if x=='FLAG':
                        tmp = np.ones((len(allMJD), len(res['WL'])), dtype='bool')
                    else:
                        tmp = np.zeros((len(allMJD), len(res['WL'])))
                    tmp[wv,:] = res['OI_VIS'][k][x]
                    if x in res['OI_VIS2'][k] and x!='FLAG':
                        tmp[wv2,:] = res['OI_VIS2'][k][x]
                    res['OI_VIS'][k][x] = tmp

                # -- VIS2
                res['OI_VIS2'][k]['MJD'] = np.array(allMJD)
                for x in ['u', 'v']:
                    if x in res['OI_VIS'][k]:
                        tmp = 1.*res['OI_VIS'][k][x]
                    else:
                        tmp = np.zeros(len(allMJD))
                    tmp[wv2] = res['OI_VIS2'][k][x]
                    res['OI_VIS2'][k][x] = tmp
                for x in ['V2', 'EV2', 'u/wl', 'v/wl', 'FLAG', 'B/wl', 'PA', 'MJD2']:
                    if x in res['OI_VIS'][k] and x!='FLAG':
                        tmp = res['OI_VIS'][k][x].copy()
                    elif x=='FLAG':
                        tmp = np.ones((len(allMJD), len(res['WL'])), dtype='bool')
                    else:
                        tmp = np.zeros((len(allMJD), len(res['WL'])))
                    tmp[wv2,:] = res['OI_VIS2'][k][x].copy()
                    res['OI_VIS2'][k][x] = tmp
                # -- make sure it worked
                if debug:
                    print(' MJD in  VIS[%s]:'%k,  res['OI_VIS'][k]['MJD'])
                    print(' MJD in VIS2[%s]:'%k,  res['OI_VIS2'][k]['MJD'])

    return res

def wTarg(hdu, targname, targets):
    if type(targets[targname]) == list:
        return np.array([x in targets[targname] for x in hdu.data['TARGET_ID']])
    else:
        return hdu.data['TARGET_ID']==targets[targname]

def _binVec_flag(_wl, WL, T, F, E=None, medFilt=None, retFlag=False, phase=False):
    """
    _wl: new WL vector
    WL: actual WL vector
    T: data table (2D)
    F: flag table (2D)
    """
    res = np.zeros((T.shape[0], len(_wl)))
    flag = np.zeros((T.shape[0], len(_wl)), dtype=bool)
    for i in range(T.shape[0]):
        w = ~F[i,:]
        # -- half of the points in the bin are valid
        #flag[i,:] = np.bool_(_binVec(_wl, WL, np.float64(F[i,:]))>0.5)

        # -- 2/3 of the points in the bin are valid
        #flag[i,:] = np.bool_(_binVec(_wl, WL, np.float64(F[i,:]))>2/3)

        # -- at least one point in the bin is valid
        flag[i,:] = ~np.bool_(_binVec(_wl, WL, np.float64(~F[i,:]), phase=phase)>0)
        if E is None:
            res[i,:] = _binVec(_wl, WL[w], T[i,:][w], medFilt=medFilt, phase=phase)
        else:
            try:
                res[i,:] = _binVec(_wl, WL[w], T[i,:][w], E=E[i,:][w], medFilt=medFilt, phase=phase)
            except:
                res[i,:] = np.nan
    if retFlag:
        return res, flag
    return res

def _binVec(x, X, Y, E=None, medFilt=None, phase=False):
    """
    bin Y(X) with new x. E is optional error bars (wor weighting)
    """
    if E is None:
        E = np.ones(len(Y))
    # if phase:
    #     Y = np.unwrap(Y, period=360)
    # -- X can be irregular, so traditionnal convolution may not work
    y = np.zeros(len(x))
    Gx = np.gradient(x)
    #dx = np.median(np.diff(x))
    for i,x0 in enumerate(x):
        # -- kernel
        #k = np.exp(-(X-x0)**2/(0.6*dx)**2)
        k = np.exp(-(X-x0)**2/(0.6*Gx[i])**2)
        no = np.sum(k/E) # normalisation
        if no!=0 and np.isfinite(no):
            y[i] = np.sum(k/E*Y)/no
        else:
            y[i] = np.sum(k*Y)/np.sum(k)
        if phase:
            # if no!=0 and np.isfinite(no):
            #     y[i] = np.sum(k/E*((Y-y[i]+180)%360 - 180 + y[i]))/no
            # else:
            #     y[i] = np.sum(k*((Y-y[i]+180)%360 - 180 + y[i]))/np.sum(k)
            if no!=0 and np.isfinite(no):
                y[i] = np.angle(np.sum(k/E*np.exp(1j*Y*np.pi/180))/no)*180/np.pi
            else:
                y[i] = np.angle(np.sum(k*np.exp(1j*Y*np.pi/180))/np.sum(k))*180/np.pi
    return y

def mergeOI(OI, collapse=True, groups=None, verbose=False, debug=False, dMJD=None):
    """
    takes OI, a list of oifits files readouts (from loadOI), and merge them into
    a smaller number of entities based with same spectral coverage

    collapse=True -> all telescope / baseline / triangle in a single dict for
        faster computation

    groups=[...] list sub-strings for grouping insnames

    TODO: how polarisation in handled?
    """
    # -- create unique identifier for each setups
    setups = [oi['insname']+str(oi['WL']) for oi in OI]
    merged = [] # list of unique setup which have been merged so far
    master = [] # same length as OI, True if element hold merged data for the setup
    res = [] # result

    # -- for each data dict
    for i, oi in enumerate(OI):
        if debug:
            print('data set to merge:', i, setups[i])
        if not setups[i] in merged:
            # -- this will hold the data for this setup
            merged.append(setups[i])
            master.append(True)
            if debug:
                print('super oi')
            res.append(copy.deepcopy(oi))
            # -- filter in place
            for l in [r for r in res[-1].keys() if r.startswith('OI_')]:
                for k in sorted(res[-1][l].keys()):
                    for t in res[-1][l][k].keys():
                        if t.startswith('E') and t[1:] in res[-1][l][k] and 'fit' in res[-1]:
                            # -- errors -> allow editing based on 'fit'
                            res[-1][l][k][t] = _filtErr(t[1:], res[-1][l][k], res[-1]['fit'], debug=0)
                        elif t == 'FLAG' and 'fit' in res[-1]:
                            # -- flags -> allow editing based on 'fit'
                            res[-1][l][k][t] = _filtFlag(res[-1][l][k], res[-1]['fit'], debug=0)
            continue # exit for loop, nothing else to do
        else:
            # -- find data holder for this setup
            #i0 = np.argmax([setups[j]==setups[i] and master[j] for j in range(i)])
            i0 = merged.index(setups[i])
            master.append(False)
            # -- start merging with filename
            res[i0]['filename'] += ';'+oi['filename']

        if debug:
            print('  will be merged with', i0)
            print('  merging...')
        # -- extensions to be merged:
        exts = ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX', 'OI_CF']
        for l in filter(lambda e: e in exts, oi.keys()):
            if not l in res[i0].keys():
                # -- extension not present, using the one from oi
                res[i0][l] = copy.deepcopy(oi[l])
                if debug:
                    print('    ', i0, 'does not have', l, '!')
                # -- no additional merging needed
                dataMerge = False
            else:
                dataMerge = True

            # -- merge list of tel/base/tri per MJDs:
            for mjd in oi['configurations per MJD']:
                if not mjd in res[i0]['configurations per MJD']:
                    res[i0]['configurations per MJD'][mjd] = \
                         oi['configurations per MJD'][mjd]
                else:
                    for k in oi['configurations per MJD'][mjd]:
                        if not k in res[i0]['configurations per MJD'][mjd]:
                            res[i0]['configurations per MJD'][mjd].append(k)

            if not dataMerge:
                continue

            # -- merge data in the extension:
            for k in sorted(oi[l].keys()):
                # -- for each telescpe / baseline / triangle
                if not k in res[i0][l].keys():
                    # -- unknown telescope / baseline / triangle
                    # -- just add it in the the dict
                    res[i0][l][k] = copy.deepcopy(oi[l][k])
                    if 'FLUX' in l:
                        res[i0]['telescopes'].append(k)
                    elif 'VIS' in l or 'CF' in l:
                        res[i0]['baselines'].append(k)
                    elif 'T3' in l:
                        res[i0]['triangles'].append(k)
                    if debug:
                        print('    adding', k, 'to', l, 'in', i0)
                    # -- filter in place
                    for t in res[i0][l][k].keys():
                        if t.startswith('E') and t[1:] in res[i0][l][k] and 'fit' in res[i0]:
                            # -- errors -> allow editing
                            res[i0][l][k][t] = _filtErr(t[1:], res[i0][l][k], res[i0]['fit'])
                        elif t == 'FLAG' and 'fit' in res[i0]:
                            # -- flags -> editing based on errors
                            res[i0][l][k][t] = _filtFlag(res[i0][l][k], res[i0]['fit'])
                else:
                    # -- merging data (most complicated case)
                    # -- ext1 are scalar
                    # -- ext2 are vector of length WL
                    if l=='OI_FLUX':
                        ext1 = ['MJD']
                        ext2 = ['FLUX', 'EFLUX', 'FLAG', 'RFLUX', 'MJD2']
                    elif l=='OI_VIS2':
                        ext1 = ['u', 'v', 'MJD']
                        ext2 = ['V2', 'EV2', 'FLAG', 'u/wl', 'v/wl', 'B/wl', 'MJD2', 'PA']
                    elif l=='OI_VIS':
                        ext1 = ['u', 'v', 'MJD']
                        ext2 = ['|V|', 'E|V|', 'PHI', 'EPHI', 'FLAG', 'u/wl', 'v/wl', 'B/wl', 'MJD2', 'PA']
                    elif l=='OI_CF':
                        ext1 = ['u', 'v', 'MJD']
                        ext2 = ['CF', 'ECF', 'PHI', 'EPHI', 'FLAG', 'u/wl', 'v/wl', 'B/wl', 'MJD2', 'PA']
                    if l=='OI_T3':
                        ext1 = ['u1', 'v1', 'u2', 'v2', 'MJD', 'B1', 'B2', 'B3']
                        ext2 = ['T3AMP', 'ET3AMP', 'T3PHI', 'ET3PHI', 'FLAG', 'Bmax/wl', 'Bmin/wl', 'Bavg/wl', 'MJD2']
                    if debug:
                        print(l, k, res[i0][l][k].keys())
                    for t in ext1:
                        # -- append len(MJDs) data
                        if t in res[i0][l][k]:
                            res[i0][l][k][t] = np.append(res[i0][l][k][t], oi[l][k][t])
                    for t in ext2:
                        # -- append (len(MJDs),len(WL)) data
                        oops = False
                        try:
                            s1 = res[i0][l][k][t].shape # = (len(MJDs),len(WL))
                            s2 = oi[l][k][t].shape # = (len(MJDs),len(WL))
                        except:
                            #print('WARNING! in merging', l, k, t)
                            pass

                        if t.startswith('E') and t[1:] in oi[l][k] and 'fit' in oi:
                            # -- errors -> allow editing
                            tmp = _filtErr(t[1:], oi[l][k], oi['fit'])
                            #print(oi[l][k][t], '->', tmp)
                        elif t == 'FLAG' and 'fit' in oi:
                            # -- flags -> editing based on errors
                            tmp = _filtFlag(oi[l][k], oi['fit'])
                        elif t in oi[l][k]:
                            tmp = oi[l][k][t]
                        else:
                            oops = True
                            #print('!', l, k, t)
                        if not oops:
                            res[i0][l][k][t] = np.append(res[i0][l][k][t], tmp)
                            try:
                                res[i0][l][k][t] = res[i0][l][k][t].reshape(s1[0]+s2[0], s1[1])
                            except:
                                print('!!!', i0, l, k, t, s1 ,s2, res[i0][l][k]['u/wl'])

    for r in res:
        for k in ['telescopes', 'baselines', 'triangles']:
            if k in r:
                r[k] = list(set(r[k]))

    # -- make sure there is a 1-to-1 correspondance between VIS and VIS2:
    # -- should be handled at loading!!!
    # for r in res:
    #     if 'OI_VIS' in r and 'OI_VIS2' in r:
    #         kz = list(set(list(r['OI_VIS'].keys())+list(r['OI_VIS2'].keys())))
    #         #print(kz)
    #         for k in kz:
    #             if not k in r['OI_VIS']:
    #                 print(k, 'missing from OI_VIS!')
    #                 r['OI_VIS'][k]={}
    #                 for x in ['MJD', 'u', 'v', 'u/wl', 'v/wl', 'B/wl', 'PA',
    #                          'FLAG']:
    #                     r['OI_VIS'][k][x] = r['OI_VIS2'][k][x].copy()
    #                 r['OI_VIS'][k]['FLAG'] = np.logical_or(r['OI_VIS'][k]['FLAG'],
    #                                                         True)
    #                 r['OI_VIS'][k]['|V|'] = 1+0*r['OI_VIS2'][k]['V2']
    #                 r['OI_VIS'][k]['E|V|'] = 1+0*r['OI_VIS2'][k]['V2']
    #                 r['OI_VIS'][k]['PHI'] = 1+0*r['OI_VIS2'][k]['V2']
    #                 r['OI_VIS'][k]['EPHI'] = 1+0*r['OI_VIS2'][k]['V2']
    #             elif not k in r['OI_VIS2']: # -- NOT important...
    #                 print(k, 'missing from OI_VIS2!')
    #                 r['OI_VIS2'][k]={}
    #                 for x in ['MJD', 'u', 'v', 'u/wl', 'v/wl', 'B/wl', 'PA',
    #                          'FLAG']:
    #                     r['OI_VIS2'][k][x] = r['OI_VIS'][k][x].copy()
    #                 #r['OI_VIS2'][k]['FLAG'] += True
    #                 r['OI_VIS2'][k]['FLAG'] = np.logical_or(r['OI_VIS2'][k]['FLAG'],
    #                                                         True)
    #
    #                 r['OI_VIS2'][k]['V2'] = 1+0*r['OI_VIS'][k]['|V|']
    #                 r['OI_VIS2'][k]['EV2'] = 1+0*r['OI_VIS'][k]['|V|']
    #             mjd = list(set(list(r['OI_VIS'][k]['MJD'])+list(r['OI_VIS2'][k]['MJD'])))
    #             print(sorted(mjd), sorted(r['OI_VIS'][k]['MJD']), sorted(r['OI_VIS2'][k]['MJD']))
    #             if len(r['OI_VIS'][k]['MJD']) != len(mjd):
    #                 #print(k, 'VIS is missing data! (%d/%d)'%(len(r['OI_VIS'][k]['MJD']), len(mjd)))
    #                 w = [x not in r['OI_VIS'][k]['MJD'] for x in r['OI_VIS2'][k]['MJD']]
    #                 w = np.array(w)
    #
    #                 for x in ['MJD', 'u', 'v']:
    #                     r['OI_VIS'][k][x] = np.append(r['OI_VIS'][k][x],
    #                                                   r['OI_VIS2'][k][x][w])
    #
    #                 for x in ['u/wl', 'v/wl', 'B/wl', 'FLAG', 'PA',
    #                           '|V|', 'E|V|', 'PHI', 'EPHI']:
    #                     if x in r['OI_VIS2'][k]:
    #                         r['OI_VIS'][k][x] = np.concatenate((r['OI_VIS'][k][x],
    #                                                             r['OI_VIS2'][k][x][w].reshape(w.sum(), -1)))
    #                     else:
    #                         r['OI_VIS'][k][x] = np.concatenate((r['OI_VIS'][k][x],
    #                                                             1+0*r['OI_VIS2'][k]['V2'][w].reshape(w.sum(), -1)))
    #                 r['OI_VIS'][k]['FLAG'][-sum(w):,:] = np.logical_or(
    #                                     r['OI_VIS'][k]['FLAG'][-sum(w):,:],True)
    #
    #                 # -- check:
    #                 # for x in r['OI_VIS'][k]:
    #                 #     print(x, r['OI_VIS'][k][x].shape, end=' ')
    #                 #     if x in r['OI_VIS2'][k]:
    #                 #         print(r['OI_VIS2'][k][x].shape)
    #                 #     else:
    #                 #         print(r['OI_VIS2'][k]['V2'].shape)
    #
    #             if len(r['OI_VIS2'][k]['MJD']) != len(mjd):
    #                 #print(k, 'VIS2 is missing data! (%d/%d)'%(len(mjd)-len(r['OI_VIS2'][k]['MJD']), len(mjd)))
    #                 # -- TODO: copy what is above!
    #                 w = [x not in r['OI_VIS2'][k]['MJD'] for x in r['OI_VIS'][k]['MJD']]
    #                 w = np.array(w)
    #                 print(w)
    #                 for x in ['MJD', 'u', 'v']:
    #                     r['OI_VIS2'][k][x] = np.append(r['OI_VIS2'][k][x],
    #                                                    r['OI_VIS'][k][x][w])
    #
    #                 for x in ['u/wl', 'v/wl', 'B/wl', 'FLAG', 'PA',
    #                           'V2', 'EV2']:
    #                     if x in r['OI_VIS'][k]:
    #                         r['OI_VIS2'][k][x] = np.concatenate((r['OI_VIS2'][k][x],
    #                                                             r['OI_VIS'][k][x][w].reshape(w.sum(), -1)))
    #                     else:
    #                         r['OI_VIS2'][k][x] = np.concatenate((r['OI_VIS2'][k][x],
    #                                                             1+0*r['OI_VIS'][k]['|V|'][w].reshape(w.sum(), -1)))
    #                 r['OI_VIS2'][k]['FLAG'][-sum(w):,:] = np.logical_or(
    #                                     r['OI_VIS2'][k]['FLAG'][-sum(w):,:], True)
    #
    # -- special case for T3 formulas
    # -- match MJDs for T3 computations:
    for r in res:
        if 'OI_T3' in r.keys():
            rmkey = []
            for k in r['OI_T3'].keys():
                s, t, w0, w1, w2 = r['OI_T3'][k]['formula']
                _w0, _w1, _w2 = [], [], []
                if 'OI_VIS' in r.keys() and t[0] in r['OI_VIS'] and \
                        t[1] in r['OI_VIS'] and t[2] in r['OI_VIS']:
                    key = 'OI_VIS'
                elif 'OI_VIS2' in r.keys() and t[0] in r['OI_VIS2'] and \
                        t[1] in r['OI_VIS2'] and t[2] in r['OI_VIS2']:
                    key = 'OI_VIS2'
                else:
                    # I Should never arrive here!!!
                    key = None
                try:
                    #print(k, r[key].keys())
                    for i,mjd in enumerate(r['OI_T3'][k]['MJD']):
                        _w0.append(np.argmin((r[key][t[0]]['MJD']-mjd)**2+
                                            (r[key][t[0]]['u']-s[0]*r['OI_T3'][k]['u1'][i])**2+
                                            (r[key][t[0]]['v']-s[0]*r['OI_T3'][k]['v1'][i])**2))
                        _w1.append(np.argmin((r[key][t[1]]['MJD']-mjd)**2+
                                            (r[key][t[1]]['u']-s[1]*r['OI_T3'][k]['u2'][i])**2+
                                            (r[key][t[1]]['v']-s[1]*r['OI_T3'][k]['v2'][i])**2))
                        _w2.append(np.argmin((r[key][t[2]]['MJD']-mjd)**2+
                                            (r[key][t[2]]['u']+s[2]*r['OI_T3'][k]['u1'][i]+
                                                    s[2]*r['OI_T3'][k]['u2'][i])**2+
                                            (r[key][t[2]]['v']+s[2]*r['OI_T3'][k]['v1'][i]+
                                                    s[2]*r['OI_T3'][k]['v2'][i])**2))
                    r['OI_T3'][k]['formula'] = [s, t, _w0, _w1, _w2]
                except:
                    tmp = sorted(list(r[key].keys()))
                    print('OOPS: cannot compute T3 formula!!!')
                    print(k, t, t[0] in tmp, t[1] in tmp, t[2] in tmp, key, tmp)
                    rmkey.append(k)
                    r['OI_T3'][k]['formula'] = None
            for k in rmkey:
               r['OI_T3'].pop(k)

    # -- keep only master oi's, which contains merged data:
    if verbose:
        print('mergeOI:', len(OI), 'data files merged into', len(res), end=' ')
        if len(res)>1:
            print('dictionnaries with unique setups:')
        else:
            print('dictionnary with a unique setup:')
        for i, m in enumerate(res):
            print(' [%d]'%i, m['insname'], ' %d wavelengths:'%len(m['WL']),
                    m['WL'][0], '..', m['WL'][-1], 'um', end=' ')
            print(m['baselines'], end='\n  ')
            print('\n  '.join(m['filename'].split(';')))
    if collapse:
        # -- all baselines in a single key (faster computations)
        res = _allInOneOI(res, verbose=verbose, debug=debug)

    # -- keep fitting context, not need to keep error-based ones, except DPHI!
    for r in res:
        if 'fit' in r:
            tmp = {}
            for k in r['fit'].keys():
                # -- for differential quantities, these are globally defined
                for p in ['DPHI', 'NFLUX', 'N|V|']:
                    if type(r['fit'][k]) is dict and p in r['fit'][k].keys():
                        if k in tmp:
                            if p in tmp[k] and tmp[k][p]!=r['fit'][k][p]:
                                print('mergeOI: WARNING, merging cannot merge "'+
                                      k+'" for "'+p+'"')
                            tmp[k][p] = r['fit'][k][p]
                        else:
                            tmp[k] = {p:r['fit'][k][p]}
            r['fit'] = {k:r['fit'][k] for k in ['obs', 'wl ranges', 'baseline ranges',
                                                'MJD ranges', 'continuum ranges', 'prior',
                                                'Nr', 'DPHI order', 'N|V| order',
                                                'NFLUX order', 'ignore negative flux',
                                                'correlations', 'wl kernel', 'spatial kernel', 'smear']
                        if k in r['fit']}
            r['fit'].update(tmp)

    for r in res:
        if 'configurations per MJD' in r:
            r['MJD'] = np.array(sorted(set(r['configurations per MJD'].keys())))

    if not dMJD is None:
        tmp = []
        for r in res:
            tmp.extend(splitOIbyMJD(r, dMJD=dMJD))
        return tmp
    else:
        return res

def _filtErr(t, ext, filt, debug=False):
    """
    t: name of the observable (e.g. 'T3PHI', 'V2', etc)
    ext: dictionnary containing the data (OI extension)
    filt: what to apply to data ('fit' dict from OIDATA)
    """
    # -- original errors:
    err = ext['E'+t].copy()

    # == this is now in oimodels.computeNormFluxOI
    # -- this is a bit of a Kludge => impacts badly bootstrapping!!!
    # if t=='FLUX' and 'min error' in filt.keys() and 'NFLUX' in filt['min error']:
    #     filt['min error']['FLUX'] = filt['min error']['NFLUX']*np.median(ext['FLUX'])
    #
    # # -- this one is correct
    # if t=='FLUX' and 'mult error' in filt.keys() and 'NFLUX' in filt['mult error']:
    #     filt['mult error']['FLUX'] = filt['mult error']['NFLUX']

    if 'mult error' in filt.keys() and t in filt['mult error'].keys():
        if debug:
            print('mult error:', t, end=' ')
        err *= filt['mult error'][t]

    if 'min error' in filt.keys() and t in filt['min error'].keys():
        if debug:
            print('min error:', t, end=' ')
            print(np.sum(err<filt['min error'][t]), '/', err.size)
        err = np.maximum(filt['min error'][t], err)

    if 'min relative error' in filt.keys() and t in filt['min relative error'].keys():
        if debug:
            print('min relative error:', t, end=' ')
            print(np.sum(err<filt['min relative error'][t]*np.abs(ext[t])),
                    '/', err.size)
        err = np.maximum(filt['min relative error'][t]*np.abs(ext[t]), err)
    return err

def _filtFlag(ext, filt, debug=False):
    """
    ext: dictionnary containing data (OI extension)
    """
    flag = ext['FLAG'].copy()
    if not 'max error' in filt and not 'max relative error' in filt:
        return flag

    # -- this is a bit of a Kludge :S
    if 'FLUX' in ext and 'max error' in filt.keys() and 'NFLUX' in filt['max error']:
        filt['max error']['FLUX'] = filt['max error']['NFLUX']*ext['FLUX'].mean()

    if 'max error' in filt:
        for k in filt['max error']:
            if k == 'DPHI':
                k = 'PHI'
            if k == 'D|V|':
                k = '|V|'
            if k in filt['max error'] and 'E'+k in ext:
                if debug:
                    print('flag: max', k, filt['max error'])
                flag += ext['E'+k]>=filt['max error'][k]

    if 'max relative error' in filt:
        for k in filt['max relative error']:
            if k in filt['max relative error'] and 'E'+k in ext:
                if debug:
                    print('flag: max relative', k)
                flag += ext['E'+k]>=filt['max relative error'][k]*np.abs(ext[k])
    return flag

def _allInOneOI(oi, verbose=False, debug=False):
    """
    allInOne:
    - averages fluxes per MJD
    - puts all baselines / triangles in same dict

    """
    if type(oi)==list:
        return [_allInOneOI(o, verbose=verbose, debug=debug) for o in oi]

    for e in filter(lambda x: x in oi, ['OI_FLUX', 'NFLUX']):
        fluxes = {}
        weights = {}
        key = {'OI_FLUX':'FLUX', 'NFLUX':'NFLUX'}[e]
        names = {}
        for k in oi[e].keys():
            for j,mjd in enumerate(oi[e][k]['MJD']):
                mask = ~oi[e][k]['FLAG'][j,:]
                if not mjd in fluxes:
                    fluxes[mjd] = np.zeros(len(oi['WL']))
                    weights[mjd] = np.zeros(len(oi['WL']))
                    names[mjd] = []
                fluxes[mjd][mask] += oi[e][k][key][j,mask]/\
                        oi[e][k]['E'+key][j,mask]
                weights[mjd][mask] += 1/oi[e][k]['E'+key][j,mask]
                names[mjd].append(k)
        flags = {}
        efluxes = {}
        for mjd in fluxes.keys():
            names[mjd] = ';'.join(names[mjd])
            mask = weights[mjd]>0
            fluxes[mjd][mask] /= weights[mjd][mask]
            flags[mjd] = ~mask
            efluxes[mjd] = np.zeros(len(oi['WL']))
            efluxes[mjd][mask] = 1/weights[mjd][mask]
        oi[e]['all'] = {
            key: np.array([fluxes[mjd] for mjd in fluxes.keys()]),
            'E'+key: np.array([efluxes[mjd] for mjd in fluxes.keys()]),
            'FLAG': np.array([flags[mjd] for mjd in fluxes.keys()]),
            'NAME': np.array([names[mjd] for mjd in fluxes.keys()]),
            'MJD': np.array(list(fluxes.keys())),
            }
        oi[e]['all']['MJD2'] = oi[e]['all']['MJD'][:,None]+\
                            0*oi['WL'][None,:]

    for e in filter(lambda x: x in oi.keys(), ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_CF']):
        tmp = {'NAME':[]}
        for k in filter(lambda x: x!='all', sorted(oi[e].keys())): # each Tel/B/Tri
            tmp['NAME'].extend([k for i in range(len(oi[e][k]['MJD']))])
            for d in oi[e][k].keys(): # each data type
                if not d in tmp:
                    if type(oi[e][k][d])==list:
                        tmp[d] = [tuple(oi[e][k][d])]*len(oi[e][k]['MJD'])
                    else:
                        tmp[d] = oi[e][k][d]
                else:
                    if type(oi[e][k][d])==list:
                        tmp[d].extend([tuple(oi[e][k][d])]*len(oi[e][k]['MJD']))
                    elif type(oi[e][k][d])==np.ndarray:
                        if oi[e][k][d].ndim==1:
                            tmp[d] = np.append(tmp[d], oi[e][k][d])
                        elif oi[e][k][d].ndim==2:
                            try:
                                tmp[d] = np.append(tmp[d], oi[e][k][d], axis=0)
                            except:
                                #print('DEBUG:', e, k, d, tmp[d].shape, oi[e][k][d].shape)
                                pass
                    else:
                        print('allInOneOI warning: unknow data', e, d)
        tmp['NAME'] = np.array(tmp['NAME'])
        oi[e]['all'] = tmp

    if 'OI_T3' in oi:
        key = 'OI_VIS'
        if not key in oi:
            key = 'OI_VIS2'
        # -- recompute formula for T3
        _w0, _w1, _w2 = [], [], []
        _s0, _s1, _s2 = [], [], []
        for i,mjd in enumerate(oi['OI_T3']['all']['MJD']):
            s, t, w0, w1, w2 = oi['OI_T3']['all']['formula'][i]
            _s0.append(s[0])
            _s1.append(s[1])
            _s2.append(s[2])
            # _w0.append(np.argmin(abs(oi[key]['all']['MJD']-mjd)+(oi[key]['all']['NAME']!=t[0])))
            # _w1.append(np.argmin(abs(oi[key]['all']['MJD']-mjd)+(oi[key]['all']['NAME']!=t[1])))
            # _w2.append(np.argmin(abs(oi[key]['all']['MJD']-mjd)+(oi[key]['all']['NAME']!=t[2])))

            _w0.append(np.argmin((oi[key]['all']['MJD']-mjd)**2+
                                (oi[key]['all']['u']-s[0]*oi['OI_T3']['all']['u1'][i])**2+
                                (oi[key]['all']['v']-s[0]*oi['OI_T3']['all']['v1'][i])**2))
            _w1.append(np.argmin((oi[key]['all']['MJD']-mjd)**2+
                                (oi[key]['all']['u']-s[1]*oi['OI_T3']['all']['u2'][i])**2+
                                (oi[key]['all']['v']-s[1]*oi['OI_T3']['all']['v2'][i])**2))
            _w2.append(np.argmin((oi[key]['all']['MJD']-mjd)**2+
                                (oi[key]['all']['u']+s[2]*oi['OI_T3']['all']['u1'][i]+s[2]*oi['OI_T3']['all']['u2'][i])**2+
                                (oi[key]['all']['v']+s[2]*oi['OI_T3']['all']['v1'][i]+s[2]*oi['OI_T3']['all']['v2'][i])**2))

        s = np.array(_s0), np.array(_s1), np.array(_s2)
        oi['OI_T3']['all']['formula'] = [s, ('all', 'all', 'all'), _w0, _w1, _w2]
    for e in filter(lambda x: x.startswith('OI_'), oi.keys()):
        for k in list(oi[e].keys()):
            if k!='all':
                oi[e].pop(k)
    oi['telescopes'] = ['all']
    oi['baselines'] = ['all']
    oi['triangles'] = ['all']
    return oi

def medianFilt(oi, kernel_size=None):
    """
    kernel_size is the half width
    """
    if type(oi) == list:
        return [medianFilt(o, kernel_size=kernel_size) for o in oi]

    if 'OI_FLUX' in oi.keys():
        # -- make sure the tellurics are handled properly
        if 'TELLURICS' in oi.keys():
            t = oi['TELLURICS']
        else:
            t = np.ones(np.len(oi['WL']))
        for k in oi['OI_FLUX'].keys():
            for i in range(len(oi['OI_FLUX'][k]['MJD'])):
                mask = ~oi['OI_FLUX'][k]['FLAG'][i,:]
                oi['OI_FLUX'][k]['FLUX'][i,mask] = scipy.signal.medfilt(
                    oi['OI_FLUX'][k]['FLUX'][i,mask]/t[mask],
                    kernel_size=kernel_size)*t[mask]
                oi['OI_FLUX'][k]['EFLUX'][i,mask] /= np.sqrt(kernel_size)
    if 'OI_VIS' in oi.keys():
        for k in oi['OI_VIS'].keys():
            for i in range(len(oi['OI_VIS'][k]['MJD'])):
                mask = ~oi['OI_VIS'][k]['FLAG'][i,:]
                oi['OI_VIS'][k]['|V|'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS'][k]['|V|'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS'][k]['E|V|'][i,mask] /= np.sqrt(kernel_size)

                oi['OI_VIS'][k]['PHI'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS'][k]['PHI'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS'][k]['EPHI'][i,mask] /= np.sqrt(kernel_size)

    if 'OI_VIS2' in oi.keys():
        for k in oi['OI_VIS2'].keys():
            for i in range(len(oi['OI_VIS2'][k]['MJD'])):
                mask = ~oi['OI_VIS2'][k]['FLAG'][i,:]
                oi['OI_VIS2'][k]['V2'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS2'][k]['V2'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS2'][k]['EV2'][i,mask] /= np.sqrt(kernel_size)

    if 'OI_T3' in oi.keys():
        for k in oi['OI_T3'].keys():
            for i in range(len(oi['OI_T3'][k]['MJD'])):
                mask = ~oi['OI_T3'][k]['FLAG'][i,:]
                oi['OI_T3'][k]['T3PHI'][i,mask] = scipy.signal.medfilt(
                    oi['OI_T3'][k]['T3PHI'][i,mask], kernel_size=kernel_size)
                oi['OI_T3'][k]['ET3PHI'][i,mask] /= np.sqrt(kernel_size)

                oi['OI_T3'][k]['T3AMP'][i,mask] = scipy.signal.medfilt(
                    oi['OI_T3'][k]['T3AMP'][i,mask], kernel_size=kernel_size)
                oi['OI_T3'][k]['ET3AMP'][i,mask] /= np.sqrt(kernel_size)
    return oi

def n_JHK(wl, T=None, P=None, H=None):
    """
    wl: wavelength in microns (only valid from 1.3 to 2.5um)
    T: temperature in K
    P: pressure in mbar
    H: relative humidity in %

    from https://arxiv.org/pdf/physics/0610256.pdf
    """
    nu = 1e4/wl
    nuref = 1e4/2.25 # cm−1

    # -- https://arxiv.org/pdf/physics/0610256.pdf
    # -- table 1
    # -- i; ciref / cmi; ciT / cmiK;  ciTT / [cmiK2]; ciH / [cmi/%]; ciHH / [cmi/%2]
    table1a=[[0, 0.200192e-3, 0.588625e-1, -3.01513, -0.103945e-7, 0.573256e-12],
             [1, 0.113474e-9, -0.385766e-7, 0.406167e-3, 0.136858e-11, 0.186367e-16],
             [2, -0.424595e-14, 0.888019e-10, -0.514544e-6, -0.171039e-14, -0.228150e-19],
             [3, 0.100957e-16, -0.567650e-13, 0.343161e-9, 0.112908e-17, 0.150947e-22],
             [4, -0.293315e-20, 0.166615e-16, -0.101189e-12, -0.329925e-21, -0.441214e-26],
             [5, 0.307228e-24, -0.174845e-20, 0.106749e-16, 0.344747e-25, 0.461209e-30]]

    # -- cip / [cmi/Pa]; cipp / [cmi/Pa2]; ciTH / [cmiK/%]; ciTp / [cmiK/Pa]; ciHp / [cmi/(% Pa)]
    table1b = [[0, 0.267085e-8, 0.609186e-17, 0.497859e-4, 0.779176e-6, -0.206567e-15],
               [1, 0.135941e-14, 0.519024e-23, -0.661752e-8, 0.396499e-12, 0.106141e-20],
               [2, 0.135295e-18, -0.419477e-27, 0.832034e-11, 0.395114e-16, -0.149982e-23],
               [3, 0.818218e-23, 0.434120e-30, -0.551793e-14, 0.233587e-20, 0.984046e-27],
               [4, -0.222957e-26, -0.122445e-33, 0.161899e-17, -0.636441e-24, -0.288266e-30],
               [5, 0.249964e-30, 0.134816e-37, -0.169901e-21, 0.716868e-28, 0.299105e-34]]

    Tref, Href, pref = 273.15+17.5, 10., 75e3
    if T is None:
        T = Tref
    if P is None:
        P = pref/100
    if H is None:
        H = Href

    n = 0.0
    p = P*100 # formula in Pa, not mbar

    for k,ca in enumerate(table1a):
        i = ca[0]
        ciref = ca[1]
        ciT = ca[2]
        ciTT = ca[3]
        ciH = ca[4]
        ciHH = ca[5]
        cb = table1b[k]
        cip = cb[1]
        cipp = cb[2]
        ciTH = cb[3]
        ciTp = cb[4]
        ciHp = cb[5]
        # -- equation 7
        ci = ciref + ciT*(1/T - 1/Tref) + ciTT*(1/T - 1/Tref)**2 +\
            ciH*(H-Href) + ciHH*(H-Href)**2 + cip*(p-pref) + cipp*(p-pref)**2 +\
            ciTH*(1/T-1/Tref)*(H-Href) + ciTp*(1/T-1/Tref)*(p-pref) +\
            ciHp*(H-Href)*(p-pref)
        # -- equation 6
        #print('mathar:', i, ciref, ci)
        n += ci*(nu - nuref)**i
    return n+1.0

def OI2FITS(oi, fitsfile):
    pass

def getESOPipelineParams(H, verbose=True):
    """
    H is a header
    """
    P = {}
    last_rec = ''
    p = OrderedDict()
    for k in filter(lambda x: 'PARAM' in x and 'NAME' in x, H.keys()):
        rec = k.split('ESO PRO ')[1].split()[0]
        if rec!=last_rec:
            if verbose:
                if last_rec!='':
                    print()
                print('\033[46m'+rec, H['ESO PRO '+rec+' ID'],
                        '['+H['ESO PRO '+rec+' PIPE ID']+']\033[0m', end=' ')
            # -- find files:
            F = []
            for f in filter(lambda x: 'ESO PRO '+rec+' RAW' in x and 'NAME' in x, H.keys()):
                F.append(H[f]+' ('+H[f.replace('NAME', 'CATG')]+')')
            for f in filter(lambda x: 'ESO PRO '+rec+' CAL' in x and 'NAME' in x, H.keys()):
                F.append(H[f]+' ('+H[f.replace('NAME', 'CATG')]+')')

            if verbose:
                print(', '.join(F))
            if last_rec!='':
                P[last_rec] = {'ID':H['ESO PRO '+rec+' ID'],
                            'DRS':H['ESO PRO '+rec+' DRS ID'],
                            'PIPE ID': H['ESO PRO '+rec+' PIPE ID'],
                            'parameters':p,
                            'files':F}
                p = OrderedDict()
            last_rec = rec
        if H[k.replace('NAME', 'VALUE')]=='true':
            c = '\033[32m'
        elif H[k.replace('NAME', 'VALUE')]=='false':
            c = '\033[31m'
        else:
            c = '\033[34m'
        p[H[k]] = H[k.replace('NAME', 'VALUE')]
        if verbose:
            print(H[k]+'='+c+H[k.replace('NAME', 'VALUE')]+'\033[0m', end=' ')
    if verbose:
        print()
    if p!={} and last_rec!='':
        P[last_rec] = {'ID':H['ESO PRO '+rec+' ID'],
                    'DRS':H['ESO PRO '+rec+' DRS ID'],
                    'PIPE ID': H['ESO PRO '+rec+' PIPE ID'],
                    'parameters':p,
                    'files':F}
    return P
