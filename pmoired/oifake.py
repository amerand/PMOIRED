import numpy as np
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
import itertools

try:
    # Necessary while Python versions below 3.9 are supported.
    import importlib_resources as resources
except ImportError:
    from importlib import resources

import scipy.special

from pmoired import oimodels

# -- generate fake VLTI data

# -- some problems found by G. Duvert in the 'A0' terms (last col):
# @wvgvlti:Appl_data:VLTI:vltiParameter:telStations.nomCentre
# -- altitude: W=4.5397 fot ATs, W=13.044 for the UTs
vlti_latitude = -24.62743941
vlti_longitude = -70.40498688
layout_orientation = -18.984  # degrees
layout = {'A0': (-32.0010, -48.0130, -14.6416, -55.8116, 129.8495),
          'A1': (-32.0010, -64.0210, -9.4342, -70.9489, 150.8475),
          'B0': (-23.9910, -48.0190, -7.0653, -53.2116, 126.8355),
          'B1': (-23.9910, -64.0110, -1.8631, -68.3338, 142.8275),
          'B2': (-23.9910, -72.0110, 0.7394, -75.8987, 150.8275, ),
          'B3': (-23.9910, -80.0290, 3.3476, -83.4805, 158.8455),
          'B4': (-23.9910, -88.0130, 5.9449, -91.0303, 166.8295),
          'B5': (-23.9910, -96.0120, 8.5470, -98.5942, 174.8285),
          'C0': (-16.0020, -48.0130, 0.4872, -50.6071, 118.8405),
          'C1': (-16.0020, -64.0110, 5.6914, -65.7349, 134.8385),
          'C2': (-16.0020, -72.0190, 8.2964, -73.3074, 142.8465),
          'C3': (-16.0020, -80.0100, 10.8959, -80.8637, 150.8375),
          'D0': (0.0100, -48.0120, 15.6280, -45.3973, 97.8375),
          'D1': (0.0100, -80.0150, 26.0387, -75.6597, 134.8305),
          'D2': (0.0100, -96.0120, 31.2426, -90.7866, 150.8275),
          'E0': (16.0110, -48.0160, 30.7600, -40.1959, 81.8405),
          'G0': (32.0170, -48.0172, 45.8958, -34.9903, 65.8357),
          'G1': (32.0200, -112.0100, 66.7157, -95.5015, 129.8255),
          'G2': (31.9950, -24.0030, 38.0630, -12.2894, 73.0153),
          'H0': (64.0150, -48.0070, 76.1501, -24.5715, 58.1953),
          'I1': (72.0010, -87.9970, 96.7106, -59.7886, 111.1613),
          # -- XY are correct, A0 is guessed!
          'I2': (80, -24, 83.456, 3.330, 90),
          'J1': (88.0160, -71.9920, 106.6481, -39.4443, 111.1713),
          'J2': (88.0160, -96.0050, 114.4596, -62.1513, 135.1843),
          'J3': (88.0160, 7.9960, 80.6276, 36.1931, 124.4875),
          'J4': (88.0160, 23.9930, 75.4237, 51.3200, 140.4845),
          'J5': (88.0160, 47.9870, 67.6184, 74.0089, 164.4785),
          'J6': (88.0160, 71.9900, 59.8101, 96.7064, 188.4815),
          'K0': (96.0020, -48.0060, 106.3969, -14.1651, 90.1813),
          'L0': (104.0210, -47.9980, 113.9772, -11.5489, 103.1823),
          'M0': (112.0130, -48.0000, 121.5351, -8.9510, 111.1763),
          'U1': (-16.0000, -16.0000, -9.9249, -20.3346, 189.0572),
          'U2': (24.0000, 24.0000, 14.8873, 30.5019, 190.5572),
          'U3': (64.0000, 48.0000, 44.9044, 66.2087, 199.7447),
          'U4': (112.0000, 8.0000, 103.3058, 43.9989, 209.2302),
          # 'N0':[188,     -48,     0,        0, 188],
          # 'J7':[ 88,    -167,     0,        0, 200]
          }
# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.delayLineParams
# --            U      Vin      Vout
DL_UV = {'DL1': (58.92, -37.245, -37.005),
         'DL2': (58.92, -37.995, -37.755),
         'DL3': (45.08, -38.505, -38.745),
         'DL4': (45.08, -39.255, -39.495),
         'DL5': (58.92, -40.745, -40.505),
         'DL6': (58.92, -41.495, -41.255)}

# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.LabInputCoord
# -- U coordinates of IP 1 through 8:
IP_U = {'IP1': 52.32, 'IP2': 52.56, 'IP3': 52.80, 'IP4': 53.04,
        'IP5': 53.28, 'IP6': 53.52, 'IP7': 53.76, 'IP8': 54.00}

# @wvgvlti:Appl_data:VLTI:vltiParameter:optics.SwitchYardParams
# -- OPL per input: direct, BC, DDL, BC+DDL
OPL = {'IP1': (2.105, 11.5625, 11.5298, 20.9873),
       'IP2': (1.385, 12.7625,  9.6098, 20.9873),
       'IP3': (2.825, 11.8025, 11.2898, 20.2673),
       'IP4': (2.105, 13.0025,  9.3698, 20.2673),
       'IP5': (3.545, 12.0425, 11.0498, 19.5473),
       'IP6': (2.825, 13.2425,  9.1298, 19.5473),
       'IP7': (4.265, 12.2825, 10.8098, 18.8273),
       'IP8': (3.545, 13.4825,  8.8898, 18.8273), }

# ---------- horizons ---------------
# -- seems to come from VLT-TRE-ESO-15000-2551
# -- https://pdm.eso.org/kronodoc/1100/Get/265745/VLT-TRE-ESO-15000-2551_1.PDF


def loadHorizon(filename=None):
    # -- use custom computation hzlib.tabulateAll()
    with resources.as_file(resources.files('pmoired').joinpath('newVltiHzn_obsDoors.txt')) as p:
        if filename is None:
            filename = p
        with open(filename, 'r') as f:
            # print('- reading VLTI horizons from '+filename)
            for l in f.readlines():
                if not l.startswith('#'):
                    az.append(float(l.split()[0]))
                    for i, k in enumerate(keys):
                        horizon[k].append(float(l.split()[i+1]))
                elif l.startswith('# az'):
                    keys = l.split('# az')[1].split()
                    az = []
                    horizon = {k: [] for k in keys}
                else:
                    pass
    for k in keys:
        horizon[k] = (az, horizon[k])
    return horizon


horizon = loadHorizon()


def nTelescopes(T, target, lst, ip=None, DLconstraints=None,
                min_altitude=30, max_OPD=100, max_vcmPressure=None,
                max_airmass=None, flexible=True, plot=False, STS=True):
    """
    Compute u, v, alt, az, airmass, tracks for a nT configurartion

    - T: list of telescopes stations OR stationDL. example: ['A0','K1']
      or ['A0DL1','K0DL4']. If delaylines are given, it will compute the optimum
      position for the dlines and VCM pressure.

    - target: Can be a string with the name of the target resolvable by simbad.
      Can also be a tuple of 2 decimal numbers for the coordinates (ra, dec), in
      decimal hours and decimal degrees

    - lst: a list (or ndarray) of local sidearal times.

    - [ip]: list of input channels. The order should be the same as telescopes.

    - min_altitude: for observability computation, in degrees (default=20deg)

    - max_OPD: for observability, in m (default=100m)

    - max_vcmPressure: enforce the VCM limitation (SLOW!!!)

    - flexible: if False, qperform some checks on the Array Configuration: no
      AT/UT; no more than 4 ATs west of the I track (DL1,2,4,5) and no more
      than 2 ATs east ofq the G track (DL3,4); no more than one AT per track
      (e.g. A0-A1 is not possible). *flexible=False will allow currently
      impossible configurations*

    - plot: True/False

    - DLconstraints sets min and max. For example: {'min1':35, 'max2':70}

    Returns: return a self explanatory dictionnary. units are m and degrees

    the key 'observable' is a table of booleans (same length as 'lst')
    indicating wether or not the object is observable.

    """
    tmp = []
    if ip is None:
        # -- assumes 1,3,5,7 for 4T
        ip = range(2*len(T))[1::2]

    if all(['DL' in x for x in T]):
        dl = [int(t.split('DL')[1]) for t in T]
        T = [t.split('DL')[0] for t in T]
        if len(set(dl)) < len(dl):
            print('ERROR: some DL used more than once')
            return None
        if len(set(T)) < len(T):
            print('ERROR: some STATIONS used more than once')
            return None
    else:
        dl = None

    if not isinstance(DLconstraints, dict):
        DLconstraints = {}

    if not flexible:
        # -- check is we have at most 4 Tel p<30m and at most 2T p>30m
        W = list(filter(lambda t: layout[t][0] <= 35, T))
        E = list(filter(lambda t: layout[t][0] >= 35, T))
        assert len(W) <= 4, 'Too many stations West of the laboratory'
        assert len(E) <= 2, 'Too many stations East of the laboratory'

        # -- check the delay lines association
        if not dl is None:
            for k in range(len(T)):
                assert (layout[T[k]][0] <= 35 and dl[k] in [1, 2, 5, 6]) or \
                       (layout[T[k]][0] >= 35 and dl[k] in [3, 4]), \
                    'wrong delay line / station association'

        # -- check that all telescopes are UTs if any UTs
        tracks = [t[0] for t in T]
        if 'U' in tracks:
            assert len(
                set(tracks)) == 1, 'UTs and ATs cannot be mixed, use flexible=False'
        else:
            if not any(['J' in t for t in T]):
                assert len(set(tracks)) == len(T), \
                    'only one AT per track is allowed, use flexible=False'
            else:  # special case for J stations
                _T = list(filter(lambda t: not t.startswith('J'), T))
                _tracks = [t[0] for t in _T]
                assert len(set(_tracks)) == len(_T), \
                    'only one AT per track is allowed, use flexible=False'
                _T = list(filter(lambda t: t.startswith('J'), T))
                _s = list(filter(lambda t: t in ['J1', 'J2'], T))
                _n = list(filter(lambda t: t in ['J3', 'J4', 'J5', 'J6'], T))
                assert len(
                    _s) <= 1, 'only AT on J south track, use flexible=False'
                assert len(
                    _n) <= 1, 'only AT on J north track, use flexible=False'

    if isinstance(target, str):
        # s = simbad.query(target)
        # target_name=target
        # target = [s[0]['RA.h'], s[0]['DEC.d']]
        s = Simbad.query_object(target)
        try:
            ra = np.sum(np.float64(s['RA'][0].split())
                        * np.array([1, 1/60., 1/3600.]))
            dec = np.float64(s['DEC'][0].split())*np.array([1, 1/60., 1/3600.])
            dec = np.sum(np.abs(dec))*np.sign(dec[0])
        except:
            ra, dec = 0, 0
            print('WARNING: simbad query failed for', target)
            print(s)
        target_name = target
        target = ra, dec
    else:
        # assume [ra dec]
        if int(60*(target[0]-int(target[0]))) < 10:
            target_name = 'RA=%2d:0%1d' % (int(target[0]),
                                           int(60*(target[0]-int(target[0]))),)
        else:
            target_name = 'RA=%2d:%2d' % (int(target[0]),
                                          int(60*(target[0]-int(target[0]))),)
        if int(60*(abs(target[1])-int(abs(target[1])))) < 10:
            target_name += ' DEC=%3d:0%1d' % (int(target[1]),
                                              int(60*(abs(target[1]) -
                                                      int(abs(target[1])))),)
        else:
            target_name += ' DEC=%3d:%2d' % (int(target[1]),
                                             int(60*(abs(target[1]) -
                                                     int(abs(target[1])))),)
    res = {}
    if not dl is None:
        res['config'] = [T[k]+'DL'+str(dl[k])+'IP'+str(ip[k])
                         for k in range(len(ip))]
    else:
        res['config'] = [T[k]+'IP'+str(ip[k]) for k in range(len(ip))]

    for i in range(len(T)+1):
        for j in range(len(T))[i+1:]:
            tmp = projBaseline(T[i], T[j],
                               target, lst, ip1=ip[i], ip2=ip[j],
                               DL1=None if dl is None else dl[i],
                               DL2=None if dl is None else dl[j],
                               min_altitude=min_altitude,
                               max_airmass=max_airmass,
                               max_OPD=max_OPD)
            if not 'lst' in res.keys():
                # init
                for k in ['lst', 'airmass', 'observable',
                          'ra', 'dec', 'alt', 'az', 'parang',
                          'horizon']:
                    res[k] = tmp[k]
                res['baselines'] = [tmp['baselines']]
                for k in ['B', 'PA', 'u', 'v', 'opd', 'ground']:
                    res[k] = {}
            else:
                # update
                res['observable'] = res['observable']*tmp['observable']
                res['baselines'].append(tmp['baselines'])
                res['horizon'] = np.maximum(res['horizon'],
                                            tmp['horizon'])
            for k in ['B', 'PA', 'u', 'v', 'opd', 'ground']:
                res[k][tmp['baselines']] = tmp[k]

    # -- convert OPL into DL position in the lab:
    # -- @wvgvlti:Appl_data:VLTI:vltiParameter:optics:LabInputCoord
    x_M16 = 52.  # where the center of the DL tunnel
    dx_dl = 6.92  # DL start to M16 -> real number!
    y_lab = -28
    # -- coordinate of 0-OPD in the lab
    y_ip = {'IP1': y_lab+5*0.24, 'IP2': y_lab+1*0.24,
            'IP3': y_lab+6*0.24, 'IP4': y_lab+2*0.24,
            'IP5': y_lab+7*0.24, 'IP6': y_lab+3*0.24,
            'IP7': y_lab+8*0.24, 'IP8': y_lab+4*0.24, }
    x_ip = x_M16 - 5

    wdl = .3  # width of the DL (drawing only)
    def fy_M12(x): return y_lab - 1.8*x  # M12 y as fct of DL index
    def fx_IP(x): return x_M16 + 1*x - 4  # IP x as fct of IP index

    minDL_opl = 11.  # smallest OPL for each cart

    # -- compute DL position and VCM position
    if not dl is None:  # Delay lines are given, so compute VCM limitation
        res['vcm'] = {}
        res['dl'] = {}
        for d in dl:
            res['vcm'][d] = []
            res['dl'][d] = []
        # -- compute OPD dictionnary
        for k in range(len(res['lst'])):
            opdD = {}
            for i1, t1 in enumerate(T):
                for i2, t2 in enumerate(T):
                    maxO = max_OPD
                    if t1+t2 in res['opd'].keys():
                        opdD[(dl[i1], dl[i2])] = res['opd'][t1+t2][k]
            s = solveDLposition(dl, opdD, stations=T, STS=STS,
                                dlPosMin=minDL_opl/2.,
                                dlPosMax=(max_OPD+minDL_opl)/2.,
                                constraints=DLconstraints)
            for d in dl:
                if np.isnan(s[d]):
                    res['observable'][k] = False
                res['dl'][d].append(s[d])
                res['vcm'][d].append(s['vcm'+str(d)])

        for d in dl:
            res['dl'][d] = np.array(res['dl'][d])
            res['vcm'][d] = np.array(res['vcm'][d])

        if not max_vcmPressure is None:
            maxP = 0
            for d in dl:
                maxP = np.maximum(maxP, res['vcm'][d])
            res['observable'] = res['observable']*(maxP < max_vcmPressure)

        res['x_dl'] = {}
        res['y_dl'] = {}

        for i, d in enumerate(dl):
            if d == 3 or d == 4:
                res['x_dl'][d] = x_M16 - dx_dl - res['dl'][d]
            else:
                res['x_dl'][d] = x_M16 + dx_dl + res['dl'][d]
            res['y_dl'][d] = fy_M12(d)

    # -- plot observability
    if plot:
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 0.5, 0), (0, 1, 0.5), (0.5, 0, 1)]
        colors = [(0.3+c[0]*0.5,
                   0.3+c[1]*0.5,
                   0.3+c[2]*0.5) for c in colors]

        def where(key): return [res[key][k] for k in
                                range(len(res[key]))
                                if res['observable'][k]]

        def where2(key1, key2): return [res[key1][key2][k] for k in
                                        range(len(res['lst']))
                                        if res['observable'][k]]
        if isinstance(plot, int):
            fig = plot
        else:
            fig = 0
        if not dl is None:
            plt.figure(fig, figsize=(12, 7))
            plt.clf()
            plt.subplots_adjust(hspace=0.16, top=0.94, left=0.06,
                                wspace=0.15, right=0.98, bottom=0.01)
            ax1 = plt.subplot(221)
        else:
            plt.figure(fig, figsize=(7, 5))
            plt.clf()
            plt.subplots_adjust(wspace=0.07, top=0.90, left=0.08,
                                right=0.98, bottom=0.12)
            ax1 = plt.subplot(111)

        plt.plot(res['lst'], res['lst']*0 + min_altitude, '-',
                 linestyle='dashed', color='k', linewidth=3, alpha=0.4)

        plt.plot(res['lst'], res['horizon'],
                 color=(0.1, 0.2, 0.4), alpha=0.3,
                 label='UT shadow', linewidth=2)
        plt.fill_between(res['lst'], res['horizon']*0.0,
                         res['horizon'], color=(0.1, 0.2, 0.4),
                         alpha=0.5)
        plt.plot(res['lst'], res['alt'], '+', alpha=0.9,
                 color=(0.5, 0.1, 0.3), label='not obs.',
                 markersize=8)
        plt.plot(where('lst'), where('alt'), 'o', alpha=0.5,
                 color=(0.1, 0.5, 0.3), label='observable',
                 markersize=8)
        plt.legend(prop={'size': 9}, numpoints=1)
        plt.ylim(0, 92)
        plt.xlabel('lst (h)')
        plt.ylabel('altitude (deg)')
        plt.suptitle(target_name+' '+'-'.join(res['config']) +
                     (' [STS]' if STS else ''))
        # plt.hlines(min_altitude, plt.xlim()[0], plt.xlim()[1],
        #              color='r', linewidth=1, linestyle='dotted')
        # -- make LST labels to be in 0..24
        # xl = plt.xticks()
        # xl = [str(int(x)%24) for x in xl[0]]
        # ax1.set_xticklabels(xl)
        plt.grid()
        # -- DL cart position
        if dl is None:
            return

        # -- OPL (2xDL position) for each DL:
        ax3 = plt.subplot(222, sharex=ax1)
        for i, d in enumerate(dl):
            plt.plot(where('lst'), 2*np.array(where2('dl', d)),
                     '-', label='DL'+str(d),
                     alpha=0.7, color=colors[i], linewidth=3)
            # -- plt ref DL
            wr = np.where(np.diff(where2('dl', d)) == 0)
            plt.plot(np.array(where('lst'))[wr], 2*np.array(where2('dl', d))[wr],
                     alpha=1, color=colors[i], linewidth=5,
                     linestyle='dashed')

        plt.legend(loc='upper center', prop={'size': 10}, numpoints=1,
                   ncol=6)
        plt.xlabel('lst (h)')
        plt.ylabel('OPL (m)')
        plt.hlines([minDL_opl, max_OPD+minDL_opl], res['lst'].min(), res['lst'].max(),
                   linestyle='dashed', color='k', linewidth=3, alpha=0.4)
        plt.grid()
        plt.ylim(0, 130)

        # ========================
        # == TUNNEL ==============
        # ========================

        # markers:
        DLm_W = [(-0.03, 0.04), (-0.1, 0.04), (-0.12, 0.01), (-0.12, -0.01),
                 (-0.1, -0.04), (-0.1, -0.06), (0.1, -0.06), (0.1, -0.04),
                 (-0.03, -0.04), (-0.03, 0.04)]

        DLm_E = [(-d[0], d[1]) for d in DLm_W]

        ax2 = plt.subplot(212)
        plt.text(x_M16, y_lab+0.3, 'LAB', ha='center', va='bottom',
                 color='0.5')
        # -- delay lines rails and ticks:
        for d in [1, 2, 3, 4, 5, 6]:
            s = (-1)**(d == 3 or d == 4)
            # -- range
            plt.plot([x_M16 + s*dx_dl, x_M16 + s*dx_dl + s*60],
                     [fy_M12(d)-wdl/2, fy_M12(d)-wdl/2], '-k',
                     alpha=0.5)
            plt.text(x_M16 + s*dx_dl + s*62, fy_M12(d)-wdl/2,
                     'DL'+str(d), va='center', color='0.5', size=9,
                     ha='right' if s == -1 else 'left')
            # -- 10m ticks
            plt.plot(x_M16 + s*dx_dl + s*np.linspace(0, 60, 7),
                     fy_M12(d)-wdl/2 + np.zeros(7), '|k',
                     markersize=9, color='0.6')
            # -- 5m ticks
            plt.plot(x_M16 + s*dx_dl + s*np.linspace(0, 50, 6)+s*5,
                     fy_M12(d)-wdl/2 + np.zeros(6), '|k',
                     markersize=7, color='0.4')
            for x in np.linspace(0, 60, 7):
                plt.text(x_M16 + s*dx_dl + s*x, fy_M12(d)+0.6*wdl,
                         str(int(2*x)), size=6, va='bottom',
                         ha='center', color='k')
            # -- DL restrictions
            for k in DLconstraints.keys():
                if str(d) in k:
                    # print(d, k, DLconstraints[k])
                    if 'min' in k:
                        plt.fill_between(x_M16 + s*dx_dl +
                                         s*np.array([0, DLconstraints[k]/2]),
                                         fy_M12(d)-np.array([0.9, 0.9])*wdl,
                                         fy_M12(d)-np.array([0.1, 0.1])*wdl,
                                         color='k', hatch='////', alpha=0.3)
                        plt.text(x_M16 + s*dx_dl + s*DLconstraints[k]/2,
                                 fy_M12(d)-2.5*wdl, str(DLconstraints[k]),
                                 color='r', size=8, va='bottom', ha='center')
                    elif 'max' in k:
                        plt.fill_between(x_M16 + s*dx_dl +
                                         s*np.array([60, DLconstraints[k]/2]),
                                         fy_M12(d)-np.array([0.9, 0.9])*wdl,
                                         fy_M12(d)-np.array([0.1, 0.1])*wdl,
                                         color='k', hatch='////', alpha=0.3)
                        plt.text(x_M16 + s*dx_dl + s*DLconstraints[k]/2,
                                 fy_M12(d)-2.5*wdl, str(DLconstraints[k]),
                                 color='r', size=8, va='bottom', ha='center')

        # -- used range:
        for i, d in enumerate(dl):
            # print(d, np.min(where2('dl', d)), np.max(where2('dl', d)))
            s = (-1)**(d == 3 or d == 4)
            # -- DL range
            plt.plot([x_M16 + s*dx_dl + s*np.min(where2('dl', d)),
                     x_M16 + s*dx_dl + s*np.max(where2('dl', d))],
                     [fy_M12(d)-wdl/2, fy_M12(d)-wdl/2], '-',
                     color=colors[i], alpha=0.5, linewidth=8,
                     label='range %d' % (d))

        # -- light path: T -> M12 -> cart -> M16
        for i, t in enumerate(T):
            y_M12 = fy_M12(dl[i])
            x_IP = fx_IP(ip[i])
            s = (-1)**(dl[i] == 3 or dl[i] == 4)
            x_cart = x_M16 + s*dx_dl + s * \
                np.mean(where2('dl', dl[i]))  # middle position
            # -- light path
            plt.plot([layout[t][0], layout[t][0], x_M16, x_cart+s*1, x_cart+s*1, x_IP, x_IP],
                     [layout[t][1], y_M12, y_M12, y_M12,
                         y_M12-wdl, y_M12-wdl, y_lab],
                     '-', alpha=0.5, linewidth=1.5, color=colors[i])
            # -- cart
            plt.plot(x_cart, y_M12-wdl/2, color=colors[i], markersize=22,
                     marker=DLm_W if (dl[i] == 3 or dl[i] == 4) else DLm_E)

            # -- write name of station where light is coming from
            if layout[t][1] > y_lab:  # north of the tracks
                plt.text(layout[t][0], y_lab-wdl, t, color=colors[i],
                         va='top', ha='center')
            else:  # south of the tracks
                plt.text(layout[t][0], fy_M12(7)+wdl, t, color=colors[i],
                         va='bottom', ha='center')
        # -- DL tunnel walls:
        U0 = layout['A0'][0] - 5
        U1 = layout['M0'][0] + 10
        color = (0.1, 0.2, 0.7)
        plt.fill_between([x_M16-110, x_M16+90], fy_M12(7)*np.array([1, 1]),
                         (fy_M12(7)-10)*np.array([1, 1]),
                         color=color, alpha=0.2, hatch='//')
        plt.fill_between([x_M16-110, x_M16+90], y_lab*np.array([1, 1]),
                         (y_lab+10)*np.array([1, 1]),
                         color=color, alpha=0.2, hatch='//')
        # end walls
        plt.fill_between((U0-10, U0), (fy_M12(7), fy_M12(7)), (y_lab, y_lab),
                         color=color, alpha=0.2, hatch='//')
        plt.fill_between((U1, U1+10), (fy_M12(7), fy_M12(7)), (y_lab, y_lab),
                         color=color, alpha=0.2, hatch='//')

        plt.ylim(fy_M12(7)-4*wdl, y_lab+4*wdl)
        plt.xlim(U0-4, U1+4)
        # plt.title('Delay line tunnel')
        ax2.get_yaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)

        # -- VCM pressure for each DL:
        # ax4 = plt.subplot(224, sharex=ax1)
        # for i,d in enumerate(dl):
        #     #plt.plot(res['lst'], res['vcm'][d], '-',label='VCM'+str(d))
        #     plt.plot(where('lst'), where2('vcm', d),
        #                 '-',markersize=8,label='VCM'+str(d),
        #                 alpha=0.7, color=colors[i], linewidth=3)
        #
        # plt.legend(loc='upper left', prop={'size':10}, numpoints=1,
        #            ncol=6)
        # plt.ylabel('DL VCM pressure %s(bar)'%('for AT/STS or UT' if STS else ''))
        # plt.xlabel('lst (h)')
        # ax1.set_xlim(res['lst'].min(), res['lst'].max())
        # plt.grid()
        # plt.ylim(0,5.0)
        # plt.hlines(2.5, res['lst'].min(), res['lst'].max(), color='k',
        #            alpha=0.4, linestyle='dashed', linewidth=3)

        print('tracking time:', res['lst'][np.where(res['observable'])[0][-1]] -
              res['lst'][np.where(res['observable'])[0][0]])
    else:
        return res


def projBaseline(T1, T2, target, lst, ip1=1, ip2=3, DL1=None, DL2=None,
                 min_altitude=30, max_OPD=100, max_airmass=None):
    """
    - T1, T2: telescopes stations (e.g. 'A0', 'U1', etc.)

    - target : [ra, dec] in decimal hour and decimal deg or name for SIMBAD

    - lst   : decimal hour (scalar or array)

    - ip1, ip2: input channels (optional)

    - DL1, DL2: delay lines (optional)

    min_altitude (default 20 degrees) sets the minimum altitude for
    observation. Checks also for shadowinf from the UTs.

    max_OPD (default 100m, from 11, to 111m) maximum OPD stroke of the delay lines.

    return a self explanatory dictionnary. units are m and degrees
    """
    if isinstance(target, str):
        # -- old simbad
        # s = simbad.query(target)[0]
        # radec = [s['RA.h'], s['DEC.d']]
        # --
        s = Simbad.query_object(target)
        ra = np.sum(np.float64(s['RA'][0].split())
                    * np.array([1, 1/60., 1/3600.]))
        dec = np.float64(s['DEC'][0].split())*np.array([1, 1/60., 1/3600.])
        dec = np.sum(np.abs(dec))*np.sign(dec[0])
        radec = ra, dec

    else:
        radec = target

    # -- hour angle
    ha = (np.array(lst) - radec[0]) * 360/24.0

    # -- alt-az
    dec = radec[1]

    tmp1 = np.sin(np.radians(dec)) * np.sin(np.radians(vlti_latitude)) +\
        np.cos(np.radians(dec)) * np.cos(np.radians(vlti_latitude)) *\
        np.cos(np.radians(ha))
    alt = np.arcsin(tmp1)*180/np.pi

    tmp1 = np.cos(np.radians(dec)) * np.sin(np.radians(ha))
    tmp2 = -np.cos(np.radians(vlti_latitude)) * np.sin(np.radians(dec)) + \
        np.sin(np.radians(vlti_latitude)) * np.cos(np.radians(dec)) * \
        np.cos(np.radians(ha))
    az = (360-np.arctan2(tmp1, tmp2)*180/np.pi) % 360

    b = [layout[T1][2]-layout[T2][2],
         layout[T1][3]-layout[T2][3],
         0.0]  # assumes you *DO NOT* combine ATs and UTs

    # -- projected baseline
    ch_ = np.cos(ha*np.pi/180.)
    sh_ = np.sin(ha*np.pi/180.)
    cl_ = np.cos(vlti_latitude*np.pi/180.)
    sl_ = np.sin(vlti_latitude*np.pi/180.)
    cd_ = np.cos(radec[1]*np.pi/180.)
    sd_ = np.sin(radec[1]*np.pi/180.)

    # -- (u,v) coordinates in m
    u = ch_ * b[0] - sl_*sh_ * b[1] + cl_*sh_ * b[2]
    v = sd_*sh_ * b[0] + (sl_*sd_*ch_+cl_*cd_) * b[1] -\
        (cl_*sd_*ch_ - sl_*cd_) * b[2]

    # -- static OPL to telescope
    if not DL1 is None and not DL2 is None:
        config = (T1+'DL%dIP%d' % (DL1, ip1), T2+'DL%dIP%d' % (DL2, ip2))
        # -- assume compressed, since it is the case for UT and AT-STS
        opl1, opl2 = computeOpl0(config, plot=False, compressed=True)
    else:
        opl1 = layout[T1][4] + 0.12*(ip1-1)
        opl2 = layout[T2][4] + 0.12*(ip2-1)

    # add telescope OPL (M3-M11 distance): only matters for UT-AT combination
    # email SMo 15/10/2013:
    # if T1.startswith('U'):
    #     opl1 += 65.80 # UT
    # else:
    #     opl1 += 18.19 # AT
    # if T2.startswith('U'):
    #     opl2 += 65.80 # UT
    # else:
    #     opl2 += 18.19 # AT

    # optical delay and optical path delay, in m
    d = -b[0]*cd_*sh_ -\
        b[1]*(sl_*cd_*ch_ - cl_*sd_) +\
        b[2]*(cl_*cd_*ch_ + sl_*sd_)
    d = np.array(d)

    # -- final OPD to correct for
    opd = d + opl2 - opl1

    # -- airmass:
    czt = np.cos(np.pi*(90-alt)/180.)
    airmass = 1/czt*(1-.0012*(1/czt**2-1))
    airmass *= (alt > 20.)
    airmass += (alt <= 20)*2.9

    # -- parallactic angle
    if np.cos(np.radians(dec)) != 0:
        parang = 180*np.arcsin(np.sin(np.radians(az)) *
                               np.cos(np.radians(vlti_latitude)) /
                               np.cos(np.radians(dec)))/np.pi
    else:
        parang = 0.

    # -- observability
    observable = (alt > min_altitude) * \
                 (np.abs(opd) < max_OPD) *\
                 (alt > np.interp(az % 360, horizon[T1][0], horizon[T1][1])) *\
                 (alt > np.interp(az % 360, horizon[T2][0], horizon[T2][1]))

    if not max_airmass is None:
        observable = observable*(airmass <= max_airmass)

    if isinstance(alt, np.ndarray):
        observable = np.array(observable)

    res = {'u': u, 'v': v, 'opd': opd, 'alt': alt, 'az': az,
           'observable': observable, 'parang': parang,
           'lst': lst, 'ra': radec[0], 'dec': radec[1],
           'B': np.sqrt(u**2+v**2), 'PA': np.arctan2(u, v)*180/np.pi,
           'airmass': airmass, 'baselines': T1+T2, 'opd': opd,
           'horizon': np.maximum(np.interp(az % 360, horizon[T1][0],
                                           horizon[T1][1]),
                                 np.interp(az % 360, horizon[T2][0],
                                           horizon[T2][1])),
           # -- baseline on the ground:
           'ground': np.array([layout[T2][0]-layout[T1][0],
                              layout[T2][1]-layout[T1][1],
                              (20. if 'U' in T2 else 0.)-(20. if 'U' in T1 else 0.)])}
    return res


def solveDLposition(dl, opd, dlPosMin=11/2., dlPosMax=111/2.,
                    stations=None, STS=False, constraints=None):
    """
    dl: list of delay lines numbers, like [1,3,2]
    opd: dict of OPDs for pairs of delay lines, in meters, for example,
    {(1,2):12.0, (3,1):-34.22, (3,2):0.0}. Convention is 2*(DL2-DL1)=12.0,
    2*(DL1-DL3)=-34.22, because the OPL is twice the DL position

    if a list of stations is given (same order as dl), VCM pressure will be
    computed.

    constraints: specific constraints for each DL, override the dlPosMin and
    dlPosMax which are global. e.g.: constraints={'min3':20, 'max5':40} means
    DL3 must be atleast at 20 m (position, i.e. 40m delay) and DL5 has to be
    at most at position 40 (i.e. delay 80)

    a bit smarter than the previous version: check len(dl) configuration,
    every time putting one of the DL closest to the lab.

    Author: amerand
    """
    # -- build default range constraints
    cons = {}
    for d in [1, 2, 3, 4, 5, 6]:
        cons['min'+str(d)] = dlPosMin
        cons['max'+str(d)] = dlPosMax
        if _test_dl34_plus30:
            # -- test the effect of extending DL3,4 for 30 OPL:
            if d in [3, 4]:
                cons['max'+str(d)] += 30

    if not constraints is None and isinstance(constraints, dict):
        for k in constraints.keys():
            if k in cons.keys():
                cons[k] = constraints[k]/2.
            else:
                print('WARNING: unknown constraint', k)

    # -- initialise dict used to track optimum solution:
    best = {}
    for d in dl:
        best[d] = np.nan
        best['vcm'+str(d)] = 5

    for i in range(len(dl)):  # -- for each DL
        # -- try to put the DL dl[i] at minimum position:
        dlPos = {dl[i]: cons['min'+str(dl[i])]}
        # -- explore all other DL positions
        while len(dlPos.keys()) != len(dl):
            for o in opd.keys():  # -- for each requested OPD
                # -- set DL pos according to OPD
                if o[0] in dlPos.keys() and not o[1] in dlPos.keys():
                    dlPos[o[1]] = dlPos[o[0]] - opd[o]/2.
                elif o[1] in dlPos.keys() and not o[0] in dlPos.keys():
                    dlPos[o[0]] = dlPos[o[1]] + opd[o]/2.
                # print(dlPos, dlPos.keys(), o[0], o[1])
        # -- check that DL carts are within optional constraints:
        test = all([dlPos[k] <= cons['max'+str(k)] for k in dlPos.keys()])
        test = all([dlPos[k] >= cons['min'+str(k)]
                   for k in dlPos.keys()]) and test

        # -- find best position
        if test:
            if not stations is None:
                # -- compute VCM pressure
                for k, d in enumerate(dl):
                    dlPos['vcm'+str(d)] = computeVcmPressure(stations[k],
                                                             d, 1, dlPos[d], STS=STS)
                # -- try to keep ALL VCM pressure to a minimum
                if np.max([dlPos['vcm'+str(d)] for d in dl]) <=\
                   np.max([best['vcm'+str(d)] for d in dl]):
                    best = dlPos.copy()
                # -- try to keep DL closer to the lab
                # if np.max([dlPos[d] for d in dl])<=np.max([best[d] for d in dl]):
                #    best = dlPos.copy()

            else:
                # -- set vcm pressure to nan
                for d in dl:
                    dlPos['vcm'+str(d)] = np.nan
                if np.mean([dlPos[d] for d in dl]) <= np.mean([best[d] for d in dl]):
                    best = dlPos.copy()
                # if np.max([dlPos[d] for d in dl]) <= np.max([best[d] for d in dl]):
                #    best = dlPos.copy()
        else:
            # print dlPos
            # print opd
            pass
    return best


_X, _Y = [], []


def visImage(image, scale, u, v, wl, debug=False):
    """
    image: 2D image, square pixels!
    scale: "scale", pixels size, in mas
    u, v: spatial coordinates (vectors, in m)
    wl: wavelength (vector or scalar, in um)
    """
    global _X, _Y
    if np.shape(_X) != np.shape(image) or \
            np.shape(_Y) != np.shape(image):
        Ny, Nx = np.shape(image)
        x, y = np.arange(Nx)-(Nx-1)/2, np.arange(Ny)-(Ny-1)/2
        _X, _Y = np.meshgrid(scale*x, scale*y)
        if debug:
            print('visImage: x', Nx, x.min(), x[Nx//2], x.max())
            print('visImage: y', Ny, y.min(), y[Ny//2], y.max())

    # -- fourier transform
    c = np.pi/180/3600/1000/1e-6
    if np.isscalar(wl):
        vis = image[:, :, None] *\
            np.exp(-2j*np.pi*c*(_X[:, :, None]*u[None, None, :] +
                                _Y[:, :, None]*v[None, None, :])/wl)
    else:
        vis = image[:, :, None, None] *\
            np.exp(-2j*np.pi*c*(_X[:, :, None, None]*u[None, None, :, None] +
                                _Y[:, :, None, None]*v[None, None, :, None]) /
                   wl[None, None, None, :])
    return np.sum(vis, axis=(0, 1))/np.sum(image)


def visCube(cube, u, v, wl):
    """
    cube:  {'image':, 'scale': 'WL':}
        image: Nwl (wavelength, optional) x Nx x Ny (spatial)
        scale: "scale", in mas
        wl: wavelength vector in um (ignored if cube is 2D)

    values for the computations:
        u, v: spatial coordinates (vectors, in m)
        wl: wavelength (vector, in um)
    """
    global _X, _Y
    res = np.zeros((len(u), len(wl)), np.complex64)
    _X, _Y = cube['X'], cube['Y']
    if False:
        # == Interpolate in visibility space =============================
        # -- compute V(u,v) for each wl of the cube:
        tmp = np.zeros((len(u), len(cube['WL'])), np.complex64)
        for i, x in enumerate(cube['WL']):
            tmp[:, i] = visImage(cube['image'][i, :, :], cube['scale'],
                                 u, v, cube['WL'][i])
        # -- interpolate
        for i in range(len(u)):
            res[i, :] = np.interp(wl, cube['WL'], tmp[i, :])
    else:
        # == Interpolate in image space ==================================
        for i, x in enumerate(wl):
            # -- find 2 closest images
            i1, i2 = np.argsort(np.abs(x-cube['WL']))[:2]
            # -- linear interpolation
            im = cube['image'][i1, :, :] + \
                (x-cube['WL'][i1])/(cube['WL'][i2]-cube['WL'][i1]) *\
                (cube['image'][i2, :, :]-cube['image'][i1, :, :])

            res[:, i] = visImage(im, cube['scale'], u, v, x)
    return res


def fluxCube(cube, wl):
    """
    cube:  {'image':, 'scale': 'WL':}
        image: Nx x Ny spatial pixels (x Nwl optional )
        scale: "scale", in mas (ignored)
        wl: wavelength vector in um (ignored if cube is 2D)
    wl: wavelength (vector, in um)
    """
    tmp = np.sum(cube['image'], axis=(1, 2))
    return np.interp(wl, cube['WL'], tmp)


def checkCubeShapes(cube, name='cube'):
    err = '\033[41mERROR\033[0m: '
    if type(cube) != dict:
        print(err+'"%s" should be dictionnary' % name)
        return False

    K = ['X', 'Y', 'image']
    if not all([k in cube for k in K]):
        print(err+'"%s" should at least contain "X", "Y" and "image"' % name)
        return False

    if type(cube['image']) != np.ndarray:
        print(err+'%s["image"] should be of type "np.ndarray"' % (name))
        return False

    if len(cube['image'].shape) == 2:
        L = {'X': 2, 'Y': 2, 'image': 2}
    else:
        L = {'WL': 1, 'X': 2, 'Y': 2, 'image': 3}

    if not all([type(cube[k]) == np.ndarray for k in L]):
        print(err+'%s should be of type "np.ndarray"' %
              (', '.join(['"%s"' % k for k in L])))
        for k in L:
            if type(cube[k]) != np.ndarray:
                print('  ', k, type(cube[k]), type(cube[k]) == np.ndarray)
        return False

    if not all([len(cube[k].shape) == L[k] for k in L]):
        print(err+'wrong shapes!')
        for k in L:
            if L[k] != len(cube[k].shape):
                print('  len(%s[%s].shape)=%d -> should be %d' % (name, k, L[k],
                                                                  len(cube[k].shape)))
        return False

    if cube['X'].shape != cube['Y'].shape:
        print(err+'"X" and "Y" should have same shapes!')
        print('  %s["X"].shape=%s != %s["Y"].shape=%s' % (name, str(cube['X'].shape),
                                                          name, str(cube['Y'].shape)))
        return False

    if L['image'] == 2:
        if cube['X'].shape != cube['image'].shape:
            print(err+'"image" has wrong dimensions!')
            print('  %s["X"].shape=%s != %s["image"].shape=%s' % (name, str(cube['X'].shape),
                                                                  name, str(cube['image'].shape)))
            return False
    else:
        _tup = tuple(list(cube['WL'].shape)+list(cube['X'].shape))
        if cube['image'].shape != _tup:
            print(err+'"image" has wrong dimensions!')
            print('  %s["image"].shape = %s, should be %s' % (name, str(cube['image'].shape),
                                                              str(_tup)))
            return False

    return True


__mjd0 = 57000


def makeFakeVLTI(t, target, lst, wl, mjd0=None, lst0=0,
                 diam=None, cube=None, noise=None, thres=None,
                 model=None, insname='synthetic', debug=False,
                 doubleDL=False):
    """
    for VLTI!

    t = list of stations
    targe = name of target, or (ra, dec) in (in hours, degs)
    lst = np.array of LST (in hours)
    wl = np.array of wavelength (in um)
    diam = UD diam, in mas. If set, returns simple UD model
    cube = {'image', 'scale', 'WL', 'spectrum'}
        image: Nx x Ny spatial pixels (x Nwl optional )
        scale: "scale", in mas (ignoired)
        wl: wavelength vector in um (ignored if cube is 2D only)
        spectrum: spectrum (same lemgth as wl), should be sum(image, axis=(0,1))
    model = a dictionnary describing a model (usual PMOIRED syntax)
    doubleDL = use double passage delay line (default=False)
    noise = noise to apply. If ==0, no noise; if None, typical noise (relative / degrees):
          {'V2': 0.01, '|V|':0.01, 'PHI':1., 'FLUX':0.01, 'T3PHI':1., 'T3AMP':0.01}
    """
    global __mjd0
    if noise == 0:
        noise = {k: 0 for k in ['V2', '|V|', 'PHI', 'FLUX', 'T3PHI', 'T3AMP']}
    if noise is None:
        # -- typical for GRAVITY good SNR
        noise = {'V2': 0.01, '|V|': 0.01, 'PHI': 0.5, 'FLUX': 0.01,
                 'T3PHI': 1., 'T3AMP': 0.01}
        # -- high precision
        # noise = {'V2': 0.001, '|V|':0.001, 'PHI':0.1, 'FLUX':0.001,
        #         'T3PHI':.1, 'T3AMP':0.001}
        #
    if type(noise)!=dict:
       raise Exception('noise should be a dictionnary such as '+
            str({'V2': 0.01, '|V|': 0.01, 'PHI': 0.5, 'FLUX': 0.01,'T3PHI': 1., 'T3AMP': 0.01}))

    # -- try to complete the noise dictionnary
    if 'V2' in noise and not '|V|' in noise:
        noise['|V|'] = noise['V2']
    if '|V|' in noise and not 'V2' in noise:
        noise['V2'] = noise['|V|']
    if '|V|' in noise and not 'T3AMP' in noise:
            noise['T3AMP'] = 1.5*noise['|V|']*1.5
    if 'T3PHI' in noise and not 'PHI' in noise:
        noise['PHI'] = noise['T3PHI']/1.5
    if 'PHI' in noise and not 'T3PHI' in noise:
            noise['T3PHI'] = noise['T3PHI']*1.5
    if not 'FLUX' in noise:
        noise['FLUX'] = 0.01

    if thres is None:
        # -- threshold for noise behavior (absolute)
        thres = {'V2': 0.01, '|V|': 0.02, 'T3AMP': 0.05, 'FLUX': 0.01}

    if doubleDL:
        tmp = nTelescopes(t, target, lst, max_OPD=200)
    else:
        tmp = nTelescopes(t, target, lst)
    if any(~tmp['observable']):
        print('WARNING: %2d out of %d LST are not observable!' %
              (np.sum(~tmp['observable']), len(lst)), end=' ')
        print('-'.join(t), np.array(lst)[~tmp['observable']])
        tmp['observable'] = np.array(tmp['observable'])
        for k in ['airmass', 'lst', 'alt', 'az', 'parang', 'horizon']:
            tmp[k] = np.array(tmp[k])[tmp['observable']]
        for k in ['u', 'v', 'B', 'PA', 'opd']:
            for b in tmp[k].keys():
                tmp[k][b] = tmp[k][b][tmp['observable']]
        lst = np.array(lst)[tmp['observable']]
        tmp['observable'] = tmp['observable'][tmp['observable']]
        # print(tmp)

    # -- fake MJD
    if mjd0 == None:
        # -- needed to simulate data taken at different dates
        mjd0 = __mjd0 + 1
        __mjd0 += 1
    tmp['MJD'] = (np.array(lst)-lst0)/24 + mjd0
    res = {'insname': '%s_%.3f_%.3fum_R%.0f' % (insname, min(wl), max(wl),
                                                np.mean(wl/np.gradient(wl))),
           'filename': 'synthetic',
           'targname': target if type(target) == str else
           '%.3f %.3f' % tuple(target),
           'pipeline': 'None',
           'header': {},
           'WL': wl,
           'telescopes': list(t),
           'baselines': tmp['baselines'],
           'TELLURICS': np.ones(len(wl)),
           'MJD': tmp['MJD'],
           }
    # print('res:', res)

    # == complex visibility function
    # -- default == unresolved gray object
    def fvis(u, v, l): return np.ones((len(u), len(l)))
    def fflux(l): return np.ones((len(lst), len(l)))

    if not cube is None:
        if not checkCubeShapes(cube):
            raise Exception('data cube is ill formed')
        if len(cube['image'].shape) == 2:
            def fvis(u, v, l): return visImage(
                cube['image'], cube['scale'], u, v, l)

            def fflux(l): return np.ones((len(lst), len(l)))
        else:
            def fvis(u, v, l): return visCube(cube, u, v, l)
            if 'spectrum' in cube:
                def fflux(l): return np.interp(l, cube['WL'], cube['spectrum'])
            else:
                def fflux(l): return np.interp(l, cube['WL'],
                                               np.sum(cube['image'], axis=(1, 2)))
    elif model is None and diam is None:
        # -- so we have at least somethinh to model!
        diam = 1.0

    if not diam is None:
        # -- simple uniform disk
        c = np.pi/180/3600/1000/1e-6

        def fvis(u, v, l):
            x = c*np.pi*diam*np.sqrt(u**2+v**2)[:, None]/wl[None, :]
            return 2*scipy.special.j1(x)/x

        def fflux(l): return np.ones((len(lst), len(l)))

    VIS = {}  # complex visibility
    # -- OI_VIS2
    OIV2 = {}
    # -- OI_VIS
    OIV = {}
    conf = {m: [] for m in tmp['MJD']}
    for b in tmp['baselines']:
        # -- complex visibility for this baseline
        VIS[b] = fvis(tmp['u'][b], tmp['v'][b], wl)
        nv2 = noise['V2']*np.maximum(np.abs(VIS[b])**2, thres['V2'])
        OIV2[b] = {'u': tmp['u'][b], 'v': tmp['v'][b],
                   'u/wl': tmp['u'][b][:, None]/wl[None, :],
                   'v/wl': tmp['v'][b][:, None]/wl[None, :],
                   'B/wl': tmp['B'][b][:, None]/wl[None, :],
                   # 'PA': tmp['PA'][b],
                   'PA': np.angle(tmp['v'][b][:, None]/wl[None, :] +
                                  1j*tmp['u'][b][:, None]/wl[None, :], deg=True),
                   'MJD': tmp['MJD'],
                   'MJD2': tmp['MJD'][:, None]+0*wl[None, :],
                   'FLAG': np.zeros((len(lst), len(wl)), bool),
                   'V2': np.abs(VIS[b])**2 +
                   nv2*np.random.randn(len(lst), len(wl)),
                   'EV2': nv2,
                   }
        if debug:
            print(b, list(OIV2[b].keys()))

        nv = noise['|V|']*np.maximum(np.abs(VIS[b]), thres['|V|'])
        OIV[b] = {'u': tmp['u'][b], 'v': tmp['v'][b],
                  'u/wl': tmp['u'][b][:, None]/wl[None, :],
                  'v/wl': tmp['v'][b][:, None]/wl[None, :],
                  'B/wl': tmp['B'][b][:, None]/wl[None, :],
                  # 'PA': tmp['PA'][b],
                  'PA': np.angle(tmp['v'][b][:, None]/wl[None, :] +
                                 1j*tmp['u'][b][:, None]/wl[None, :], deg=True),
                  'MJD': tmp['MJD'],
                  'MJD2': tmp['MJD'][:, None]+0*wl[None, :],
                  'FLAG': np.zeros((len(lst), len(wl)), bool),
                  '|V|': np.abs(VIS[b]) + nv*np.random.randn(len(lst), len(wl)),
                  'E|V|': nv,
                  'PHI': np.angle(VIS[b])*180/np.pi +
                  noise['PHI']*np.random.randn(len(lst), len(wl)),
                  'EPHI': noise['PHI']*np.ones((len(lst), len(wl))),
                  }
        for m in tmp['MJD']:
            conf[m].append(b)
    res['OI_VIS2'] = OIV2
    res['OI_VIS'] = OIV

    # -- OI_FLUX
    OIF = {}
    for s in t:
        OIF[s] = {'FLUX': fflux(wl)*(1 +
                                     noise['FLUX']*np.random.randn(len(lst), len(wl))),
                  'RFLUX': fflux(wl)*(1 +
                                      noise['FLUX']*np.random.randn(len(lst), len(wl))),
                  'EFLUX': noise['FLUX']*np.ones((len(lst), len(wl)))*fflux(wl)[None, :],
                  'FLAG': np.zeros((len(lst), len(wl)), bool),
                  'MJD': tmp['MJD'],
                  'MJD2': tmp['MJD'][:, None]+0*wl[None, :],
                  }
        for m in tmp['MJD']:
            conf[m].append(s)
    res['OI_FLUX'] = OIF

    # -- OI_T3
    OIT3 = {}
    res['triangles'] = []
    if len(t) >= 3:
        # print(tmp['baselines'])
        for i, tri in enumerate(itertools.combinations(t, 3)):
            res['triangles'].append(''.join(tri))
            if tri[0]+tri[1] in tmp['baselines']:
                b1 = tri[0]+tri[1]
                s1 = 1
                T3 = VIS[b1].copy()
            elif tri[1]+tri[0] in tmp['baselines']:
                b1 = tri[1]+tri[0]
                s1 = -1
                T3 = np.conj(VIS[b1])

            if tri[1]+tri[2] in tmp['baselines']:
                b2 = tri[1]+tri[2]
                s2 = 1
                T3 *= VIS[b2]
            elif tri[2]+tri[1] in tmp['baselines']:
                b2 = tri[2]+tri[1]
                s2 = -1
                T3 *= np.conj(VIS[b2])

            if tri[2]+tri[0] in tmp['baselines']:
                b3 = tri[2]+tri[0]
                s3 = 1
                T3 *= VIS[b3]
            elif tri[0]+tri[2] in tmp['baselines']:
                b3 = tri[0]+tri[2]
                s3 = -1
                T3 *= np.conj(VIS[b3])

            # -- minimum visibility
            minV = np.minimum(np.minimum(np.abs(VIS[b1]), np.abs(VIS[b2])),
                              np.abs(VIS[b3]))

            form = ('+' if s1 > 0 else '-')+b1 +\
                   ('+' if s2 > 0 else '-')+b2 +\
                   ('+' if s3 > 0 else '-')+b3
            # print('debug:', i, tri, form)

            # -- KLUDGE -> account for T3 amplitude
            ncp = 1  # np.maximum(1, (thres['T3AMP']/np.abs(T3))**(1/3))
            # ncp = np.maximum(1, thres['T3AMP']/minV)
            if debug:
                print(tri, 'NCP:', ncp.min(), ncp.max(),
                      'noise T3PHI:', noise['T3PHI'])

            OIT3[''.join(tri)] = {
                'MJD': tmp['MJD'],
                'MJD2': tmp['MJD'][:, None]+0*wl[None, :],
                'FLAG': np.zeros((len(lst), len(wl)), bool),
                'T3AMP': np.abs(T3)*(1 +
                                     noise['T3AMP']*np.random.randn(len(lst), len(wl))),
                'ET3AMP': noise['T3AMP']*np.ones((len(lst), len(wl)))*np.abs(T3),
                'T3PHI': np.angle(T3)*180/np.pi +
                ncp*noise['T3PHI']*np.random.randn(len(lst), len(wl)),
                'ET3PHI': ncp*noise['T3PHI']*np.ones((len(lst), len(wl))),
                'u1': s1*tmp['u'][b1], 'v1': s1*tmp['v'][b1],
                'u2': s2*tmp['u'][b2], 'v2': s2*tmp['v'][b2],
                'u1/wl': s1*tmp['u'][b1][:, None]/wl[None, :],
                'v1/wl': s1*tmp['v'][b1][:, None]/wl[None, :],
                'u2/wl': s2*tmp['u'][b2][:, None]/wl[None, :],
                'v2/wl': s2*tmp['v'][b2][:, None]/wl[None, :],
                'B1': tmp['B'][b1],
                'B2': tmp['B'][b2],
                'B3': tmp['B'][b3],
                'Bmin/wl': np.minimum(tmp['B'][b1], tmp['B'][b2], tmp['B'][b3])[:, None] /
                wl[None, :],
                'Bmax/wl': np.maximum(tmp['B'][b1], tmp['B'][b2], tmp['B'][b3])[:, None] /
                wl[None, :],
                'Bavg/wl': (tmp['B'][b1]+tmp['B'][b2]+tmp['B'][b3])[:, None] /
                wl[None, :]/3,
                'formula': [[s1, s2, s3], [b1, b2, b3],
                            list(range(len(tmp['MJD']))),
                            list(range(len(tmp['MJD']))),
                            list(range(len(tmp['MJD']))),]
            }
            # print(''.join(tri), sorted(OIT3[''.join(tri)].keys()))
            for m in tmp['MJD']:
                conf[m].append(''.join(tri))
    res['OI_T3'] = OIT3

    if not model is None:
        param = model
        res['fit'] = {'obs': ['V2', '|V|', 'T3PHI', 'T3AMP', 'PHI', 'FLUX']}
        res = oimodels.VmodelOI(res, param, fullOutput=True)

        # --
        for k in ['PA']:
            for b in res['OI_VIS']:
                res['OI_VIS'][b][k] = tmp[k][b][:, None]+0*res['WL'][None, :]
            for b in res['OI_VIS2']:
                res['OI_VIS2'][b][k] = tmp[k][b][:, None]+0*res['WL'][None, :]

        # -- add noise
        def addnoise(e, k, o):
            res[e][k][o] += np.random.randn(res[e][k][o].shape[0],
                                            res[e][k][o].shape[1]) *\
                res[e][k]['E'+o]

        if 'OI_FLUX' in res:
            for k in res['OI_FLUX'].keys():
                res['OI_FLUX'][k]['EFLUX'] = noise['FLUX']*np.maximum(res['OI_FLUX'][k]['FLUX'],
                                                                      thres['FLUX'])
                addnoise('OI_FLUX', k, 'FLUX')

        for k in res['OI_VIS'].keys():
            res['OI_VIS'][k]['E|V|'] = noise['|V|'] * \
                np.maximum(res['OI_VIS'][k]['|V|'], thres['|V|'])
            # res['OI_VIS'][k]['EPHI'] = noise['PHI']*(thres['|V|']/np.sqrt(thres['|V|']**2 +
            #                                np.maximum(res['OI_VIS'][k]['|V|'], thres['|V|'])**2))
            res['OI_VIS'][k]['EPHI'] = noise['PHI'] * \
                np.ones(res['OI_VIS'][k]['PHI'].shape)
            res['OI_VIS2'][k]['EV2'] = noise['V2'] * \
                np.maximum(res['OI_VIS2'][k]['V2'], thres['V2'])
            addnoise('OI_VIS', k, '|V|')
            addnoise('OI_VIS', k, 'PHI')
            addnoise('OI_VIS2', k, 'V2')

        for k in res['OI_T3'].keys():
            res['OI_T3'][k]['ET3AMP'] = noise['T3AMP'] * \
                np.maximum(res['OI_T3'][k]['T3AMP'], thres['T3AMP'])
            # res['OI_T3'][k]['ET3PHI'] = noise['T3PHI']*(thres['T3AMP']/np.sqrt(thres['T3AMP']**2+
            #                     np.maximum(res['OI_T3'][k]['T3AMP'], thres['T3AMP'])**2))
            res['OI_T3'][k]['ET3PHI'] = noise['T3PHI'] * \
                np.ones(res['OI_T3'][k]['T3PHI'].shape)

            addnoise('OI_T3', k, 'T3AMP')
            addnoise('OI_T3', k, 'T3PHI')

    res['configurations per MJD'] = conf
    return res


makeFake = makeFakeVLTI  # legacy
