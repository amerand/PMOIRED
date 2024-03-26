# check transfer function
import pmoired.oifits as oifits

from matplotlib import pyplot as plt
import numpy as np


def showTF(files, insname=None, mode=None, wlmin=None, wlmax=None):
    """
    mode as function of insname:

    GRAVITY_SC: ('ESO INS SPEC RES', 'ESO INS POLA MODE', 'ESO DET2 SEQ1 DIT')
        e.g. ('MEDIUM', 'COMBINED', 3.0)
    GRAVITY_FT: ('ESO FT POLA MODE')
        e.g. ('COMBINED')

    """
    print('loading...', end=' ')
    files = oifits._globlist(files)
    data = oifits.loadOI(files, insname=insname, withHeader=True, verbose=False)
    print('done')
    if insname is None:
        insname = set([d['insname'] for d in data])
        if len(insname)>1:
            print('more that on instrument!', insname)
            return
        else:
            insname = insname[0]

    if insname=='GRAVITY_SC':
        keys = {'target':'ESO OBS TARG NAME',
                'cal':'ESO PRO CATG', # contains "CAL" if a calibrator, otherwise not
                'mode':('ESO INS SPEC RES', 'ESO INS POLA MODE', 'ESO DET2 SEQ1 DIT'),
           }
        nT = 4
        obs = [('OI_VIS', '|V|'), ('OI_T3', 'T3PHI')]

    elif insname=='GRAVITY_FT':
        keys = {'target':'ESO OBS TARG NAME',
                'cal':'ESO PRO CATG', # contains "CAL" if a calibrator, otherwise not
                #'mode':('ESO FT POLA MODE', 'ESO FT MODE'),
                'mode':('ESO FT POLA MODE'),
                }
        obs = [('OI_VIS', '|V|'), ('OI_T3', 'T3PHI')]
        nT = 4
    else:
        print('I need rules for "insname"=', insname)

    def _mode(h):
        _m = []
        for m in keys['mode']:
            if m in h:
                tmp = h[m]
            else:
                tmp = 'no found'
            if type(tmp)==str:
                tmp = tmp.strip()
            _m.append(tmp)
        return tuple(_m)

    # == plots
    axV, pV = {}, 1
    axT, pT = {}, 3

    modes = [_mode(d['header']) for d in data]

    if len(set(modes))==1:
        dataM = data
    else:
        if mode is None:
            print('more than one mode!', set(modes))
            return
        # == select data according to mode ==
        dataM = [d for d in data if _mode(d['header'])==mode]

    MJD0 = int(min([min(d['MJD']) for d in dataM]))

    #plt.close(0)
    #figv = plt.figure(0, figsize=(10,6))
    #plt.close(1)
    #figt = plt.figure(1, figsize=(10,6))

    plt.close(0)
    plt.figure(0, figsize=(10,6))

    #wlmin, wlmax = 2.15, 2.2
    if wlmin is None:
        wlmin = max([min(d['WL']) for d in dataM])
    if wlmax is None:
        wlmax = min([max(d['WL']) for d in dataM])
    ic, dc = 0, {}
    color = ['r', 'g', 'b', 'orange', 'm', 'c', 'y']
    nc = 10 # number of colors to sample in the color map
    color = [plt.cm.tab10((i+0.5)/nc) for i in range(nc)]

    last_targ = ''
    for d in dataM:
        targ = d['header'][keys['target']]
        catg = d['header'][keys['cal']]
        if not targ in dc:
            dc[targ] = color[ic%len(color)]
            ic+=1
        # -- below specific to GRAVITY, needs to be generalised!
        # --  V --
        for k in d['OI_VIS'].keys():
            if not k in axV:
                if nT==4:
                    #axV[k] = figv.add_subplot(3,2,pV)
                    axV[k] = plt.subplot(3,4,pV)
                    if pV>=9:
                        axV[k].set_xlabel('MJD-%d'%MJD0)
                axV[k].set_title('|V| '+k, fontsize=8, x=0.2, y=0.8)
                pV+=1
                if nT==4 and pV in [3,7]:
                    pV+=2
            for i,V in enumerate(d['OI_VIS'][k]['|V|']): # for each baseline
                w = (d['WL']>=wlmin)*(d['WL']<=wlmax)
                if np.nanmean(V[w])==0:
                    m = 'x'
                    c = '0.5'
                else:
                    m = '^' if 'CAL' in catg else '*'
                    c = dc[targ]
                axV[k].plot(d['OI_VIS'][k]['MJD'][i]-MJD0, np.nanmean(V[w]), m, color=c)
                if targ!=last_targ:
                    axV[k].text(d['OI_VIS'][k]['MJD'][i]-MJD0, np.nanmean(V[w]),
                                targ, rotation=90, color=dc[targ], fontsize=6)
        # -- T3 --
        for k in d['OI_T3'].keys():
            if not k in axT:
                if nT==4:
                    #axT[k] = figt.add_subplot(2,2,pT)
                    axT[k] = plt.subplot(2,4,pT)
                    if pT>=7:
                        axT[k].set_xlabel('MJD-%d'%MJD0)

                axT[k].set_title('T3PHI '+k, fontsize=8, x=0.3, y=0.9)
                pT+=1
                if nT==4 and pT in [5]:
                    pT+=2

            for i,T3 in enumerate(d['OI_T3'][k]['T3PHI']): # for each triangle
                w = (d['WL']>=wlmin)*(d['WL']<=wlmax)
                if np.nanmean(T3[w])==0:
                    m = 'x'
                    c='0.5'
                else:
                    m = '^' if 'CAL' in catg else '*'
                    c = dc[targ]
                axT[k].plot(d['OI_T3'][k]['MJD'][i]-MJD0, np.nanmean(T3[w]), m, color=c)
                if targ!=last_targ:
                    axT[k].text(d['OI_T3'][k]['MJD'][i]-MJD0, np.nanmean(T3[w]),
                                targ, rotation=90, color=dc[targ], fontsize=6)
        if targ!=last_targ:
            last_targ = targ

    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.3, top=.99)
    return
