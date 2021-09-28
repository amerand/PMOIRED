import numpy as np

def _testT3formula(D, tri, key='OI_VIS'):
    tmp = {}
    s, t, w0, w1, w2 = D['OI_T3'][tri]['formula']
    keys = [k for k in ['OI_VIS', 'OI_VIS2'] if k in D]
    for key in keys:
        tmp[key+' mjd0'] = D['OI_T3'][tri]['MJD']-D[key][t[0]]['MJD'][w0]
        tmp[key+' u0'] = D['OI_T3'][tri]['u1']-s[0]*D[key][t[0]]['u'][w0]
        tmp[key+' v0'] = D['OI_T3'][tri]['v1']-s[0]*D[key][t[0]]['v'][w0]

        tmp[key+' mjd1'] = D['OI_T3'][tri]['MJD']-D[key][t[1]]['MJD'][w1]
        tmp[key+' u1'] = D['OI_T3'][tri]['u2']-s[1]*D[key][t[1]]['u'][w1]
        tmp[key+' v1'] = D['OI_T3'][tri]['v2']-s[1]*D[key][t[1]]['v'][w1]

        tmp[key+' mjd2'] = D['OI_T3'][tri]['MJD']-D[key][t[2]]['MJD'][w2]
        tmp[key+' u2'] = -D['OI_T3'][tri]['u1']-D['OI_T3'][tri]['u2']-s[2]*D[key][t[2]]['u'][w2]
        tmp[key+' v2'] = -D['OI_T3'][tri]['v1']-D['OI_T3'][tri]['v2']-s[2]*D[key][t[2]]['v'][w2]
    return tmp

def checkT3formula(oi, tol=1e-9):
    """
    oi is a PMOIRED object. Test if the OI_T3 formula are correct. That is, it
    checks if each OI_T3 is proprly associated

    """
    test = {} # keyed by filename
    for D in oi.data:
        if 'OI_T3' in D:
            test[D['filename']] = {}
            for tri in D['OI_T3']:
                test[D['filename']][tri] = _testT3formula(D, tri)
    try:
        for i,D in enumerate(oi._merged):
            if 'OI_T3' in D:
                test['MERGED'+str(i)] = {}
                for tri in D['OI_T3']:
                    test['MERGED'+str(i)][tri] = _testT3formula(D, tri)
    except:
        print('no merged data')
        pass

    err, msg = False, []
    for F in test: # for each filename
        for tri in test[F]: # for each triangle
            for k in test[F][tri]:
                if any(np.abs(test[F][tri][k])>tol):
                    msg += [(F, tri, k, np.abs(test[F][tri][k]))]
                    err = True
    return not err, msg

def checkShapes(oi):
    test = {}
    for i,D in enumerate(oi.data):
        test[D['filename']] = {}
        nWL = len(D['WL'])
        for e in ['OI_VIS', 'OI_VIS2', 'OI_T3']:
            for k in D[e]:
                nMJD = len(D[e][k]['MJD'])
                for x in D[e][k]:
                    if type(D[e][k][x]) == np.ndarray:
                        s = D[e][k][x].shape
                        #print(s, nWL, nMJD)
                        if len(s)==1 and s[0]==nMJD:
                            pass
                        elif len(s)==2 and s[0]==nMJD and s[1]==nWL:
                            pass
                        else:
                            test[D['filename']][e+k+x] = (s, nWL, nMJD)
    return test
