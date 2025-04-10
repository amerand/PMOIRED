import numpy as np
"""
Daubechies Wavelet, D4 is the default behaviour, but other orders are available.

http://en.wikipedia.org/wiki/Daubechies_wavelet
http://www.bearcave.com/misl/misl_tech/wavelets/daubechies/index.html
http://cam.mathlab.stthomas.edu/wavelets/pdffiles/UST06/Lecture6.pdf
"""

coefH = {}
coefG = {}

# http://en.wikipedia.org/wiki/Daubechies_wavelet
# coefficients have been double checked
coefH[2]=np.array([1.0,1.0])
coefH[4]=np.array([0.6830127,1.1830127,0.3169873,-0.1830127])
coefH[6]=np.array([0.47046721,1.14111692,0.650365,-0.19093442,-0.12083221,0.0498175])
coefH[8]=np.array([0.32580343,1.01094572,0.8922014,-0.03957503,
                  -0.26450717,0.0436163,0.0465036,-0.01498699])
coefH[10]=np.array([0.22641898,0.85394354,1.02432694,0.19576696,-0.34265671,
                    -0.04560113,0.10970265,-0.00882680,-0.01779187,4.71742793e-3])
coefH[12]=np.array([0.15774243,0.69950381,1.06226376,0.44583132,
                    -0.31998660,-0.18351806,0.13788809,0.03892321,
                    -0.04466375,7.83251152e-4,6.75606236e-3,-1.52353381e-3])
coefH[14]=np.array([0.11009943,0.56079128,1.03114849,0.66437248,
                    -0.20351382,-0.31683501,0.1008467,0.11400345,
                    -0.05378245,-0.02343994,0.01774979,
                    6.07514995e-4,-2.54790472e-3,5.00226853e-4])
coefH[16]=np.array([0.07695562,0.44246725,0.95548615,0.82781653,
                    -0.02238574,-0.40165863,6.68194092e-4,0.18207636,
                    -0.02456390,-0.06235021,0.01977216,0.01236884,
                    -6.88771926e-3,-5.54004549e-4,9.55229711e-4,-1.66137261e-4])
coefH[18]=np.array([0.05385035,0.34483430,0.85534906,0.92954571,
                    0.18836955,-0.41475176,-0.13695355,0.21006834,
                    0.043452675,-0.09564726,3.54892813e-4,0.03162417,
                    -6.67962023e-3,-6.05496058e-3,2.61296728e-3,
                    3.25814671e-4,-3.56329759e-4,5.5645514e-5])
coefH[20]=np.array([0.03771716,0.26612218,0.74557507,0.97362811,
                    0.39763774,-0.35333620,-0.27710988,0.18012745,
                    0.13160299,-0.10096657,-0.04165925,0.04696981,
                    5.10043697e-3,-0.01517900,1.97332536e-3,2.81768659e-3,
                    -9.69947840e-4,-1.64709006e-4,1.32354367e-4,-1.875841e-5])

for k in coefH.keys():
    # -- direct transform
    coefH[k] /= np.sqrt(2)
    # -- inverse transform
    coefG[k] = (-1.)**np.arange(k)*coefH[k][::-1]

# import scipy.signal
# for i in range(34):
#     k = 2*i+2
#     coefH[k] = scipy.signal.daub(i+1)
#     coefG[k] = (-1.)**np.arange(k)*coefH[k][::-1]

fMats = {} # stores forward transform matrices
iMats = {} # stores inverse transform matrices

def test():
    import scipy.misc
    from matplotlib import pyplot as plt

    plt.close(1)
    plt.figure(1)
    plt.clf()
    # -- 1D signal
    y = np.sin(np.arange(128)/4.)+0.1*np.random.randn(128)
    plt.subplot(121)
    plt.title('signal')
    plt.plot(y)
    plt.subplot(122)
    plt.title('transform')
    for o in range(4):
        plt.plot(oneD(y, 5, order=2*(o+1)), label='O=%d'%(2*(o+1)))
    plt.legend()

    plt.close(2)
    plt.figure(2, figsize=(12,6))
    plt.clf()

    # -- test image
    image = scipy.misc.ascent()
    #image = scipy.misc.face()

    plt.subplot(121, frameon=False)
    plt.imshow(image, cmap='bone')
    plt.subplot(122, frameon=False)
    plt.imshow(np.abs(twoD(image, 1, order=2))**0.5, cmap='gist_stern')
    return

def forwardMat(n, p=4, debug=False):
    """
    compute matrix
    """
    assert (p in coefH.keys()), "coef for order "+str(p)+" are not implemented"
    if (n,p) in fMats.keys() and not debug:
        return fMats[(n,p)]
    mat = np.zeros((n,n))
    deb = [['--' for k in range(n)] for l in range(n)]
    deb = np.array(deb)
    for k in range(n)[::2]:
        for i in range(p):
            mat[k,(k+i)%n]   += coefH[p][i]
            mat[k+1,(k+i)%n] += coefG[p][i]
            deb[k,(k+i)%n]   = 'H'+str(i)
            deb[k+1,(k+i)%n] = 'G'+str(i)
    mat = np.roll(mat, 0, axis=0)
    deb = np.roll(deb, 0, axis=0)

    fMats[(n,p)] = mat
    iMats[(n,p)] = np.linalg.inv(mat)
    if debug:
        return deb
    return mat

def inverseMat(n,p=4, debug=False):
    if not (n,p) in iMats.keys():
        forwardMat(n,p, debug)
    return iMats[(n,p)]

#@numba.jit
def oneD(x,n,order=4):
    """
    1D D4 transform of x, ran n times recursively. If n<0, then the inverse
    transform is applied.

    len(x) should be a power of 2
    """
    res = np.array(x).copy()
    p = len(x)//2
    if n>0: # -- forward transform
        tmp = np.dot(forwardMat(len(x),order), x)
        res[p:] = tmp[1::2] # -- wavelet
        if n>1: # -- recursion
            res[:p] = oneD(tmp[0::2], n-1, order=order)/np.sqrt(2)
        else:
            res[:p] = tmp[0::2] # -- scaling
    elif n<0: # -- inverse transform
        tmp = np.array(x).copy()
        if n<-1: # -- recursion
            tmp[:p] = oneD(res[:p], n+1, order=order)
        res[0::2] = tmp[:p]*np.sqrt(2) # -- scaling
        res[1::2] = tmp[p:] # -- wavelet
        res = np.dot(inverseMat(len(x), order), res)
    return res

def build1DFilt(N,filt):
    """
    N is the size of the signal, filt is the filter. First value is the
    transmission for the lowest freq, the last value is for the highest
    frequency.

    for example:
    filt=[1,0.5,0] is a low pass filter
    filt=[0,0.2,0.4,0.8,1] is a high pass filter

    """
    res = np.zeros(N)
    p = N//2
    res[p:]=filt[-1]
    if len(filt)>2:
        res[:p]=build1DFilt(p,filt[:-1])
    else:
        res[:p]=filt[0]
    return res

#@numba.jit
def filter1D(x,filt,order=4):
    """
    x = 1D signal, filt is a transmission vector.

    x can have any dims (will be padded if needed)

    filt can be, for example:
    [1,0.5,0] for a low pass filter
    [0,0.2,0.4,0.8,1] for a high pass filter
    """
    N = np.log(len(x))/np.log(2)
    if N>int(N):
        N=int(N+1)
    else:
        N=int(N)
    tmp = np.zeros(2**N)
    P = 2**N-len(x)
    # -- zero padding
    #tmp[:len(x)]=x
    # -- mirror padding
    if P>0:
        tmp[P//2:P//2+len(x)]=x
        tmp[:P//2+1] = x[P//2::-1]
        tmp[-P//2:] = x[-2:-P//2-2:-1]
    else:
        tmp = x

    res = oneD(oneD(tmp,len(filt)+1,order)*
                build1DFilt(2**N,filt),
                -(len(filt)+1),order)
    # -- zero padding:
    if P>0:
        return res[P//2:P//2+len(x)]
    else:
        return res

def twoD(x,n,order=4):
    """
    2D D4 transform of x, ran n times recursively. If n<0, then the inverse
    transform is applied.

    only for x a square array, of dims being a power of 2
    """
    res = np.array(x).copy()
    p = x.shape[0]//2
    if n>0:
        for k in range(x.shape[0]):
            res[:,k] = oneD(res[:,k],1,order)
        for k in range(x.shape[0]):
            res[k,:] = oneD(res[k,:],1,order)
        if n>1:
            res[:p,:p] = twoD(res[:p,:p],n-1,order)
    elif n<0:
        if n<-1:
            res[:p,:p] = twoD(res[:p,:p],n+1,order)
        for k in range(x.shape[0]):
            res[k,:]=oneD(res[k,:],-1,order)
        for k in range(x.shape[0]):
            res[:,k]=oneD(res[:,k],-1,order)
    return res

def build2DFilt(N,filt):
    """
    NxN is the size of the image, filt is the filter. First value is the
    transmission for the lowest freq, the last value is for the highest
    frequency.

    for example:
    filt=[1,0.5,0] is a low pass filter
    filt=[0,0.2,0.4,0.8,1] is a high pass filter

    """
    res = np.zeros((N,N))
    p = N//2
    res[p:,p:]=filt[-1]
    res[:p,p:]=filt[-1]
    res[p:,:p]=filt[-1]
    if len(filt)>2:
        res[:p,:p]=build2DFilt(p,filt[:-1])
    else:
        res[:p,:p]=filt[0]
    return res

def filter2D(x,filt,order=4):
    """
    x = 2D image, filt is a transmission vector.

    x can have any dims (will be padded if needed)

    filt can be, for example:
    [1,0.5,0] for a low pass filter
    [0,0.2,0.4,0.8,1] for a high pass filter
    """
    N = int(np.log(max(x.shape))/np.log(2))+1
    tmp = np.zeros((2**N,2**N))
    tmp[:x.shape[0],:x.shape[1]]=x
    return twoD(twoD(tmp,len(filt)+1,order)*
                build2DFilt(tmp.shape[0],filt),
                -(len(filt)+1),order)[:x.shape[0],:x.shape[1]]

def struct2D(x,order=4):
    """
    structured 2D transform. x must have (2**N, 2**N) shape.

    returns a dictionnary with separate frequencies and indexes for
    reconstruction. fx_X where x s a number from 1...N, 1 being the lowest
    frequency, X=D, H or V for diagonal, horizontal or vertical frequencies.f0
    are the lowest frequencies (residuals).

    keys in the dictionnary:
    N = log2(width)
    T = unstructured transform
    frequencies for i=0 to i=N-1:
        f0: residuals (2x2)
        f1_D, f1_H, f1_V: diagonal, horizontal and vertical frequencies (2x2 each)
        ...
        fi_D, fi_H, fi_V: diagonal, horizontal and vertical frequencies (2**ix2**i each)
    """
    N = int(np.log(min(x.shape))/np.log(2))
    t = twoD(x, N, order)
    res = {'N':N, 'T':t}
    n = 0
    for k in range(N):
        if k==0: # -- residuals
            #res['f'+str(k)] = t[n:n+2**k, n:n+2**k] # -- amplitudes
            #res['i'+str(k)] = (n,n+2**k, n,n+2**k) # -- indices
            res['f'+str(k)] = t[0:2, 0:2] # -- amplitudes
            res['i'+str(k)] = (0,2, 0,2) # -- indices
            n += 2
        else: # -- frequencies: Diagonal, Horizontal and Vertical
            res['f'+str(k)+'_D'] = t[n:n+2**k, n:n+2**k]
            res['i'+str(k)+'_D'] =  (n,n+2**k, n,n+2**k)
            res['f'+str(k)+'_H'] = t[n:n+2**k, 0:2**k]
            res['i'+str(k)+'_H'] =  (n,n+2**k, 0,2**k)
            res['f'+str(k)+'_V'] = t[0:2**k, n:n+2**k]
            res['i'+str(k)+'_V'] =  (0,2**k, n,n+2**k)
            n += 2**k
    return res

def _trimStruct2D(s):
    """
    remove highest frequencies from structured 2D trasnform
    """
    res = {}; res.update(s)
    N = s['N']
    for x in ['D', 'H', 'V']:
        res.pop('f%i_%s'%(N-1, x))
        res.pop('i%i_%s'%(N-1, x))
    res['N']-=1
    return res

def _expandStruct2D(s):
    """
    remove highest frequencies from structured 2D trasnform
    """
    res = {}; res.update(s)
    N = s['N']
    res['f%i_D'%(N-1)] = np.zeros((2**N, 2**N))
    res['i%i_D'%(N-1)] = (2**(N-1), 2**N, 2**(N-1), 2**N)
    res['f%i_V'%(N-1)] = np.zeros((2**N, 2**N))
    res['i%i_V'%(N-1)] = (0, 2**(N-1), 2**(N-1), 2**N)
    res['f%i_H'%(N-1)] = np.zeros((2**N, 2**N))
    res['i%i_H'%(N-1)] = (2**(N-1), 2**N, 0, 2**(N-1))
    res['N']+=1
    return res

def structInv2D(x, order=4, retT=False):
    """
    structured inverse 2D transform.

    x is a dictionnary with separate frequencies and indexes for
    reconstruction (see 'struct2D'). """
    #N = max([int(k[1:].split('_')[0]) for k in x.keys()])+1
    N = x['N']
    res = np.zeros((2**N, 2**N))
    # -- reconstruct 2D transform from dict:
    for i in filter(lambda k: k.startswith('i'), x.keys()):
        #print(i, x[i])
        res[x[i][0]:x[i][1], x[i][2]:x[i][3]] = x['f'+i[1:]]
    if retT:
        return res
    else:
        #print('N=', N, 'o=', order)
        return twoD(res, -N, order=order)

def sparse2Dfrac(x, order=4, frac=10, retT=False):
    """
    x is a 2D image of shape (2**N, 2**N) shape.
    keep fraction "frac" in percent of each frequencies
    """
    S = struct2D(x, order=order)
    actual_frac = 0.
    for i in filter(lambda k: k.startswith('f'), S.keys()):
        o = float(i.split('_')[0][1:])
        if o>0:
            f = 1-(frac/100.)**(o*1./S['N'])
            #print(f)
            mask = np.abs(S[i])>=np.percentile(np.abs(S[i]), 100*f)
            S[i] *= mask
    return structInv2D(S, order=order, retT=retT)

def sparse2Dsigma(x, order=4, nsigma=10, retT=False):
    """
    x is a 2D image of shape (2**N, 2**N) shape.
    keep fraction "frac" in percent of each frequencies
    """
    S = struct2D(x, order=order)
    for i in filter(lambda k: k.startswith('f'), S.keys()):
        if float(i.split('_')[0][1:])>0:
            sigma = 0.5*(np.percentile(S[i], 84)-
                         np.percentile(S[i], 16))
            mask = np.abs(S[i])>=nsigma*sigma
            S[i] *= mask
    return structInv2D(S, order=order, retT=retT)

def psd2D(x):
    """
    x = 2D (image)
    returns: ?
    """
    N = int(np.log(max(x.shape))/np.log(2))+1
    tmp = np.zeros((2**N,2**N))
    # -- zero padding:
    tmp[:x.shape[0],:x.shape[1]]=x

    dwt = twoD(tmp,N)
    res = []
    for k in range(N+1):
        filt = np.zeros(N+1)
        filt[k]=1.0
        filt2D = build2DFilt(2**N,filt)
        res.append((dwt**2*filt2D).sum()/filt2D.sum())
    return res
