# /Library/Frameworks/Python.framework/Versions/3.8/bin/python3.8 -m cProfile -o out.txt speedtest.py
# > import pstats
# > p = pstats.Stats('out.txt')
# > p.sort_stats('time').print_stats(20)

import oifits, oimodels, os

directory = '/Users/amerand/DATA/Science/FSCMa/'
files = os.listdir(directory)
files = list(filter(lambda x: x.startswith('PIO'), files))

data = [oifits.loadOI(os.path.join(directory, f)) for f in files]

for d in data:
    # -- observable to fit
    d['fit'] = {'obs':['V2','T3PHI'], 
               'min error':{'V2':0.03, 'T3PHI':1.},
               'max error':{'V2':0.06, 'T3PHI':10.0},
               'mult error':{'V2':3., 'T3PHI':1.},
               }

param = {'disk,amp1':     -0.978, # +/- 0.075
         'disk,amp2':     -0.14, # +/- 0.14
         'disk,amp3':     0.074, # +/- 0.026
         'disk,diam':     38.03, # +/- 0.67
         'disk,f0':       0.804, # +/- 0.065
         'disk,f1':       1.31, # +/- 0.17
         'disk,i':        61.83, # +/- 0.95
         'disk,pa':       73.09, # +/- 0.81
         'disk,pa1':      84.19, # +/- 0.89
         'disk,pa2':      15.2, # +/- 17.6
         'disk,pa3':      -20.05, # +/- 6.30
         'disk,profile':  'doughnut' ,
         'disk,thick':    0.838, # +/- 0.068
         'inner,amp1':    -0.67, # +/- 0.24
         'inner,amp2':    -0.65, # +/- 0.77
         'inner,diam':    22.76, # +/- 1.77
         'inner,f0':      0.142, # +/- 0.063
         'inner,f1':      'inner,f0 * disk,f1 / disk,f0',
         'inner,i':       'disk,i' ,
         'inner,pa':      'disk,pa' ,
         'inner,pa1':     85.03, # +/- 6.07
         'inner,pa2':     20.1, # +/- 19.0
         'inner,profile': 'doughnut' ,
         'inner,thick':   0.97, # +/- 0.24
         'star,f0':       1.0 ,
         'star,f1':       -1 ,
         'star,ud':       0.20, # +/- 0.45
        }

res = oimodels.residualsOI(data, param, timeit=False)
