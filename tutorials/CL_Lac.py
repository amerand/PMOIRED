
load = """directory = './CL_Lac/'
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('viscal.fits')]
display(files)
oi = pmoired.OI(files)
oi.show(logV=True) # possible with 'logV=True' to see low visibilities"""

udfit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
oi.doFit({'star,ud':3}) # try several first values
oi.show(logV=True)"""

ldc2000fit = """oi.setupFit({'obs':['V2'], 'max relative error':{'V2':0.5}})
# -- Teff=3500, logg=1.0, fixed parameters
oi.doFit({'star,diam':2.5, 
          'star,profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
          'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389}, 
          doNotFit=['A1', 'A2', 'A3', 'A4'])
oi.show(logV=True, imFov=5)

# -- Teff=3500, logg=1.0, fit the 4 limb-darkening parameters (it does not converge)
oi.doFit({'star,diam':2.5, 
          'star,profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
          'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389})
oi.show(logV=True, imFov=5)
oi.showFit()"""

ldc2000fitprior = """oi.setupFit({'obs':['V2'], 'max relative error':{'V2':0.5}})
m = {'star,diam':2.5, 
    'star,profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
    'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389}
oi.doFit(m, prior=[('np.abs(A1)', '<', 2), ('np.abs(A2)', '<', 2), ('np.abs(A3)', '<', 2), ('np.abs(A4)', '<', 2)])
oi.show(logV=True, imFov=5)
oi.showFit()"""


ldalphafit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
oi.doFit({'star,diam':2.5, 'star,profile':'$MU**$alpha', 'alpha':0.5})
oi.show(logV=True, imFov=5)"""

ldalphaoblatefit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
oi.doFit({'star,diam':2.5, 'star,profile':'$MU**$alpha', 'alpha':0.5, 'star,projang':45, 'star,incl':20})
oi.show(logV=True, imFov=5)"""
    
ldspotfit = """# -- we also fit the closure phase
oi.setupFit({'obs':['T3PHI', 'V2'],'max relative error':{'V2':0.5}, 
                'max error':{'T3PHI':30}, 
                })
# -- taking the best model previously fitted
m = {'alpha':       0.533, # +/- 0.017
    'star,diam':   2.6611, # +/- 0.0075
    'star,incl':   20.93, # +/- 0.62
    'star,projang':85.31, # +/- 1.66
    'star,profile':'$MU**$alpha',
    }
# -- adding a spot
m.update({'spot,diam':0.8, 'spot,x':0.1, 'spot,y':0.1, 'spot,f':0.1})

prior = [('spot,diam', '<', 'star,diam/4'), 
          ('spot,diam', '>', 'star,diam/8'),
          ('spot,x**2+spot,y**2', '<', 'max(star,diam/2 - spot,diam/2, 0)**2')]

oi.doFit(m, prior=prior)
# -- using imMax to be able to see the stellar surface
oi.show(imFov=5, logV=True, imMax='99')
oi.showFit()"""

ldspotgrid="""# -- we also fit the closure phase
oi.setupFit({'obs':['T3PHI', 'V2'],
                'max relative error':{'V2':0.5}, # ignore large uncertainties
                'max error':{'T3PHI':30}, # ignore large uncertainties
                })
# -- taking the best model previously fitted
m = {'alpha':       0.533, # +/- 0.017
    'star,diam':   2.6611, # +/- 0.0075
    'star,incl':   20.93, # +/- 0.62
    'star,projang':85.31, # +/- 1.66
    'star,profile':'$MU**$alpha',
    }
# -- adding a spot
m.update({'spot,diam':0.8, 'spot,x':0.0, 'spot,y':0.0, 'spot,f':0.1})

# -- prior on the size of the spot
prior = [('spot,diam', '<', 'star,diam/4'), 
          ('spot,diam', '>', 'star,diam/8'),
          ('spot,x**2+spot,y**2', '<', 'max(star,diam/2 - spot,diam/2, 0)**2')]

# -- we define our exploration pattern (random uniform)
expl = {'rand':{'spot,x':(-m['star,diam']/2, m['star,diam']/2), 'spot,y':(-m['star,diam']/2, m['star,diam']/2)}}

# -- we constrain the exploration pattern so the spot is on the star
constrain = [('spot,x**2+spot,y**2', '<', '(star,diam/2)**2')]

# -- grid fit: 64 randomisation
oi.gridFit(expl, 64, model=m, prior=prior, constrain=constrain)

# -- show result of the grid:
oi.showGrid()

# -- show best fit model
oi.show(imFov=5, imMax='99', logV=1)"""

bootstrap = """oi.bootstrapFit(100)
oi.showBootstrap()"""