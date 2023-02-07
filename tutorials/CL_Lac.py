
load = """directory = './CL_Lac/'
files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('viscal.fits')]
display(files)
oi = pmoired.OI(files)
oi.show(logV=True) # possible with 'logV=True' to see low visibilities"""

udfit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
oi.doFit({'ud':1}) # try several first values
oi.show(logV=True)"""

ldc2000fit = """oi.setupFit({'obs':['V2'], 'max relative error':{'V2':0.5}})
# -- Teff=3500, logg=1.0, fixed parameters
oi.doFit({'diam':2.5, 
          'profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
          'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389}, 
          doNotFit=['A1', 'A2', 'A3', 'A4'])
oi.show(logV=True, imFov=3, showUV=False)

# -- Teff=3500, logg=1.0, fit the 4 limb-darkening parameters (it does not converge)
oi.doFit({'diam':2.5, 
          'profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
          'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389})
oi.show(logV=True, imFov=3, showUV=False)
oi.showFit()"""

ldc2000fitprior = """oi.setupFit({'obs':['V2'], 'max relative error':{'V2':0.5}})
m = {'diam':2.5, 
    'profile':'1 - $A1*(1-$MU**1/2) - $A2*(1-$MU**2/2) - $A3*(1-$MU**3/2) - $A4*(1-$MU**4/2)', 
    'A1':1.5899, 'A2':-1.6835, 'A3':1.0073, 'A4':-0.2389}
oi.doFit(m, prior=[('np.abs(A1)', '<', 2), ('np.abs(A2)', '<', 2), ('np.abs(A3)', '<', 2), ('np.abs(A4)', '<', 2)])
oi.show(logV=True, imFov=3, showUV=False)
oi.showFit()"""


ldalphafit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
prior=[('alpha', '>', 0)]
oi.doFit({'diam':2.5, 'profile':'$MU**$alpha', 'alpha':0.5}, prior=prior)
oi.show(logV=True, imFov=3, showUV=False)"""

ldalphaoblatefit = """oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
prior=[('alpha', '>', 0)]
oi.doFit({'star,diam':2.5, 'star,profile':'$MU**$alpha', 'alpha':0.5, 'star,projang':45, 'star,incl':20, 'res,f':0.05},
           prior=prior)
oi.show(logV=True, imFov=3, showUV=False)"""
    
ldspotfit = """# -- we also fit the closure phase
oi.setupFit({'obs':['T3PHI', 'V2'],'max relative error':{'V2':0.5}, 
                'max error':{'T3PHI':30}, 
                })
# -- taking the best model previously fitted
m = {'alpha':       0.521, # +/- 0.017
    'res,f':       0.0202, # +/- 0.0020
    'star,diam':   2.6349, # +/- 0.0079
    'star,incl':   20.94, # +/- 0.62
    'star,projang':89.64, # +/- 1.62
    'star,profile':'$MU**$alpha',
    }
# -- adding a spot
m.update({'spot,diam':0.8, 'spot,x':0.1, 'spot,y':0.1, 'spot,f':0.1})

prior = [('alpha', '>', 0),
         ('spot,diam', '<', 'star,diam/4'), 
         ('spot,diam', '>', 'star,diam/8'),
         ('spot,x**2+spot,y**2', '<', 'max(star,diam/2 - spot,diam/2, 0)**2')]

oi.doFit(m, prior=prior)
# -- using imMax to be able to see the stellar surface
oi.show(imFov=3, logV=True, imMax='99', showUV=False)
oi.showFit()"""

ldspotgrid="""# -- we also fit the closure phase
oi.setupFit({'obs':['T3PHI', 'V2'],
                'max relative error':{'V2':0.5}, # ignore large uncertainties
                'max error':{'T3PHI':30}, # ignore large uncertainties
                })
# -- taking the best model previously fitted
m = {'alpha':       0.521, # +/- 0.017
    'res,f':       0.0202, # +/- 0.0020
    'star,diam':   2.6349, # +/- 0.0079
    'star,incl':   20.94, # +/- 0.62
    'star,projang':89.64, # +/- 1.62
    'star,profile':'$MU**$alpha',
    }
# -- adding a spot
m.update({'spot,diam':0.8, 'spot,x':0.0, 'spot,y':0.0, 'spot,f':0.1})

# -- prior on the size of the spot
prior = [('alpha', '>', 0),
         ('spot,diam', '<', 'star,diam/4'), 
         ('spot,diam', '>', 'star,diam/8'),
         ('spot,x**2+spot,y**2', '<', 'max(star,diam/2 - spot,diam/2, 0)**2')]

# -- we define our exploration pattern (grid with step a fraction of angular resolution)
expl = {'grid':{'spot,x':(-m['star,diam']/2, m['star,diam']/2, 0.3), 
                'spot,y':(-m['star,diam']/2, m['star,diam']/2, 0.3)}}

# -- we constrain the exploration pattern so the spot is on the star
constrain = [('spot,x**2+spot,y**2', '<', '(star,diam/2)**2')]

# -- grid fit
oi.gridFit(expl, model=m, prior=prior, constrain=constrain)

# -- show result of the grid:
oi.showGrid()

# -- show best fit model
oi.show(imFov=3, imMax='99', logV=1, showUV=False)"""

bootstrap = """oi.bootstrapFit(100)
oi.showBootstrap()"""

realisticprofile ="""# -- profile to look like Fig 4 in paper
p = {'star,diam':'(2+10/$k)*$Rstar', 'Rstar':1.5, 'res,f':0.02,
     'star,profile': '1-$u*(1-np.sqrt(1-(($R<$Rstar)*$R/$Rstar)**2)) + ($R>=$Rstar)*((1-$u)*np.exp(-$k*($R-$Rstar)/$Rstar)-1)',
     'u':0.2, 'k':10, 'star,projang':-90, 'star,incl':20}

oi.setupFit({'obs':['V2'],'max relative error':{'V2':0.5}})
# -- cannot fit 'k', but 10 looks OK compared to fig 4
oi.doFit(p, doNotFit=['k'])

# -- show profile
plt.close(0); plt.figure(0)
p2 = pmoired.oimodels.computeLambdaParams(oi.bestfit['best'])
r = np.linspace(0, p2['star,diam']/2, 100)
plt.plot(r/(p2['Rstar']), eval(p2['star,profile'].replace('$R', 'r')))
plt.xlabel('r/R$_\star$')

# -- show data
oi.show(logV=1, imFov=1.1*p2['star,diam'], showUV=False)"""