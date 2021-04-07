![banner](banner/banner/banner.001.png)

## Overview

`PMOIRED` is a Python3 module which allows to model astronomical spectro-interferometric data stored in the OIFITS format ([Duvert et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract)). Parametric modeling is used to describe the observed scene as blocks such as disks, rings and Gaussians which can be combined and their parameters linked. It includes plotting, least-square fitting and bootstrapping estimation of uncertainties. For spectroscopic instruments (such as GRAVITY), tools are provided to model spectral lines and correct spectra for telluric lines.

The modeling of data is based on several principles:
- The model is composed of a combination of basic building blocks
- Simple building blocks include uniform disks, uniform rings, Gaussians.
- Building blocks can be deformed (analytically), including stretched in one preferred direction, or slanted. This can efficiently simulate inclined components.
- More complicated blocks are available, such as disks/rings with arbitrary radial profile, with possibility to include azimuthal intensity variations.
- Each component has a spectrum, including modelling of emission or absorption lines (Gaussian or Lorentzian)
- In order for the computation to be fast (a requirement to perform data fitting), basic blocks have analytical or semi-analytical complex visibilities. Moreover, for the same reason, their spectral component is independent of the geometry.

The principles are close to tools such as [LITpro](https://www.jmmc.fr/english/tools/data-analysis/litpro). However, `PMOIRED` offers additional features:
- `PMOIRED` extends the modelling in the spectral dimension. For this reason, it contains a module to do basic telluric correction (only for GRAVITY at the moment)
- Models' parameters can be expressed a function of each others, which allows to build complex geometrical shapes: astronomical realistic models (e.g. approximate Keplerian rotating disks) can be build this way, without compromising on execution speed.

## Install

Run the following command in the the root directory of the sources:
```
python setup.py install --record files.txt
```
if you do not have the root rights to your system, you can alternatively run:
```
python setup.py install --user --record files.txt
```
To uninstall (assuming you have recorded the install files at described above):
```
xargs rm -rf < files.txt
```

## Examples

Check out the examples provided in the package in the directory `examples`, in the form of Jupyter notebooks:
- [Alpha Cen A](https://github.com/amerand/PMOIRED/blob/master/examples/alphaCenA.ipynb) PIONIER data from [Kervalla et al. A&A 597, 137 (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.137K/abstract). Fitting V2 with uniform disk or limb-darkened disks, including with parametrised darkening.
- [FU Ori](https://github.com/amerand/PMOIRED/blob/master/examples/FUOri.ipynb) GRAVITY data from [Liu et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884...97L/abstract). Fitting a 2 chromatic components model.
- [AX Cir](https://github.com/amerand/PMOIRED/blob/master/examples/AXCir.ipynb) demonstrates how to do a companion search using a grid of positions, inspired from [CANDID](https://github.com/amerand/CANDID).

Below is a skeleton documentation (more to come!). The best is to go through examples, as most of the syntax is covered.

## Loading data

`self.OI`

## Plotting data

`self.show`

## Model syntax

Models are defined as dictionaries. The keys describe the model and the values are the actual parameters. Models can have many components (see "Combining blocks to build complex models" below). The model also support expressing parameters as function of other parameters (including from another component). The parametrisation applies to both geometrical and spectral dimensions, although by default models are grey.

The example notebook linked above provide examples of usage of the modeling capability.

Possible keys in the model dictionary:

### Position:
  - `x`, `y`: define position in the field, in mas
      if not give, default will be `'x':0, 'y':0`. `x` is offset in RA, towards East and `y` is offset in declination, towards North.  

### Size (will decide the model):
  - `ud`: uniform disk diameter (in mas)
  - or `fwhm`: Full width at half maximum for a Gaussian (in mas)
  - or `diam`: outside diameter for a ring (in mas) and `thick`: fractional thickness of the ring (0..1)

if none of these is given, the component will be fully resolved (V=0)

### Stretching:
  Object can be stretched along one direction, using 2 additional parameters:
  - `projang`: projection angle from N (positive y) to E (positive x), in deg
  - `incl`: inclination, in deg. 0 is face-on, 90 is edge-on. The dimensions will be reduced by a factor cos(incl) along the direction perpendicular to the projection angle

### Slant:
  `slant`: slant in flux along the object (only for uniform disks and rings, not Gaussian). between 0 (None) to 1 (maximum, i.e. one side of the object has 0 flux).
  `slant projang`: projection angle of the slant, in degrees. 0 is N and 90 is E.

### Complex rings and disks:
  Rings are by default uniform in brightness. This can be altered by using
  different keywords:
- `diam` is the diameter (in mas) and `thick` the thickness. By default, thickness is 1 if omitted.
- `profile`: radial profile.

Profiles can be arbitrary defined. To achieve this, you can define the profile as a string, as function of other parameters in the dictionary. There are also two special variables: `$R` and `$MU` where `R`=2*r/`diam`  and `MU`=sqrt(1-`R`^2). For example, to model a limb-darkened disk with a power law profile (in `MU`), one will define the model as:
```
param = {'diam':1.0, 'profile':'$MU**$alpha', '$alpha':0.1}
```
The parsing of `profile` is very basic, so do not create variable names with common name (e.g. `np` or `sqrt`).

- `az ampi`, `az projangi`: defines the cos variation amplitude and phase for i nodes along the azimuth.

### Flux modeling:
If nothing is specified, flux is assume equal to 1 at all wavelengths. Flux can be described using several parameters:

- `f`: if the constant flux (as function of wavelength). If not given, default will be implicitly `'f':1.0`

- spectral lines can be defined as:
  - 'line_1_f': amplitude of line (emission >0, absorption <0), same unit as 'f'
  - `line_1_wl0`: central wavelength in microns
  - `line_1_gaussian`: full width at half maximum, in nm (not microns!!!)
  - or `line_1_lorentzian`: width at half maximum, in nm (not microns!!!). `1/(1+($WL-wl0)**2/(lorentzian/1000)**2)`

You can define several lines by using `line_2_...`, `line_3_...` etc. or even be more explicit and use `line_H_...`, `line_Fe_...` etc.

Note that using lines defined this way will allow for fitting differential quantities such as `NFLUX` (flux normalised to the continuum) or `DPHI` (differential phase, with respect to continuum). The continuum is automatically determined using the lines' parameters.

Arbitrary chromatic variations can be achieved using the `spectrum` parameter, very much like the `profile` parameter for rings. Note that the spectrum will be added to any over spectral information present in the parameters:
```
{'A0':1.0, 'A2':0.2, 'spectrum':'A0 + A2*($WL-np.min($WL))**2'}
```

## Combining blocks to build complex models

Blocks can be combined by using a single dictionary, but tagging each component as using syntax `tag,parameter`. For example, to define a simple binary model:
```
{'primary,ud':0.1,
 'secondary,ud':0.1,
 'secondary,x':5.0,
 'secondary,y':-12,
 'secondary,f':0.1}
```
Note that in this example, the following is implicit: `{'primary,x':0.0, 'primary,y':0.0, 'primary,f':1.0}`.

Any parameter can be describe as a formula. To parametrise the binary in polar coordinates, with both stars (primary and secondary) having the same uniform disk diameter:
```
{'primary,ud':'$diam0',
 'secondary,ud':'$diam0',
 'secondary,x':'$sep*np.sin($pa*np.pi/180)',
 'secondary,y':'$sep*np.cos($pa*np.pi/180)',
 'sep':10, 'pa': 45,
 'secondary,f':0.1,
 'diam0':0.1}
```
The only limitation in the syntax and variable creation is to avoid using names used in `python`, `numpy` or `pmoired`.

## Model fitting

Once your model is defined, you can fit it to the data.

### Setting up the fit

`self.setupFit`

### Fitting

`self.doFit`
