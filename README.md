![banner](banner/banner/banner.001.png)


## Preamble: using and quoting `PMOIRED`

This code is still in development and not yet fully documented. An article is in preparation describing the algorithms and features of `PMOIRED`. Using this code at the present should be done on a collaborative basis:
> ***Until a refereed article describing `PMOIRED` is published, if you are preparing an article using it, you should agree to add [me](mailto:amerand@eso.org) as a co-author***.

## Overview

`PMOIRED` is a Python3 module which allows to model astronomical spectro-interferometric data stored in the OIFITS format ([Duvert et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract)). Parametric modeling is used to describe the observed scene as blocks such as disks, rings and Gaussians which can be combined and their parameters linked. It includes plotting, least-square fitting and bootstrapping estimation of uncertainties. For spectroscopic instruments (such as GRAVITY), tools are provided to model spectral lines and correct spectra for telluric lines.

The modelling of data is based on several principles:
- The model is composed of a combination of basic building blocks
- Simple building blocks include uniform disks, uniform rings, Gaussians.
- Building blocks can be deformed (analytically), including stretched in one preferred direction, or slanted. This can efficiently simulate inclined components.
- More complicated blocks are available, such as disks/rings with arbitrary radial profile, and possibility to include azimuthal intensity variations.
- Each component has a spectrum, including modelling of emission or absorption lines (Gaussian or Lorentzian)
- In order for the computation to be fast (a requirement to perform data fitting), basic blocks have analytical or semi-analytical complex visibilities. Moreover, for the same reason, their spectral component is independent of the geometry.

The principles are close to tools such as [LITpro](https://www.jmmc.fr/english/tools/data-analysis/litpro). However, `PMOIRED` offers additional features:
- `PMOIRED` extends the modelling in the spectral dimension. For this reason, it contains a module to do basic telluric correction (only for GRAVITY at the moment)
- Models' parameters can be expressed a function of each others, which allows to build complex geometrical shapes: astronomical realistic models can be build this way, without compromising on execution speed.
- Uncertainties can be estimated using bootstrapping (data resampling by date+telescope) to mitigate the effects of correlations between data.
- The values of parameters can be explored using grid and or random search.


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
- [FU Ori](https://github.com/amerand/PMOIRED/blob/master/examples/FUOri.ipynb) GRAVITY data from [Liu et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884...97L/abstract). Fitting 2-components model with chromatic flux ratio.
- [AX Cir](https://github.com/amerand/PMOIRED/blob/master/examples/AXCir.ipynb) an implementation of [CANDID](https://github.com/amerand/CANDID)'s companion grid search and estimation of detection limit for a third component.

## Limitations and known issues

PMOIRED uses the `multiprocessing` libraries to parallelise some computation (e.g. bootstrapping, grid search). This library has some issues if you call a script containing such computation is an interactive shell (e.g. ipython, Spyder). The provided examples as notebooks do not suffer from this problem.
