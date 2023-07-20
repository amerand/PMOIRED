![banner](banner/banner/banner.001.png)

## Overview

`PMOIRED` is a Python3 module which allows to model astronomical spectro-interferometric data stored in the OIFITS format ([Duvert et al. 2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A...8D/abstract)). Parametric modeling is used to describe the observed scene as blocks such as disks, rings and Gaussians which can be combined and their parameters linked. It includes plotting, least-square fitting and bootstrapping estimation of uncertainties. For spectroscopic instruments (such as GRAVITY), tools are provided to model spectral lines and correct spectra for telluric lines.

The modelling of data is based on several principles:
- The model is composed of a combination of basic building blocks (see the [model definition](https://github.com/amerand/PMOIRED/blob/master/examples/Model%20definitions%20and%20examples.ipynb) notebook)
- Simple building blocks include uniform disks, uniform rings, Gaussians.
- Building blocks can be deformed (analytically), including stretched in one preferred direction, or slanted. This can efficiently simulate inclined components.
- More complicated blocks are available, such as disks/rings with arbitrary radial profile, and possibility to include azimuthal intensity variations.
- Each component has a spectrum, including modelling of emission or absorption lines (Gaussian or Lorentzian)
- In order for the computation to be fast (a requirement to perform data fitting), basic blocks have analytical or semi-analytical complex visibilities. Moreover, for the same reason, their spectral component is independent of the geometry.

The principles are close to tools such as [LITpro](https://www.jmmc.fr/english/tools/data-analysis/litpro). However, `PMOIRED` offers additional features:
- Modelling in the spectral dimension. For this reason, it contains a module to do basic telluric correction (only for VLTI/GRAVITY at the moment)
- Models' parameters can be expressed a function of each others, which allows to build complex geometrical shapes: astronomical realistic models can be build this way, without compromising on execution speed.
- Uncertainties can be estimated using bootstrapping (data resampling by date+telescope) to mitigate the effects of correlations between data.
- The values of parameters can be explored using grid and or random search.
- Synthetic data for VLTI array observations can be generated from data cubes.

## Install

_`PMOIRED` can be used without installing it. Look at the examples to see how to call it_.

Run the following command in the the root directory of the sources:
```
python3 -m pip install .
```

To uninstall (assuming you have recorded the install files at described above):
```
python3 -m pip uninstall pmoired
```

## Examples

Check out the examples provided in the package in the directory `examples`, in the form of Jupyter notebooks:
- [Model definition](https://github.com/amerand/PMOIRED/blob/master/examples/Model%20definitions%20and%20examples.ipynb) Illustrates the syntax for model definition. Start here!
- [Alpha Cen A](https://github.com/amerand/PMOIRED/blob/master/examples/angular%20diameter%20alphaCenA.ipynb) PIONIER data from [Kervella et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...597A.137K/abstract). Fitting V2 with uniform disk or limb-darkened disks, including with parametrised darkening. Most `PMOIRED` basics are covered there.
- [FU Ori](https://github.com/amerand/PMOIRED/blob/master/examples/chromatic%20multi%20components%20FUOri.ipynb) GRAVITY data from [Liu et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...884...97L/abstract). Fitting 2-components model with chromatic flux ratio.
- [AX Cir](https://github.com/amerand/PMOIRED/blob/master/examples/companion%20search%20AXCir.ipynb) PIONIER data from [Gallenne et al. 2015](https://ui.adsabs.harvard.edu/abs/2015A%26A...579A..68G/abstract), shows how `PMOIRED` can be used to cover most of the features of [CANDID](https://github.com/amerand/CANDID)'s companion grid search and estimation of detection limit for a third component.
- [Keplerian disks](https://github.com/amerand/PMOIRED/blob/master/examples/Be%20model%20comparison%20with%20AMHRA.ipynb) shows how to generate VLTI simulated data from a data cube (image x wavelength) and how to fit them with `PMOIRED`. The data cube is a [Be star Model](https://amhra.oca.eu/AMHRA/bedisk/input.htm) from [AMHRA](https://amhra.oca.eu/AMHRA/index.htm). The model has been re-implemented in `PMOIRED`: data can be directly fitted. This notebook shows a comparison between the 2 implementations. Please refer to [Delaa et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011A%26A...529A..87D/abstract) for a description of the Keplerian disk model. `PMOIRED` is also used to fit VLTI/AMBER observations of alpha Col, and results are compared to [Meilland et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012A%26A...538A.110M/abstract) which use the Delaa etl al. (2011) model. 

## Using and quoting `PMOIRED`

References to `PMOIRED` should point to the [2022 SPIE Telescopes+Instrumentation conference proceeding paper](https://ui.adsabs.harvard.edu/abs/2022SPIE12183E..1NM/abstract). 

This code is a __research project in continuous development__ and not yet properly fully documented. If you want to get the best analysis for your data, do not hesitate to contact me: I try to be responsive, in particular with junior scientists. New features are driven by collaborations: if you think  `PMOIRED` is missing something, definitely contact me!

I maintain a list of works published using `PMOIRED` in a [curated bibliography](https://ui.adsabs.harvard.edu/public-libraries/dz7RG915Swq5yAB1KwmgTA).

> ***I used to to request co-authorship on publications using PMOIRED (until Feb 2023): this is no longer the case, unless I contributed directly to the work in question***

## Limitations and known issues

### Running a script in ipython hangs

PMOIRED uses the `multiprocessing` library to parallelise some computations (e.g. bootstrapping, grid search). This library has some issues if you run a script containing such computation is an interactive shell (using `%run` or `run` in ipython or Spyder). The provided examples as notebooks do not suffer from this problem. If you want to use PMOIRED in `.py` scripts you run in iPython, you should structure your `.py` script more or less as follow:
```
import pmoired
import matplotlib

matplotlib.interactive(True)
__spec__ = None
if __name__=='__main__':
    [code]
```
in iPython, you can now type `%run myscript.py`.

### Dependencies

The latest version of `PMOIRED` has been tested on:
- python 3.11.2
- numpy 1.24.2
- scipy 1.10.1
- astropy 5.2.1
- astroquery 0.4.6
- matplotlib 3.7.0
- jupyterlab 3.5.0
