# AISLENS

AISLENS (The Antarctic Ice Sheet Large Ensemble) is a [project](https://iceclimate.eas.gatech.edu/research/antarctic-ice-sheet-large-ensemble-project-aislens/) to simulate a large ensemble of possible scenarios of future evolution of the Antarctic Ice Sheet under realistic variability in oceanic (and atmospheric?) processes. The purpose of this project is to understand the role of climate variability in the uncertainty of future sea level projections.

More details on the ensemble generator used in this work can be found in the [aislens_emulation](https://github.com/mshiv/aislens_emulation) repository.[[1]](#1)

The work makes use of the [MALI (MPAS-Albany Land Ice)](https://github.com/MALI-Dev/E3SM)[[2]](#2) model for running simulations and generating model outputs. This repository contains workflows necessary to generate the initial conditions and forcing files required for the same.



## Installation

Use a `conda` environment that can be created:
```shell
git clone https://github.com/mshiv/AISLENS.git
cd AISLENS
conda create --name aislens --file requirements.txt
```

The variability generator makes use of a forked version of the `xeofs` package, and details regarding changes made therein can be found [here](https://github.com/mshiv/xeofs-rand.git).


## Data

Create a `data/` directory in the root project repository, with the following structure:
/AISLENS/
├── data/                         # Main, CLI interface
    ├── external/
    ├── interim/
    ├── processed/
    ├── tmp/

Copy the data files provided in the Zenodo object into this directory.

## Workflow

Refer docs.

## References

<a id="1">[1]</a>
S. Muruganandham, A. A. Robel, M. J. Hoffman and S. F. Price, "[Statistical Generation of Ocean Forcing With Spatiotemporal Variability for Ice Sheet Models](https://ieeexplore.ieee.org/document/10201387)," in Computing in Science & Engineering, vol. 25, no. 3, pp. 30-41, May-June 2023, doi: 10.1109/MCSE.2023.3300908.

<a id="2">[2]</a>
Hoffman, M. J., Perego, M., Price, S. F., Lipscomb, W. H., Zhang, T., Jacobsen, D., Tezaur, I., Salinger, A. G., Tuminaro, R., and Bertagna, L.: [MPAS-Albany Land Ice (MALI): a variable-resolution ice sheet model for Earth system modeling using Voronoi grids](https://doi.org/10.5194/gmd-11-3747-2018), Geosci. Model Dev., 11, 3747–3780, 2018, doi: 10.5194/gmd-11-3747-2018.