# OptMSP

OptMSP is a toolbox for optimal design of multi-stage processes, that determines the optimal combinations of distinct models that are based on experimental data (e.g. including growth rate, substrate uptake and product formation rates) and switching time(s) between those stages such that an objective function is optimized (e.g. volumetric productivity). 

### This repository includes:
MultiStagePackage
  - ```OptMSPfunctions.py``` = including all functions for OptMS to work
  - ```models.py``` = the models for the case study (rate values based on [Wichmann *et al.*](https://doi.org/10.1016/j.ymben.2023.04.006))
    
```ModelingTutorial.ipynb``` = Notebook that describes how models are implemented and how custom models could be included

```OptMSP_MainFunctions.ipynb``` = Notebook that describes all essential functions of OptMSP with examples

```OptMSP_SupportFunctions.ipynb``` = Notebook that descibes a few supporting functions of OptMSP with examples

```environment.yml``` = Dependency file for conda environment

### Install instructions
I recommend to install the [```conda-libmamba-solver```](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for faster installation of the environment. Type in terminal:
`CONDA_EXPERIMENTAL_SOLVER=classic conda install -n base conda-libmamba-solver=23.3`

`conda config --set solver libmamba`

Create environment:
`conda env create -f environment.yml`




