# OptMSP

OptMSP is a toolbox for optimal design of multi-stage processes, that determines the optimal combinations of distinct models that are based on experimental data (e.g. including growth rate, substrate uptake and product formation rates) and switching time(s) between those stages such that an objective function is optimized (e.g. volumetric productivity). 

### This repository includes:
- MultiStagePackage
  - ```OptMSPfunctions.py``` = including all functions for OptMS to work
  - ```models.py``` = the models for CaseStudy_wichmann2023.ipynb (rate values based on [Wichmann *et al.*](https://doi.org/10.1016/j.ymben.2023.04.006))
  - ```auxstates.csv``` = the auxiliary variables for CaseStudy_klamt2018.ipynb (values based on [Klamt *et al.*](https://doi.org/10.1002/biot.201700539))
    
- ```ModelingTutorial.ipynb``` = Notebook that describes how models are implemented and how custom models could be included (but also take a look at the case studies)

- ```OptMSP_MainFunctions.ipynb``` = Notebook that describes all essential functions of OptMSP with examples

- ```OptMSP_SupportFunctions.ipynb``` = Notebook that descibes a few supporting functions of OptMSP with examples
  
- ```CaseStudy_wichmann2023.ipynb``` = Different tests of finding optimal 1-, 2- and 3-Stage processes with OptMSP based on data from [Wichmann *et al.*](https://doi.org/10.1016/j.ymben.2023.04.006)

- ```CaseStudy_gotsmy2023.ipynb``` = Optimization for volumetric productivity in a fed batch setup based on data from [Gotsmy *et al.*](https://doi.org/10.1186/s12934-023-02248-2)

- ```CaseStudy_klamt2018.ipynb``` = Optimization for volumetric productivity with yield constraint based on data from [Klamt *et al.*](https://doi.org/10.1002/biot.201700539)

- ```environment.yml``` = Dependency file for conda environment

### Install instructions
I recommend to install the [```conda-libmamba-solver```](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for faster installation of the environment. 

Type in terminal:

`CONDA_EXPERIMENTAL_SOLVER=classic conda install -n base conda-libmamba-solver=23.3`

`conda config --set solver libmamba`

Create environment:

`conda env create -f environment.yml`




