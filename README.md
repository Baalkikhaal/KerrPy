# KerrPy
An openCV based Python project for processing Kerr microscopy image. KerrPy is used to extract ellipse fits of  bubble domains created under the application of out-of-plane and in-plane fields. These fits are used to estimate the underlying Dzyaloshinskii Moriya interaction (DMI)

# Installation requirements

- Anaconda (preferably 64 bit) for managing Python environments.

- A Python environment

- Installation of dependencies

## Python environment : envKerr

Every Python project is recommended to have a separate python environment. There are two ways to create a new environment.

## Through [Navigator]((https://docs.anaconda.com/anaconda/navigator/tutorials/create-python35-environment/?highlight=environment%20create#creating-a-python-3-5-environment-from-anaconda2-or-anaconda3))

A walkthrough of the installation is given in  [YouTube video of installation walkthrough](https://www.youtube.com/watch?v=8bzp04TeE3I)

- Activate the environment

- Install packages in the environment

### Packages

core packages |	image processing packages | IDE packages
--- | ---- | ---
numpy	| 		opencv	|	spyder
scipy	|	|
matplotlib	|	|

Anaconda maintains "channels" for storing repositories.
Core packages are available in standard/default channel.
So can be directly downloaded
whereas
opencv is present in "conda-forge" channel.
conda-forge is a community maintained channel.
The community is comprised mostly of researchers.
So we need to add conda-forge channel to the environment.


Newer versions of Anaconda has `opencv` in the default channel
So no need to add `conda-forge`
However latest versions of `opencv` may not be supported in the default.

However we will install the default channel version.

## command line way

- through [command line](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). Use the `environment.yml` in the top level directory.

		conda env create -f environmentKerrPy.yml

- activate the environments

		conda activate envKerr
