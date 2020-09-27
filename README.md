# KerrPy
An openCV based Python project for processing Kerr microscopy image. KerrPy is used to extract ellipse fits of  bubble domains created under the application of out-of-plane and in-plane fields. These fits are used to estimate the underlying Dzyaloshinskii Moriya interaction (DMI)

# Installation requirements

## Anaconda for managing Python environments

- preferably 64 bit.

- open Navigator.

- default environment is base(root)

## Create a Python environment : envKerr

- Every project based on Python is recommended to have a separate python environment.

- Activate the environment

- Install packages in the environment

## Packages


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


Looks like newer versions of Anaconda has opencv in the default channel
So no need to add conda-forge
However latest versions of opencv may not be supported in the defualt

However we will install the default channel version.
