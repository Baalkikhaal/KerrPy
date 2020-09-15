

0. Install Anaconda for managing Python environments


	preferably 64 bit.
	
	
	open Navigator.
	
	default environment is base(root)
	

1. Create a Python environment : envKerr

	Every project based on Python is recommended to have
	
	a separate python environment.


2. Activate the environment




3. Install packages in the environment

	-------------
	core packages
	-------------
	0. numpy
	1. scipy
	2. matplotlib

	-------------------------
	image processing packages
	-------------------------
	0. opencv

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
	
	-------------------------------------------
	Integrated Development Environment packages
	-------------------------------------------
	spyder


	So as we can see each python package is built upon other packages.
	
	This packages called dependencies.
	
	For this project, there are around 150 dependencies.
	
	And mostly all of these are OPEN SOURCE.
	
	
	Even matplotlib has interesting history.
	
	Its creator John Hunter, who was a neuroscientist
	
	was not happy with closed nature of MATLAB 
	
	and not satisfied with limited features of plotting
	
	his Brain EEG graphs.
	
	So he decided to create his own package.
	
	He called it Matplotlib as it is based on MATLAB's plotting
	interface
	
	
	The package is so extensive that even the documentation is
	
	more than 4000 pages.
	
	
	OK envKerr is installed. Now access spyder from windows toolbar
	
	
	
	
	we have cleaned the code so that we dont depend on skimage.
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	