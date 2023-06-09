echo Script executed by `whoami` at `date` in `hostname`
python --version
conda --version
echo The conda environment is $CONDA_PREFIX
python -c 'import numpy as np; print("numpy version", np.__version__)'
python -c 'import scipy as sp; print("scipy version", sp.__version__)'
python -c 'import astropy as ap; print("astropy version", ap.__version__)'
python -c 'import matplotlib as mpl; print("matplotlib version", mpl.__version__)'
python -c 'import corner; print("corner version", corner.__version__)'
python -c 'import pint; print("PINT version", pint.__version__)'
echo tempo2 version is `tempo2 -v`
python -c 'import libstempo; print("libstempo version", libstempo.__version__)' 2> /dev/null
python -c 'import enterprise; print("ENTERPRISE version", enterprise.__version__)'
python -c 'import enterprise_extensions; print("enterprise_extensions version", enterprise_extensions.__version__)'
python -c 'import PTMCMCSampler; print("PTMCMCSampler version", PTMCMCSampler.__version__)' | grep version
julia --version
python -c 'import enterprise_gwecc as gwecc; print("GWecc version", gwecc.__version__)'