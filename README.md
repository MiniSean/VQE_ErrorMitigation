# VQE_ErrorMitigation
Supervised project that emulates a noisy variational quantum eigensolver. Error mitigation techniques are applied in order to improve performance.

For windows systems that want to run Psi4 library:
Either use a docker setup where all molecular data calculation is done in a linux environment.
Or setup an (Ubuntu) subsystem with virtual environment including the psi4 package (e.g. the conda package manager).
To comply with python's way of referencing modules be sure to (temporarily) add the project directory to your python path.
```
bash $ export PYTHONPATH="$HOME/path_to_project"
```