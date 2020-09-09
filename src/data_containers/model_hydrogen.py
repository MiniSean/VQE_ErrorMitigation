from typing import Sequence
import cirq
import numpy as np
import os
from openfermion import MolecularData
from openfermionpsi4 import run_psi4

from src.data_containers.helper_interfaces.i_wave_function import IGeneralizedUCCSD
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.circuit_noise_extension import Noisify


class HydrogenAnsatz(IGeneralizedUCCSD):

    def __init__(self):
        molecule_params = IParameter({'r0': .7414})  # .7414, 'r1': 1.
        super().__init__(molecule_params)

    def _generate_molecule(self, p: IParameter) -> MolecularData:
        """Produce molecule that can be used by the hamiltonian.
        Using a singlet state with S = 0 to specify we are looking for the lowest singlet energy state.
        multiplicity = 2S + 1
        """
        geometry = [('H', (0., 0., 0.)), ('H', (0., 0., p['r0']))]  # , ('H', (0., p['r1'], 0.))
        molecule = MolecularData(geometry=geometry, basis='sto-3g', multiplicity=1, charge=0, description=format(p['r0']))
        # Make sure parameter 'shell=true' is specified in the subprocess.Popen() call in _run_psi4.py
        # (process = subprocess.Popen(['psi4', input_file, output_file], shell=True))
        # Source 1: https://stackoverflow.com/questions/39269675/python-trouble-with-popen-filenotfounderror-winerror-2
        # Source 2: https://stackoverflow.com/questions/42572582/winerror-2-the-system-cannot-find-the-file-specified-python
        # Run (bash): cmake -H. -Bbuilddir -D CMAKE_INSTALL_PREFIX=D:\sean\programs\PyCharmProjects -G "Visual Studio 15 2017 Win64"
        # Run (conda): cmake -B D:\sean\programs\PyCharmProjects\psi4 -D CMAKE_INSTALL_PREFIX=D:\sean\programs\PyCharmProjects -G "Visual Studio 15 2017 Win64"
        # -DBLAS_TYPE=SYSTEM_NATIVE -DLAPACK_TYPE=SYSTEM_NATIVE
        # (specify 64 bit VS compiler)
        # Build from binary perhaps? http://www.psicode.org/psi4manual/master/conda.html
        # molecule = run_psi4(molecule,
        #                     run_scf=True,
        #                     run_mp2=True,
        #                     run_cisd=True,
        #                     run_ccsd=True,
        #                     run_fci=True,
        #                     verbose=False,
        #                     tolerate_error=True,
        #                     memory=4000)
        molecule.save()
        molecule.load()
        print(f'Reference (Full Configuration Energy: {molecule.fci_energy})')
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        description = str(p['r0'])

        # Change to whatever directory you want
        cwd = os.getcwd()
        data_directory = cwd+'/mol_data'

        if not os.path.exists(data_directory):
            os.mkdir(data_directory)

        filename = data_directory+'/H2_'+description

        run_scf = 1
        run_mp2 = 1
        run_cisd = 1
        run_ccsd = 1
        run_fci = 1
        delete_input = False
        delete_output = False
        verbose = False

        molecule = MolecularData(
            geometry,
            basis,
            multiplicity,
            description=description,
            filename=filename)

        if os.path.exists('{}.hdf5'.format(filename)):
            molecule.load()

        else:
            molecule = run_psi4(molecule,
                                verbose=verbose,
                                run_scf=run_scf,
                                run_mp2=run_mp2,
                                run_cisd=run_cisd,
                                run_ccsd=run_ccsd,
                                run_fci=run_fci)

        return molecule

    # IWaveFunction
    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """
        Using Hartree Fock state:
        |1100>
        instead of the most general state:
        |0011> + |1100> + |1001> + |0110> (disregarding normalization)

        :param qubits: Circuit qubits, (0, 1, 2, 3)
        :return:  X_0 X_1
        """
        yield [cirq.rx(np.pi).on(qubits[0]),
               cirq.rx(np.pi).on(qubits[1])]


class NoisyHydrogen(HydrogenAnsatz):

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_tree(IGeneralizedUCCSD.operations(self, qubits=qubits))

    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_tree(HydrogenAnsatz.initial_state(self, qubits=qubits))
