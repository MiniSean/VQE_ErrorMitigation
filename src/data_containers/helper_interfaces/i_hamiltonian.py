import openfermioncirq
from openfermion import jordan_wigner, QubitOperator

from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction


class IHamiltonian:

    @staticmethod
    def get_qubit_operator(w: IWaveFunction) -> QubitOperator:
        molecule = w.molecule
        molecule.load()
        return jordan_wigner(molecule.get_molecular_hamiltonian())

    @staticmethod
    def get_variational_study(w: IWaveFunction, name: str) -> openfermioncirq.VariationalStudy:
        qubit_operator = IHamiltonian.get_qubit_operator(w)
        objective = openfermioncirq.HamiltonianObjective(qubit_operator)
        return openfermioncirq.VariationalStudy(name=name, ansatz=w, objective=objective)
