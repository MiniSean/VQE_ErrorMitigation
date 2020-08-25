import numpy as np
import cirq
import openfermioncirq
from openfermioncirq.optimization import ScipyOptimizationAlgorithm, OptimizationParams, OptimizationTrialResult
from openfermion import jordan_wigner, QubitOperator

from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter


class QPU:

    @staticmethod
    def get_expectation_value(t_r: cirq.TrialResult, w: IWaveFunction):
        qubit_operator = QPU.get_hamiltonian_evaluation_operator(w)
        objective = openfermioncirq.HamiltonianObjective(qubit_operator)
        return objective.value(t_r.measurements['x'])

    @staticmethod
    def get_trial_results(r_c: cirq.circuits.circuit, r: cirq.study.resolver, max_iter: int) -> cirq.TrialResult:
        simulator = cirq.Simulator()  # Setup simulator
        return simulator.run(r_c, r, max_iter)  # Probabilistic result

    @staticmethod
    def get_initial_state_circuit(w: IWaveFunction) -> cirq.circuits.circuit:
        circuit = cirq.Circuit(w.initial_state(qubits=w.qubits), strategy=cirq.InsertStrategy.EARLIEST)
        return circuit

    @staticmethod
    def get_resolved_circuit(c: cirq.circuits.circuit, p: IParameter) -> cirq.circuits.circuit:
        return cirq.resolve_parameters(c, p.get_resolved())

    @staticmethod
    def get_optimized_state(s: openfermioncirq.VariationalStudy, max_iter: int) -> OptimizationTrialResult:
        algorithm = ScipyOptimizationAlgorithm(kwargs={'method': 'COBYLA'}, options={'maxiter': max_iter}, uses_bounds=False)
        optimization_params = OptimizationParams(algorithm=algorithm)
        seed_array = [np.random.randint(2 ** 31) for i in optimization_params.initial_guess] if optimization_params.initial_guess is not None else [np.random.randint(2 ** 31)]
        return s.optimize(optimization_params, seeds=seed_array)

    @staticmethod
    def get_hamiltonian_evaluation_operator(w: IWaveFunction) -> QubitOperator:
        molecule = w.molecule
        molecule.load()
        return jordan_wigner(molecule.get_molecular_hamiltonian())

    @staticmethod
    def get_variational_study(w: IWaveFunction, p_c: cirq.Circuit, name: str) -> openfermioncirq.VariationalStudy:
        evaluator = QPU.get_hamiltonian_evaluation_operator(w)
        objective = openfermioncirq.HamiltonianObjective(evaluator)
        return openfermioncirq.VariationalStudy(name=name, ansatz=w, objective=objective, preparation_circuit=p_c)


