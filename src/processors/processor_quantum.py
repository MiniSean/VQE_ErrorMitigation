import numpy as np
import cirq
# from openfermioncirq import HamiltonianObjective, VariationalStudy
from scipy.sparse import csc_matrix
# from openfermioncirq.optimization import ScipyOptimizationAlgorithm, OptimizationParams, OptimizationTrialResult
from openfermion import jordan_wigner, QubitOperator

from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter
# from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper


class QPU:

    # @staticmethod
    # def get_expectation_value(t_r: cirq.TrialResult, w: IWaveFunction):
    #     qubit_operator = QPU.get_hamiltonian_evaluation_operator(w)
    #     objective = HamiltonianObjective(qubit_operator)
    #     return objective.value(t_r.measurements['x'])

    # @staticmethod
    # def get_hamiltonian_objective_operator(w: IWaveFunction) -> np.ndarray:
    #     qubit_operator = QPU.get_hamiltonian_evaluation_operator(w)
    #     return HamiltonianObjective(qubit_operator)._hamiltonian_linear_op

    # How to calculate an Expected Value of some operator acting on qubits?
    # https://quantumcomputing.stackexchange.com/questions/6940/how-to-calculate-an-expected-value-of-some-operator-acting-on-qubits
    # @staticmethod
    # def get_simulated_noisy_expectation_value(w: IWaveFunction, r_c: cirq.circuits.circuit, r: cirq.study.resolver) -> float:
    #     simulator = cirq.DensityMatrixSimulator(ignore_measurement_results=False)  # Mixed state simulator
    #     simulated_result = simulator.simulate(program=r_c, param_resolver=r)  # Include final density matrix
    #
    #     qubit_operator = QPU.get_hamiltonian_evaluation_operator(w)
    #     objective = HamiltonianObjective(qubit_operator)
    #     # Perform trace between Hamiltonian operator and final state density matrix
    #     H_operator = objective._hamiltonian_linear_op  # Observable
    #     sparse_density_matrix = csc_matrix(simulated_result.final_density_matrix)
    #     # Tr( rho * H )
    #     trace = (sparse_density_matrix * H_operator).diagonal().sum()  # Complex stored value that should always be real
    #     return trace.real  # objective.value(simulated_result)  # If simulation result is cirq.WaveFunctionTrialResult

    @staticmethod
    def get_realistic_noisy_expectation_value(n_w: 'INoiseWrapper', c: cirq.Circuit, p: IParameter, max_iter: int) -> float:
        # Resolve circuit
        circuit_to_run = QPU.get_resolved_circuit(c=c, p=p)
        _func, _cost = n_w.observable_measurement()
        return _func(circuit_to_run, n_w._noise_channel, max_iter)

    @staticmethod
    def get_trial_results(r_c: cirq.circuits.circuit, r: cirq.study.resolver, max_iter: int) -> cirq.TrialResult:
        simulator = cirq.Simulator()  # Setup simulator
        trial_result = simulator.run(program=r_c, param_resolver=r, repetitions=max_iter)  # Probabilistic result
        return trial_result

    @staticmethod
    def get_noisy_trial_result(r_c: cirq.circuits.circuit, r: cirq.study.resolver, max_iter: int, noise_model: 'cirq.NOISE_MODEL_LIKE' = None) -> cirq.TrialResult:
        return cirq.sample(program=r_c, noise=noise_model, param_resolver=r, repetitions=max_iter)  # Probabilistic result

    @staticmethod
    def get_initial_state_circuit(w: IWaveFunction) -> cirq.circuits.circuit:
        circuit = cirq.Circuit(w.initial_state(qubits=w.qubits), strategy=cirq.InsertStrategy.EARLIEST)
        return circuit

    @staticmethod
    def get_resolved_circuit(c: cirq.circuits.circuit, p: IParameter) -> cirq.circuits.circuit:
        return cirq.resolve_parameters(c, p.get_resolved())

    # @staticmethod
    # def get_optimized_state(s: VariationalStudy, max_iter: int) -> OptimizationTrialResult:
    #     algorithm = ScipyOptimizationAlgorithm(kwargs={'method': 'COBYLA'}, options={'maxiter': max_iter}, uses_bounds=False)
    #     optimization_params = OptimizationParams(algorithm=algorithm)
    #     seed_array = [np.random.randint(2 ** 31) for i in optimization_params.initial_guess] if optimization_params.initial_guess is not None else [np.random.randint(2 ** 31)]
    #     return s.optimize(optimization_params, seeds=seed_array)

    @staticmethod
    def get_hamiltonian_evaluation_operator(w: IWaveFunction) -> QubitOperator:
        molecule = w.molecule
        molecule.load()
        return jordan_wigner(molecule.get_molecular_hamiltonian())

    # @staticmethod
    # def get_variational_study(w: IWaveFunction, p_c: cirq.Circuit, name: str) -> VariationalStudy:
    #     evaluator = QPU.get_hamiltonian_evaluation_operator(w)
    #     objective = HamiltonianObjective(evaluator)
    #     return VariationalStudy(name=name, ansatz=w, objective=objective, preparation_circuit=p_c)


