# Goal: estimate the minimum energy of a given hamiltonian and trail wave function
from typing import Callable, Tuple

from src.data_containers.qubit_circuit import *
from src.data_containers.helper_interfaces.i_hamiltonian import IHamiltonian, T
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.model_ising import IsingModel
# ISettingsML
# IExpectationValue(IOperator_i -> IValue)
import cirq
import copy
import random
import numpy as np
from scipy import optimize


class Estimator:

    def __init__(self, h: IHamiltonian, qpu_iter: int, cpu_iter: int):
        self._hamiltonian = h
        # self._wave_function = w
        qpu = QPU(h, simulation_repetitions=qpu_iter)
        self._min_energy = CPU.minimise_expectation_value(h.parameters, qpu, max_iter=cpu_iter)


class QPU:

    def __init__(self, h: IHamiltonian, simulation_repetitions: int):
        self._sim = cirq.Simulator()  # Setup simulator
        # Define circuit (unresolved)
        self._circuit = cirq.Circuit()
        self._circuit.append(h.observable(None), strategy=cirq.InsertStrategy.EARLIEST)
        self._circuit.append(cirq.measure(*h.qubits, key='x'))  # Temporary forced measurement on x axis
        # Define measurement iterations
        self._iter = simulation_repetitions
        # Define energy function
        self._energy_function = h.energy_func

    # def get_expectation_values(self, h: IHamiltonian, p: IParameter) -> [float]:
    #     sweep = QPU.get_parameter_sweep(p)  # Parameter resolver
    #     resolved_circuit = QPU.get_resolved_circuit(h, p)  # Resolve symbol molecule in circuit
    #     print(resolved_circuit)
    #     trial_results = self.get_simulation_trial_results(resolved_circuit, sweep, self._iter)  # Probabilistic results
    #     return [IHamiltonian.expectation_value(result, h.energy_func()) for result in trial_results]  # Statistic result

    def get_expectation_value(self, p: IParameter) -> float:
        sweep = QPU.get_parameter_sweep(p)  # Parameter resolver
        resolved_circuit = QPU.get_resolved_circuit(self._circuit, p)  # Resolve symbol molecule in circuit
        # print(resolved_circuit)
        trial_result = self._sim.run(resolved_circuit, sweep, self._iter)  # Probabilistic result
        return IHamiltonian.expectation_value(trial_result, self._energy_function)  # Statistic result

    def get_simulation_trial_results(self, c: cirq.circuits.circuit, s: cirq.study.resolver, r: int) -> [cirq.study.trial_result]:
        return self._sim.run_sweep(c, params=s, repetitions=r)

    @staticmethod
    def get_parameter_sweep(p: IParameter) -> cirq.study.resolver:
        return p.get_resolved()

    @staticmethod
    def get_resolved_circuit(c: cirq.circuits.circuit, p: IParameter) -> cirq.circuits.circuit:
        return cirq.resolve_parameters(c, p.get_resolved())


class CPU:

    @staticmethod
    def get_total_expectation_value(structure: IHamiltonian, w: IWaveFunction) -> float:
        pass

    @staticmethod
    def minimise_expectation_value(p: IParameter, processor: QPU, max_iter: int) -> float:
        return CPU.scipy_optimize(p, processor, max_iter)

    @staticmethod
    def scipy_optimize(p: IParameter, processor: QPU, max_iter: int) -> float:
        def optimize_func(x: np.ndarray) -> float:
            for i, key in enumerate(p):
                p.dict[key] = x[i]
            return processor.get_expectation_value(p)
        # Optimization settings
        initial_values = np.fromiter(p.dict.values(), dtype=float)
        method = 'Nelder-Mead'
        options = {'maxiter': max_iter}
        return optimize.minimize(fun=optimize_func, x0=initial_values, method=method, options=options)

    @staticmethod
    def random_walk(p: IParameter, processor: QPU, max_iter: int) -> float:
        delta_step = .1
        circuit_parameter = p
        expectation_value = processor.get_expectation_value(circuit_parameter)

        def sigmoid(v: float) -> float:
            return 1 / (1 + np.exp(v))  # sigmoid acceptance

        def iteration(p_old: IParameter, v_old: float) -> (IParameter, float):
            p_new = CPU.get_random_parameter_walk(p_old, delta_step)
            v_new = processor.get_expectation_value(p_new)  # Create probability value (0, 1]
            try:  # Catch zero division errors if new state is not allowed
                prob = (sigmoid(v_new) / sigmoid(v_old)) ** 2  # Create probability value (0, 1]
                print(f'from: {v_old}, to: {v_new}. With prob: {prob}')
                if prob >= 1 or (random.uniform(0, 1) < prob):
                    return (p_new, v_new)  # Accept new state under certain probability
            except ZeroDivisionError:
                pass
            return (p_old, v_old)

        for i in range(max_iter):
            circuit_parameter, expectation_value = iteration(circuit_parameter, expectation_value)
            print(expectation_value)
            print(circuit_parameter)
        return expectation_value

    @staticmethod
    def get_random_parameter_walk(p: IParameter, delta: float) -> IParameter:
        dict_copy = copy.deepcopy(p.dict)
        for key in dict_copy:
            dict_copy[key] += random.uniform(-delta, delta)  # random walk
        return IParameter(parameters=dict_copy)


class Model(IHamiltonian):

    def __init__(self, grid_size: int):
        qubits = cirq.GridQubit.square(grid_size)  # Determine qubits to be used to represent Hamiltonian
        parameters = IParameter({'alpha': 1})  # Determine tweakable molecule
        super().__init__(qubits, parameters)

    def observable(self, w: IWaveFunction) -> [cirq.ops.moment]:
        pass

    def energy_func(self, measurements: np.ndarray) -> Callable[[Tuple], T]:
        pass


if __name__ == '__main__':
    # Define qubits and circuit
    H = IsingModel(3)
    P = H.parameters
    # for i, key in enumerate(P):
    #     P.dict[key] = .1
    print(P)
    print(H.h)
    print(H.jr)
    print(H.jc)

    Processor = QPU(H, simulation_repetitions=30)
    exp_value = CPU.minimise_expectation_value(P, Processor, max_iter=200)
    print(f'Minimization result: \n{exp_value}')

    # print(P.dict)
    # # Show observable circuit and get expectation value
    # values = Processor.get_expectation_value(H, P)
    # print(values)
    #
    # # Change molecule and update circuit?
    # P.dict['alpha'] = .1
    # P.dict['beta'] = .2
    # P.dict['gamma'] = .3
    # print(P.dict)
    # values = Processor.get_expectation_value(H, P)
    # print(values)

    # value = QPU.get_expectation_values(H, None)
    # print(value)

    # sweep = (cirq.Linspace(key='alpha', start=0.1, stop=0.9, length=5)
    #          * cirq.Linspace(key='beta', start=0.1, stop=0.9, length=5)
    #          * cirq.Linspace(key='gamma', start=0.1, stop=0.9, length=5))
    # print(([cirq.Linspace(key=key, start=0.1, stop=0.9, length=5) for key in h._par.resolved.param_dict]))
    # print(sweep)
