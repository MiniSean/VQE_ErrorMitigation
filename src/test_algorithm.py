# Goal: estimate the minimum energy of a given hamiltonian and trail wave function
from typing import Callable, Tuple

from src.data_containers.qubit_circuit import *
from src.data_containers.helper_interfaces.i_hamiltonian import IHamiltonian, T
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter
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
    #     resolved_circuit = QPU.get_resolved_circuit(h, p)  # Resolve symbol parameters in circuit
    #     print(resolved_circuit)
    #     trial_results = self.get_simulation_trial_results(resolved_circuit, sweep, self._iter)  # Probabilistic results
    #     return [IHamiltonian.expectation_value(result, h.energy_func()) for result in trial_results]  # Statistic result

    def get_expectation_value(self, p: IParameter) -> float:
        sweep = QPU.get_parameter_sweep(p)  # Parameter resolver
        resolved_circuit = QPU.get_resolved_circuit(self._circuit, p)  # Resolve symbol parameters in circuit
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


# Temporary test class
class IsingModel(IHamiltonian):
    # H = np.sum(h_i Z_i) + np.sum(J_{i,j} Z_i Z_j)

    def __init__(self, grid_size: int):
        qubits = cirq.GridQubit.square(grid_size)  # Determine qubits to be used to represent Hamiltonian
        parameters = IParameter({'alpha': 1, 'beta': 1, 'gamma': 1})  # Determine tweakable parameters
        super().__init__(qubits, parameters)
        self._name = 'Randomized Ising-model Hamiltonian'

        self.x_half_turns, self.h_half_turns, self.j_half_turns = self.get_symbol_parameters()

        self.h, self.jr, self.jc = IsingModel.hamiltonian_coupling_constants(grid_size)
        self.length = len(self.h)

    def observable(self, w: IWaveFunction) -> [cirq.ops.moment]:
        # yield IsingModel.rot_x_layer(self.length, 1)
        yield IsingModel.rot_x_layer(self.length, self.x_half_turns)
        yield IsingModel.rot_z_layer(self.h, self.h_half_turns)
        yield IsingModel.rot_11_layer(self.jr, self.jc, self.j_half_turns)

    # Randomizing h and J weights
    @staticmethod
    def rand2d(rows, cols) -> [[float]]:
        return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def value2d(value: float, rows: int, cols: int) -> [[float]]:
        return [[value for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def hamiltonian_coupling_constants(diameter):
        # transverse field terms
        h = IsingModel.rand2d(diameter, diameter)
        # links within a row
        jr = IsingModel.rand2d(diameter - 1, diameter)
        # links within a column
        jc = IsingModel.rand2d(diameter, diameter - 1)
        return (h, jr, jc)

    @staticmethod
    def rot_x_layer(length, half_turns):
        """Yields X rotations by half_turns on a square grid of given length."""

        # Define the gate once and then re-use it for each Operation.
        rot = cirq.XPowGate(exponent=half_turns)

        # Create an X rotation Operation for each qubit in the grid.
        for i in range(length):
            for j in range(length):
                yield rot(cirq.GridQubit(i, j))

    @staticmethod
    def rot_z_layer(h, half_turns):
        """Yields Z rotations by half_turns conditioned on the field h."""
        gate = cirq.ZPowGate(exponent=half_turns)
        for i, h_row in enumerate(h):
            for j, h_ij in enumerate(h_row):
                if h_ij == 1:
                    yield gate(cirq.GridQubit(i, j))

    @staticmethod
    def rot_11_layer(jr, jc, half_turns):
        """Yields rotations about |11> conditioned on the jr and jc fields."""
        cz_gate = cirq.CZPowGate(exponent=half_turns)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                q = cirq.GridQubit(i, j)
                q_1 = cirq.GridQubit(i + 1, j)
                if jr_ij == -1:
                    yield cirq.X(q)
                    yield cirq.X(q_1)
                yield cz_gate(q, q_1)
                if jr_ij == -1:
                    yield cirq.X(q)
                    yield cirq.X(q_1)

        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                q = cirq.GridQubit(i, j)
                q_1 = cirq.GridQubit(i, j + 1)
                if jc_ij == -1:
                    yield cirq.X(q)
                    yield cirq.X(q_1)
                yield cz_gate(q, q_1)
                if jc_ij == -1:
                    yield cirq.X(q)
                    yield cirq.X(q_1)

    def energy_func(self, measurements: np.ndarray) -> T:
        # Reshape circuit (qubit) measurement into array that matches grid shape.
        pm_meas = [measurements[i * self.length:(i + 1) * self.length] for i in range(self.length)]
        pm_meas = IHamiltonian.bool_to_spin(pm_meas)  # Convert true/false (0/1) to (+1/-1).
        # print('Qubit spin measurement')
        # print(pm_meas)

        tot_energy = np.sum(pm_meas * self.h)
        for i, jr_row in enumerate(self.jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
        for i, jc_row in enumerate(self.jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]
        return tot_energy


class Model(IHamiltonian):

    def __init__(self, grid_size: int):
        qubits = cirq.GridQubit.square(grid_size)  # Determine qubits to be used to represent Hamiltonian
        parameters = IParameter({'alpha': 1})  # Determine tweakable parameters
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
    # # Change parameters and update circuit?
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
