from src.data_containers.helper_interfaces.i_hamiltonian import IHamiltonian, T
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter
import cirq
import random
import numpy as np


# Temporary test class from the cirq documentation:
# https://cirq.readthedocs.io/en/stable/tutorials/variational_algorithm.html
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