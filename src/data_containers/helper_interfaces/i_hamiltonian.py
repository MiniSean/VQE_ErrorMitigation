from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_parameter import IParameter
from abc import abstractmethod
from typing import (Callable, Tuple, TypeVar)
import numpy as np
import cirq
import sympy
T = TypeVar('T')


class IHamiltonian:

    def get_init_parameters(self) -> IParameter:
        return self._par

    def get_symbol_parameters(self) -> [sympy.core.symbol]:
        return [sympy.Symbol(key) for key in self.parameters]

    parameters = property(get_init_parameters)

    def __init__(self, q: [cirq.GridQubit], p: IParameter):
        self.qubits = q  # Qubits, used to represent Hamiltonian
        self._par = p  # Initial parameter set, used for optimizing
        self._name = 'Arbitrary Hamiltonian'  # For string representation

    @abstractmethod
    def observable(self, w: IWaveFunction) -> [cirq.ops.moment]:
        """
        Combines the hamiltonian operators with the (trial) wave function.
        Maps these operator-expectation values to a list of circuit operations.
        :param w: (Trial) Wave function
        :return: List of quantum circuit gates representing <phi| H |phi>
        """
        yield NotImplemented

    @abstractmethod
    def energy_func(self, measurements: np.ndarray) -> float:  # Callable[[Tuple], T]
        """
        Maps individual qubit measurements (0/1) (true/false) to a single energy value
        :param measurements: Array of qubit projection measurement (0/1)
        :return: A callable function with single input (measurements) to retrieve an energy value
        """
        yield NotImplemented

    @staticmethod
    def bool_to_spin(measurements: np.ndarray) -> np.ndarray:
        """
        Converts true/false measurements (0/1) to (+1/-1) spin values.
        :param measurements: Array of qubit projection measurement (0/1)
        :return: Array of qubit spin values (+1/-1)
        """
        return 1 - 2 * np.array(measurements).astype(np.int32)

    @staticmethod
    def expectation_value(result: cirq.study.trial_result, energy_function: Callable[[Tuple], T] = cirq.value.big_endian_bits_to_int):
        energy_hist = result.histogram(key='x', fold_func=energy_function)
        return np.sum([k * v for k, v in energy_hist.items()]) / result.repetitions

    def __str__(self):
        return self._name
