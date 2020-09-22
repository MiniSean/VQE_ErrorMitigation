import cirq
import openfermion as of
from typing import Sequence, List, Callable

from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.processors.processor_quantum import QPU
from src.circuit_noise_extension import Noisify


class INoiseModel:
    def get_callable(self) -> Callable[[], List[cirq.Gate]]:
        return lambda: self._noise_gates

    def get_description(self) -> str:
        return self._description

    def __init__(self, noise_gates: List[cirq.Gate], description: str):
        self._noise_gates = noise_gates
        self._description = description


class INoiseWrapper(IWaveFunction, cirq.NoiseModel):

    # Implement abstract functions
    def _generate_qubits(self) -> Sequence[cirq.Qid]:
        return self._ideal_wave_function._generate_qubits()

    def operations(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_type(self._ideal_wave_function.operations(qubits=qubits), self._noise_channel)

    def initial_state(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        yield Noisify.introduce_noise_type(self._ideal_wave_function.initial_state(qubits=qubits), self._noise_channel)

    def _generate_molecule(self, p: IParameter) -> of.MolecularData:
        return self._ideal_wave_function._generate_molecule(p=p)

    def __init__(self, w_class: IWaveFunction, noise_channel: INoiseModel):  # List[cirq.Gate]
        """
        Noise Wrapper Constructor.
        Extents functionality of IWaveFunction class by introducing noise channels on both initial state preparation as ansatz operations.
        :param w_class: any class that inherits directly or indirectly from IWaveFunction
        :param noise_channel: List of cirq noise channel(s)
        """
        self._ideal_wave_function = w_class
        self._noise_channel = noise_channel.get_callable()  # Callable[[], List[cirq.Gate]]
        IWaveFunction.__init__(self, w_class.operator_parameters, w_class.molecule_parameters)
        cirq.NoiseModel.__init__(self)

    # cirq.NoiseModel
    def noisy_operation(self, op):
        result = [op]
        for gates in self._noise_channel():
            result.append(gates.on(op.qubits[0]))
        return result

    # quick access
    def get_clean_circuit(self) -> cirq.circuits.circuit:
        result = QPU.get_initial_state_circuit(w=self._ideal_wave_function)  # Initial state
        result.append(self._ideal_wave_function.operations(self._ideal_wave_function.qubits))  # Ansatz operators
        return result

    def get_noisy_circuit(self) -> cirq.circuits.circuit:
        result = QPU.get_initial_state_circuit(w=self)  # Initial state
        result.append(self.operations(self._ideal_wave_function.qubits))  # Ansatz operators
        return result
