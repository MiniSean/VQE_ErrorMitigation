import cirq
import openfermion as of
from typing import Sequence, List

from src.data_containers.helper_interfaces.i_parameter import IParameter
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.circuit_noise_extension import Noisify


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

    def __init__(self, w_class: IWaveFunction, noise_channel: List[cirq.Gate]):
        """
        Noise Wrapper Constructor.
        Extents functionality of IWaveFunction class by introducing noise channels on both initial state preparation as ansatz operations.
        :param w_class: any class that inherits directly or indirectly from IWaveFunction
        :param noise_channel: List of cirq noise channel(s)
        """
        self._ideal_wave_function = w_class
        self._noise_channel = lambda: noise_channel  # Callable[[], List[cirq.Gate]]
        IWaveFunction.__init__(self, w_class.operator_parameters, w_class.molecule_parameters)
        cirq.NoiseModel.__init__(self)

    def noisy_operation(self, op):
        result = [op]
        for gates in self._noise_channel():
            result.append(gates.on(op.qubits[0]))
        return result