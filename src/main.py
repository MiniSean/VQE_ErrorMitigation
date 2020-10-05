import cirq

from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper, INoiseModel
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get resolved circuit
    print('Show circuit with initial state preparation:')
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append(uccsd_ansatz.circuit)
    # noise_channel = INoiseModel(noise_gates=[cirq.depolarize(p=.5)], description=f'Depolarize (p={0.5})')
    # circuit = Noisify.introduce_noise(circuit, noise_channel.get_callable())
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # Get custom optimization
    noise_channel = INoiseModel(noise_gates_1q=[cirq.bit_flip(p=.5), cirq.phase_flip(p=.5)], noise_gates_2q=[], description=f'Bit and Phase flip (p={0.5})')  # [cirq.AmplitudeDampingChannel(gamma=.1)]
    noisy_ansatz = INoiseWrapper(uccsd_ansatz, noise_channel)
    values, params = CPU.get_custom_optimized_state(n_w=noisy_ansatz, max_iter=10)
    print(f'Operator expectation value: {values}\nOperator parameters: {params}')

    for i, key in enumerate(parameters.dict.keys()):  # Dirty set parameter values
        parameters[key] = params[i]
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)
