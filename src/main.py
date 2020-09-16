import cirq  # TEMP

from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.calculate_minimisation import calculate_and_write
from src.plot_minimisation import read_and_plot
from src.circuit_noise_extension import Noisify


if __name__ == '__main__':

    # print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    # print("Using a generated operator tree based on the UCCSD theorem")
    # uccsd_ansatz = HydrogenAnsatz()
    # parameters = uccsd_ansatz.operator_parameters
    #
    # # Get resolved circuit
    # print('Show circuit with initial state preparation:')
    # circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    # circuit.append(uccsd_ansatz.circuit)
    # print(circuit)
    #
    # # Get variational study
    # result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    # print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')
    #
    # parameters.update(r=result)
    # resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    # print(resolved_circuit)
    #
    # # Get custom optimization
    # noise_channel = [cirq.bit_flip(p=.5), cirq.phase_flip(p=.5)]  # [cirq.AmplitudeDampingChannel(gamma=.1)]
    # noisy_ansatz = INoiseWrapper(uccsd_ansatz, noise_channel)
    # values, params = CPU.get_custom_optimized_state(n_w=noisy_ansatz, cpu_iter=10)
    # print(f'Operator expectation value: {values}\nOperator parameters: {params}')

    # for i, key in enumerate(parameters.dict.keys()):  # Dirty set parameter values
    #     parameters[key] = params[i]
    # resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    # print(resolved_circuit)

    # --------------------------

    # # First test in applying noise to circuits or molecule classes directly
    # print("\nTaking the existing circuit and apply a noise filter:")
    # clean_hydrogen = HydrogenAnsatz()
    # clean_circuit = QPU.get_initial_state_circuit(clean_hydrogen)  # Append state initialisation
    # clean_circuit.append(clean_hydrogen.circuit)  # Append operations
    # # Test random noise models
    # # random_noise_channel = lambda: Noisify.uniform_rotation_gate(1000)
    # def phase_flip_channel(): return [cirq.phase_flip(p=.5)]
    # noise_circuit = Noisify.introduce_noise(clean_circuit, phase_flip_channel)  # 'Noisify'
    # print(noise_circuit)
    #
    # # Test noise wrapper
    # print("\nIntroduce noise directly by applying noise wrapper with specified noise channel around the Hydrogen model:")
    # noise_channel = [cirq.bit_flip(p=.1), cirq.phase_flip(p=.1)]  # Example noise channel
    # noisy_hydrogen = NoiseWrapper(clean_hydrogen, noise_channel)
    # noise_circuit = QPU.get_initial_state_circuit(noisy_hydrogen)  # Append state initialisation
    # noise_circuit.append(noisy_hydrogen.circuit)  # Append operations
    # print(noise_circuit)

    # Perform minimisation calculation with noise channel
    noise_channel = [cirq.bit_flip(p=.1), cirq.phase_flip(p=.1)]  # Example noise channel
    noise_channel2 = [cirq.AmplitudeDampingChannel(gamma=.1)]  # Example noise channel
    filename = 'H2_noisy_semi_minimisation_02'
    clean_class = HydrogenAnsatz()
    noisy_class = INoiseWrapper(clean_class, noise_channel)
    # Run optimization and store data in a json format
    calculate_and_write(wave_class=noisy_class, filename=filename)
    # Read stored json format and express as a good old matlab plot
    plt_obj = read_and_plot(filename=filename)
    plt_obj.show()

    # --------------------------

    # filename = 'H2_semi_minimisation'
    # noisy_class = HydrogenAnsatz()
    # # Run optimization and store data in a json format
    # calculate_and_store(wave_class=noisy_class, filename=filename)
    # # Read stored json format and express as a good old matlab plot
    # plt_obj = read_and_plot(filename=filename)
    # plt_obj.show()

    # --------------------------

