import os
import json
import jsonpickle  # TEMP
import matplotlib.pyplot as plt  # TEMP

from src.data_containers.model_hydrogen import HydrogenAnsatz, NoisyHydrogen
from src.processors.processor_quantum import QPU
from src.processors.processor_classic import CPU
from src.circuit_noise_extension import Noisify


if __name__ == '__main__':

    print("H2-Molecule example. Finding expectation value for molecule with optimal atom distance (.7414 angstrom)")
    print("Using a generated operator tree based on the UCCSD theorem")
    uccsd_ansatz = HydrogenAnsatz()
    parameters = uccsd_ansatz.operator_parameters

    # Get resolved circuit
    print('Show circuit with initial state preparation:')
    circuit = QPU.get_initial_state_circuit(uccsd_ansatz)
    circuit.append(uccsd_ansatz.circuit)
    print(circuit)

    # Get variational study
    result = CPU.get_optimized_state(w=uccsd_ansatz, max_iter=1000)
    print(f'Operator expectation value: {result.optimal_value}\nOperator parameters: {result.optimal_parameters}')

    parameters.update(r=result)
    resolved_circuit = QPU.get_resolved_circuit(circuit, parameters)
    print(resolved_circuit)

    # --------------------------

    # # First test in applying noise to circuits or molecule classes directly
    # print("\nTaking the existing circuit and apply a noise filter:")
    # # Noisy circuit
    # clean_hydrogen = HydrogenAnsatz()
    # noise_circuit = QPU.get_initial_state_circuit(clean_hydrogen)  # Append state initialisation
    # noise_circuit.append(clean_hydrogen.circuit)  # Append operations
    # noise_circuit = Noisify.introduce_noise(noise_circuit)  # 'Noisify'
    # print(noise_circuit)
    #
    # print("\nIntroduce noise directly into the initial state and operator preparations of the Hydrogen model:")
    # # Noisy initial state
    # noisy_hydrogen = NoisyHydrogen()
    # noise_circuit = cirq.Circuit(noisy_hydrogen.initial_state(noisy_hydrogen.qubits))
    # noise_circuit.append(noisy_hydrogen.operations(noisy_hydrogen.qubits))
    # print(noise_circuit)

    # --------------------------

    # Run 'optimization and store data in a json format
    data_directory = os.getcwd() + '/classic_minimisation'
    filename = data_directory + '/H2_semi_minimisation'
    # container_list = CPU.get_semi_optimized_ground_state(w=uccsd_ansatz, qpu_iter=10)
    # # Temporarily store results
    # if not os.path.exists(data_directory):  # If non existing -> create
    #     os.mkdir(data_directory)
    # # Something like pickle or json or jsonpickle?
    # # Writing JSON object
    # with open(filename, 'w') as wf:
    #     # json.dump(container_list.__dict__, f, indent=4)
    #     wf.write(jsonpickle.encode(value=container_list, indent=4))

    # --------------------------

    # Read stored json format and express as a good old matlab plot
    # Read JSON object
    with open(filename, 'r') as rf:
        obj = jsonpickle.decode(rf.read())

    x = [collection.molecule_param for collection in obj]
    y = [collection.measured_value for collection in obj]
    z = [collection.fci_value for collection in obj]

    fig, ax = plt.subplots()
    # Set plot layout
    ax.title.set_text("Eigen Energy depending on Atomic distance for H2")
    ax.set_xlabel("Interatomic Distance [$\AA$]")
    ax.set_ylabel("Energy (Hartree) [$a.u.$]")
    # Set plot points
    ax.plot(x, y, '-', label="STO-3G")
    ax.plot(x, z, 'o', label="fci energy")
    ax.legend()

    plt.show()
    # --------------------------

