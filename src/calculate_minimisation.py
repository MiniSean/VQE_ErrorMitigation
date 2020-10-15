import os
import jsonpickle
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.helper_interfaces.i_collection import IMeasurementCollection
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_classic import CPU

DATA_DIR = os.getcwd() + '/classic_minimisation'
QPU_ITER = 10
CPU_ITER = 10  # 20


def check_dir(rel_dir: str) -> bool:
    """Checks if relative directory exists"""
    return os.path.exists(rel_dir)


def create_dir(rel_dir: str) -> bool:
    """Checks if relative directory exists. If not, create file path"""
    result = check_dir(rel_dir)
    if not result:
        os.mkdir(rel_dir)
    return result


# # Deprecated
# def calculate_and_write(wave_class: IWaveFunction, filename: str, **kwargs):
#     container_obj = CPU.get_specific_ground_states(p_space=kwargs['p_space'], w=wave_class, cpu_iter=CPU_ITER, qpu_iter=QPU_ITER)
#
#     # Temporarily store results
#     create_dir(DATA_DIR)  # If non existing -> create
#     file_path = f'{DATA_DIR}/{filename}'
#     # Writing JSON object
#     with open(file_path, 'w') as wf:
#         wf.write(jsonpickle.encode(value=container_obj, indent=4))


def calculate_and_write_collection(collection: IMeasurementCollection, filename: str):
    # Populate IMeasurementCollection
    collection.container = CPU.get_collection_ground_states(collection=collection, cpu_iter=CPU_ITER, qpu_iter=QPU_ITER)
    container_obj = collection.container

    # Temporarily store results
    create_dir(DATA_DIR)  # If non existing -> create
    file_path = f'{DATA_DIR}/{filename}'
    # Writing JSON object
    with open(file_path, 'w') as wf:
        wf.write(jsonpickle.encode(value=container_obj, indent=4))


def write_object(obj: object, filename: str):
    create_dir(DATA_DIR)  # If non existing -> create
    file_path = f'{DATA_DIR}/{filename}'
    # Writing JSON object
    with open(file_path, 'w') as wf:
        wf.write(jsonpickle.encode(value=obj, indent=4))


def read_object(filename: str) -> object:
    # Read stored json format and express as a good old matlab plot
    file_path = f'{DATA_DIR}/{filename}'
    try:
        if check_dir(file_path):
            # Read JSON object
            with open(file_path, 'r') as rf:
                obj = jsonpickle.decode(rf.read())
        else:
            raise FileNotFoundError()
        return obj
    except FileNotFoundError:
        raise Exception(f'Indicated file path does not exist: {file_path}')


if __name__ == '__main__':
    import cirq
    import numpy as np
    from src.error_mitigation import SingleQubitPauliChannel, TwoQubitPauliChannel
    from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseModel
    from src.plot_minimisation import read_and_plot
    # Calculate noise models near ground state energy
    clean_ansatz = HydrogenAnsatz()
    filename = 'H2_temptest6'
    parameter_space = [.7414]  # np.round(np.linspace(0.1, 3.0, 15), 1)  # [.6, .7, .7414, .8, .9]
    noise_space = [INoiseModel(noise_gates_1q=[SingleQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)], noise_gates_2q=[TwoQubitPauliChannel(p_x=p, p_y=p, p_z=6 * p)], description=f'asymmetric depolarization (p_tot={16 * p})') for p in [0.000, 1e-4]]
    measure_collection = IMeasurementCollection(w=clean_ansatz, p_space=parameter_space, n_space=noise_space)
    calculate_and_write_collection(collection=measure_collection, filename=filename)

    # calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, [cirq.depolarize(p=0.001)]), filename=filename, p_space=parameter_space)  # cirq.bit_flip(p=.05)
    plt_obj = read_and_plot(filename=filename)
    plt_obj.show()
