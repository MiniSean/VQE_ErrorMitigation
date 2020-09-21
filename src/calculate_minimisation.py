import os
import jsonpickle
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_classic import CPU

DATA_DIR = os.getcwd() + '/classic_minimisation'
OPT_ITER = 10
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


def calculate_and_write(wave_class: IWaveFunction, filename: str):
    file_path = f'{DATA_DIR}/{filename}'
    container_list = CPU.get_semi_optimized_ground_state(w=wave_class, cpu_iter=CPU_ITER, qpu_iter=OPT_ITER)
    # Temporarily store results
    create_dir(DATA_DIR)  # If non existing -> create
    # Writing JSON object
    with open(file_path, 'w') as wf:
        wf.write(jsonpickle.encode(value=container_list, indent=4))


if __name__ == '__main__':
    import cirq
    from src.data_containers.helper_interfaces.i_noise_wrapper import INoiseWrapper
    from src.plot_minimisation import read_and_plot
    # Calculate noise models near ground state energy
    clean_ansatz = HydrogenAnsatz()
    filename = 'H2_temptest4'
    calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, []), filename=filename)  # cirq.bit_flip(p=.05)
    plt_obj = read_and_plot(filename=filename)
    plt_obj.show()

    # calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, [cirq.bit_flip(p=.05)]), filename='H2_bitflip_005')
    # calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, [cirq.bit_flip(p=.1)]), filename='H2_bitflip_010')
    # calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, [cirq.bit_flip(p=.15)]), filename='H2_bitflip_015')
    # calculate_and_write(wave_class=INoiseWrapper(clean_ansatz, [cirq.bit_flip(p=.2)]), filename='H2_bitflip_020')
