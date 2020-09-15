import os
import jsonpickle
from src.data_containers.helper_interfaces.i_wave_function import IWaveFunction
from src.data_containers.model_hydrogen import HydrogenAnsatz
from src.processors.processor_classic import CPU

DATA_DIR = os.getcwd() + '/classic_minimisation'
OPT_ITER = 10


def check_dir(rel_dir: str) -> bool:
    """Checks if relative directory exists"""
    return os.path.exists(rel_dir)


def create_dir(rel_dir: str) -> bool:
    """Checks if relative directory exists. If not, create file path"""
    result = check_dir(rel_dir)
    if not result:
        os.mkdir(rel_dir)
    return result


def calculate_and_store(wave_class: IWaveFunction, filename: str):
    file_path = f'{DATA_DIR}/{filename}'
    container_list = CPU.get_semi_optimized_ground_state(w=wave_class, qpu_iter=OPT_ITER)
    # Temporarily store results
    create_dir(DATA_DIR)  # If non existing -> create
    # Writing JSON object
    with open(file_path, 'w') as wf:
        wf.write(jsonpickle.encode(value=container_list, indent=4))


if __name__ == '__main__':
    wave_function_class = HydrogenAnsatz()
    calculate_and_store(wave_class=wave_function_class, filename='H2_semi_minimisation')
