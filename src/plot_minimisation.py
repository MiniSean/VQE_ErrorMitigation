import jsonpickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Subplot
from typing import List, Tuple
from src.calculate_minimisation import DATA_DIR, check_dir
from src.data_containers.helper_interfaces.i_collection import IContainer


def import_file(rel_dir: str) -> List[IContainer]:
    """Checks if file path exists. If so, un-pickle json data and return object"""
    if check_dir(rel_dir):
        # Read JSON object
        with open(rel_dir, 'r') as rf:
            obj = jsonpickle.decode(rf.read())
    else:
        raise FileNotFoundError()
    return obj


def get_plot(plot_obj: plt, data: List[IContainer]) -> (Figure, Subplot):
    """Returns plt object"""
    # Data
    labeled_dict = {}
    for collection in data:
        if collection.label not in labeled_dict:
            labeled_dict[collection.label] = list()
        labeled_dict[collection.label].append(collection)
    fig, ax = plot_obj.subplots()
    # Set plot layout
    ax.title.set_text("Expectation value for different noise models and molecule parameters")  # Title "Eigen Energy depending on Noise Channel"
    ax.set_xlabel("Interatomic Distance [$\AA$]")  # X axis label
    ax.set_ylabel("Energy (Hartree) [$a.u.$]")  # Y axis label

    # Set plot points
    reference_key = None
    for key in labeled_dict.keys():
        reference_key = key
        x = [collection.molecule_param for collection in labeled_dict[key]]
        y = [collection.measured_value for collection in labeled_dict[key]]
        e = [collection.measured_std for collection in labeled_dict[key]]
        ax.errorbar(x, y, yerr=e, linestyle='None', marker='^', label=key)  # "STO-3G"

    x = [collection.molecule_param for collection in labeled_dict[reference_key]]
    f = [collection.fci_value for collection in labeled_dict[reference_key]]
    h = [collection.hf_value for collection in labeled_dict[reference_key]]
    ax.plot(x, f, 'o', label="fci energy")
    ax.plot(x, h, 'o', label="HF energy")
    ax.legend(loc=0)
    return fig, ax


def plot_with_inset(plot_obj: plt, data: List[IContainer], inset_region: Tuple[float, float, float, float]) -> plt:
    """Returns plt object with inset"""
    # Data
    x = [collection.molecule_param for collection in data]
    y = [collection.measured_value for collection in data]
    z = [collection.fci_value for collection in data]
    # Clean plot
    fig, ax = get_plot(plot_obj=plot_obj, data=data)
    # inset axes....
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.27])
    axins.plot(x, y, 'o')
    axins.plot(x, z, 'o')

    # sub region of the original image
    x1, x2, y1, y2 = inset_region
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    ax.indicate_inset_zoom(axins)
    return plot_obj


def read_and_plot(filename: str) -> plt:
    """Imports from file path and returns a matplotlib-pyplot object"""
    # Read stored json format and express as a good old matlab plot
    file_path = f'{DATA_DIR}/{filename}'
    try:
        container_data = import_file(file_path)
    except FileNotFoundError:
        raise Exception(f'Indicated file path does not exist: {file_path}')
    # Matplot magic
    plot_obj = plt
    f, a = get_plot(plot_obj=plot_obj, data=container_data)
    # plot_obj = plot_with_inset(plot_obj=plot_obj, data=container_data, inset_region=(.65, .85, -1.14, -1.125))
    return plot_obj


if __name__ == '__main__':
    container_data_005 = import_file(f'{DATA_DIR}/H2_temptest1')
    container_data_010 = import_file(f'{DATA_DIR}/H2_temptest2')
    container_data_015 = import_file(f'{DATA_DIR}/H2_temptest3')
    container_data_020 = import_file(f'{DATA_DIR}/H2_temptest4')

    # Data
    x = [collection.molecule_param for collection in container_data_005]
    y1 = [collection.measured_value for collection in container_data_005]
    e1 = [collection.measured_std for collection in container_data_005]
    z = [collection.fci_value for collection in container_data_005]
    h = [float('NaN') for collection in container_data_005]

    y2 = [collection.measured_value for collection in container_data_010]
    e2 = [collection.measured_std for collection in container_data_010]

    y3 = [collection.measured_value for collection in container_data_015]
    e3 = [collection.measured_std for collection in container_data_015]

    y4 = [collection.measured_value for collection in container_data_020]
    e4 = [collection.measured_std for collection in container_data_020]

    fig, ax = plt.subplots()
    # Set plot layout
    ax.title.set_text("Depolarizing noise at different probabilities")  # Title "Eigen Energy depending on Noise Channel"
    ax.set_xlabel("Interatomic Distance [$\AA$]")  # X axis label
    ax.set_ylabel("Energy (Hartree) [$a.u.$]")  # Y axis label
    # Set plot points
    ax.errorbar(x, y1, yerr=e1, linestyle='None', marker='^', label="p = 0.05")
    ax.errorbar(x, y2, yerr=e2, linestyle='None', marker='^', label="p = 0.10")
    ax.errorbar(x, y3, yerr=e3, linestyle='None', marker='^', label="p = 0.15")
    ax.errorbar(x, y4, yerr=e4, linestyle='None', marker='^', label="p = 0.20")
    ax.plot(x, z, 'o', label="fci energy")
    ax.plot(x, h, 'o', label="HF energy")
    ax.legend(loc=0)
    plt.show()

    # plt_obj = read_and_plot(filename='H2_bitflip_005')
    # plt_obj.show()
