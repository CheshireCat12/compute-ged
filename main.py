import argparse
import json
import re
from os.path import isfile
from os.path import join
from pathlib import Path
from typing import Tuple
from time import time

import numpy as np
from cyged import Coordinator
from cyged import MatrixDistances
from cyged import load_graphs

NODE_ATTRIBUTE = 'x'

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

GRAPH_FORMAT = 'graphml'


def write_distances(filename: str, distances: np.ndarray) -> None:
    """
    Save the GEDs in `.npy` file

    Args:
        filename: File where to save the GEDs.
        distances: `np.array` containing the GEDs

    Returns:

    """
    with open(filename, 'wb') as file:
        np.save(file, distances)


def load_distances(coordinator: Coordinator,
                   alpha: float,
                   n_cores: int,
                   folder_distances: str) -> np.ndarray:
    """
    The function loads or computes the GED matrices for each alpha value in the given list of alphas.
    If the GED matrices for a particular alpha value already exist in the specified folder, it loads the matrices from there.
    Otherwise, it computes the GED matrices using the MatrixDistances class and the coordinator.graphs attribute.
    The matrices are then stored in the specified folder for future use.

    Args:
        coordinator:
        alpha: A list of floats representing the alpha values for which the GED matrices need
                to be computed or loaded.
        n_cores: An integer representing the number of cores to be used for parallel computation of GED matrices
        folder_distances: A string representing the path of the folder where the GED matrices
                        will be stored or loaded from

    Returns:
        A list of numpy arrays representing the GED matrices for all the alpha values.
    """
    is_parallel = n_cores > 0
    matrix_dist = MatrixDistances(coordinator.ged,
                                  parallel=is_parallel)

    file_distances = join(folder_distances,
                          f'distances_alpha_{alpha}.npy')

    # Check if the file containing the distances for the particular alpha exists
    if isfile(file_distances):
        # If yes load the distances
        dist = np.load(file_distances)
    else:
        # Otherwise compute the GEDs
        coordinator.edit_cost.update_alpha(alpha)

        t0 = time()
        dist = np.array(matrix_dist.calc_matrix_distances(coordinator.graphs,
                                                          coordinator.graphs,
                                                          num_cores=n_cores))
        t1 = time()
        computation_time = t1 - t0

        with open(join(folder_distances, f'computation_time_n_cores_{n_cores}.txt'), 'w') as file:
            file.write(str(computation_time))

    write_distances(file_distances, dist)

    return dist


def compute_distances_labels(root_dataset: str,
                             graph_format: str,
                             parameters_edit_cost: Tuple[float, float, float, float, str],
                             alpha: float,
                             n_cores: int,
                             folder_distances: str) -> np.ndarray:
    graphs, labels = load_graphs(root_dataset=root_dataset,
                                 file_extension=graph_format,
                                 load_classes=True)
    coordinator = Coordinator(parameters_edit_cost,
                              graphs=graphs,
                              classes=labels)
    distance_matrix = load_distances(coordinator, alpha, n_cores, folder_distances)

    return distance_matrix


def get_dataset(root_dataset: str) -> str:
    root = root_dataset[:-1] if root_dataset.endswith('/') else root_dataset
    split_root = root.split('/')
    dataset_name = split_root[2]

    return dataset_name


def make_folder_results(root_dataset: str) -> str:
    dataset_name = get_dataset(root_dataset)

    folder_results = join('./results', dataset_name)
    Path(folder_results).mkdir(parents=True, exist_ok=True)

    return folder_results


def main(root_dataset: str,
         graph_format: str,
         alpha: float,
         n_cores: int):
    folder_results = make_folder_results(root_dataset)

    parameters_edit_cost = (1., 1., 1., 1., 'euclidean')

    matrix_distance = compute_distances_labels(root_dataset=root_dataset,
                                               graph_format=graph_format,
                                               parameters_edit_cost=parameters_edit_cost,
                                               alpha=alpha,
                                               n_cores=n_cores,
                                               folder_distances=folder_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Classification Using KNN with GED')
    subparser = parser.add_subparsers()

    parser.add_argument('--root-dataset',
                        type=str,
                        required=True,
                        help='Root of the dataset')
    parser.add_argument('--graph-format',
                        type=str,
                        default='graphml',
                        help='Root of the dataset')

    # Hyperparameters to test
    parser.add_argument('--alpha',
                        # nargs='*',
                        # default=ALPHAS,
                        type=float,
                        help='Alpha parameter of GED')

    parser.add_argument('--n-cores',
                        default=0,
                        type=int,
                        help='Set the number of cores to use.'
                             'If n_cores == 0 then it is run without parallelization.'
                             'If n_cores > 0 then use this number of cores')

    parse_args = parser.parse_args()

    main(**vars(parse_args))