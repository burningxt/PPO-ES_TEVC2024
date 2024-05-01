import os
import numpy as np


def find_project_root(current_dir, anchor):
    """
    Find the project root directory by looking for a specific anchor file or directory.
    """
    while not os.path.exists(os.path.join(current_dir, anchor)) and current_dir != os.path.dirname(current_dir):
        current_dir = os.path.dirname(current_dir)
    return current_dir

def linear_schedule(initial_value, min_value=1e-4):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :param min_value: (float) Minimum learning rate.
    :return: (function) Schedule function that takes the current progress (from 1 to 0) and returns the learning rate.
    """
    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0 (end).
        The learning rate will linearly decrease to the min_value.
        """
        return max(min_value, progress * (initial_value - min_value) + min_value)

    return func


def save_data(file_path, data):
    np.save(file_path, data)


def load_data(file_path):
    return np.load(file_path)


def cmaes_min(baselines_dir, instance):
    optimums = []
    for problem_index in range(1, 25):
        global_min_fitness = float('inf')  # Initialize to a very high value
        # Load and calculate convergence rates for CMA-ES
        fitness_values_cma_es = load_data(
            os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        if fitness_values_cma_es.size > 0:
            global_min_fitness = min(global_min_fitness, np.min(fitness_values_cma_es))
        optimums.append(global_min_fitness)
    return optimums


def cmaes_data(baselines_dir, instance):
    data = []
    for problem_index in range(1, 25):
        # Load and calculate convergence rates for CMA-ES
        fitness_values_cma_es = load_data(
            os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        data.append(np.min(fitness_values_cma_es, axis=0))
    return np.array(data)

