import os
import numpy as np
import cocoex
from src.utilities.tools import save_data
from src.config.config import FES_MAX, NUM_RUN, POP_SIZE, SIGMA_0
from src.environment.vanilla_es import ES


class OneFifthES(ES):
    def __init__(self, dim, sigma_0=0.5, lambda_=None, target_success_rate=0.2, success_factor=1.5):
        super().__init__(dim, sigma_0, lambda_)  # Initialize the base ES class
        self.target_success_rate = target_success_rate
        self.success_factor = success_factor
        self.success_history = []

    def tell(self, arx, fitnesses):
        # First, call the base class's tell method
        super().tell(arx, fitnesses)

        # Then, implement the one-fifth rule specific logic
        sorted_idx = np.argsort(fitnesses)
        successful = np.sum(sorted_idx[:self.mu] < self.lambda_ // 5)
        self.success_history.append(successful / self.mu > self.target_success_rate)

        if len(self.success_history) > 5:  # Check history length to adjust step size
            success_rate = np.mean(self.success_history)
            if success_rate > self.target_success_rate:
                self.sigma *= self.success_factor
            elif success_rate < self.target_success_rate:
                self.sigma /= self.success_factor
            self.success_history = []  # Reset history after adjustment


def test_one_fifth_es(results_dir, problem_type, test_problem_dimension, problemIndex, instance):
    suite = cocoex.Suite(problem_type, "",
                         f"dimensions: {test_problem_dimension} function_indices:1-24 instance_indices:{instance}")
    problem = suite.get_problem(problemIndex - 1)

    baselines_dir = os.path.join(results_dir, f'baselines', f'DIM_{test_problem_dimension}')
    os.makedirs(baselines_dir, exist_ok=True)

    all_fitness_values = []

    for _ in range(NUM_RUN):
        es = OneFifthES(test_problem_dimension, sigma_0=SIGMA_0, lambda_=POP_SIZE)
        lst_fitness_values = []
        fes = 0
        current_best_fitness = None

        while fes <= FES_MAX:
            solutions = es.ask()
            fitness_values = np.apply_along_axis(problem, 1, solutions)
            es.tell(solutions, fitness_values)

            current_generation_best_fitness = np.min(fitness_values)
            if current_best_fitness is not None:
                current_best_fitness = min(current_best_fitness, current_generation_best_fitness)
            else:
                current_best_fitness = current_generation_best_fitness

            fes += es.lambda_
            lst_fitness_values.append(current_best_fitness)

        all_fitness_values.append(lst_fitness_values)

    all_fitness_values_arr = np.array(all_fitness_values)
    save_data(os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problemIndex}_instance_{instance}.npy'), all_fitness_values_arr)
