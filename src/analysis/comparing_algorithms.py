from src.models.ppo_es_model import PPO_ES
from src.models.test_cma_es import test_cma_es
from src.models.test_one_fifth_es import test_one_fifth_es
from src.utilities.plotting import Draw
from src.utilities.friedman import perform_friedman_test
from src.utilities.tools import find_project_root
import os


def comparing_algorithms(need_train=False,
                         test_problem_type='bbob',
                         test_instance=1,
                         test_dimension=40,
                         need_test_models=False,
                         need_test_cma_es=False,
                         need_test_one_fifth_es=False,
                         cuda_device='cuda:0'):
    base_dir = find_project_root(os.path.dirname(os.path.abspath(__file__)), 'run.py')
    results_dir = os.path.join(base_dir, 'output_data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(results_dir, 'plots', f'DIM_{test_dimension}', f'instance_{test_instance}')
    os.makedirs(plot_dir, exist_ok=True)

    # Create new directories for each problem index
    episodes_tested_dir = os.path.join(results_dir, f'episodes_tested', f'DIM_{test_dimension}')
    baselines_dir = os.path.join(results_dir, f'baselines', f'DIM_{test_dimension}')
    baselines_dir_train = os.path.join(results_dir, f'baselines', f'DIM_{40}')
    os.makedirs(episodes_tested_dir, exist_ok=True)
    os.makedirs(baselines_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    for problemIndex in range(1, 25):
        if need_test_cma_es:
            test_cma_es(results_dir, test_problem_type, test_dimension, problemIndex, test_instance)
        if need_test_one_fifth_es:
            test_one_fifth_es(results_dir, test_problem_type, test_dimension, problemIndex, test_instance)

    ppo_es = PPO_ES(base_dir=base_dir, cuda_device=cuda_device)
    if need_train:
        ppo_es.train_ppo_es()

    for problemIndex in range(1, 25):
        if need_test_models:
            ppo_es.test_ppo_es(test_problem_type, test_dimension, problemIndex, test_instance)

        # plot convergence figure
        Draw().plot_convergence_data_mean_ci(problemIndex, episodes_tested_dir, baselines_dir, plot_dir, test_instance)
        Draw().plot_convergence_data(problemIndex, episodes_tested_dir, baselines_dir, plot_dir, test_instance)

    Draw().plot_radar_chart(episodes_tested_dir, baselines_dir, plot_dir, test_instance)
    # plot overall box figure
    # Draw().plot_standardized_performance_boxplot(episodes_tested_dir, baselines_dir, plot_dir, test_instance)

    # generate Friedman test table
    perform_friedman_test(episodes_tested_dir, baselines_dir, plot_dir, test_instance)


