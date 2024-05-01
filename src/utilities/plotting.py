import numpy as np
import os
from src.config.config import POP_SIZE, EPISODES
from src.utilities.tools import load_data
from scipy.stats import sem, rankdata
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import pi


class Draw:
    @staticmethod
    def plot_loss_data(base_dir, loss_history):
        results_dir = os.path.join(base_dir, 'output_data', 'results')
        plt.plot(loss_history)
        plt.title('Loss over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.tight_layout()  # Adjust the layout
        # plt.savefig(os.path.join(results_dir, 'loss_data.png'), dpi=1200)  # Save the plot
        plt.show()  # Display the plot

    # @staticmethod
    # def plot_episode_data(base_dir, episode_data, problem_index):
    #     """
    #     Plots [episode, best_fitness_value] and [episode, total_reward] in two subplots.
    #
    #     :param episode_data: List of [episode, best_fitness_value, total_reward].
    #     :param save_path: Path to save the plot.
    #     """
    #     results_dir = os.path.join(base_dir, 'output_data', 'results')
    #     # Convert the input output_data to numpy arrays for easier handling
    #     episode_data = np.array(episode_data)
    #     episodes = episode_data[:, 0]
    #     best_fitness_values = episode_data[:, 2]
    #     total_rewards = episode_data[:, 3]
    #
    #     # Initialize the plot
    #     plt.figure(figsize=(8, 6))
    #
    #     # Subplot for best fitness value
    #     plt.subplot(2, 1, 1)
    #     plt.plot(episodes, best_fitness_values, marker='o', color='cadetblue')
    #     plt.yscale('symlog')  # Set the y-axis to a logarithmic scale
    #     plt.xlabel('Episode')
    #     plt.ylabel('Cumulative Reward')
    #     plt.title('Cumulative Reward by Best Fitness Value per Episode (Logarithmic Scale)')
    #
    #     # Subplot for total reward
    #     plt.subplot(2, 1, 2)
    #     plt.plot(episodes, total_rewards, marker='o', color='darksalmon')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Cumulative Reward')
    #     plt.title('Cumulative Reward by Convergence Rate per Episode')
    #
    #     plt.tight_layout()  # Adjust the layout
    #     plt.savefig(os.path.join(results_dir, f'episode_data_problem.pdf'), dpi=1200)  # Save the plot

    @staticmethod
    def plot_episode_data(base_dir, episode_data, problem_index):
        """
        Plots [episode, best_fitness_value] and [episode, total_reward] with two y-axes.

        :param episode_data: List of [episode, best_fitness_value, total_reward].
        :param problem_index: Index of the problem being plotted, used for labeling the plot.
        """
        results_dir = os.path.join(base_dir, 'output_data', 'results')
        os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists

        # Convert the input data to numpy arrays for easier handling
        episode_data = np.array(episode_data)
        episodes = episode_data[:, 0]
        best_fitness_values = episode_data[:, 2]
        total_rewards = episode_data[:, 3]

        # Initialize the plot
        fig, ax1 = plt.subplots(figsize=(8, 3))

        # Create a second y-axis for the total rewards

        color = '#845EC2'
        ax1.set_ylabel('Cumulative Reward (Convergence Rate)', color=color)  # we already handled the x-label with ax1
        ax1.plot(episodes, total_rewards, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Plotting best fitness values on the primary y-axis
        color = '#B0A8B9'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward (Fitness Value)', color=color)
        ax2.plot(episodes, best_fitness_values, marker='o', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_yscale('symlog')  # Set the y-axis to a logarithmic scale



        # Title and layout
        plt.title(f'Cumulative Reward by Convergence Rate and Fitness Value per Episode')
        fig.tight_layout()  # Adjust the layout to make room for the second y-axis

        # Save the plot
        plt.savefig(os.path.join(results_dir, f'episode_data.pdf'),
                    dpi=1200)  # Save the plot in high resolution
        plt.close()

    # @staticmethod
    # def plot_convergence_data_median_variance(problem_index, episodes_tested_dir, baselines_dir, plot_dir, instance):
    #     plt.figure(figsize=(10, 6))
    #     plt.title(f'Convergence Comparison Across Episodes, CMA-ES, and One-Fifth ES on Problem F{problem_index}')
    #     plt.yscale('symlog')
    #     plt.xlabel('Number of Evaluations')
    #     plt.ylabel('Best Fitness Value (log scale)')
    #     plt.grid(True)
    #
    #     # Define a list of colors for the episodes, ensure there are enough distinct colors for all elements
    #     episode_colors = ['steelblue', 'sandybrown', 'firebrick']
    #
    #     for i, episode in enumerate(EPISODES):
    #         fitness_values = load_data(
    #             os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
    #         median_fitness_values = np.mean(fitness_values, axis=0)
    #         min_fitness_values = np.min(fitness_values, axis=0)
    #         max_fitness_values = np.max(fitness_values, axis=0)
    #         evaluations = np.arange(len(median_fitness_values)) * POP_SIZE
    #         color = episode_colors[i % len(episode_colors)]  # Use distinct colors for each episode
    #         if episode == 1200:
    #             plt.plot(evaluations, median_fitness_values, label=f'Episode {episode}', color=color, linewidth=2)
    #         else:
    #             plt.plot(evaluations, median_fitness_values, label=f'Episode {episode}', color=color)
    #         plt.fill_between(evaluations, min_fitness_values, max_fitness_values, color=color, alpha=0.2)
    #
    #     # Plotting for CMA-ES
    #     fitness_values_cma_es = load_data(
    #         os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
    #     median_cma_es_fitness = np.median(fitness_values_cma_es, axis=0)
    #     max_cma_es_fitness = np.max(fitness_values_cma_es, axis=0)
    #     min_cma_es_fitness = np.min(fitness_values_cma_es, axis=0)
    #     evaluations = np.arange(len(median_cma_es_fitness)) * POP_SIZE
    #     plt.plot(evaluations, median_cma_es_fitness, label='CMA-ES', color='black', linewidth=2)
    #     plt.fill_between(evaluations, min_cma_es_fitness, max_cma_es_fitness, color='grey', alpha=0.2)
    #
    #     # Plotting for One-Fifth ES
    #     fitness_values_one_fifth_es = load_data(
    #         os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
    #     median_one_fifth_es_fitness = np.median(fitness_values_one_fifth_es, axis=0)
    #     max_one_fifth_es_fitness = np.max(fitness_values_one_fifth_es, axis=0)
    #     min_one_fifth_es_fitness = np.min(fitness_values_one_fifth_es, axis=0)
    #     evaluations = np.arange(len(median_one_fifth_es_fitness)) * POP_SIZE
    #     plt.plot(evaluations, median_one_fifth_es_fitness, label='One-Fifth ES', color='darkseagreen', linewidth=2)
    #     plt.fill_between(evaluations, min_one_fifth_es_fitness, max_one_fifth_es_fitness, color='darkseagreen', alpha=0.2)
    #
    #     plot_filename = os.path.join(plot_dir, f'convergence_comparison_plot_F{problem_index}.pdf')
    #     plt.legend()
    #     plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
    #     plt.close()

    @staticmethod
    def plot_convergence_data_mean_ci(problem_index, episodes_tested_dir, baselines_dir, plot_dir, instance):
        plt.figure(figsize=(5, 3))
        plt.title(f'Convergence Comparison on Problem F{problem_index}')
        plt.yscale('symlog')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Fitness Value (Mean with 95% CI)')
        plt.grid(True)

        episode_colors = ['steelblue', 'sandybrown', 'firebrick']

        for i, episode in enumerate(EPISODES):
            fitness_values = load_data(
                os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
            mean_fitness_values = np.mean(fitness_values, axis=0)
            ci = 1.96 * sem(fitness_values, axis=0)  # 95% CI
            evaluations = np.arange(len(mean_fitness_values)) * POP_SIZE
            color = episode_colors[i % len(episode_colors)]
            plt.plot(evaluations, mean_fitness_values, label=f'Episode {episode}', color=color, linewidth=2)
            plt.fill_between(evaluations, mean_fitness_values - ci, mean_fitness_values + ci, color=color, alpha=0.2)

        # Plotting for CMA-ES
        fitness_values_cma_es = load_data(os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        mean_cma_es_fitness = np.mean(fitness_values_cma_es, axis=0)
        ci_cma_es = 1.96 * sem(fitness_values_cma_es, axis=0)
        evaluations = np.arange(len(mean_cma_es_fitness)) * POP_SIZE
        plt.plot(evaluations, mean_cma_es_fitness, label='CMA-ES', color='black', linewidth=2)
        plt.fill_between(evaluations, mean_cma_es_fitness - ci_cma_es, mean_cma_es_fitness + ci_cma_es, color='grey',
                         alpha=0.2)

        # Plotting for One-Fifth ES
        fitness_values_one_fifth_es = load_data(
            os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
        mean_one_fifth_es_fitness = np.mean(fitness_values_one_fifth_es, axis=0)
        ci_one_fifth_es = 1.96 * sem(fitness_values_one_fifth_es, axis=0)
        evaluations = np.arange(len(mean_one_fifth_es_fitness)) * POP_SIZE
        plt.plot(evaluations, mean_one_fifth_es_fitness, label='One-Fifth ES', color='darkseagreen', linewidth=2)
        plt.fill_between(evaluations, mean_one_fifth_es_fitness - ci_one_fifth_es,
                         mean_one_fifth_es_fitness + ci_one_fifth_es, color='darkseagreen', alpha=0.2)

        plot_filename = os.path.join(plot_dir, f'convergence_comparison_plot_F{problem_index}.pdf')
        plt.legend()
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()

    def calculate_mean_and_ci(self, data, confidence=0.95):
        mean = np.mean(data, axis=1)
        sem = stats.sem(data, axis=1)
        n = data.shape[0]
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
        return mean, h

    def plot_convergence_data(self, problem_index, episodes_tested_dir, baselines_dir, plot_dir, instance):
        plt.figure(figsize=(5, 3))
        plt.title(f'Convergence Rate on Problem F{problem_index}')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Convergence Rate')
        plt.grid(True)

        global_min_fitness = float('inf')  # Initialize to a very high value

        # Load and calculate convergence rates for episodes
        for episode in EPISODES:
            fitness_values = load_data(os.path.join(episodes_tested_dir,
                                                    f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
            if fitness_values.size > 0:
                global_min_fitness = min(global_min_fitness, np.min(fitness_values))

        # Load and calculate convergence rates for CMA-ES
        fitness_values_cma_es = load_data(
            os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        if fitness_values_cma_es.size > 0:
            global_min_fitness = min(global_min_fitness, np.min(fitness_values_cma_es))

        # Load and calculate convergence rates for One-Fifth ES
        fitness_values_one_fifth_es = load_data(
            os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
        if fitness_values_one_fifth_es.size > 0:
            global_min_fitness = min(global_min_fitness, np.min(fitness_values_one_fifth_es))

        global_min_fitness = global_min_fitness - 0.1 * abs(global_min_fitness)

        # Reiterate over the episodes to plot convergence rates
        episode_colors = ['steelblue', 'sandybrown', 'firebrick']
        episode_markers = ['s', 'D', 'o']  # Square, Diamond, Circle
        for i, episode in enumerate(EPISODES):
            fitness_values = load_data(os.path.join(episodes_tested_dir,
                                                    f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
            f0 = fitness_values[:, 0] if fitness_values.size > 0 else global_min_fitness
            convergence_rates = [
                1 - (abs(fitness_values[:, t] - global_min_fitness) / abs(f0 - global_min_fitness)) ** (1 / (t + 1)) for
                t in range(fitness_values.shape[1])]
            mean_convergence_rates, ci = self.calculate_mean_and_ci(np.array(convergence_rates))
            evaluations = np.arange(len(mean_convergence_rates)) * POP_SIZE
            color = episode_colors[i % len(episode_colors)]
            marker = episode_markers[i % len(episode_markers)]
            if episode == 0:
                plt.plot(evaluations, mean_convergence_rates, label=f'Episode Best',
                         color=color, linewidth=2, marker=marker, markevery=2)
            else:
                plt.plot(evaluations, mean_convergence_rates, label=f'Episode {episode}',
                         color=color, linewidth=2, marker=marker, markevery=2)
            # plt.fill_between(evaluations, mean_convergence_rates - ci, mean_convergence_rates + ci, color=color,
            #                  alpha=0.2)

        # Calculate and plot convergence rate for CMA-ES
        f0 = fitness_values_cma_es[:, 0] if fitness_values_cma_es.size > 0 else global_min_fitness
        convergence_rates_cma_es = [
            1 - (abs(fitness_values_cma_es[:, t] - global_min_fitness) / abs(f0 - global_min_fitness)) ** (1 / (t + 1))
            for t in range(fitness_values_cma_es.shape[1])]
        mean_convergence_rates_cma_es, ci_cma_es = self.calculate_mean_and_ci(np.array(convergence_rates_cma_es))
        evaluations = np.arange(len(mean_convergence_rates_cma_es)) * POP_SIZE
        plt.plot(evaluations, mean_convergence_rates_cma_es, label='CMA-ES',
                 color='black', linewidth=2, marker='^', markevery=2)

        # plt.fill_between(evaluations, mean_convergence_rates_cma_es - ci_cma_es,
        #                  mean_convergence_rates_cma_es + ci_cma_es, color='grey', alpha=0.2)

        # Calculate and plot convergence rate for One-Fifth ES
        f0 = fitness_values_one_fifth_es[:, 0] if fitness_values_one_fifth_es.size > 0 else global_min_fitness
        convergence_rates_one_fifth_es = [
            1 - (abs(fitness_values_one_fifth_es[:, t] - global_min_fitness) / abs(f0 - global_min_fitness)) ** (
                        1 / (t + 1)) for t in range(fitness_values_one_fifth_es.shape[1])]
        mean_convergence_rates_one_fifth_es, ci_one_fifth_es = self.calculate_mean_and_ci(
            np.array(convergence_rates_one_fifth_es))
        evaluations = np.arange(len(mean_convergence_rates_one_fifth_es)) * POP_SIZE
        plt.plot(evaluations, mean_convergence_rates_one_fifth_es, label='One-Fifth ES',
                 color='darkseagreen', linewidth=2, marker='p', markevery=2)
        # plt.fill_between(evaluations, mean_convergence_rates_one_fifth_es - ci_one_fifth_es,
        #                  mean_convergence_rates_one_fifth_es + ci_one_fifth_es, color='darkseagreen', alpha=0.2)

        plt.legend()
        plot_filename = os.path.join(plot_dir, f'convergence_rate_plot_F{problem_index}.pdf')
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()

    @staticmethod
    def standardize_data(values):
        mean_val = np.mean(values)
        std_val = np.std(values)
        standardized = (values - mean_val) / std_val
        return standardized

    # def plot_standardized_performance_boxplot(self, episodes_tested_dir, baselines_dir, plot_dir, instance):
    #     all_fitness_values = {'Episode 1200': [], 'Episode 4800': [], 'Episode 9600': [], 'CMA-ES': [], 'One-Fifth ES': []}
    #
    #     # Collect and standardize data for each EP
    #     for episode in EPISODES:  # EPISODES should be something like [1, 600, 1200]
    #         key = f'Episode {episode}'
    #         for problem_index in range(1, 25):  # Assuming 24 problems
    #             fitness_values = load_data(
    #                 os.path.join(episodes_tested_dir, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'))
    #             standardized_values = self.standardize_data(fitness_values[:, -1])
    #             all_fitness_values[key].extend(standardized_values)
    #
    #     # Collect and standardize data for CMA-ES
    #     for problem_index in range(1, 25):
    #         fitness_values = load_data(os.path.join(baselines_dir, f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
    #         standardized_values = self.standardize_data(fitness_values[:, -1])
    #         all_fitness_values['CMA-ES'].extend(standardized_values)
    #
    #     # Collect and standardize data for One-Fifth ES
    #     for problem_index in range(1, 25):
    #         fitness_values = load_data(os.path.join(baselines_dir, f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))
    #         standardized_values = self.standardize_data(fitness_values[:, -1])
    #         all_fitness_values['One-Fifth ES'].extend(standardized_values)
    #
    #     # Plotting the box plot with standardized data
    #     plt.figure(figsize=(12, 8))
    #     plt.title('Standardized Overall Performance Comparison')
    #     plt.ylabel('Standardized Final Fitness Value (z-score)')
    #     plt.boxplot(all_fitness_values.values(), labels=all_fitness_values.keys())
    #     plt.grid(True)
    #
    #     plot_filename = os.path.join(plot_dir, 'standardized_overall_performance_comparison_boxplot.pdf')
    #     plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
    #     plt.close()

    def plot_radar_chart(self, episodes_tested_dir, baselines_dir, plot_dir, instance):
        categories = [f'F{index}' for index in range(1, 25)]
        N = len(categories)

        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)

        labels = ['PPO-ES (Best Episode)', 'CMA-ES', 'One-Fifth ES']
        colors = ['#C34A36', '#9B89B3', '#D5CABD']
        markers = ['o', 's', '^']
        global_min_fitness = [float('inf')] * 24  # Initialize to a very high value
        for problem_index in range(1, 25):
            for label in labels:
                filename = self.construct_filename(label, problem_index, instance, episodes_tested_dir, baselines_dir)
                fitness_values = load_data(filename)
                global_min_fitness[problem_index - 1] = min(global_min_fitness[problem_index - 1],
                                                            np.min(fitness_values))
                global_min_fitness[problem_index - 1] = (global_min_fitness[problem_index - 1]
                                                         - 0.1 * abs(global_min_fitness[problem_index - 1]))


        # First pass to find the max convergence rate for each problem across all algorithms
        mean_convergence_rates = np.zeros((3, 25))
        for i, label in enumerate(labels):
            for problem_index in range(1, 25):
                filename = self.construct_filename(label, problem_index, instance, episodes_tested_dir, baselines_dir)
                fitness_values = load_data(filename)
                convergence_rates = (1 - abs((fitness_values[:, -1] - global_min_fitness[problem_index - 1])
                                             / (fitness_values[:, 0] - global_min_fitness[problem_index - 1]))
                                     ** (1 / (fitness_values.shape[1] + 1)))
                mean_convergence_rates[i, problem_index - 1] = np.mean(convergence_rates)
            mean_convergence_rates[i, -1] = mean_convergence_rates[i, 0]
        normalized_rates = np.zeros((3, 25))
        for i, label in enumerate(labels):
            for problem_index in range(1, 25):
                normalized_rates[i, problem_index - 1] = ((mean_convergence_rates[i, problem_index - 1] - 0)
                                                          / (np.max(mean_convergence_rates[:, problem_index - 1]) - 0))
            normalized_rates[i, -1] = normalized_rates[i, 0]
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]  # Ensure the closure of the radar chart

            ax.plot(angles, normalized_rates[i, :], color=colors[i], linewidth=2, linestyle='solid',
                    label=label, marker=markers[i], markersize=5)
            # ax.fill(angles, normalized_rates[i, :], color=colors[i], alpha=0.1)

        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.0"], color="grey", size=7)
        plt.ylim(0, 1.05)

        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plot_filename = os.path.join(plot_dir, f'convergence_radar_plot.pdf')
        plt.savefig(plot_filename, dpi=1200, bbox_inches='tight')
        plt.close()

    @staticmethod
    def construct_filename(label, problem_index, instance, episodes_tested_dir, baselines_dir):
        if label == 'PPO-ES (Best Episode)':
            data_dir = episodes_tested_dir
            filename = f'fitness_episode_Best_problem_{problem_index}_instance_{instance}.npy'
        else:
            data_dir = baselines_dir
            if label == 'CMA-ES':
                filename = f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'
            elif label == 'One-Fifth ES':
                filename = f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'
        return os.path.join(data_dir, filename)





