import numpy as np
import pandas as pd
import os
from scipy.stats import stats
from src.utilities.tools import load_data


def perform_friedman_test(episodes_dir, baselines_dir, plot_dir, instance):
    # Initialize the results list
    full_results = []

    # Process data for each problem
    for problem_index in range(1, 25):  # 1 to 24
        # Load the data for episode_best, cma_es, and one_fifth_es
        ep_best_data = load_data(os.path.join(episodes_dir,
                                              f'fitness_episode_Best_problem_{problem_index}_instance_{instance}.npy'))
        cma_es_data = load_data(os.path.join(baselines_dir,
                                             f'fitness_cma_es_problem_{problem_index}_instance_{instance}.npy'))
        one_fifth_es_data = load_data(os.path.join(baselines_dir,
                                                   f'fitness_one_fifth_es_problem_{problem_index}_instance_{instance}.npy'))

        # Combine the data from the three algorithms for ranking
        combined_data = np.stack((ep_best_data[:, -1], cma_es_data[:, -1], one_fifth_es_data[:, -1]), axis=-1)

        # Rank the data along the last axis (algorithms)
        ranks = stats.rankdata(combined_data, axis=1)

        # Perform the Friedman test on the ranks for this problem
        data_for_problem = ranks[:]
        statistic, p_value = stats.friedmanchisquare(
            data_for_problem[:, 0],
            data_for_problem[:, 1],
            data_for_problem[:, 2]
        )

        # Calculate average rank for each algorithm on this problem
        average_ranks_problem = np.mean(data_for_problem, axis=0)

        # Calculate the mean fitness for each algorithm on this problem
        mean_fitness_episode_best = np.mean(ep_best_data)
        mean_fitness_cma_es = np.mean(cma_es_data)
        mean_fitness_one_fifth_es = np.mean(one_fifth_es_data)

        # Find the best (minimum) mean fitness value among the three algorithms
        best_mean_fitness = min(mean_fitness_episode_best, mean_fitness_cma_es, mean_fitness_one_fifth_es)

        # Append the results to the list
        full_results.append([
            problem_index,  # Problem number
            average_ranks_problem[0],  # Avg Rank for episode_best
            average_ranks_problem[1],  # Avg Rank for cma_es
            average_ranks_problem[2],  # Avg Rank for one_fifth_es
            statistic,  # Friedman Statistic
            p_value  # p-value
        ])

    # Create a DataFrame from the results
    results_df = pd.DataFrame(full_results, columns=[
        'Problem Number', 'Avg Rank (episode_best)', 'Avg Rank (cma_es)', 'Avg Rank (one_fifth_es)',
        'Friedman Statistic', 'p-value'
    ])

    # Calculate overall Friedman test across all problems
    all_ranks = np.array(full_results)[:, 1:6]  # Assuming ranks are in columns 1, 2, and 3
    # Calculate mean ranks for each algorithm across all problems
    mean_rank_ep_best = np.mean(all_ranks[:, 0])
    mean_rank_cma_es = np.mean(all_ranks[:, 1])
    mean_rank_one_fifth_es = np.mean(all_ranks[:, 2])
    mean_statistics = np.mean(all_ranks[:, 3])
    mean_p_value = np.mean(all_ranks[:, 4])

    # Append the overall results to the DataFrame
    overall_row = {
        'Problem Number': 'Mean',
        'Avg Rank (episode_best)': mean_rank_ep_best,
        'Avg Rank (cma_es)': mean_rank_cma_es,
        'Avg Rank (one_fifth_es)': mean_rank_one_fifth_es,
        'Friedman Statistic': mean_statistics,
        'p-value': mean_p_value
    }
    results_df = results_df._append(overall_row, ignore_index=True)

    # Add a step to save the DataFrame to a CSV file
    csv_file_path = os.path.join(plot_dir, 'friedman_test_results.csv')
    results_df.to_csv(csv_file_path, index=False)

    # Generate the LaTeX table string with formatting
    latex_table_str = format_latex_table(results_df)

    # Save the LaTeX table to a file
    latex_file_path = os.path.join(plot_dir, 'Friedman_test_results.tex')
    with open(latex_file_path, 'w') as f:
        f.write(latex_table_str)


def format_latex_table(df):
    """
    Formats the data frame into a LaTeX table with two rows for headers.
    Boldface the smallest average rank for each problem if p-value < 0.05.
    Split headers over two rows and align 'p-value' in the center of two rows.
    """

    # Prepare the LaTeX table header with two rows
    header = """
        \\begin{tabular}{ccccccc}
        \\toprule
        \\multirow{2}{*}{Problem} & \\multicolumn{3}{c}{Avg Rank} & Friedman & \\multirow{2}{*}{p-value} \\\\
        \\cline{2-4}
        & PPO-ES & CMA-ES & one-fifth ES & Statistic & \\\\
        \\midrule
        """

    # Start formatting table body
    body = ""
    for idx, row in df.iterrows():
        # Bold the smallest average rank if p-value < 0.05
        ranks = [row['Avg Rank (episode_best)'], row['Avg Rank (cma_es)'], row['Avg Rank (one_fifth_es)']]
        min_rank = min(ranks)
        formatted_ranks = [f"\\textbf{{{rank:.2f}}}" if rank == min_rank and row['p-value'] < 0.05 else f"{rank:.2f}"
                           for rank in ranks]

        # Add row data
        if idx <= 23:
            body += f"{int(row['Problem Number'])} & {' & '.join(formatted_ranks)} & {row['Friedman Statistic']:.2f} & {row['p-value']:.2e} \\\\\n"
        else:
            body += "\\midrule\n"
            body += f"{row['Problem Number']} & {' & '.join(formatted_ranks)} & {row['Friedman Statistic']:.2f} & {row['p-value']:.2e} \\\\\n"

    # Calculate mean values for the summary row
    mean_ranks = [df[f'Avg Rank (episode_best)'].mean(), df[f'Avg Rank (cma_es)'].mean(),
                  df[f'Avg Rank (one_fifth_es)'].mean()]
    mean_friedman = df['Friedman Statistic'].mean()

    # Close the LaTeX table
    footer = "\\bottomrule\n\\end{tabular}"

    # Combine header, body, and footer
    latex_str = header + body + footer
    return latex_str

