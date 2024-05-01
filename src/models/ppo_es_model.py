import random
from src.environment.es_env import ES_Env
from src.callbacks.callbacks import LearningRateScheduler, SaveOnBestTrainingRewardCallback
from src.config.config import NUM_RUN, TRAIN_INSTANCE, EPISODES
from src.utilities.tools import linear_schedule, save_data
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.utilities.tools import cmaes_data
import numpy as np
import os


def test_model(env, model_path, data_path, episode, problem_index, instance):
    model = PPO.load(model_path, env=env)
    all_fitness_values = []
    for _ in range(NUM_RUN):
        obs, info = env.envs[0].reset()
        fitness_values = []
        while env.envs[0].unwrapped.countevals <= env.envs[0].unwrapped.fes_max:  # Adjust the number of steps as needed
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(np.array([action]))
            obs = obs[0]
            if dones:
                break
            fitness_values.append(env.envs[0].unwrapped.current_best_fitness)
        all_fitness_values.append(np.array(fitness_values))
    all_fitness_values_arr = np.array(all_fitness_values)
    save_data(os.path.join(data_path, f'fitness_episode_{episode}_problem_{problem_index}_instance_{instance}.npy'), all_fitness_values_arr)


class PPO_ES:
    def __init__(self, base_dir, cuda_device):
        self.base_dir = base_dir
        self.cuda_device = cuda_device
        self.seeds = [42]
        # Initialize paths for saving results and plots
        self.results_dir = os.path.join(base_dir, 'output_data', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def train_ppo_es(self):
        baselines_dir_train = os.path.join(self.results_dir, f'baselines', f'DIM_{40}')
        baseline_for_train = cmaes_data(baselines_dir_train, instance=1)
        for seed in self.seeds:
            env = make_vec_env(lambda: ES_Env(instance=TRAIN_INSTANCE, seed=seed, cmaes_data=baseline_for_train), n_envs=1)
            # Reset the model with the new environment to ensure it's training from scratch
            model = PPO(
                policy='MlpPolicy',
                env=env,
                device=self.cuda_device,
                learning_rate=3e-4,
                verbose=1,
                n_steps=24 * 250,  # Number of steps to run for each environment per update
                batch_size=40,  # Batch size for training
                n_epochs=20,  # Number of epochs to run for each update
                gamma=0.99,  # Discount factor
                gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                clip_range=0.2
            )
            episodes_trained_dir = os.path.join(self.results_dir, f'episodes_trained')
            os.makedirs(episodes_trained_dir, exist_ok=True)
            callback = SaveOnBestTrainingRewardCallback(save_path=episodes_trained_dir, seed=seed)
            total_timesteps = 24 * 5000
            # scheduler = linear_schedule(initial_value=3e-5)
            # lr_scheduler_callback = LearningRateScheduler(
            #     initial_learning_rate=3e-5,
            #     scheduler=scheduler
            # )
            # lr_scheduler_callback.total_timesteps = total_timesteps
            model.learn(total_timesteps=total_timesteps, callback=[callback])

    def test_ppo_es(self, problem_type, test_problem_dimension, problem_index, instance):
        baselines_dir_test = os.path.join(self.results_dir, f'baselines', f'DIM_{test_problem_dimension}')
        baseline_for_test = cmaes_data(baselines_dir_test, instance=1)
        # Randomly select one seed for testing
        seed = random.choice(self.seeds)
        episodes_tested_dir = os.path.join(self.results_dir, f'episodes_tested', f'DIM_{test_problem_dimension}')
        os.makedirs(episodes_tested_dir, exist_ok=True)

        seed_env = make_vec_env(lambda: ES_Env(problem_type=problem_type,
                                               instance=instance,
                                               dim=test_problem_dimension,
                                               problem_index=problem_index,
                                               cmaes_data=baseline_for_test,
                                               seed=seed), n_envs=1)

        for episode in EPISODES:
            model_filename = f"model_seed_{seed}_episode_{episode}.zip"
            test_model_path = os.path.join(self.results_dir, 'episodes_trained', model_filename)

            for single_env in seed_env.envs:
                single_env.unwrapped.set_mode('testing')

            test_model(seed_env, test_model_path, episodes_tested_dir, episode, problem_index, instance)

