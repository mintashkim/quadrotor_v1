import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.flappy_env import FlappyEnv


log_path = os.path.join('logs')
save_path = os.path.join('saved_models')
best_model_save_path = os.path.join('saved_models', 'best_model')
env = FlappyEnv(render_mode="human")
env = VecMonitor(DummyVecEnv([lambda: env]))

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=best_model_save_path,
                             verbose=1)

net_arch = {'pi': [512,256,256,128],
            'vf': [512,256,256,128]}

model = PPO('MlpPolicy', 
            env=env,
            learning_rate=3e-4,
            n_steps=2048, # The number of steps to run for each environment per update
            batch_size=256,
            gamma=0.98,  # 0.99 # look forward 1.65s
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            policy_kwargs={'net_arch':net_arch},
            tensorboard_log=log_path)

model.learn(total_timesteps=100000, # The total number of samples (env steps) to train on
            progress_bar=True,
            callback=eval_callback)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

model.save(save_path)

obs_sample = model.env.observation_space.sample()
print("Pre saved", model.predict(obs_sample, deterministic=True))
del model # delete trained model to demonstrate loading
loaded_model = PPO.load(save_path)
print("Loaded", loaded_model.predict(obs_sample, deterministic=True))