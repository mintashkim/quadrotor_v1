import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/flappy_v2')
from envs.ppo.ppo import PPO # Customized
# from stable_baselines3 import PPO # Naive
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.flappy_env import FlappyEnv


log_path = os.path.join('logs')
save_path = os.path.join('saved_models')
best_model_save_path = os.path.join('saved_models', 'best_model')
env = FlappyEnv(render_mode="human")
env = DummyVecEnv([lambda: env])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

net_arch = {'pi': [512,256,128],
            'vf': [512,256,128]}

model = PPO('MlpPolicy', 
            env=env,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.98,  # 0.99 # look forward 1.65s
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1, 
            policy_kwargs={'net_arch':net_arch},
            tensorboard_log=log_path)

# model = PPO.load(best_model_save_path, env=env)
model.learn(total_timesteps=100,
            progress_bar=True,
            callback=eval_callback)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()