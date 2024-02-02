import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.flappy_env import FlappyEnv


log_path = os.path.join('logs')
_env = FlappyEnv()
env = DummyVecEnv([lambda: _env])

save_path = os.path.join('train', 'saved_models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

net_arch = {'pi': [128,128,128,128],
            'vf': [128,128,128,128]}

model = PPO('MlpPolicy', 
            env,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.98,  # 0.99 # look forward 1.65s
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1, 
            policy_kwargs={'net_arch':net_arch},
            tensorboard_log=log_path)

model.learn(total_timesteps=100,
            progress_bar=True,
            callback=eval_callback)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()