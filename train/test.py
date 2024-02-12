import os
import sys
sys.path.append('/Users/mintaekim/Desktop/HRL/Flappy/Integrated/Flappy_Integrated/quadrotor_v1')
from envs.quadrotor_env import QuadrotorEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ppo.ppo import PPO # Customized
from stable_baselines3.common.evaluation import evaluate_policy


env = QuadrotorEnv(render_mode="human")
env = VecMonitor(DummyVecEnv([lambda: env]))
save_path = os.path.join('saved_models')
loaded_model = PPO.load(save_path+"/best_model")

print("Evaluation start")
evaluate_policy(loaded_model, env, n_eval_episodes=100, render=True)
env.close()