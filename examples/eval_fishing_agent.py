import argparse

import gym
import curriculum_learning
from gym_fishing.envs.shared_env import plot_mdp, simulate_mdp
from curriculum_learning.envs.simulate_vec_sb2 import simulate_mdp_vec
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import LstmPolicy

if __name__ == "__main__":
    env_kwargs = {
        "Tmax": 50,
    }
    model = PPO2.load(
        f"agents/fishing_agent.zip"
    )
    env_name = "fishing-v1"
    env = make_vec_env(
        env_name, 8, env_kwargs=env_kwargs
    )
    eval_env = gym.make(env_name, **env_kwargs)
    plot_mdp(
        env,
        simulate_mdp_vec(env, eval_env, model, 10),
        output="trash.png",
    )
