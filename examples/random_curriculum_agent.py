import argparse
import os
import random

import gym
import gym_fishing
import stable_baselines
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy

import curriculum_learning

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Name of model to load from agents/",
    default="fishing_agent_random",
)
parser.add_argument(
    "--range",
    type=float,
    help="+/- range to use around r=0.3 and K=1",
    default=0.05,
)
args = parser.parse_args()

# To avoid GPU memory hogging by TF
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Handling network architecture options for ease of use
net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "med": dict(pi=[256, 256], vf=[256, 256]),
    "large": dict(pi=[400, 400], vf=[400, 400]),
}

# Creating custom LSTM policy to use for fishing agent
class CustomLSTMPolicy(LstmPolicy):
    def __init__(
        self,
        sess,
        ob_space,
        ac_space,
        n_env,
        n_steps,
        n_batch,
        n_lstm=25,
        reuse=False,
        **_kwargs,
    ):
        super().__init__(
            sess,
            ob_space,
            ac_space,
            n_env,
            n_steps,
            n_batch,
            n_lstm,
            reuse,
            net_arch=[100, "lstm", net_arch["med"]],
            layer_norm=True,
            feature_extraction="mlp",
            **_kwargs,
        )


fishing_agent_hypers = {
    "cliprange": 0.1,
    "ent_coef": 0.0008280502090570286,
    "gamma": 1,
    "lam": 0.8,
    "learning_rate": 0.0002012041680316291,
    "noptepochs": 5,
}

if __name__ == "__main__":
    if not os.path.exists("agents"):
        os.makedirs("agents")
    # Instantiating the first agent
    env = make_vec_env(
        lambda: gym.make("fishing-v1"),
        n_envs=8,
    )
    for i in range(20):
        env.set_attr("r", random.uniform(0.3 - args.range, 0.3 + args.range))
        env.set_attr("K", random.uniform(1 - args.range, 1 + args.range))
        if i > 0:
            model = PPO2.load(f"agents/{args.model}")
            model.set_env(env)
        else:
            model = PPO2(
                CustomLSTMPolicy, env, verbose=2, **fishing_agent_hypers
            )
        model.learn(total_timesteps=100000, log_interval=1)
        model.save(f"agents/{args.model}")
        del model
