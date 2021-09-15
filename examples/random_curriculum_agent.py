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
        env.set_attr(
            "params",
            {"r": random.uniform(0.25, 0.35), "K": random.uniform(0.95, 1.05)},
        )
        if i > 0:
            model = PPO2.load("agents/fishing_agent_random")
            model.set_env(env)
        else:
            model = PPO2(
                CustomLSTMPolicy, env, verbose=2, **fishing_agent_hypers
            )
        model.learn(total_timesteps=100000, log_interval=1)
        model.save("agents/parameter_agent_random")
        del model
