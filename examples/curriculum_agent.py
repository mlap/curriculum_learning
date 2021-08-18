import os

import gym
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

net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "med": dict(pi=[256, 256], vf=[256, 256]),
    "large": dict(pi=[400, 400], vf=[400, 400]),
}


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


if __name__ == "__main__":
    if not os.path.exists("agents"):
        os.makedirs("agents")
    # Instantiating the first agent
    env = make_vec_env(
        lambda: gym.make("curriculum_fishing-v0"),
        n_envs=8,
    )
    model = PPO2(CustomLSTMPolicy, env, verbose=0)
    model.learn(total_timesteps=10000)
    model.save(f"agents/parameter_agent")
