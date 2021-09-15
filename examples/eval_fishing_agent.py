import argparse
import math

import gym
import gym_fishing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym_fishing.envs.shared_env import plot_mdp, simulate_mdp
from stable_baselines import A2C, ACKTR, PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import LstmPolicy

from curriculum_learning.envs.simulate_vec_sb2 import simulate_mdp_vec

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Name of model to load from agents/",
    default="fishing_agent",
)
parser.add_argument(
    "--r",
    type=float,
    help="r value to plot",
    default=0.3,
)
parser.add_argument(
    "--K",
    type=float,
    help="K value to plot",
    default=1,
)
args = parser.parse_args()

if __name__ == "__main__":
    # Making a df for constant escapement policy
    env = gym.make("fishing-v1", r=args.r, K=args.K, sigma=0.0)

    ## inits
    row = []
    rep = 0
    env.reset()
    obs = env.state

    # Note that we must map between the "observed" space (on -1, 1) and model
    # space (0, 2K) for both actions and states with the get_* methods

    for t in range(env.Tmax):
        fish_population = env.get_fish_population(obs)
        ## The escapement rule
        Q = max(fish_population - args.K / 2, 0)
        action = env.get_action(Q)
        quota = env.get_quota(action)
        obs, reward, done, info = env.step(action)
        row.append([t, fish_population, quota, reward, int(rep), "const_esc"])

    const_esc_df = pd.DataFrame(
        row, columns=["time", "state", "action", "reward", "rep", "type"]
    )
    #############
    # Now making a df for the curriculum agent
    env_kwargs = {
        "Tmax": 100,
        "r": args.r,
        "K": args.K,
    }
    model = PPO2.load(f"agents/{args.model}.zip")
    env_name = "fishing-v1"
    env = make_vec_env(env_name, 8)
    env.set_attr("r", args.r)
    env.set_attr("K", args.K)
    eval_env = gym.make(env_name, **env_kwargs)
    curriculum_df = simulate_mdp_vec(env, eval_env, model, 10)
    import pdb; pdb.set_trace()

    ##############
    # Plotting both dfs
    fig, axs = plt.subplots(3, 1)
    for index, df in enumerate([curriculum_df, const_esc_df]):
        for i in np.unique(df.rep):
            results = df[df.rep == i]
            episode_reward = np.cumsum(results.reward)
            if index == 0:
                color = "blue"
                label = "curriculum"
            else:
                color = "red"
                label = "const. esc."
            axs[0].plot(
                results.time,
                results.state,
                color=color,
                alpha=0.3,
                label=label,
            )
            axs[1].plot(
                results.time,
                results.action,
                color=color,
                alpha=0.3,
                label=label,
            )
            axs[2].plot(
                results.time,
                episode_reward,
                color=color,
                alpha=0.3,
                label=label,
            )

    axs[0].set_ylabel("state")
    axs[1].set_ylabel("action")
    axs[2].set_ylabel("reward")
    fig.tight_layout()
    plt.legend()
    plt.savefig("trash.png")
    plt.close("all")
