import gym
import stable_baselines2
from stable_baselines import PPO2

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
            net_arch=[100, "lstm", net_arch["medium"]],
            layer_norm=True,
            feature_extraction="mlp",
            **_kwargs,
        )


if __name__ == "__main__":
    # Instantiating the first agent
    env = make_vec_env(
        lambda: gym.make("curriculum_fishing-v0", Tmax=self.Tmax),
        n_envs=params["n_envs"],
    )
    model = PPO2(CustomLSTMPolicy, env, verbose=0)
    model.learn(total_timesteps=self.inter_tsteps, env=env)
    model.save(f"agents/{self.fishing_agent_name}")
