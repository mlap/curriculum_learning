import gym
import gym_fishing
from gym import spaces
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import LstmPolicy

from envs.simulate_vec_sb2 import simulate_mdp_vec

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


class CurriculumFishingEnv(gym.Env):
    def __init__(
        self,
        env_name="fishing-v1",
        fishing_agent_name="fishing_agent",
        # Using hyperparameters from tuned LSTM agent on fishing-v1
        fishing_agent_hypers={
            "batch_size": 128,
            "cliprange": 0.1,
            "ent_coef": 0.0008280502090570286,
            "gamma": 1,
            "lam": 0.8,
            "learning_rate": 0.0002012041680316291,
            "noptepochs": 5,
        },
        inter_tsteps=100,
        Tmax=100,
    ):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.num_episodes = 0
        self.fishing_agent_name = fishing_agent_name
        self.inter_tsteps = inter_tsteps
        self.env_name = env_name
        self.Tmax = Tmax
        self.done = False

        # Instantiating the first  fishing agent
        env = make_vec_env(
            lambda: gym.make(self.env_name, Tmax=self.Tmax),
            n_envs=8,
        )
        model = PPO2(CustomLSTMPolicy, env, verbose=1)
        model.learn(total_timesteps=self.inter_tsteps)
        model.save(f"agents/{self.fishing_agent_name}")
        del model

    def step(self, action):
        # Mapping action to env kwargs and creating environment
        env_kwargs = self.rescale_params(action)

        env = make_vec_env(
            lambda: gym.make(
                self.env_name,
                Tmax=self.Tmax,
                r=env_kwargs[0],
                K=env_kwargs[1],
            ),
            n_envs=8,
        )

        # load agent and train on new set of env kwargs
        model = PPO2.load(f"agents/{self.fishing_agent_name}")
        import pdb

        pdb.set_trace()
        model.set_env(env)
        model.learn(total_timesteps=self.inter_tsteps)
        model.save(f"agents/{self.fishing_agent_name}")

        eval_env = gym.make(
            self.env_name,
            Tmax=self.Tmax,
            r=env_kwargs[0],
            K=env_kwargs[1],
        )
        eval_df = simulate_mdp_vec(env, eval_env, model, 10)
        mean_reward = eval_df.groupby(["rep"]).sum().mean(axis=0)["reward"]
        del model

        return action, -mean_reward, self.done, {}

    def reset(self):
        pass

    def rescale_params(self, action):
        # Remaps r and K to 0 to 2
        env_kwargs = 2 * (action + 1)
        return env_kwargs
