import gym
import gym_fishing
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

class CurriculumFishing(gym.Env):
    def __init__(self, env_name = "fishing-v0", fishing_agent_name = "agent", fishing_agent_hypers={"ent_coef": 0.01, }, inter_tsteps=100, Tmax=100):
        self.action_space = spaces.Box(low=-1, high=1, shape = (2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape = (2,))
        self.num_episodes = 0
        self.fishing_agent_name = fishing_agent_name
        self.inter_tsteps = inter_tsteps
        self.env_name = env_name
        self.Tmax = Tmax
        self.done = False
        
        # Instantiating the first agent 
        env = gym.make(self.env_name, Tmax=self.Tmax, verbose=0)
        model.learn(total_timesteps=self.inter_tsteps, env=env)
        model.save(f"agents/{self.fishing_agent_name}")
        del model
    
    def step(self, action):
        # Mapping action to env kwargs and creating environment
        env_kwargs = self.rescale_params(action)
        import pdb; pdb.set_trace()
        env = gym.make(self.env_name, self.Tmax, params={"r": env_kwargs[0], "K": env_kwargs[1]})
        
        # load agent and train on new set of env kwargs
        model = SAC.load(f"agents/{self.fishing_agent_name}")
        model.set_env(env)
        model.learn(total_timesteps=self.inter_tsteps, env=env)
        model.save(f"agents/{self.fishing_agent_name}")
        
        mean_reward, std_reward = evaluate_policy(model, env)
    
        return action, -mean_reward, self.done, {}
    
    def reset(self):
        pass
    
    def rescale_params(self, action):
        # Remaps r and K to 0 to 2
        env_kwargs = 2 * (action + 1) 
        return env_kwargs
