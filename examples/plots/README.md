`fishing_agent_random_[parameters]` PNGs come from a fishing agent trained via random domain generation over r=0.3+/-0.05 and K=1+/-0.05. Recurrent policy-based PPO2 agent.

`random_fishing_agent_2t_range_longertr_[parameters]` PNGs come from a fishing agent trained via random domain generation over r=0.3+/-0.2 and K=1+/-0.2. The suffix `longertr` denotes that this agent was trained for a total of 5 million timesteps; without the suffix, the agent is trained for 2 million timesteps. Recurrent policy-based PPO2 agents here.

`sac_fishingv1_sigma0_singletask` PNGs come from a SAC trained agent trained on the default parameters on gym_fishing, r=0.3 and K=1.
