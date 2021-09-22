`curriculum_agent.py` trains a parameter agent using the `curriculum_learning` environment. Goal is to train a fishing agent that performs well over a range of model parameters; the parameter agent selects what model parameters to train on for a certain number of timesteps. All agents use a recurrent policy here. Not working right now and I imagine that this is going to be tricky to get right. Requires SB2.

`eval_fishing_agent_sb3.py` plots SAR trajectories compared to optimal for a SAC SB3 agent from `agents/`. Requires SB3.

`eval_fishing_agent.py` plots SAR trajectories compared to optimal for a PPO2 recurrent agent from `agents/`. Requires SB2.

`make_plots_sb3.sh` is a bash script that uses `eval_fishing_agent_sb3.py` to create a range of plots over r and K. Requires SB3.

`make_plots.sh` is a bash script that uses `eval_fishing_agent.py` to create a range of plots over r and K. Requires SB2.

`plot_comparisons.py` takes PNGs from `plots/` according to a name pattern id'ed via glob and makes pairwise comparisons. Does not require SB2 or SB3. 

`random_curriculum_agent.py` trains a recurrent policy-based agent on random selection of model parameters. Requires SB2.
