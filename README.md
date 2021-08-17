# curriculum_learning
Testing out curriculum learning inspired by [https://arxiv.org/pdf/2012.02096.pdf](https://arxiv.org/pdf/2012.02096.pdf)

Main idea here is that I have a dual loop RL, where in the inner loop in curriculum env, a fishing agent is being trained. On the outerloop, an agent is being trained that picks parameters. The objective for the inner agent is to maximize the cumulative catch by selecting the appropriate quotas. The task for the outer agent is to select parameters that are difficult for the fishing agent. The idea here is that the MaxMin approach should create a curriculum that speeds up learning for the inner agent, so the inner agent will learn over a series of model parameters. 
