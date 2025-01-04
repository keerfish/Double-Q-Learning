<kbd><img src='results/world_1_1.gif' align="center" width=384/></kbd>
<kbd><img src='results/world_1_2.gif' align="center" width=384/></kbd>
<kbd><img src='results/world_2_3.gif' align="center" width=384/></kbd>

# Play Super Mario with Double Q Learning
This toy example of Reinforcement Learning demonstrates an implementation of Double Q-Learning to play Super Mario. [Super Mario](https://en.wikipedia.org/wiki/Super_Mario_Bros.) is an arcade game developed and released by Nintendo in 1983. I chose Mario Bros. as a replay implementation example to get familar to reinforce learning because [Pytorch](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) supplied a detailed tutorial with this implementation for a comprehensive understanding of the Double Q-learning algorithm. It gives very big convenience of environment setting, software installation, programming running, and results tuning. The implementation here is mostly a replay of the tutorial example from [Pytorch](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

## AI-Powered Mario Agent
Mario Bros. has two characters, Mario and Luigi, who work as plumbers. The game Mario Bros. includes eight worlds with a series of increasingly difficult stages in which there are different types of enemies and obstacles. Mario and Luigi's mission is to cross the obstacles on their way and defeat the enemies they met on the way to the flag in each stage. The example here will replay the implementation of the Double Q-learning algorithm using PyTorch, but it will only train one agent to play the Mario Bros. game. Thus only Mario will appear in the implementation.

## Mario Agent Setting
In the reinforcement learning framework, an agent is trained to control the character Mario in the game Mario Bros. The agent aims to navigate through Mario Bros.'s world within a proper time. The state of the game environment includes enemy positions, enemy states, obstacle positions, etc. When the agent is going through the game environment, it should avoid obstacles, collect coins, defeat enemies, and ultimately complete the levels by reaching the end flag. 

The agent's actions are decided by the current state of the environment and the current action policy. For each state, Mario's agent either selects an action to explore potential strategies or exploits the neural network, which is based on previously learned information. The exploration rate gives the possibility of exploring the world or exploiting the experience information. We will set the exploration rate to one as the initial condition to guarantee the agent is doing a random action at the beginning. After a while, the exploration rate will be reduced. The agent will begin to exploit by using his neural network rather than explore with random actions. The agent improves his actions (or action policy) based on the repeat learning process. 

## Architecture and Algorithm
The proposed Double Q-learning method in the paper [[1]](#1) addresses the overestimation problem from Q-learning [[2]](#2). This is achieved through the use of two independent networks, namely the Q-Network and the target Network in the implementation; see the framework pipeline below.

<img src="https://github.com/keerfish/Double-Q-Learning/blob/main/imgs/architecture.jpg" align="center" width="500px"/>

The start state is the initial state as the input $s$ of the Q-Network to predict the action $a$ that yields the highest Q-value. The Mario Bros. environment returns to the next state after the action. These two elements, along with the current state and action, are stored in a replay buffer.

During the training, a batch of data containing state, next state, reward, and done status is randomly sampled from the replay buffer. Both networks process this batch to compute their respective output values $Q^*(s,a)$ and $r+\gamma\max_{a'}Q^{**}(s',a')$, and update each other. This entire complete state action sequence constitutes an episode. To effectively train the agents to learn an optimal policy function, the process is repeated across multiple episodes.

The procedure of the algorithm is from the paper [[3]](#3), which is represented as a completed pseudocode in Algorithm.

<img src="https://github.com/keerfish/Double-Q-Learning/blob/main/imgs/algorithm.jpg" align="center" width="700px"/>

For each episode, the agent repeats steps 3 to 12, and continues this over many episodes, progressively reducing $\gamma$ and refining the policy. Depending on the complexity of the game state, different architectures (e.g., Convolutional neural networks (CNNs) for image-based states, simple feed-forward networks for processed numerical states) may be required. 

## References
<a id="1">[1]</a>
Hasselt, Hado van and Guez, Arthur and Silver, David. 
Deep reinforcement learning with double Q-Learning. 
AAAI Press, 
Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, 
pages 2094–2100, 
Phoenix, Arizona, 
2016.

<a id="2">[2]</a>
Watkins, Christopher J. C. H. and Dayan, Peter. 
Q-Learning.
Machine Learning, 
pages 279--292, 
volume = 8, 
1992.

<a id="3">[3]</a>
Hasselt, Hado van. 
Double Q-learning. 
Curran Associates Inc, 
Proceedings of the 23rd International Conference on Neural Information Processing Systems, 
Volume 2, 
pages 2613–2621, 
Vancouver, British Columbia, Canada, 
2010.


