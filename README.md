<kbd><img src='results/world_1_1.gif' align="center" width=384/></kbd>
<kbd><img src='results/world_1_2.gif' align="center" width=384/></kbd>
<kbd><img src='results/world_2_3.gif' align="center" width=384/></kbd>

# Play Super Mario with Double Q Learning
This toy example of Reinforcement Learning demonstrates an implementation of Double Q-Learning to play Super Mario. [Super Mario](https://en.wikipedia.org/wiki/Super_Mario_Bros.) is an arcade game developed and released by Nintendo in 1983. I chose Mario Bros. as a replay implementation example to get familar to reinforce learning because [Pytorch](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) supplied a detailed tutorial with this implementation for a comprehensive understanding of the Double Q-learning algorithm. It gives very big convenience of environment setting, software installation, programming running, and results tuning. The implementation here is mostly a replay of the tutorial example from [Pytorch](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

## AI-Powered Mario Agent
Mario Bros. has two characters, Mario and Luigi, who work as plumbers. The game Mario Bros. includes eight worlds with a series of increasingly difficult stages in which there are different types of enemies and obstacles. Mario and Luigi's mission is to cross the obstacles on their way and defeat the enemies they met on the way to the flag in each stage. The example here will replay the implementation of the Double Q-learning algorithm using PyTorch, but it will only train one agent to play the Mario Bros. game. Thus only Mario will appear in the implementation.

## Mario Agent Setting
In the reinforcement learning framework, an agent is trained to control the character Mario in the game Mario Bros. The agent aims to navigate through Mario Bros.'s world within a proper time. The state of the game environment includes enemy positions, enemy states, obstacle positions, etc. When the agent is going through the game environment, it should avoid obstacles, collect coins, defeat enemies, and ultimately complete the levels by reaching the end flag. 

