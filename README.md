# demo of MARL algorithms
This codebase is a reimplementation of some MARL algorithms.  
We did not tune the hyperparameters, and your are welcomed to do this for better performance.  



## code structure
D_* represent the code related to Discrete action space.  
C_* represent the code related to Continuous action space.  

C/D_env_* represent the Environment, which takes action as input and returns the (r, s', d, info).  
Please do not change the C/D_env_* files (if you want to make a fair comparison).    

C/D_models represent the Agent.  
You can modify this file to define your Agent with different DNN-structures or different optimization methods/losses.  

C/D_settings represent the parameter settings.  
You can modify this file to tune the parameters for better performance.  

C/D_main is the main file to run the code.  


## run the code
You can run IDQN Agent in navigation3v3 Environment by: 
```
python D_main.py --agent-name IDQN --env-name navigation3v3.  
python C_main.py --agent-name MADDPG --env-name routing6v4.  
```



## please cite the following papers
1. Neighborhood Cognition Consistent Multi-Agent Reinforcement Learning, AAAI-2020.  
2. Modelling the dynamic joint policy of teammates with attention multi-agent DDPG, AAMAS-2019.  
3. Multi-agent actor-critic for mixed cooperative-competitive environments, NIPS-2017.  
4. Graph convolutional reinforcement learning for multi-agent cooperation, ICLR-2020.  
5. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning, ICML-2018.  
6. Value-decomposition networks for cooperative multi-agent learning based on team reward, AAMAS-2018.  

