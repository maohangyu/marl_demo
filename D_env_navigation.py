# highly based on:
# [1] https://github.com/agakshat/multiagent-particle-envs
# [2] https://github.com/agakshat/multiagent-particle-envs/blob/master/multiagent/scenarios/simple_spread.py

import numpy as np
from common import ParticleEntity


class Environment(object):
    def __init__(self, args):
        # We can change the following to create complex environments
        self.agent_count = args.agent_count
        self.landmark_count = args.landmark_count
        self.env_bound = args.env_bound
        self.position_dim = args.env_dim
        self.velocity_dim = args.env_dim
        self.action_effective_step = args.action_effective_step

        self.agents = [ParticleEntity() for _ in range(self.agent_count)]
        self.landmarks = [ParticleEntity() for _ in range(self.landmark_count)]
        for i, agent in enumerate(self.agents):
            agent.color = np.array([0.35, 0.85, 0.35])
        for i, landmark in enumerate(self.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        self.entities = self.agents + self.landmarks

        # landmarks are not movable, so the following code should be put in __init__()
        # see MADDPG with https://sites.google.com/site/multiagentac/
        for i, landmark in enumerate(self.landmarks):
            # landmark.position = np.random.uniform(0, self.env_bound, self.position_dim)
            landmark.position = np.random.randint(0, self.env_bound, self.position_dim)  # discrete space [low, high)
            # in [2], the position is random sampled from [-1, +1], which may be easy to learn a good policy

    def reset(self):
        for i, agent in enumerate(self.agents):
            # agent.position = np.random.uniform(0, self.env_bound, self.position_dim)
            agent.position = np.random.randint(0, self.env_bound, self.position_dim)  # discrete space [low, high)
        observation_all = self._get_current_observation()
        return observation_all

    def step(self, action_all):
        self._set_action(action_all)  # must firstly decide the new actions before call self._simulate_one_step()
        for _ in range(self.action_effective_step):
            self._simulate_one_step()
        return self._get_influence_of_last_action()

    def _observation(self, agent):
        # get positions of all entities in this agent's reference frame, see observation() function in [2]
        relative_position = []
        for landmark in self.landmarks:
            relative_position.append(landmark.position - agent.position)
        for other in self.agents:
            if other is agent: continue
            relative_position.append(other.position - agent.position)
        cur_observation = np.concatenate([agent.position] + relative_position).reshape(1, -1)
        # do not consider velocity as the observation, because we do not consider damping
        return cur_observation * 1.0 / self.env_bound  # always normalize the observation
    def _get_current_observation(self):
        cur_observation_all = []
        for agent in self.agents:
            cur_observation_all.append(self._observation(agent))
        return cur_observation_all

    def _set_action(self, action_all):
        for i, agent in enumerate(self.agents):
            if action_all[i] == 0:  # stop
                agent.velocity = np.array([0, 0])
            elif action_all[i] == 1:  # up
                agent.velocity = np.array([0, 1])
            elif action_all[i] == 2:  # right
                agent.velocity = np.array([1, 0])
            elif action_all[i] == 3:  # down
                agent.velocity = np.array([0, -1])
            elif action_all[i] == 4:  # left
                agent.velocity = np.array([-1, 0])

    def _simulate_one_step(self):
        # as the agents can move, we must bound them in the env_bound
        for i, agent in enumerate(self.agents):
            agent.position = agent.position + agent.velocity
            agent.position = agent.position % self.env_bound  # keep the entity in the env_bound

    def _get_influence_of_last_action(self):
        reward = 0
        next_observation_all = self._get_current_observation()
        terminal = False
        info = None
        for landmark in self.landmarks:
            distance_list = []
            for agent in self.agents:
                distance = np.sqrt(np.sum(np.square(agent.position - landmark.position)))
                distance_list.append(distance)
            reward += -1.0 * min(distance_list)  # -1.0
            # we do not consider the collision penalty, see reward() function in [1]
        rewards = [reward] * (self.agent_count + 1)  # rewards[0] is global reward
        return rewards, next_observation_all, terminal, info
