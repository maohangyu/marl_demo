import numpy as np
np.random.seed(0)
from common import Link, Router


'''
6v4: 6 routers and 4 paths
The topology is as follows:
A--------------C
 \            /
  \          / 
    E-------F
   /         \  
  /           \
B--------------D
The flow directions are as follows:
AC, AEFC
BD, BEFD
'''


class Environment(object):
    def __init__(self, args):
        self.args = args

        self.router_count = 2 + 4
        self.controllable_router_list = ["A", "B"]
        self.routers = {
            "A": Router(router_delay=args.router_delay, upper_links=[], direct_links=["AC", "AE"]),
            "B": Router(router_delay=args.router_delay, upper_links=[], direct_links=["BE", "BD"]),
            "E": Router(router_delay=args.router_delay, upper_links=["AE", "BE"], direct_links=["EF"]),
            "F": Router(router_delay=args.router_delay, upper_links=["EF"], direct_links=["FC", "FD"]),
            "C": Router(router_delay=args.router_delay, upper_links=["AC", "FC"], direct_links=[]),
            "D": Router(router_delay=args.router_delay, upper_links=["FD", "BD"], direct_links=[]),
        }
        self.router_downstream_capacity_sum_max = 200.0  # here is linkAC+linkAE=100+100=200
        # for each router, calculate the capacity-sum of all downstream links
        # router_downstream_capacity_sum_max is the max of those capacity-sums
        # _sum_MAX is used for normalizing the flow_demand ==> all routers have a same view of the flow_demand

        self.link_count = 7
        self.link_names = ["AC", "AE", "BE", "BD", "EF", "FC", "FD"]
        self.links = {
            "AC": Link(link_capacity=100.0, link_delay=args.link_delay),
            "AE": Link(link_capacity=100.0, link_delay=args.link_delay),
            "BE": Link(link_capacity=100.0, link_delay=args.link_delay),
            "BD": Link(link_capacity=100.0, link_delay=args.link_delay),
            "EF": Link(link_capacity=200.0, link_delay=args.link_delay),
            "FC": Link(link_capacity=100.0, link_delay=args.link_delay),
            "FD": Link(link_capacity=100.0, link_delay=args.link_delay),
        }

        if args.flow_type == "synthetic":
            self.point_count = 628
            self.flow_AC = np.array([58,52,52,54,54,43,45,51,42,42,48,44,42,42,43,38,46,48,41,44,39,37,40,44,47,39,38,42,52,42,39,45,39,39,42,38,39,39,41,42,44,40,38,44,37,36,41,30,30,39,42,44,35,20,21,23,19,18,21,18,17,16,15,16,17,16,20,25,20,20,26,19,19,18,17,18,27,22,24,32,23,26,30,26,23,25,26,23,26,20,23,30,28,27,33,29,28,24,22,24,30,21,26,30,25,27,34,28,31,30,33,43,44,40,39,46,44,49,55,46,55,61,60,46,46,36,48,51,40,39,50,50,48,50,52,55,74,52,61,70,79,56,61,56,53,57,62,55,78,58,53,63,49,46,61,46,51,50,51,56,70,51,49,51,47,60,61,52,61,55,66,60,69,54,56,86,87,89,99,82,86,79,82,86,98,78,42,50,46,54,56,64,53,67,77,56,72,69,58,65,57,49,46,47,57,53,60,58,55,54,40,44,36,42,48,40,46,38,39,39,58,44,40,45,43,39,53,43,44,42,44,51,54,46,46,61,51,53,49,39,41,40,32,40,51,45,46,50,47,46,52,41,49,46,53,37,41,35,48,62,58,59,64,57,56,51,51,56,60,53,54,62,55,49,49,49,40,62,67,70,65,55,52,60,55,58,51,46,47,46,44,48,49,43,41,48,45,42,43,37,35,32,26,37,42,40,41,48,39,29,39,39,36,34,39,34,46,45,41,47,37,38,37,32,28,28,27,25,31,23,23,28,31,29,32,29,28,28,22,35,34,10,8,14,9,9,14,9,9,11,11,10,13,11,8,16,11,12,15,12,13,15,15,14,16,11,16,24,15,18,19,25,15,13,13,12,20,15,13,23,11,13,40,15,14,17,15,16,19,18,25,38,36,35,42,33,33,37,41,34,33,30,29,42,37,40,42,38,47,40,42,42,51,48,52,59,54,54,64,53,54,49,45,31,43,51,45,35,49,46,54,44,42,57,50,55,59,60,60,64,56,48,57,51,51,48,47,46,59,53,58,48,78,82,65,48,47,52,60,74,57,55,52,50,65,53,83,110,108,98,91,93,92,100,107,104,93,81,91,79,78,104,78,76,68,66,57,57,75,67,57,52,46,44,36,42,42,32,35,36,34,34,36,31,35,31,34,28,41,37,35,25,27,28,40,33,32,29,31,32,38,32,28,26,32,31,47,31,28,25,29,30,36,28,31,22,24,25,31,22,19,19,18,27,37,33,31,25,39,38,42,63,77,46,40,40,44,38,43,53,52,48,56,58,58,57,45,59,61,55,53,55,46,48,55,43,44,46,41,33,31,25,29,37,41,34,40,35,36,34,27,28,39,38,43,33,31,35,39,30,34,31,25,32,50,50,53,48,28,31,46,33,44,45,40,40,48,41,40,39,37,35,37,37,34,33,26,25])
            self.flow_BD = np.array([50,42,34,36,39,46,59,48,50,47,46,49,50,50,45,43,49,45,50,46,44,46,49,55,58,60,56,56,59,58,58,50,56,50,51,57,56,57,49,47,55,63,70,60,58,61,76,84,76,66,61,64,66,67,74,70,71,67,64,68,68,67,70,69,64,65,67,60,54,53,57,60,61,61,54,55,56,53,58,56,57,64,65,66,66,65,63,71,62,53,56,55,51,51,56,59,59,71,71,60,63,60,69,64,56,56,63,58,63,64,62,61,61,54,58,51,55,53,51,55,59,63,65,64,64,63,58,54,50,48,45,43,47,49,44,40,45,44,47,43,39,39,39,38,38,40,32,40,41,42,43,42,49,39,42,43,42,43,38,38,41,39,47,62,65,51,46,46,52,52,44,41,46,51,52,49,56,45,45,50,53,51,48,47,52,46,67,72,73,66,67,68,67,57,47,62,65,55,58,56,60,59,57,60,56,47,40,45,44,37,43,34,40,39,43,46,52,51,52,48,56,52,53,47,52,49,47,48,53,54,49,47,57,61,54,44,50,40,54,47,52,55,49,46,44,44,51,48,53,45,50,55,54,51,51,47,49,43,51,44,50,41,58,56,56,56,54,61,58,48,50,49,59,39,41,44,45,38,34,38,39,44,48,41,49,41,44,45,50,43,44,48,43,42,53,47,51,52,62,60,55,54,48,51,60,57,57,49,56,57,55,60,74,72,62,68,57,54,61,54,62,48,50,53,59,60,56,57,66,64,66,66,81,75,78,68,72,74,70,72,81,69,71,72,80,72,77,75,79,79,79,77,73,70,74,74,85,73,77,82,89,90,73,69,73,70,75,52,62,54,59,58,63,64,62,62,62,59,69,66,64,54,59,58,62,61,54,60,63,57,59,57,64,54,58,62,51,53,44,49,53,47,50,53,61,51,52,51,49,52,47,48,53,51,54,42,47,40,42,45,47,48,43,45,46,41,45,39,47,36,42,48,47,45,42,42,49,47,50,43,51,42,46,46,46,49,42,43,49,44,46,42,51,39,43,42,42,45,43,42,52,51,48,49,51,44,47,51,58,52,52,49,43,45,63,56,54,57,63,71,73,74,64,54,50,44,47,45,48,40,47,38,37,36,35,34,33,30,31,34,40,32,37,42,43,44,43,45,44,38,46,43,45,44,44,43,47,46,43,45,50,50,42,33,38,39,48,46,47,44,43,47,46,45,44,42,44,43,48,49,50,50,40,43,43,36,41,36,40,44,42,47,49,51,45,49,50,43,43,33,42,44,45,44,46,49,41,42,42,50,47,48,46,39,39,40,43,43,33,33,38,34,42,34,36,38,40,39,40,43,37,41,52,46,46,47,46,47,52,54,54,57,47,49,52,50,49,53,59,67,58,58,61,58,66,50,55,54,55,54,54,47,44,59,58,54,65,60])
        elif args.flow_type == "Abilene":
            raise ValueError("Abilene is a real trace, and it is not defined in the public codebase!")
        else:
            raise ValueError("args.flow_type is not defined!")

    def reset(self):
        self.data_step = np.random.randint(self.point_count)  # set the started flow point by random

        self.router_action = {"A": [0.5, 0.5],
                              "B": [0.5, 0.5]}  # the latest action token by this agent

        self.link_usage_average = {}
        for link_name in self.link_names:
            self.link_usage_average[link_name] = 0.5

        state_all = self._get_current_state()  # it is the observation of each agent, not the whole state of the system
        return state_all

    def step(self, action_all):
        self._set_action(action_all)  # must firstly set the new actions before call self._simulate_one_step()
        for _ in range(self.args.action_effective_step):
            self._simulate_one_step()
        return self._get_influence_of_last_action()

    def _get_current_state(self):
        current_state_all = []
        for i, router_name in enumerate(self.controllable_router_list):
            flow_demand_in_router_buffer = \
                np.array(self.routers[router_name].buffer.get_history_data()) / self.router_downstream_capacity_sum_max
            direct_links_usage_history = []
            direct_links_usage_average = []
            for link_name in self.routers[router_name].direct_links:
                link_usage_history = \
                    np.array(self.links[link_name].buffer.get_history_data()) / self.links[link_name].link_capacity
                direct_links_usage_history.extend(list(link_usage_history))
                direct_links_usage_average.append(self.link_usage_average[link_name])
            state = np.array(
                list(flow_demand_in_router_buffer) +
                direct_links_usage_history +
                list(self.router_action[router_name]) +
                direct_links_usage_average
            ).reshape(1, self.args.state_dim_list[i])
            current_state_all.append(state)
        return current_state_all

    def _set_action(self, action_all):
        for i, router_name in enumerate(self.controllable_router_list):
            self.router_action[router_name] = action_all[i]
        self.action_A, self.action_B = action_all
        self.action_AC_for_toC, self.action_AE_for_toC = self.action_A
        self.action_BE_for_toD, self.action_BD_for_toD = self.action_B

    def _simulate_one_step(self):
        self.data_step += 1
        self.data_step %= self.point_count

        # must call get_data() before add_data(), or some data will be overwrite
        # but we can ignore this, and we just take the following as the situation where router_delay-1 and link_delay-1
        self.routers["A"].buffer.add_data([self.flow_AC[self.data_step], 0.0])
        flow_toC, flow_toD = self.routers["A"].buffer.get_data()  # flow_toD==0
        tmp = flow_toC * self.action_AC_for_toC
        self.links["AC"].buffer.add_data([tmp, 0.0])
        self.links["AC"].total_flow += tmp
        tmp = flow_toC * self.action_AE_for_toC
        self.links["AE"].buffer.add_data([tmp, 0.0])
        self.links["AE"].total_flow += tmp

        self.routers["B"].buffer.add_data([0.0, self.flow_BD[self.data_step]])
        flow_toC, flow_toD = self.routers["B"].buffer.get_data()  # flow_toC==0
        tmp = flow_toD * self.action_BE_for_toD
        self.links["BE"].buffer.add_data([0.0, tmp])
        self.links["BE"].total_flow += tmp
        tmp = flow_toD * self.action_BD_for_toD
        self.links["BD"].buffer.add_data([0.0, tmp])
        self.links["BD"].total_flow += tmp

        flow_AtoC = self.links["AE"].buffer.get_data()[0]
        flow_BtoD = self.links["BE"].buffer.get_data()[1]  # links.get_data, we use flow_StartPoint_to_EndPoint
        self.routers["E"].buffer.add_data([flow_AtoC, flow_BtoD])  # see [flow_AtoC, flow_BtoD] as aggregated flow
        flow_toC, flow_toD = self.routers["E"].buffer.get_data()  # routers.get_data, we use flow_to_EndPoint
        self.links["EF"].buffer.add_data([flow_toC, flow_toD])
        self.links["EF"].total_flow += (flow_toC + flow_toD)

        flow_EtoC, flow_EtoD = self.links["EF"].buffer.get_data()
        self.routers["F"].buffer.add_data([flow_EtoC, flow_EtoD])
        flow_toC, flow_toD = self.routers["F"].buffer.get_data()
        self.links["FC"].buffer.add_data([flow_toC, 0.0])
        self.links["FC"].total_flow += flow_toC
        self.links["FD"].buffer.add_data([0.0, flow_toD])
        self.links["FD"].total_flow += flow_toD

    def _get_influence_of_last_action(self):
        # the average usage in the last control cycle (i.e., action_effective_step)
        for link_name in self.link_names:
            self.link_usage_average[link_name] = \
                self.links[link_name].total_flow / self.links[link_name].link_capacity / self.args.action_effective_step
            self.links[link_name].total_flow = 0.0  # reset total_flow for the next control cycle

        # calculate the reward
        max_usage_rate = max(self.link_usage_average.values())
        global_reward = float(1.0 - max_usage_rate)  # total_flow is a str ==> search "surprise" in env_routing12v20.py
        if max_usage_rate > 1.0:
            done = True
            global_reward += -1.0
        else:
            done = False
        # rewards[0] is always the global reward, i.e., our objective reward;
        # the training reward, namely rewards[1 ~ agent_count], can be Heuristic Rewards like lR or lgMixedR etc.
        rewards = [global_reward] + [global_reward] * len(self.controllable_router_list)

        next_state_all = self._get_current_state()

        info = {"link_usage_average": self.link_usage_average}
        return rewards, next_state_all, done, info
