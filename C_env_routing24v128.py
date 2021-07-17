import numpy as np
from common import Link, Router
np.random.seed(0)


'''
24v128: 24 routers and 128 paths
The topology is shown in "Neighborhood Cognition Consistent Multi-Agent Reinforcement Learning"
'''


class Environment(object):
    def __init__(self, args):
        self.args = args

        self.router_count = 17 + 7
        self.controllable_router_list = ["A", "B", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
        self.routers = {
            "A": Router(router_delay=args.router_delay, upper_links=[], direct_links=["A1", "A2"], egress_count=3),
            "B": Router(router_delay=args.router_delay, upper_links=[], direct_links=["B2", "B3"], egress_count=3),
            "C": Router(router_delay=args.router_delay, upper_links=[], direct_links=["C3", "C4"], egress_count=3),
            "1": Router(router_delay=args.router_delay, upper_links=["A1"], direct_links=["18", "15"], egress_count=3),
            "2": Router(router_delay=args.router_delay, upper_links=["A2", "B2"], direct_links=["25", "26"], egress_count=3),
            "3": Router(router_delay=args.router_delay, upper_links=["B3", "C3"], direct_links=["36", "37"], egress_count=3),
            "4": Router(router_delay=args.router_delay, upper_links=["C4"], direct_links=["47", "411"], egress_count=3),
            "5": Router(router_delay=args.router_delay, upper_links=["15", "25"], direct_links=["58", "59"], egress_count=3),
            "6": Router(router_delay=args.router_delay, upper_links=["26", "36"], direct_links=["69", "610"], egress_count=3),
            "7": Router(router_delay=args.router_delay, upper_links=["37", "47"], direct_links=["710", "711"], egress_count=3),
            "8": Router(router_delay=args.router_delay, upper_links=["18", "58"], direct_links=["815", "812"], egress_count=3),
            "9": Router(router_delay=args.router_delay, upper_links=["59", "69"], direct_links=["912", "913"], egress_count=3),
            "10": Router(router_delay=args.router_delay, upper_links=["610", "710"], direct_links=["1013", "1014"], egress_count=3),
            "11": Router(router_delay=args.router_delay, upper_links=["711", "411"], direct_links=["1114", "1118"], egress_count=3),
            "12": Router(router_delay=args.router_delay, upper_links=["812", "912"], direct_links=["1215", "1216"], egress_count=3),
            "13": Router(router_delay=args.router_delay, upper_links=["913", "1013"], direct_links=["1316", "1317"], egress_count=3),
            "14": Router(router_delay=args.router_delay, upper_links=["1014", "1114"], direct_links=["1417", "1418"], egress_count=3),
            "15": Router(router_delay=args.router_delay, upper_links=["815", "1215"], direct_links=["15D"], egress_count=3),
            "16": Router(router_delay=args.router_delay, upper_links=["1216", "1316"], direct_links=["16D", "16E"], egress_count=3),
            "17": Router(router_delay=args.router_delay, upper_links=["1317", "1417"], direct_links=["17E", "17F"], egress_count=3),
            "18": Router(router_delay=args.router_delay, upper_links=["1418", "1118"], direct_links=["18F"], egress_count=3),
            "D": Router(router_delay=args.router_delay, upper_links=["15D", "16D"], direct_links=[], egress_count=3),
            "E": Router(router_delay=args.router_delay, upper_links=["16E", "17E"], direct_links=[], egress_count=3),
            "F": Router(router_delay=args.router_delay, upper_links=["17F", "18F"], direct_links=[], egress_count=3),
        }
        self.router_downstream_capacity_sum_max = 1200.0  # here is link25+link26=600+600=1200
        # for each router, calculate the capacity-sum of all downstream links
        # router_downstream_capacity_sum_max is the max of those capacity-sums
        # _sum_MAX is used for normalizing the flow_demand ==> all routers have a same view of the flow_demand

        self.link_count = 40
        self.link_names = ["A1", "A2", "B2", "B3", "C3", "C4",
                           "18", "15", "25", "26", "36", "37", "47", "411",
                           "58", "59", "69", "610", "710", "711",
                           "815", "812", "912", "913", "1013", "1014", "1114", "1118",
                           "1215", "1216", "1316", "1317", "1417", "1418",
                           "15D", "16D", "16E", "17E", "17F", "18F"]
        self.links = {
            "A1": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "A2": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "B2": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "B3": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "C3": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "C4": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "18": Link(link_capacity=100.0, link_delay=args.link_delay, egress_count=3),
            "15": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "25": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "26": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "36": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "37": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "47": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "411": Link(link_capacity=100.0, link_delay=args.link_delay, egress_count=3),
            "58": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "59": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "69": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "610": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "710": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "711": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "815": Link(link_capacity=100.0, link_delay=args.link_delay, egress_count=3),
            "812": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "912": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "913": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "1013": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "1014": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "1114": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "1118": Link(link_capacity=100.0, link_delay=args.link_delay, egress_count=3),
            "1215": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "1216": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "1316": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "1317": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "1417": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "1418": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "15D": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "16D": Link(link_capacity=300.0, link_delay=args.link_delay, egress_count=3),
            "16E": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
            "17E": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "17F": Link(link_capacity=600.0, link_delay=args.link_delay, egress_count=3),
            "18F": Link(link_capacity=400.0, link_delay=args.link_delay, egress_count=3),
        }
        # these links make up of a total number of more than 100 paths !!!

        if args.flow_type == "synthetic":
            self.point_count = 628
            self.flow_AD = 3 * np.array([58,52,52,54,54,43,45,51,42,42,48,44,42,42,43,38,46,48,41,44,39,37,40,44,47,39,38,42,52,42,39,45,39,39,42,38,39,39,41,42,44,40,38,44,37,36,41,30,30,39,42,44,35,20,21,23,19,18,21,18,17,16,15,16,17,16,20,25,20,20,26,19,19,18,17,18,27,22,24,32,23,26,30,26,23,25,26,23,26,20,23,30,28,27,33,29,28,24,22,24,30,21,26,30,25,27,34,28,31,30,33,43,44,40,39,46,44,49,55,46,55,61,60,46,46,36,48,51,40,39,50,50,48,50,52,55,74,52,61,70,79,56,61,56,53,57,62,55,78,58,53,63,49,46,61,46,51,50,51,56,70,51,49,51,47,60,61,52,61,55,66,60,69,54,56,86,87,89,99,82,86,79,82,86,98,78,42,50,46,54,56,64,53,67,77,56,72,69,58,65,57,49,46,47,57,53,60,58,55,54,40,44,36,42,48,40,46,38,39,39,58,44,40,45,43,39,53,43,44,42,44,51,54,46,46,61,51,53,49,39,41,40,32,40,51,45,46,50,47,46,52,41,49,46,53,37,41,35,48,62,58,59,64,57,56,51,51,56,60,53,54,62,55,49,49,49,40,62,67,70,65,55,52,60,55,58,51,46,47,46,44,48,49,43,41,48,45,42,43,37,35,32,26,37,42,40,41,48,39,29,39,39,36,34,39,34,46,45,41,47,37,38,37,32,28,28,27,25,31,23,23,28,31,29,32,29,28,28,22,35,34,10,8,14,9,9,14,9,9,11,11,10,13,11,8,16,11,12,15,12,13,15,15,14,16,11,16,24,15,18,19,25,15,13,13,12,20,15,13,23,11,13,40,15,14,17,15,16,19,18,25,38,36,35,42,33,33,37,41,34,33,30,29,42,37,40,42,38,47,40,42,42,51,48,52,59,54,54,64,53,54,49,45,31,43,51,45,35,49,46,54,44,42,57,50,55,59,60,60,64,56,48,57,51,51,48,47,46,59,53,58,48,78,82,65,48,47,52,60,74,57,55,52,50,65,53,83,110,108,98,91,93,92,100,107,104,93,81,91,79,78,104,78,76,68,66,57,57,75,67,57,52,46,44,36,42,42,32,35,36,34,34,36,31,35,31,34,28,41,37,35,25,27,28,40,33,32,29,31,32,38,32,28,26,32,31,47,31,28,25,29,30,36,28,31,22,24,25,31,22,19,19,18,27,37,33,31,25,39,38,42,63,77,46,40,40,44,38,43,53,52,48,56,58,58,57,45,59,61,55,53,55,46,48,55,43,44,46,41,33,31,25,29,37,41,34,40,35,36,34,27,28,39,38,43,33,31,35,39,30,34,31,25,32,50,50,53,48,28,31,46,33,44,45,40,40,48,41,40,39,37,35,37,37,34,33,26,25])
            self.flow_AE = 3 * np.array([67,63,58,53,54,58,65,60,62,59,54,53,59,64,68,61,67,56,61,72,62,71,60,58,64,68,81,73,79,78,80,77,78,78,78,74,66,68,73,61,65,61,70,70,72,75,71,60,58,60,66,62,60,57,76,75,80,70,60,56,52,57,73,69,65,62,59,64,68,70,60,68,65,80,79,74,75,63,67,64,65,63,64,55,53,56,64,59,61,54,54,58,58,59,52,47,49,63,56,47,47,39,50,47,51,46,38,38,44,42,49,47,52,59,68,75,77,72,66,63,64,61,64,56,59,54,62,60,66,67,58,53,64,56,49,45,47,42,47,53,51,49,45,42,46,45,48,43,41,37,39,41,46,49,41,38,48,46,49,46,47,43,45,49,50,60,54,54,55,47,54,53,52,49,44,58,52,52,50,48,49,54,59,50,48,44,47,53,60,64,59,51,42,36,44,42,42,31,38,45,42,41,36,41,50,61,54,45,49,45,46,45,48,47,39,45,48,48,49,40,42,37,45,41,44,48,40,43,40,38,42,41,41,39,42,41,44,47,39,38,45,39,51,43,42,34,40,43,46,42,39,35,43,42,43,42,47,41,47,52,50,43,35,40,37,34,35,33,37,47,51,52,51,46,46,50,52,58,65,59,56,49,50,56,55,47,46,52,51,52,59,53,52,52,55,46,43,56,53,57,65,65,65,58,63,62,69,69,73,66,56,58,59,62,70,66,75,71,75,85,74,75,76,74,70,71,77,74,86,75,73,69,69,73,68,68,65,73,83,78,92,82,70,75,74,72,65,74,73,73,75,62,70,74,68,75,77,70,65,67,80,76,66,62,61,54,54,64,66,58,62,59,61,62,55,57,60,54,56,53,60,63,48,54,54,67,64,55,54,48,51,53,54,57,48,49,59,54,56,54,54,56,61,64,61,55,53,57,58,59,55,49,56,48,48,53,58,52,43,53,52,49,60,57,63,54,52,56,56,57,49,57,63,57,59,48,50,52,57,54,56,55,47,46,49,42,50,56,62,52,61,64,63,67,64,61,56,55,58,57,61,61,69,67,66,64,51,49,51,47,48,43,49,45,45,44,52,56,50,50,48,45,47,43,51,46,48,50,53,53,55,57,57,50,58,46,54,44,45,52,48,48,45,40,44,44,47,43,49,41,54,43,48,46,41,46,59,58,59,49,50,42,49,54,62,54,47,50,53,48,49,41,46,38,47,53,55,55,53,53,56,60,59,57,62,40,40,47,48,40,36,37,40,36,40,35,42,32,36,38,38,40,40,42,48,48,51,40,45,44,48,47,48,50,47,44,41,41,48,54,72,50,49,53,55,54,53,51,58,54,58,53,63,61,58,53,57,61,57,50,56,52,59,55,61,54,63,66,67,63,58,58,64,61,57,57,63,57,63,62,60,59,55,56,67,66,60,61])
            self.flow_BD = 3 * np.array([13,14,14,13,14,14,12,15,15,16,17,18,20,17,14,15,18,11,12,11,12,13,16,14,12,12,17,23,18,22,20,29,21,18,23,19,22,26,34,29,43,38,41,37,37,41,48,42,39,43,48,60,47,40,47,52,46,45,61,61,57,54,71,54,69,45,45,72,55,59,72,55,61,76,74,75,78,89,60,72,69,58,81,61,71,49,59,54,58,43,44,50,53,44,61,50,54,48,40,54,43,40,57,77,75,51,58,52,54,57,61,60,64,54,70,82,81,74,58,58,57,74,52,63,79,67,58,64,49,50,66,58,69,67,58,56,64,63,55,54,56,54,46,32,35,41,33,37,45,28,27,33,24,25,31,23,31,22,19,29,32,28,30,33,27,25,46,36,37,39,40,26,28,21,26,31,31,24,30,20,20,25,22,23,24,23,25,30,25,27,44,44,45,44,40,41,55,42,35,43,31,30,36,28,30,38,31,33,42,35,41,41,16,30,40,36,33,36,30,29,31,21,23,30,32,29,44,41,33,39,29,33,38,37,36,40,30,26,37,26,29,31,23,33,54,45,52,66,56,54,64,55,40,40,29,29,30,21,26,28,26,22,28,25,20,25,22,26,26,15,16,21,14,12,12,15,11,10,11,9,18,12,15,17,11,12,16,16,16,18,14,18,24,16,18,23,16,15,18,14,14,12,11,9,12,9,14,17,7,10,19,11,12,9,11,15,21,23,25,24,20,17,22,22,20,21,20,20,30,28,27,32,36,38,49,42,44,54,50,52,64,48,54,59,54,54,55,45,47,49,58,44,43,40,62,67,62,62,70,62,59,57,53,44,64,44,48,58,56,52,56,63,87,74,74,77,87,73,78,89,71,73,82,74,63,63,58,73,77,68,60,98,78,86,84,86,73,76,82,58,65,61,54,71,55,62,69,70,62,60,58,66,54,47,44,53,56,56,53,59,64,51,45,45,51,45,30,50,49,47,49,44,54,46,41,38,34,34,42,40,31,34,30,34,39,28,28,50,61,49,40,57,35,40,31,25,24,25,27,29,35,33,31,42,42,47,60,45,45,43,45,44,44,41,38,47,43,45,46,35,39,49,45,47,55,48,51,69,54,55,57,48,48,51,42,35,39,37,57,59,35,56,47,36,37,34,31,38,37,51,50,55,33,36,45,40,45,48,37,36,31,32,34,36,35,34,38,33,36,36,35,43,42,50,42,47,36,38,39,36,36,35,31,35,32,23,22,29,30,26,32,36,31,35,33,31,35,19,17,19,14,12,11,12,15,14,13,15,29,20,20,20,12,8,14,12,13,13,13,14,17,13,14,14,15,14,20,12,12,17,18,13,16,17,16,20,16,14,21,37,64,19,18,23,30,23,25,31,23,52,34,30,24,33,27,41,71,29,25,47,52,45,50,47,47,49,43,49])
            self.flow_BF = 3 * np.array([50,42,34,36,39,46,59,48,50,47,46,49,50,50,45,43,49,45,50,46,44,46,49,55,58,60,56,56,59,58,58,50,56,50,51,57,56,57,49,47,55,63,70,60,58,61,76,84,76,66,61,64,66,67,74,70,71,67,64,68,68,67,70,69,64,65,67,60,54,53,57,60,61,61,54,55,56,53,58,56,57,64,65,66,66,65,63,71,62,53,56,55,51,51,56,59,59,71,71,60,63,60,69,64,56,56,63,58,63,64,62,61,61,54,58,51,55,53,51,55,59,63,65,64,64,63,58,54,50,48,45,43,47,49,44,40,45,44,47,43,39,39,39,38,38,40,32,40,41,42,43,42,49,39,42,43,42,43,38,38,41,39,47,62,65,51,46,46,52,52,44,41,46,51,52,49,56,45,45,50,53,51,48,47,52,46,67,72,73,66,67,68,67,57,47,62,65,55,58,56,60,59,57,60,56,47,40,45,44,37,43,34,40,39,43,46,52,51,52,48,56,52,53,47,52,49,47,48,53,54,49,47,57,61,54,44,50,40,54,47,52,55,49,46,44,44,51,48,53,45,50,55,54,51,51,47,49,43,51,44,50,41,58,56,56,56,54,61,58,48,50,49,59,39,41,44,45,38,34,38,39,44,48,41,49,41,44,45,50,43,44,48,43,42,53,47,51,52,62,60,55,54,48,51,60,57,57,49,56,57,55,60,74,72,62,68,57,54,61,54,62,48,50,53,59,60,56,57,66,64,66,66,81,75,78,68,72,74,70,72,81,69,71,72,80,72,77,75,79,79,79,77,73,70,74,74,85,73,77,82,89,90,73,69,73,70,75,52,62,54,59,58,63,64,62,62,62,59,69,66,64,54,59,58,62,61,54,60,63,57,59,57,64,54,58,62,51,53,44,49,53,47,50,53,61,51,52,51,49,52,47,48,53,51,54,42,47,40,42,45,47,48,43,45,46,41,45,39,47,36,42,48,47,45,42,42,49,47,50,43,51,42,46,46,46,49,42,43,49,44,46,42,51,39,43,42,42,45,43,42,52,51,48,49,51,44,47,51,58,52,52,49,43,45,63,56,54,57,63,71,73,74,64,54,50,44,47,45,48,40,47,38,37,36,35,34,33,30,31,34,40,32,37,42,43,44,43,45,44,38,46,43,45,44,44,43,47,46,43,45,50,50,42,33,38,39,48,46,47,44,43,47,46,45,44,42,44,43,48,49,50,50,40,43,43,36,41,36,40,44,42,47,49,51,45,49,50,43,43,33,42,44,45,44,46,49,41,42,42,50,47,48,46,39,39,40,43,43,33,33,38,34,42,34,36,38,40,39,40,43,37,41,52,46,46,47,46,47,52,54,54,57,47,49,52,50,49,53,59,67,58,58,61,58,66,50,55,54,55,54,54,47,44,59,58,54,65,60])
            self.flow_CE = 0.5 * (self.flow_AD + self.flow_AE)
            self.flow_CF = 0.5 * (self.flow_BD + self.flow_BF)
        elif args.flow_type == "Abilene":
            raise ValueError("Abilene is a real trace, and it is not defined in the public codebase!")
        else:
            raise ValueError("args.flow_type is not defined!")

    def reset(self):
        self.data_step = np.random.randint(self.point_count) # set the started flow point

        self.router_action = {"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5], "C": [0.5, 0.5, 0.5, 0.5],
                              "1": [0.5, 0.5, 0.5, 0.5], "2": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                              "3": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "4": [0.5, 0.5, 0.5, 0.5],
                              "5": [0.5, 0.5, 0.5, 0.5], "6": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "7": [0.5, 0.5, 0.5, 0.5],
                              "8": [0.5, 0.5], "9": [0.5, 0.5, 0.5, 0.5], "10": [0.5, 0.5, 0.5, 0.5], "11": [0.5, 0.5],
                              "12": [0.5, 0.5], "13": [0.5, 0.5], "14": [0.5, 0.5]}

        self.link_usage_average = {}
        for link_name in self.link_names:
            self.link_usage_average[link_name] = 0.5

        state_all = self._get_current_state()
        return state_all

    def step(self, action_all):
        self._set_action(action_all)
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
        self.action_A, self.action_B, self.action_C, self.action_1, self.action_2, self.action_3, self.action_4, \
            self.action_5, self.action_6, self.action_7, self.action_8, self.action_9, self.action_10, self.action_11, \
            self.action_12, self.action_13, self.action_14 = action_all
        self.action_A1_for_toD, self.action_A2_for_toD, self.action_A1_for_toE, self.action_A2_for_toE = self.action_A
        self.action_B2_for_toD, self.action_B3_for_toD, self.action_B2_for_toF, self.action_B3_for_toF = self.action_B
        self.action_C3_for_toE, self.action_C4_for_toE, self.action_C3_for_toF, self.action_C4_for_toF = self.action_C
        self.action_18_for_toD, self.action_15_for_toD, self.action_18_for_toE, self.action_15_for_toE = self.action_1
        self.action_25_for_toD, self.action_26_for_toD, self.action_25_for_toE, self.action_26_for_toE, self.action_25_for_toF, self.action_26_for_toF = self.action_2
        self.action_36_for_toD, self.action_37_for_toD, self.action_36_for_toE, self.action_37_for_toE, self.action_36_for_toF, self.action_37_for_toF = self.action_3
        self.action_47_for_toE, self.action_411_for_toE, self.action_47_for_toF, self.action_411_for_toF = self.action_4
        self.action_58_for_toD, self.action_59_for_toD, self.action_58_for_toE, self.action_59_for_toE = self.action_5
        self.action_69_for_toD, self.action_610_for_toD, self.action_69_for_toE, self.action_610_for_toE, self.action_69_for_toF, self.action_610_for_toF = self.action_6
        self.action_710_for_toE, self.action_711_for_toE, self.action_710_for_toF, self.action_711_for_toF = self.action_7
        self.action_815_for_toD, self.action_812_for_toD = self.action_8
        self.action_912_for_toD, self.action_913_for_toD, self.action_912_for_toE, self.action_913_for_toE = self.action_9
        self.action_1013_for_toE, self.action_1014_for_toE, self.action_1013_for_toF, self.action_1014_for_toF = self.action_10
        self.action_1114_for_toF, self.action_1118_for_toF = self.action_11
        self.action_1215_for_toD, self.action_1216_for_toD = self.action_12
        self.action_1316_for_toE, self.action_1317_for_toE = self.action_13
        self.action_1417_for_toF, self.action_1418_for_toF = self.action_14

    def _inject_flow_to_ingress(self):
        # must call get_data before set_data(add_data), or some data will be overwrite
        # however, we can ignore this, and just take the following as the situation where router/link_delay-1
        self.routers["A"].buffer.add_data([self.flow_AD[self.data_step], self.flow_AE[self.data_step], 0.0])
        flow_toD, flow_toE, flow_toF = self.routers["A"].buffer.get_data()  # flow_toF==0
        tmp1 = flow_toD * self.action_A1_for_toD
        tmp2 = flow_toE * self.action_A1_for_toE
        self.links["A1"].buffer.add_data([tmp1, tmp2, 0.0])
        self.links["A1"].total_flow += (tmp1 + tmp2 + 0.0)
        tmp1 = flow_toD * self.action_A2_for_toD
        tmp2 = flow_toE * self.action_A2_for_toE
        self.links["A2"].buffer.add_data([tmp1, tmp2, 0.0])
        self.links["A2"].total_flow += (tmp1 + tmp2 + 0.0)

        self.routers["B"].buffer.add_data([self.flow_BD[self.data_step], 0.0, self.flow_BF[self.data_step]])
        flow_toD, flow_toE, flow_toF = self.routers["B"].buffer.get_data()  # flow_toE==0
        tmp1 = flow_toD * self.action_B2_for_toD
        tmp2 = flow_toF * self.action_B2_for_toF
        self.links["B2"].buffer.add_data([tmp1, 0.0, tmp2])
        self.links["B2"].total_flow += (tmp1 + 0.0 + tmp2)
        tmp1 = flow_toD * self.action_B3_for_toD
        tmp2 = flow_toF * self.action_B3_for_toF
        self.links["B3"].buffer.add_data([tmp1, 0.0, tmp2])
        self.links["B3"].total_flow += (tmp1 + 0.0 + tmp2)

        self.routers["C"].buffer.add_data([0.0, self.flow_CE[self.data_step], self.flow_CF[self.data_step]])
        flow_toD, flow_toE, flow_toF = self.routers["C"].buffer.get_data()  # flow_toD==0
        tmp1 = flow_toE * self.action_C3_for_toE
        tmp2 = flow_toF * self.action_C3_for_toF
        self.links["C3"].buffer.add_data([0.0, tmp1, tmp2])
        self.links["C3"].total_flow += (0.0 + tmp1 + tmp2)
        tmp1 = flow_toE * self.action_C4_for_toE
        tmp2 = flow_toF * self.action_C4_for_toF
        self.links["C4"].buffer.add_data([0.0, tmp1, tmp2])
        self.links["C4"].total_flow += (0.0 + tmp1 + tmp2)
    def _split_flow_to_links(self, router_name, flow_toD, flow_toE, flow_toF):
        if router_name in ["1", "5", "9"]: # two destinations: toD, toE
            tmp1 = flow_toD * self.router_action[router_name][0] # self.action_18_for_toD
            tmp2 = flow_toE * self.router_action[router_name][2] # self.action_18_for_toE
            tmp3 = 0.0
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = flow_toD * self.router_action[router_name][1] # self.action_15_for_toD
            tmp2 = flow_toE * self.router_action[router_name][3] # self.action_15_for_toE
            tmp3 = 0.0
            flow_on_link1 = [tmp1, tmp2, tmp3]
        elif router_name in ["2", "3", "6"]: # three destinations: toD, toE, toF
            tmp1 = flow_toD * self.router_action[router_name][0] # self.action_25_for_toD
            tmp2 = flow_toE * self.router_action[router_name][2] # self.action_25_for_toE
            tmp3 = flow_toF * self.router_action[router_name][4] # self.action_25_for_toF
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = flow_toD * self.router_action[router_name][1] # self.action_26_for_toD
            tmp2 = flow_toE * self.router_action[router_name][3] # self.action_26_for_toE
            tmp3 = flow_toF * self.router_action[router_name][5] # self.action_26_for_toF
            flow_on_link1 = [tmp1, tmp2, tmp3]
        elif router_name in ["4", "7", "10"]: # two destinations: toE, toF
            tmp1 = 0.0
            tmp2 = flow_toE * self.router_action[router_name][0] # self.action_47_for_toE
            tmp3 = flow_toF * self.router_action[router_name][2] # self.action_47_for_toF
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = 0.0
            tmp2 = flow_toE * self.router_action[router_name][1] # self.action_411_for_toE
            tmp3 = flow_toF * self.router_action[router_name][3] # self.action_411_for_toF
            flow_on_link1 = [tmp1, tmp2, tmp3]
        elif router_name in ["8", "12"]: # ONE destination: toD
            tmp1 = flow_toD * self.router_action[router_name][0] # self.action_815_for_toD
            tmp2 = 0.0
            tmp3 = 0.0
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = flow_toD * self.router_action[router_name][1] # self.action_812_for_toD
            tmp2 = 0.0
            tmp3 = 0.0
            flow_on_link1 = [tmp1, tmp2, tmp3]
        elif router_name in ["11", "14"]: # ONE destination: toF
            tmp1 = 0.0
            tmp2 = 0.0
            tmp3 = flow_toF * self.router_action[router_name][0] # self.action_1114_for_toF
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = 0.0
            tmp2 = 0.0
            tmp3 = flow_toF * self.router_action[router_name][1] # self.action_1118_for_toF
            flow_on_link1 = [tmp1, tmp2, tmp3]
        elif router_name in ["13"]: # ONE destination: toE
            tmp1 = 0.0
            tmp2 = flow_toE * self.router_action[router_name][0] # self.action_1316_for_toE
            tmp3 = 0.0
            flow_on_link0 = [tmp1, tmp2, tmp3]
            tmp1 = 0.0
            tmp2 = flow_toE * self.router_action[router_name][1] # self.action_1317_for_toE
            tmp3 = 0.0
            flow_on_link1 = [tmp1, tmp2, tmp3]
        else:
            raise ValueError("router_name is not defined!")
        return flow_on_link0, flow_on_link1
    def _discharge_flow_from_egress(self, router_name, flow_toD, flow_toE, flow_toF):
        if router_name == "15":
            self.links["15D"].buffer.add_data([flow_toD, 0.0, 0.0])
            self.links["15D"].total_flow += flow_toD
        elif router_name == "16":
            self.links["16D"].buffer.add_data([flow_toD, 0.0, 0.0])
            self.links["16D"].total_flow += flow_toD
            self.links["16E"].buffer.add_data([0.0, flow_toE, 0.0])
            self.links["16E"].total_flow += flow_toE
        elif router_name == "17":
            self.links["17E"].buffer.add_data([0.0, flow_toE, 0.0])
            self.links["17E"].total_flow += flow_toE
            self.links["17F"].buffer.add_data([0.0, 0.0, flow_toF])
            self.links["17F"].total_flow += flow_toF
        elif router_name == "18":
            self.links["18F"].buffer.add_data([0.0, 0.0, flow_toF])
            self.links["18F"].total_flow += flow_toF
    def _simulate_one_step(self):
        self.data_step += 1
        self.data_step %= self.point_count

        self._inject_flow_to_ingress()

        for router_name in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]:
            # get the aggregated flow from upper links
            aggregated_flow_toD, aggregated_flow_toE, aggregated_flow_toF = 0.0, 0.0, 0.0
            for link_name in self.routers[router_name].upper_links:
                temp_toD, temp_toE, temp_toF = self.links[link_name].buffer.get_data()
                aggregated_flow_toD += temp_toD
                aggregated_flow_toE += temp_toE
                aggregated_flow_toF += temp_toF

            # set the aggregated flow into this router's buffer
            self.routers[router_name].buffer.add_data([aggregated_flow_toD, aggregated_flow_toE, aggregated_flow_toF])

            # get the flow in this router's buffer
            flow_toD, flow_toE, flow_toF = self.routers[router_name].buffer.get_data()

            if router_name in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]:
                # split the flow into this router's direct down-stream links
                flow_on_link0, flow_on_link1 = self._split_flow_to_links(router_name, flow_toD, flow_toE, flow_toF)
                link_names = self.routers[router_name].direct_links
                self.links[link_names[0]].buffer.add_data(flow_on_link0)
                self.links[link_names[0]].total_flow += sum(flow_on_link0)
                self.links[link_names[1]].buffer.add_data(flow_on_link1)
                self.links[link_names[1]].total_flow += sum(flow_on_link1)
            elif router_name in ["15", "16", "17", "18"]:
                self._discharge_flow_from_egress(router_name, flow_toD, flow_toE, flow_toF)

    def _get_influence_of_last_action(self):
        # the average usage in the last control cycle (i.e., action_effective_step)
        for link_name in self.link_names:
            self.link_usage_average[link_name] = \
                self.links[link_name].total_flow / self.links[link_name].link_capacity / self.args.action_effective_step
            self.links[link_name].total_flow = 0.0  # reset total_flow for the next control cycle

        # calculate the reward
        max_usage_rate = max(self.link_usage_average.values())
        global_reward = float(1.0 - max_usage_rate)
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
