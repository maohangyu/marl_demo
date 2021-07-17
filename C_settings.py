import argparse
import numpy as np
from common import preprocess_graph


def parse_arguments():
    parser = argparse.ArgumentParser("Deep RL for Cooperative Multi-Agent Control with Continuous Action Space")
    parser.add_argument("--env-name", type=str, default="routing6v4")
    parser.add_argument("--agent-name", type=str, default="IND_AC", help="IND_AC, MADDPG, ATT_MADDPG, NCC_AC")
    parser.add_argument("--head-count", type=int, default=4, help="number of heads in ATT_MADDPG, MAAC, etc.")
    parser.add_argument("--hidden-layer-count", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--results-dir", type=str, default="./results/")
    args = parser.parse_args()
    args.exp_name = args.env_name + "-" + args.agent_name

    if args.env_name in ["routing6v4", "routing12v20", "routing24v128"]:
        if args.env_name == "routing6v4":
            controllable_router_list = ["A", "B"]
            args.agent_count = len(controllable_router_list)
            args.router_delay = 2
            args.link_delay = 5
            args.action_effective_step = args.router_delay + args.link_delay
            action_dim_A = 2  # toC
            action_dim_B = 2  # toC
            args.action_dim_list = [action_dim_A, action_dim_B]
            # (flow_demand + usage_history*direct_link_count)*destination + action + usage_ave*direct_link_count
            observation_dim_A = (args.router_delay + args.link_delay * 2) * 2 + action_dim_A + 1 * 2  # 28
            observation_dim_B = (args.router_delay + args.link_delay * 2) * 2 + action_dim_B + 1 * 2  # 28
            args.observation_dim_list = [observation_dim_A, observation_dim_B]
            args.state_dim_list = args.observation_dim_list
            # this is used by GNN
            args.adj = [[0, 1],
                        [1, 0]]
            #
            args.episode_count = 1000  # 2000
            args.epsilon = 1.0
            args.epsilon_delta = 0.001
            args.epsilon_end = 0.0
            args.max_episode_len = 20
            args.gamma = 0.95
            #
            args.alpha_L2 = 0.1  # the weight of L2-loss in NCC-AC
            args.alpha_KL = 0.1  # the weight of KL-loss in NCC-AC
            args.alpha_CON = 0.1  # the weight of contrastive-loss
        elif args.env_name == "routing12v20":
            controllable_router_list = ["A", "B", "1", "2", "3", "4", "5"]
            args.agent_count = len(controllable_router_list)
            args.router_delay = 2
            args.link_delay = 5
            args.action_effective_step = args.router_delay + args.link_delay
            action_dim_A = 2 + 2  # toC + toD
            action_dim_B = 2 + 2  # toC + toD
            action_dim_1 = 2  # toC (toD only one path)
            action_dim_2 = 2 + 2  # toC + toD
            action_dim_3 = 2  # toD (toC only one path)
            action_dim_4 = 2  # toC (toD only one path)
            action_dim_5 = 2  # toD (toC only one path)
            args.action_dim_list = [action_dim_A, action_dim_B, action_dim_1, action_dim_2,
                                    action_dim_3, action_dim_4, action_dim_5]
            # (flow_demand + usage_history*direct_link_count)*destination + action + usage_ave*direct_link_count
            observation_dim_A = (args.router_delay + args.link_delay * 2) * 2 + action_dim_A + 1 * 2
            observation_dim_B = (args.router_delay + args.link_delay * 2) * 2 + action_dim_B + 1 * 2
            observation_dim_1 = (args.router_delay + args.link_delay * 2) * 2 + action_dim_1 + 1 * 2
            observation_dim_2 = (args.router_delay + args.link_delay * 2) * 2 + action_dim_2 + 1 * 2
            observation_dim_3 = (args.router_delay + args.link_delay * 2) * 2 + action_dim_3 + 1 * 2
            observation_dim_4 = (args.router_delay + args.link_delay * 2) * 2 + action_dim_4 + 1 * 2
            observation_dim_5 = (args.router_delay + args.link_delay * 2) * 2 + action_dim_5 + 1 * 2
            args.observation_dim_list = [observation_dim_A, observation_dim_B, observation_dim_1, observation_dim_2,
                                         observation_dim_3, observation_dim_4, observation_dim_5]
            args.state_dim_list = args.observation_dim_list
            # this is used by GNN
            args.adj = [[0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0],
                        [1, 1, 0, 0, 0, 1, 1],
                        [0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0]]
            #
            args.episode_count = 2000
            args.epsilon = 1.0
            args.epsilon_delta = 0.001
            args.epsilon_end = 0.0
            args.max_episode_len = 50
            args.gamma = 0.98
            #
            args.alpha_L2 = 0.1  # the weight of L2-loss in NCC-AC
            args.alpha_KL = 0.1  # the weight of KL-loss in NCC-AC
            args.alpha_CON = 0.1  # the weight of contrastive-loss in BCC
        elif args.env_name == "routing24v128":
            controllable_router_list = ["A", "B", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]
            args.agent_count = len(controllable_router_list)
            args.router_delay = 2
            args.link_delay = 5
            args.action_effective_step = args.router_delay + args.link_delay
            action_dim_A = 4
            action_dim_B = 4
            action_dim_C = 4
            action_dim_1 = 4
            action_dim_2 = 6
            action_dim_3 = 6
            action_dim_4 = 4
            action_dim_5 = 4
            action_dim_6 = 6
            action_dim_7 = 4
            action_dim_8 = 2
            action_dim_9 = 4
            action_dim_10 = 4
            action_dim_11 = 2
            action_dim_12 = 2
            action_dim_13 = 2
            action_dim_14 = 2
            args.action_dim_list = np.array([action_dim_A, action_dim_B, action_dim_C,
                                             action_dim_1, action_dim_2, action_dim_3, action_dim_4,
                                             action_dim_5, action_dim_6, action_dim_7,
                                             action_dim_8, action_dim_9, action_dim_10, action_dim_11,
                                             action_dim_12, action_dim_13, action_dim_14])
            # (flow_demand + usage_history*direct_link_count)*destination + action + usage_ave*direct_link_count
            # state_dim_A = (args.router_delay + args.link_delay * 2) * 3 + action_dim_A + 1 * 2
            observation_dim_base = (args.router_delay + args.link_delay * 2) * 3 + 1 * 2
            args.observation_dim_list = args.action_dim_list + observation_dim_base
            args.state_dim_list = args.observation_dim_list
            # this is used by GNN
            args.adj = np.array([[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
            #
            args.episode_count = 2000  # 3000
            args.epsilon = 1.0
            args.epsilon_delta = 0.001
            args.epsilon_end = 0.0
            args.max_episode_len = 100
            args.gamma = 0.99
            #
            args.alpha_L2 = 0.2  # the weight of L2-loss in NCC-AC
            args.alpha_KL = 0.2  # the weight of KL-loss in NCC-AC
            args.alpha_CON = 0.1  # the weight of contrastive-loss in BCC
        args.adj_norm = preprocess_graph(args.adj)
        args.buffer_size = 60000
        args.batch_size = 64
        args.exp_count = 5  # 2 if args.env_name == "routing24v128" else 5
        args.hidden_dim = 32
        args.lr_actor = 1e-3
        args.lr_critic = 1e-2
        args.tau = 1e-3
        args.clipped_norm_value = 10.0
        args.seed = 0
        args.flow_type = "synthetic"
    else:
        raise ValueError("args.env_name is not defined! ...")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
