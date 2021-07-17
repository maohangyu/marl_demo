import argparse
from common import preprocess_graph
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser("Deep RL for Cooperative Multi-Agent Control with Discrete Action Space")
    parser.add_argument("--env-name", type=str, default="navigation6v6")
    parser.add_argument("--agent-name", type=str, default="IDQN", help="IDQN, VDN, QMIX, NCC_VDN, NCC_QMIX, etc.")
    parser.add_argument("--head-count", type=int, default=4, help="number of heads in DGN, etc.")
    parser.add_argument("--hidden-layer-count", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--results-dir", type=str, default="./results/")
    args = parser.parse_args()
    args.exp_name = args.env_name + "-" + args.agent_name

    if args.env_name in ["navigation2v2", "navigation3v3", "navigation4v4", "navigation6v6", "navigation10v10"]:
        if args.env_name == "navigation2v2":
            args.agent_count = 2
            args.landmark_count = 2
            args.observation_dim_list = [8, 8]
            args.action_dim_list = [5, 5]  # stop, up, right, down, left
            args.adj = [[0, 1],
                        [1, 0]]  # this ADJ does not contain self-connections
            if args.agent_name in ['Contrastive_VDN', 'Contrastive_QMIX']:  # specially designed for Contrastive_*
                raise ValueError('navigation2v2 is unsuitable for testing Contrastive_NCC/QMIX models ...')
        elif args.env_name == "navigation3v3":
            args.agent_count = 3
            args.landmark_count = 3
            args.observation_dim_list = [12, 12, 12]
            args.action_dim_list = [5, 5, 5]
            args.adj = [[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]]
            if args.agent_name in ['Contrastive_VDN', 'Contrastive_QMIX']:  # specially designed for Contrastive_*
                raise ValueError('navigation3v3 is unsuitable for testing Contrastive_NCC/QMIX models ...')
        elif args.env_name == "navigation4v4":
            args.agent_count = 4
            args.landmark_count = 4
            args.observation_dim_list = [16, 16, 16, 16]
            args.action_dim_list = [5, 5, 5, 5]
            args.adj = [[0, 1, 1, 1],
                        [1, 0, 1, 1],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0]]
            if args.agent_name in ['Contrastive_VDN', 'Contrastive_QMIX']:  # specially designed for Contrastive_NCC
                args.adj = [[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]]
        elif args.env_name == "navigation6v6":
            args.agent_count = 6
            args.landmark_count = 6
            args.observation_dim_list = [24, 24, 24, 24, 24, 24]
            args.action_dim_list = [5, 5, 5, 5, 5, 5]
            args.adj = [[0, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 0]]
            if args.agent_name in ['Contrastive_VDN', 'Contrastive_QMIX']:  # specially designed for Contrastive_*
                args.adj = [[0, 1, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 1, 0]]
        elif args.env_name == "navigation10v10":
            args.agent_count = 10
            args.landmark_count = 10
            args.observation_dim_list = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
            args.action_dim_list = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            args.adj = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
            if args.agent_name in ['Contrastive_VDN', 'Contrastive_QMIX']:  # specially designed for Contrastive_*
                args.adj = [[0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]
        args.env_rename = "CooperativeNavigation"
        args.adj_norm = preprocess_graph(args.adj)
        args.env_bound = 10
        args.env_dim = 2  # do not change this, since it is co-related with observation_dim_list
        args.action_effective_step = 1
        #
        args.buffer_size = 100000
        args.batch_size = 64
        args.exp_count = 5
        args.hidden_dim = 32
        args.lr = 1e-3
        args.clipped_norm_value = 10.0
        args.seed = 0
        #
        args.episode_count = 2000
        args.epsilon = 1.0
        args.epsilon_delta = 0.001
        args.epsilon_end = 0.0
        args.episode_for_updating_T = 20
        args.max_episode_len = 10
        args.gamma = 0.9
        #
        args.alpha_L2 = 0.1  # the weight of L2-loss in NCC/Pikachu
        args.alpha_KL = 0.1  # the weight of KL-loss in NCC/Pikachu
        args.alpha_PCA = 0.1  # the weight of PCA-loss in Pikachu
        args.alpha_PCA_end = 0.0  # 0.1 or 0.0, where 0.1 means fixed alpha_PCA
        args.alpha_PCA_delta = (args.alpha_PCA - args.alpha_PCA_end) / args.episode_count
        args.alpha_CON = 0.1  # the weight of contrastive-loss
    elif args.env_name in ["simple", "simple_adversary", "simple_crypto", "simple_push", "simple_reference", "simple_speaker_listener", "simple_spread", "simple_tag", "simple_world_comm"]:
        from MPE import make_env
        args.env = make_env.make_env(args.env_name, benchmark=False)
        # print("********************8")
        args.env_rename = "MPE"
        args.agent_count = len(args.env.world.agents)
        args.landmark_count = len(args.env.world.landmarks)
        print('args.agent_count ==>', args.agent_count)
        print('args.landmark_count ==>', args.landmark_count)
        args.observation_dim_list = [i.shape[0] for i in args.env.observation_space]
        if args.env_name == "simple_reference":
            raise ValueError('something wrong for the [simple_reference] environment ...')
            # print("args.env.action_spaces", args.env.action_space)
            args.action_dim_list = [i.num_discrete_space for i in args.env.action_space]
        elif args.env_name == "simple_world_comm":
            raise ValueError('something wrong for the [simple_reference] environment ...')
            # print("args.env.action_spaces", args.env.action_space)
            args.action_dim_list = []
            args.action_dim_list.append(args.env.action_space[0].num_discrete_space)
            args.action_dim_list.append(args.env.action_space[1].n)
            args.action_dim_list.append(args.env.action_space[2].n)
            args.action_dim_list.append(args.env.action_space[3].n)
            args.action_dim_list.append(args.env.action_space[4].n)
            args.action_dim_list.append(args.env.action_space[5].n)
        else:
            args.action_dim_list = [i.n for i in args.env.action_space]
        args.adj = np.ones((args.agent_count, args.agent_count), dtype=int) - np.eye(args.agent_count, dtype=int)
        args.adj_norm = preprocess_graph(args.adj)
        args.env_bound = 10
        args.env_dim = 2  # do not change this, since it is co-related with observation_dim_list
        args.action_effective_step = 1
        #
        args.buffer_size = 100000
        args.batch_size = 64
        args.exp_count = 5
        args.hidden_dim = 32
        args.lr = 1e-3
        args.clipped_norm_value = 10.0
        args.seed = 0
        #
        args.episode_count = 2000
        args.epsilon = 1.0
        args.epsilon_delta = 0.001
        args.epsilon_end = 0.0
        args.episode_for_updating_T = 20
        args.max_episode_len = 10
        args.gamma = 0.9
        #
        args.alpha_L2 = 0.1  # the weight of L2-loss in NCC/Pikachu
        args.alpha_KL = 0.1  # the weight of KL-loss in NCC/Pikachu
        args.alpha_PCA = 0.1  # the weight of PCA-loss in Pikachu
        args.alpha_PCA_end = 0.0  # 0.1 or 0.0, where 0.1 means fixed alpha_PCA
        args.alpha_PCA_delta = (args.alpha_PCA - args.alpha_PCA_end) / args.episode_count
        args.alpha_CON = 0.1  # the weight of contrastive-loss
    elif args.env_name in ["2s3z", "3s5z", "1c3s5z", "3m"]:
        if args.env_name == '2s3z':
            args.n_agents = 5
            args.n_actions = 11
            args.obs_shape = 80
            args.state_shape = 120
            args.episode_limit = 120
            #
            args.agent_count = args.n_agents
            args.observation_dim_list = [args.obs_shape for _ in range(args.agent_count)]
            args.action_dim_list = [args.n_actions for _ in range(args.agent_count)]
            args.adj = [[0, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 0]]
        elif args.env_name == '3s5z':
            args.n_agents = 8
            args.n_actions = 14
            args.obs_shape = 128
            args.state_shape = 216
            args.episode_limit = 150
            #
            args.agent_count = args.n_agents
            args.observation_dim_list = [args.obs_shape for _ in range(args.agent_count)]
            args.action_dim_list = [args.n_actions for _ in range(args.agent_count)]
            args.adj = [[0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0]]
        elif args.env_name == '1c3s5z':
            args.n_agents = 9
            args.n_actions = 15
            args.obs_shape = 162
            args.state_shape = 270
            args.episode_limit = 180
            #
            args.agent_count = args.n_agents
            args.observation_dim_list = [args.obs_shape for _ in range(args.agent_count)]
            args.action_dim_list = [args.n_actions for _ in range(args.agent_count)]
            args.adj = [[0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0]]
        elif args.env_name == '3m':
            args.n_agents = 3
            args.n_actions = 9
            args.obs_shape = 30
            args.state_shape = 48
            args.episode_limit = 60
            #
            args.agent_count = args.n_agents
            args.observation_dim_list = [args.obs_shape for _ in range(args.agent_count)]
            args.action_dim_list = [args.n_actions for _ in range(args.agent_count)]
            args.adj = [[0, 1, 1],
                        [1, 0, 1],
                        [1, 1, 0]]
        args.env_rename = "StarCraftII"
        args.adj_norm = preprocess_graph(args.adj)
        args.difficulty = '7'  # the difficulty of the game, 7==VeryHard
        args.game_version = 'latest'
        args.step_mul = 8  # how many steps to make an action
        args.replay_dir = '/home/noahrl/workspace/sc2_replay/'  # MUST be an absolute path to save the replay
        args.evaluate_episode = 20  # number of episode to evaluate the agent
        args.evaluate_cycle = 100  # every evaluate_cycle episodes to evaluate the agent
        args.last_action = False  # whether to use the last action to choose action
        #
        args.buffer_size = int(5e3)
        args.batch_size = 32
        args.hidden_dim = 32
        args.hidden_dim_rnn = 64
        args.lr = 5e-4
        args.clipped_norm_value = 10.0
        args.seed = 0
        #
        args.episode_count = 10000
        args.epsilon = 1.0
        args.epsilon_end = 0.05
        args.epsilon_anneal_scale = 'step'  # step or episode
        anneal_steps = 50000
        args.epsilon_delta = (args.epsilon - args.epsilon_end) / anneal_steps
        #
        args.exp_count = 5
        args.train_steps_one_episode = 1
        args.episode_for_saving_model = 2000  # how often to save the model
        args.episode_for_updating_T = 200  # how often to update the target_net
        args.target_update_cycle = args.episode_for_updating_T
        args.max_episode_len = 1000
        args.gamma = 0.99
        #
        args.alpha_L2 = 0.1  # the weight of L2-loss in NCC/QWEIGHT
        args.alpha_KL = 0.1  # the weight of KL-loss in NCC/QWEIGHT
    else:
        raise ValueError("args.env_name is not defined! ...")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
