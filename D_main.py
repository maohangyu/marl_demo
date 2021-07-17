import time
import numpy as np
from tensorboardX import SummaryWriter

from common import (ReplayBuffer, save_1d_data)
import D_settings


def exploration(Qvalue_all, args):
    if np.random.random() < args.epsilon:
        action_all_after_exploration = []
        for i in range(args.agent_count):
            action_id_i = np.random.randint(0, args.action_dim_list[i])  # [low, high)
            action_all_after_exploration.append(action_id_i)
        print("action_all_after_exploration ==>", action_all_after_exploration)
        return action_all_after_exploration
    else:
        action_all_before_exploration = []
        for i in range(args.agent_count):
            action_all_before_exploration.append(np.argmax(Qvalue_all[i][0]))
        print("action_all_before_exploration ==>", action_all_before_exploration)
        return action_all_before_exploration


def training(args, agent, batch, train_target_network, writer=None, training_step=0):
    observation_list, action_id_list, next_observation_list = [], [], []
    for i in range(args.agent_count):
        observation_i_batch = np.concatenate([e[0][i] for e in batch])
        action_id_i_batch = np.asarray([e[1][i] for e in batch]).reshape(-1, 1)
        next_observation_i_batch = np.concatenate([e[3][i] for e in batch])
        observation_list.append(observation_i_batch)
        action_id_list.append(action_id_i_batch)
        next_observation_list.append(next_observation_i_batch)
    done_batch = np.asarray([e[4] for e in batch]).astype(int).reshape(-1, 1)

    reward_list = []
    if args.agent_name in ["IDQN", "CommNet", "DGN"]:
        for i in range(args.agent_count):
            reward_i_batch = np.asarray([e[2][i + 1] for e in batch]).reshape(-1, 1)
            # rewards[1 -- N] is the training reward of agent 1--N, respectively.
            reward_list.append(reward_i_batch)
    elif args.agent_name in ["VDN", "QMIX", "NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX"]:
        for i in range(args.agent_count):
            reward_i_batch = np.asarray([e[2][0] for e in batch]).reshape(-1, 1)
            # rewards[0] is our objective reward (i.e., the shared global reward)
            reward_list.append(reward_i_batch)
    else:
        raise ValueError("args.agent_name is not defined! ...")
    loss = agent.train(observation_list, action_id_list, reward_list, next_observation_list, done_batch, writer, training_step)
    if train_target_network:
        agent.train_target_network_hard()
    print("training loss:", loss)


def main(args, writer=None):
    global_training_step = 0

    agent = Agent(args)
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
    if args.env_rename == "CooperativeNavigation":
        from D_env_navigation import Environment
        env = Environment(args)
    elif args.env_rename == "MPE":
        env = args.env
    else:
        raise ValueError("args.env_name is not defined! ...")

    episode_reward_list = []
    for episode in range(args.episode_count):
        args.epsilon -= args.epsilon_delta
        args.alpha_PCA -= args.alpha_PCA_delta
        temp_buffer = []  # small trick: current episode experience replay
        episode_reward = 0.0
        if args.env_rename == "CooperativeNavigation":
            observation_all = env.reset()
        elif args.env_rename == "MPE":
            observation_all = env.reset()
            observation_all = [temp.reshape(1, -1) for temp in observation_all]
        for step in range(args.max_episode_len):
            # env.render()
            print("=" * 10, "episode", episode, "***** step", step)
            Qvalue_all = agent.generate_Qvalue(observation_all)
            action_id_all = exploration(Qvalue_all, args)
            if args.env_rename == "CooperativeNavigation":
                rewards, next_observation_all, done, _ = env.step(action_id_all)
            elif args.env_rename == "MPE":
                next_observation_all, rewards, done, _ = env.step(action_id_all)
                rewards = [rewards[0]] * (args.agent_count + 1)
                next_observation_all = [temp.reshape(1, -1) for temp in next_observation_all]
                done = np.any(done)
            replay_buffer.add([observation_all, action_id_all, rewards, next_observation_all, done])
            temp_buffer.append([observation_all, action_id_all, rewards, next_observation_all, done])
            observation_all = next_observation_all
            episode_reward += rewards[0]  # rewards[0] is our objective reward
            print("rewards ==>", rewards)

            # train the agent
            batch = replay_buffer.sample()
            if done or step == args.max_episode_len - 1:  # small trick: current episode experience replay
                batch.extend(temp_buffer)
            training(args, agent, batch, not(episode % args.episode_for_updating_T), writer, global_training_step)
            global_training_step += 1
            if done:
                break
        episode_reward_list.append(episode_reward)
        if writer is not None:
            writer.add_scalar("episode_reward_list", episode_reward, episode)

    save_1d_data(data=episode_reward_list, name=args.results_dir + args.exp_name + "-reward-train")
    if args.exp_id == 1:  # just save 1 agent.sess for human double check ...
        agent.save_model(args.results_dir + args.exp_name + "-train")


if __name__ == "__main__":
    args = D_settings.parse_arguments()

    if args.agent_name in ["IDQN", "VDN", "QMIX", "NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX", "CommNet", "DGN"]:
        from D_models import Agent
    else:
        raise ValueError("args.agent_name is not defined! ...")

    for exp_id in range(1, args.exp_count + 1):
        args.epsilon = 1.0  # Critically, please always reset this value!!!
        args.exp_id = exp_id
        np.random.seed(args.seed + exp_id)

        time_begin = time.time()
        writer = None  # SummaryWriter(log_dir=f"./log/{args.exp_name}/{exp_id}")
        main(args, writer=writer)
        print("time_used ==>", time.time() - time_begin)
