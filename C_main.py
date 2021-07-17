import time
import numpy as np
from tensorboardX import SummaryWriter

from common import (ReplayBuffer, save_1d_data, explore_action_2dim, explore_action_4dim, explore_action_6dim)
import C_settings


def exploration(action_all, args):
    action_all_before_exploration = []
    for i in range(args.agent_count):
        action_all_before_exploration.append(action_all[i][0])
    print("action_all_before_exploration ==>", action_all_before_exploration)

    if np.random.random() < args.epsilon:
        action_all_after_exploration = []
        if args.env_name in ["routing6v4", "routing12v20", "routing24v128"]:
            for i in range(args.agent_count):
                temp_act = None
                if args.action_dim_list[i] == 2:
                    temp_act = explore_action_2dim(action_all_before_exploration[i], args.epsilon)
                elif args.action_dim_list[i] == 4:
                    temp_act = explore_action_4dim(action_all_before_exploration[i], args.epsilon)
                elif args.action_dim_list[i] == 6:
                    temp_act = explore_action_6dim(action_all_before_exploration[i], args.epsilon)
                action_all_after_exploration.append(list(temp_act))
        else:
            raise ValueError("args.env_name is not defined! ...")
        print("action_all_after_exploration ==>", action_all_after_exploration)
        return action_all_after_exploration
    else:
        return action_all_before_exploration


def training(args, agent, batch, writer=None, training_step=0):
    observation_list, action_list, reward_list, next_observation_list = [], [], [], []
    for i in range(args.agent_count):
        observation_i_batch = np.concatenate([e[0][i] for e in batch])
        action_i_batch = np.asarray([e[1][i] for e in batch])  # if action_dim==1, need add '.reshape(-1, 1)'
        reward_i_batch = np.asarray([e[2][i + 1] for e in batch]).reshape(-1, 1)
        # rewards[0] is our objective reward, while rewards[1~N] is the training reward of agent 1~N, respectively.
        next_observation_i_batch = np.concatenate([e[3][i] for e in batch])
        observation_list.append(observation_i_batch)
        action_list.append(action_i_batch)
        reward_list.append(reward_i_batch)
        next_observation_list.append(next_observation_i_batch)
    done_list = np.asarray([e[4] for e in batch]).astype(int).reshape(-1, 1)

    loss_c = agent.train_critic(observation_list, action_list, reward_list, next_observation_list, done_list, writer, training_step)
    loss_a = agent.train_actor(observation_list, writer, training_step)
    agent.train_target_network_soft()
    # return loss_a, loss_c
    print("train_actor ==> loss:", loss_a)
    print("train_critic ==> loss:", loss_c)


def main(args, writer):
    global_training_step = 0

    agent = Agent(args)
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
    env = Environment(args)

    episode_reward_list = []
    for episode in range(args.episode_count):
        args.epsilon -= args.epsilon_delta
        temp_buffer = []  # small trick: current episode experience replay
        episode_reward = 0.0
        observation_all = env.reset()
        for step in range(args.max_episode_len):
            # env.render()
            print("=" * 10, "episode", episode, "***** step", step)
            action_all = agent.generate_action(observation_all)
            action_all = exploration(action_all, args)
            rewards, next_observation_all, done, _ = env.step(action_all)
            replay_buffer.add([observation_all, action_all, rewards, next_observation_all, done])
            temp_buffer.append([observation_all, action_all, rewards, next_observation_all, done])
            observation_all = next_observation_all
            episode_reward += rewards[0]  # rewards[0] is our objective reward
            print("rewards ==>", rewards)

            # train the agent
            batch = replay_buffer.sample()
            if done or step == args.max_episode_len - 1:  # small trick: current episode experience replay
                batch.extend(temp_buffer)
            training(args, agent, batch, writer, global_training_step)
            global_training_step += 1
            if done:
                break
        episode_reward_list.append(episode_reward)
        if writer is not None:
            writer.add_scalar("episode_reward_list", episode_reward, episode)

    save_1d_data(data=episode_reward_list, name=args.results_dir + args.exp_name + "-reward-train")
    if args.exp_id == 1:  # just save the first agent for human double check ...
        agent.save_model(args.results_dir + args.exp_name + "-train")


if __name__ == "__main__":
    args = C_settings.parse_arguments()

    if args.env_name == "routing6v4":
        from C_env_routing6v4 import Environment
    elif args.env_name == "routing12v20":
        from C_env_routing12v20 import Environment
    elif args.env_name == "routing24v128":
        from C_env_routing24v128 import Environment
    else:
        raise ValueError("args.env_name is not defined! ...")

    if args.agent_name in ["IND_AC", "MADDPG", "ATT_MADDPG", "NCC_AC", "MAAC", "Contrastive"]:
        from C_models import Agent
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
