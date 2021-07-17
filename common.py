import numpy as np
import scipy.sparse as sp
from collections import deque
import random
random.seed(0)





class ReplayBuffer(object):
    def __init__(self, buffer_size=1e6, batch_size=64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_experiences = 0
        self.buffer = deque()

    def add(self, experience):
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample(self):
        if self.num_experiences < self.batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, self.batch_size)


def save_1d_data(data, name):
    out_file = open(name + ".csv", "a") # append
    out_str = ",".join([str(d) for d in data])
    out_file.write(out_str + "\n")
    out_file.close()


def save_2d_data(data, name):
    out_file = open(name + ".csv", "a")  # append
    for i in range(data.shape[0]):
        out_str = ",".join([str(d) for d in data[i, :]])
        out_file.write(out_str + "\n")
    out_file.write("\n")
    out_file.close()





class CircularBuffer(object):
    def __init__(self, size=5, egress_count=2):
        self.size = size
        self.pointer = 0
        if egress_count == 2:  # for 6v4, 12v20
            self.buffer = [[0.0, 0.0]] * size  # [flow_toC, flow_toD]
        elif egress_count == 3:  # for 24v128
            self.buffer = [[0.0, 0.0, 0.0]] * size  # [flow_toD, flow_toE, flow_toF]
        else:
            raise ValueError("egress_count=2 is not defined!")

    def add_data(self, data_list):
        self.buffer[self.pointer] = data_list
        self.pointer += 1
        self.pointer %= self.size

    def get_data(self):
        data_list = self.buffer[self.pointer]
        return data_list

    def get_history_data(self):
        history_data = self.buffer[self.pointer:]+self.buffer[:self.pointer]
        hd_list = []
        for d in history_data: hd_list.extend(d)
        return hd_list


class Link(object):
    def __init__(self, link_capacity, link_delay=6, egress_count=2):
        self.link_capacity = link_capacity
        self.link_delay = link_delay
        self.total_flow = 0.0
        self.buffer = CircularBuffer(size=link_delay, egress_count=egress_count)


class Router(object):
    def __init__(self, router_delay=4, upper_links=[], direct_links=[], egress_count=2):
        self.router_delay = router_delay
        self.buffer = CircularBuffer(size=router_delay, egress_count=egress_count)
        self.upper_links = upper_links
        self.direct_links = direct_links  # it represents the down-stream direct links


def get_Abilene_trace():
    pass


def explore_action_2dim(action, epsilon):
    noise_action = action + epsilon
    denominator = noise_action[0] + noise_action[1]
    noise_action[0] /= denominator
    noise_action[1] /= denominator
    return noise_action


def explore_action_4dim(action, epsilon):
    noise_action = action + epsilon
    denominator = noise_action[0] + noise_action[1]
    noise_action[0] /= denominator
    noise_action[1] /= denominator
    denominator = noise_action[2] + noise_action[3]
    noise_action[2] /= denominator
    noise_action[3] /= denominator
    return noise_action


def explore_action_6dim(action, epsilon):
    noise_action = action + epsilon
    denominator = noise_action[0] + noise_action[1]
    noise_action[0] /= denominator
    noise_action[1] /= denominator
    denominator = noise_action[2] + noise_action[3]
    noise_action[2] /= denominator
    noise_action[3] /= denominator
    denominator = noise_action[4] + noise_action[5]
    noise_action[4] /= denominator
    noise_action[5] /= denominator
    return noise_action





class ParticleEntity(object):
    def __init__(self):
        self.size = 0.2
        self.color = np.array([0.0, 0.0, 0.0])
        self.collision = False  # entity collides with others, not used
        self.position = np.array([0.0, 0.0])  # x-axis, y-axis
        self.velocity = np.array([0.0, 0.0])  # x-axis, y-axis
        # for velocity, we do not consider damping (see integrate_state() function in master/multiagent/core.py/)





def preprocess_graph(adj):
    # refer Equ.(2) in 2017-ICLR-Semi-Supervised Classification with Graph Convolutional Networks
    # https://github.com/tkipf/gae/blob/master/gae/preprocessing.py#L14
    # http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).todense()
    return adj_normalized
