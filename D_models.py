import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

USE_CUDA = False  # torch.cuda.is_available()  # ==> without image input, using cuda is a bad choice
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)


class MixQ_VDN(nn.Module):
    # https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
    # https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/vdn.py
    def __init__(self):
        super(MixQ_VDN, self).__init__()

    def forward(self, Qvalue_list):
        return torch.sum(torch.cat(Qvalue_list, dim=1), dim=1, keepdim=True)


class MixQ_QMIX(nn.Module):
    # https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
    # https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py
    def __init__(self, args):
        super(MixQ_QMIX, self).__init__()
        self.state_dim = sum(args.observation_dim_list)  # concat observations of all agents to approximate the state
        self.embed_dim = args.hidden_dim
        self.agent_count = args.agent_count
        self.hyper_w1 = nn.Linear(self.state_dim, self.embed_dim * self.agent_count)
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_w2 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b2_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.hyper_b2_2 = nn.Linear(self.embed_dim, 1)

    def forward(self, Qvalue_list, state):
        # each Qvalue in Qvalue_list has a shape of [batch, 1], len(Qvalue_list)==agent_count
        # state has a shape of [batch, state_dim]
        Qvalue_all = torch.cat(Qvalue_list, dim=1)  # [batch, agent_count]
        Qvalue_all = Qvalue_all.view(-1, 1, self.agent_count)  # [batch, 1, agent_count]
        # First layer
        w1 = torch.abs(self.hyper_w1(state))  # with abs! [batch, embed_dim * agent_count]
        w1 = w1.view(-1, self.agent_count, self.embed_dim)  # [batch, agent_count, embed_dim]
        b1 = self.hyper_b1(state)  # without abs! [batch, embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)  # [batch, 1, embed_dim]
        hidden = F.elu(torch.bmm(Qvalue_all, w1) + b1)  # [batch, 1, embed_dim]
        # Second layer
        w2 = torch.abs(self.hyper_w2(state))  # with abs! [batch, embed_dim]
        w2 = w2.view(-1, self.embed_dim, 1)  # [batch, embed_dim, 1]
        # State-dependent bias
        b2_1 = F.relu(self.hyper_b2_1(state))  # with relu! [batch, embed_dim]
        b2_2 = self.hyper_b2_2(b2_1)  # without relu! [batch, 1]
        v = b2_2.view(-1, 1, 1)  # [batch, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w2) + v  # [batch, 1, 1]
        # Reshape and return
        Qtotal = y.view(-1, 1)
        return Qtotal


class NetBase(nn.Module):
    def __init__(self, args):
        super(NetBase, self).__init__()
        self.args = args
        self._define_parameters()
        if self.args.agent_name in ["VDN", "NCC_VDN", "Contrastive_VDN"]:
            self.MixQ = MixQ_VDN()
        if self.args.agent_name in ["QMIX", "NCC_QMIX", "Contrastive_QMIX"]:
            self.MixQ = MixQ_QMIX(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        pass

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()  # do not use python dict {}
            # parameters for pre-processing observations and actions
            parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            self._define_parameters_for_hidden_layers(parameters_dict, i)

            # parameters for generating Qvalues
            parameters_dict["Qvalue"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[i])
            self.parameters_all_agent.append(parameters_dict)

    def _forward_of_hidden_layers(self, out_obs_list):
        pass

    def forward(self, observation_batch_list):
        # pre-process
        out_obs_list = []
        for i in range(self.args.agent_count):
            out_obs = F.relu(self.parameters_all_agent[i]["fc_obs"](observation_batch_list[i]))
            out_obs_list.append(out_obs)

        # key part of difference MARL methods
        out_hidden_list = self._forward_of_hidden_layers(out_obs_list)
        if self.args.agent_name in ["NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX"]:
            out_hidden_list, C_hat_list, obs_hat_list = out_hidden_list

        # post-process
        Qvalue_list = []
        for i in range(self.args.agent_count):
            Qvalue = self.parameters_all_agent[i]["Qvalue"](out_hidden_list[i])  # linear activation for Q-value
            Qvalue_list.append(Qvalue)

        if self.args.agent_name in ["NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX"]:
            return Qvalue_list, C_hat_list, obs_hat_list
        else:
            return Qvalue_list

    def mix_Qvalue_VDN(self, Qvalue_list):
        Qtotal = self.MixQ(Qvalue_list)
        return Qtotal

    def mix_Qvalue_QMIX(self, Qvalue_list, state):
        Qtotal = self.MixQ(Qvalue_list, state)
        return Qtotal


class NetIndependent(NetBase):
    def __init__(self, args):
        super(NetIndependent, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        for j in range(self.args.hidden_layer_count):
            parameters_dict["fc" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def _forward_of_hidden_layers(self, out_obs_list):
        out_hidden_list = []
        for i in range(self.args.agent_count):
            out = out_obs_list[i]
            for j in range(self.args.hidden_layer_count):
                out = F.relu(self.parameters_all_agent[i]["fc" + str(j)](out))
            out_hidden_list.append(out)
        return out_hidden_list


class NetVaeNCC(NetBase):
    def __init__(self, args):
        super(NetVaeNCC, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_gcn_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_A"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_meam"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_logstd"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_out_for_Qvalue"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        #
        parameters_dict["fc_decoder_H_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_h_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_obs"] = nn.Linear(self.args.hidden_dim, self.args.observation_dim_list[agent_index])

    def _GCN_module(self, h_list, agent_index, type="obs"):
        h_all = torch.stack(h_list, dim=0)  # (agent_count, batch_size, hidden_dim)
        h_all = h_all.permute(1, 0, 2)  # (batch_size, agent_count, hidden_dim)
        adj_norm = to_tensor(self.args.adj_norm)  # (agent_count, agent_count)

        # refer Equ.(2) in 2017-ICLR-Semi-Supervised Classification with Graph Convolutional Networks
        # https://github.com/tkipf/gae/blob/master/gae/layers.py#L77
        H_all_without_adj = self.parameters_all_agent[agent_index]["fc_gcn_" + type](h_all)  # equal to H_all.shape
        H_all = F.relu(torch.matmul(adj_norm, H_all_without_adj))  # (batch_size, agent_count, hidden_dim)

        H_all = H_all.permute(1, 0, 2)  # (agent_count, batch_size, hidden_dim)
        # although H_all contains information of all agents
        # we only return the specific information for the agent_index-th agent
        # since we do not share parameters between agents
        return H_all[agent_index]

    def _Cognition_module(self, H, agent_index):
        A = self.parameters_all_agent[agent_index]["fc_cognition_A"](H)
        C_mean = self.parameters_all_agent[agent_index]["fc_cognition_C_meam"](H)
        C_logstd = self.parameters_all_agent[agent_index]["fc_cognition_C_logstd"](H)
        C_hat = C_mean + torch.exp(C_logstd) * torch.normal(mean=0.0, std=1.0, size=C_logstd.shape)
        return A, C_hat

    def _Decoder(self, C_hat_list):
        obs_hat_list = []
        for i in range(self.args.agent_count):
            recovered_Ho = F.relu(self.parameters_all_agent[i]["fc_decoder_H_obs"](C_hat_list[i]))
            recovered_ho = F.relu(self.parameters_all_agent[i]["fc_decoder_h_obs"](recovered_Ho))
            obs_hat = self.parameters_all_agent[i]["fc_decoder_obs"](recovered_ho)  # linear activation
            obs_hat_list.append(obs_hat)
        return obs_hat_list

    def _forward_of_hidden_layers(self, out_obs_list):
        # FC-module is equal to the pre-process in forward() function of NetBase(), so we skip FC-Module
        out_hidden_list, C_hat_list = [], []
        for i in range(self.args.agent_count):
            Ho = self._GCN_module(out_obs_list, i, type="obs")
            A, C_hat = self._Cognition_module(Ho, i)
            # out = A + C_hat   ==> change to the following concat (this is different from the description in paper)
            AC = torch.cat([A, C_hat], dim=1)
            out = self.parameters_all_agent[i]["fc_out_for_Qvalue"](AC)
            out_hidden_list.append(out)
            C_hat_list.append(C_hat)
        obs_hat_list = self._Decoder(C_hat_list)
        return (out_hidden_list, C_hat_list, obs_hat_list)


class NetContrastiveNCC(NetBase):
    def __init__(self, args):
        super(NetContrastiveNCC, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_gcn_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_A"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_meam"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_logstd"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_out_for_Qvalue"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        #
        parameters_dict["fc_decoder_H_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_h_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_obs"] = nn.Linear(self.args.hidden_dim, self.args.observation_dim_list[agent_index])

    def _GCN_module(self, h_list, agent_index, type="obs"):
        h_all = torch.stack(h_list, dim=0)  # (agent_count, batch_size, hidden_dim)
        h_all = h_all.permute(1, 0, 2)  # (batch_size, agent_count, hidden_dim)
        adj_norm = to_tensor(self.args.adj_norm)  # (agent_count, agent_count)

        # refer Equ.(2) in 2017-ICLR-Semi-Supervised Classification with Graph Convolutional Networks
        # https://github.com/tkipf/gae/blob/master/gae/layers.py#L77
        H_all_without_adj = self.parameters_all_agent[agent_index]["fc_gcn_" + type](h_all)  # equal to H_all.shape
        H_all = F.relu(torch.matmul(adj_norm, H_all_without_adj))  # (batch_size, agent_count, hidden_dim)

        H_all = H_all.permute(1, 0, 2)  # (agent_count, batch_size, hidden_dim)
        # although H_all contains information of all agents
        # we only return the specific information for the agent_index-th agent
        # since we do not share parameters between agents
        return H_all[agent_index]

    def _Cognition_module(self, H, agent_index):
        A = self.parameters_all_agent[agent_index]["fc_cognition_A"](H)
        C_mean = self.parameters_all_agent[agent_index]["fc_cognition_C_meam"](H)
        C_logstd = self.parameters_all_agent[agent_index]["fc_cognition_C_logstd"](H)
        C_hat = C_mean + torch.exp(C_logstd) * torch.normal(mean=0.0, std=1.0, size=C_logstd.shape)
        return A, C_hat

    def _Decoder(self, C_hat_list):
        obs_hat_list = []
        for i in range(self.args.agent_count):
            recovered_Ho = F.relu(self.parameters_all_agent[i]["fc_decoder_H_obs"](C_hat_list[i]))
            recovered_ho = F.relu(self.parameters_all_agent[i]["fc_decoder_h_obs"](recovered_Ho))
            obs_hat = self.parameters_all_agent[i]["fc_decoder_obs"](recovered_ho)  # linear activation
            obs_hat_list.append(obs_hat)
        return obs_hat_list

    def _forward_of_hidden_layers(self, out_obs_list):
        # FC-module is equal to the pre-process in forward() function of NetBase(), so we skip FC-Module
        out_hidden_list, C_hat_list = [], []
        for i in range(self.args.agent_count):
            Ho = self._GCN_module(out_obs_list, i, type="obs")
            A, C_hat = self._Cognition_module(Ho, i)
            # out = A + C_hat   ==> change to the following concat (this is different from the description in paper)
            AC = torch.cat([A, C_hat], dim=1)
            out = self.parameters_all_agent[i]["fc_out_for_Qvalue"](AC)
            out_hidden_list.append(out)
            C_hat_list.append(C_hat)
        obs_hat_list = self._Decoder(C_hat_list)
        return (out_hidden_list, C_hat_list, obs_hat_list)


class NetCommNet(NetBase):
    def __init__(self, args):
        super(NetCommNet, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        for j in range(self.args.hidden_layer_count):
            parameters_dict["fc_H" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
            parameters_dict["fc_C" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def _mean_communication(self, hidden_list, layer_index):
        output_list = []
        hidden_list = torch.stack(hidden_list, dim=0)  # (agent_count, batch_size, hidden_dim)
        hidden_sum = torch.sum(hidden_list, dim=0, keepdims=False)  # (batch_size, hidden_dim)
        for i in range(self.args.agent_count):
            message_i = (hidden_sum - hidden_list[i]) / (self.args.agent_count - 1)  # mean message operation
            param_H = self.parameters_all_agent[i]["fc_H" + str(layer_index)]  # in fact, it is a layer, not parameter
            param_C = self.parameters_all_agent[i]["fc_C" + str(layer_index)]
            output_i = torch.tanh(param_H(hidden_list[i]) + param_C(message_i))
            output_list.append(output_i)
        return output_list

    def _forward_of_hidden_layers(self, out_obs_list):
        out_hidden_list = out_obs_list
        for layer_index in range(self.args.hidden_layer_count):
            out_hidden_list = self._mean_communication(out_hidden_list, layer_index)
        return out_hidden_list


import numpy as np
class NetDGN(NetBase):
    def __init__(self, args):
        super(NetDGN, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        hidden_layer_count = 2
        hidden_dim = self.args.hidden_dim
        for j in range(hidden_layer_count):
            parameters_dict["fc_Wq" + str(j)] = nn.Linear(hidden_dim, hidden_dim)
            parameters_dict["fc_Wk" + str(j)] = nn.Linear(hidden_dim, hidden_dim)
            parameters_dict["fc_Wv" + str(j)] = nn.Linear(hidden_dim, hidden_dim)
            parameters_dict["fc_head" + str(j)] = nn.Linear(hidden_dim * self.args.head_count, hidden_dim)
        parameters_dict["fc_output"] = nn.Linear(hidden_dim * 3, hidden_dim)

    def _MHA_relation_kernel(self, hidden_list, layer_index):
        # here, we assume 'self.args.adj[i]' is unchanged; otherwise, it should be taken as input
        output_list = []
        for i in range(self.args.agent_count):  # for each agent
            attention_head_list = []
            for _ in range(self.args.head_count):  # for each attention-head
                Wq = self.parameters_all_agent[i]["fc_Wq" + str(layer_index)]  # in fact, it is a layer, not parameter
                Wk = self.parameters_all_agent[i]["fc_Wk" + str(layer_index)]
                Wv = self.parameters_all_agent[i]["fc_Wv" + str(layer_index)]

                # calculate attention across neighboring agents (i.e., Eq. 2)
                hi = hidden_list[i]
                query = Wq(hi)  # (batch_size, hidden_dim)
                query = torch.unsqueeze(query, dim=1)  # (batch_size, 1, hidden_dim)
                keys = [Wk(hidden_list[j]) for j, indicator_j in enumerate(self.args.adj[i]) if indicator_j == 1]
                keys = torch.stack(keys, dim=0)  # (count_of_neighboring_agents, batch_size, hidden_dim)
                keys = keys.permute(1, 2, 0)  # (batch_size, hidden_dim, count_of_neighboring_agents)
                attend_logits = torch.bmm(query, keys)  # (batch_size, 1, count_of_neighboring_agents)
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])  # (batch_size, 1, count_of_neighboring_agents)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, count_of_neighboring_agents)

                # calculate feature weighted summation (i.e., Eq. 3)
                values = [Wv(hidden_list[j]) for j, indicator_j in enumerate(self.args.adj[i]) if indicator_j == 1]
                values = torch.stack(values, dim=0)  # (count_of_neighboring_agents, batch_size, hidden_dim)
                values = values.permute(1, 2, 0)  # (batch_size, hidden_dim, count_of_neighboring_agents)
                attention_head = torch.mul(values, attend_weights)  # (batch_size, hidden_dim, count_of_neighboring_agents)
                attention_head = torch.sum(attention_head, dim=2, keepdim=False)  # (batch_size, hidden_dim)
                attention_head_list.append(attention_head)
            # Eq. 3: attention_heads are concatenated, and then fed into a function, i.e., one-layer MLP with ReLU
            # https://www.dropbox.com/sh/advzd6q58pp8r0d/AAB0bTbIOV-FH9pp63x1OJZla/Routing/routers_regularization.py?dl=0#L222
            concatenated_heads = torch.cat(attention_head_list, dim=1)  # (batch_size, hidden_dim*head_count_of_MHA)
            output_i = F.relu(self.parameters_all_agent[i]["fc_head" + str(layer_index)](concatenated_heads))
            output_list.append(output_i)
        return output_list

    def _forward_of_hidden_layers(self, out_obs_list):
        # two Convolutional Layers (MHA relation kernel)
        hidden0_list = self._MHA_relation_kernel(out_obs_list, layer_index=0)
        hidden1_list = self._MHA_relation_kernel(hidden0_list, layer_index=1)
        # a shortcut layer
        # https://www.dropbox.com/sh/advzd6q58pp8r0d/AAB0bTbIOV-FH9pp63x1OJZla/Routing/routers_regularization.py?dl=0#L237
        out_hidden_list = []
        for i in range(self.args.agent_count):
            cat_hidden = torch.cat([out_obs_list[i], hidden0_list[i], hidden1_list[i]], dim=1)
            out_hidden = self.parameters_all_agent[i]["fc_output"](cat_hidden)
            out_hidden_list.append(out_hidden)
        return out_hidden_list


class Agent(object):
    def __init__(self, args):
        self.args = args
        print("=" * 30, "create agent", self.args.agent_name)
        if self.args.agent_name in ["IDQN", "VDN", "QMIX"]:
            self.net = NetIndependent(args)
            self.T_net = NetIndependent(args)  # target network
        elif self.args.agent_name in ["NCC_VDN", "NCC_QMIX"]:  # please try NCC without any form of mixing Q-value
            self.net = NetVaeNCC(args)
            self.T_net = NetVaeNCC(args)
        elif self.args.agent_name in ["Contrastive_VDN", "Contrastive_QMIX"]:
            self.net = NetContrastiveNCC(args)
            self.T_net = NetContrastiveNCC(args)
        elif self.args.agent_name == "CommNet":
            self.net = NetCommNet(args)
            self.T_net = NetCommNet(args)
        elif self.args.agent_name == "DGN":
            self.net = NetDGN(args)
            self.T_net = NetDGN(args)
        else:
            raise ValueError('args.agent_name is not defined ...')

        self._init_necessary_info()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.net.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # init target network before training
        self.train_target_network_hard()

        # set target network to evaluation mode
        self.T_net.eval()

        # create optimizers
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

        if USE_CUDA:
            self._config_cuda()

    def save_cognition_for_human_understanding(self, cognition_list):
        fp = open('./results/save_cognition_for_'+self.args.exp_name+'.txt', 'a')
        save_str = ''
        cognition_list = np.array(cognition_list).reshape(1, -1)
        for cognition in cognition_list[0]:
            save_str += str(cognition) + ','
        save_str += '\n'
        fp.write(save_str)
        fp.close()

    def generate_Qvalue(self, observation_list):
        self._config_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        Qvalue_list = self.net(observation_list)
        if self.args.agent_name in ["NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX"]:
            Qvalue_list, C_hat_list, _ = Qvalue_list
            self.save_cognition_for_human_understanding([C_hat.cpu().detach().numpy() for C_hat in C_hat_list])
        return [Qvalue.cpu().detach().numpy() for Qvalue in Qvalue_list]

    def train(self, observation_list, action_id_list, reward_list, next_observation_list, done_batch, writer=None, training_step=0):
        self._config_train_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_id_list = [to_tensor(action) for action in action_id_list]
        reward_list = [to_tensor(reward) for reward in reward_list]
        next_observation_list = [to_tensor(next_observation) for next_observation in next_observation_list]
        multiplier_batch = to_tensor(1.0 - done_batch)

        chosen_Qvalue_list = []
        target_Qvalue_list = []
        Qvalue_list = self.net(observation_list)
        T_Qvalue_list = self.T_net(next_observation_list)  # use T_nets
        if self.args.agent_name in ["NCC_VDN", "NCC_QMIX", "Contrastive_VDN", "Contrastive_QMIX"]:
            Qvalue_list, C_hat_list, obs_hat_list = Qvalue_list
            T_Qvalue_list, _, _ = T_Qvalue_list

        for i in range(self.args.agent_count):
            one_hot_action = F.one_hot(action_id_list[i].to(torch.int64), num_classes=self.args.action_dim_list[i])
            # action_id_list[i] is with shape=[None, 1], one_hot_action.shape == (None, 1, self.args.action_dim),
            # so we need to squeeze the second-dim
            one_hot_action = torch.squeeze(one_hot_action, dim=1)  # removes dimensions of size 1
            chosen_Qvalue = torch.sum(Qvalue_list[i] * one_hot_action, dim=1, keepdim=True)
            target_Qvalue = torch.max(T_Qvalue_list[i], dim=1, keepdim=True)[0]  # [0]: only return max values
            chosen_Qvalue_list.append(chosen_Qvalue)
            target_Qvalue_list.append(target_Qvalue)

        chosen_Qtotal = None
        total_loss = 0.0
        if self.args.agent_name in ["IDQN", "CommNet", "DGN"]:
            for i in range(self.args.agent_count):
                TDtarget = reward_list[i] + multiplier_batch * self.args.gamma * target_Qvalue_list[i]
                total_loss += self.MSEloss(chosen_Qvalue_list[i], TDtarget.cpu().detach())  # note the detach
        elif self.args.agent_name in ["VDN", "NCC_VDN", "Contrastive_VDN"]:
            chosen_Qtotal = self.net.mix_Qvalue_VDN(chosen_Qvalue_list)
            target_Qtotal = self.T_net.mix_Qvalue_VDN(target_Qvalue_list)  # T_net
            TDtarget = reward_list[0] + multiplier_batch * self.args.gamma * target_Qtotal
            total_loss += self.MSEloss(chosen_Qtotal, TDtarget.cpu().detach())
        elif self.args.agent_name in ["QMIX", "NCC_QMIX", "Contrastive_QMIX"]:
            # concat the observations of all agents to approximate the state
            state = torch.cat(observation_list, dim=1)
            next_state = torch.cat(next_observation_list, dim=1)
            chosen_Qtotal = self.net.mix_Qvalue_QMIX(chosen_Qvalue_list, state)
            target_Qtotal = self.T_net.mix_Qvalue_QMIX(target_Qvalue_list, next_state)  # T_net
            TDtarget = reward_list[0] + multiplier_batch * self.args.gamma * target_Qtotal
            total_loss += self.MSEloss(chosen_Qtotal, TDtarget.cpu().detach())
        if writer is not None:
            writer.add_scalar("TD_loss", total_loss.cpu().detach().item(), training_step)

        if self.args.agent_name in ["NCC_VDN", "NCC_QMIX", 'PicaQ']:
            cognition_list = weight_list if self.args.agent_name == 'PicaQ' else C_hat_list
            total_KL_loss, total_L2_loss, total_PCA_loss = 0.0, 0.0, 0.0
            for i in range(self.args.agent_count):
                KL_loss = 0.0
                for j, value_j in enumerate(self.args.adj[i]):
                    if value_j == 1:  # agent j is a neighbor of agent i
                        KL_loss += self.KLDivLoss(F.log_softmax(cognition_list[i], dim=1), F.softmax(cognition_list[j], dim=1))
                        # KLDivLoss: the input given is expected to contain log-probabilities, target is probabilities
                        # print("KL_loss ==>", KL_loss)
                total_KL_loss += KL_loss / sum(self.args.adj[i])  # normalized with neighbor count
                total_L2_loss += self.MSEloss(obs_hat_list[i], observation_list[i])
            total_KL_loss = (self.args.alpha_KL * total_KL_loss) / self.args.agent_count  # normalized by agent count
            total_L2_loss = (self.args.alpha_L2 * total_L2_loss) / self.args.agent_count  # normalized by agent count
            total_loss += (total_KL_loss + total_L2_loss)

            if writer is not None:
                writer.add_scalar("KL_loss", total_KL_loss.cpu().detach().item(), training_step)
                writer.add_scalar("L2_loss", total_L2_loss.cpu().detach().item(), training_step)
        elif self.args.agent_name in ["Contrastive_VDN", "Contrastive_QMIX"]:
            C_hat_tensor = torch.stack(C_hat_list).permute(1, 0, 2)  # (batch_size, num_agents, dim_hidden)
            C_hat_tensor_Transpose = C_hat_tensor.permute(0, 2, 1)  # (batch_size, dim_hidden, num_agents)
            bilinear_similarity = torch.bmm(C_hat_tensor, C_hat_tensor_Transpose)  # (batch_size, num_agents, num_agents)
            max_v = torch.max(bilinear_similarity, 2).values.detach()  # (batch_size, num_agents)
            max_v = max_v.unsqueeze(2).repeat(1, 1, self.args.agent_count)  # (batch_size, num_agents, num_agents)
            exp_v = torch.exp(bilinear_similarity - max_v)  # (batch_size, num_agents, num_agents)

            total_Contrastive_loss, total_L2_loss = 0.0, 0.0
            for i in range(self.args.agent_count):
                exp_v_pos, exp_v_neg = 0.0, 0.0
                for j, value_j in enumerate(self.args.adj[i]):
                    if j == i:
                        continue
                    elif value_j == 1:  # agent j is a neighbor of agent i
                        exp_v_pos += exp_v[:, i, j]
                    else:
                        exp_v_neg += exp_v[:, i, j]
                exp_v_pos /= sum(self.args.adj[i])  # normalized by neighbor count
                exp_v_neg /= (self.args.agent_count - sum(self.args.adj[i]) - 1)
                contrastive_loss = -torch.log(exp_v_pos) + torch.log(exp_v_pos + exp_v_neg)
                total_Contrastive_loss += torch.mean(contrastive_loss)  # reduce the batch dimension
                total_L2_loss += self.MSEloss(obs_hat_list[i], observation_list[i])
            total_Contrastive_loss = (self.args.alpha_CON * total_Contrastive_loss) / self.args.agent_count  # normalized by agent count
            total_L2_loss = (self.args.alpha_L2 * total_L2_loss) / self.args.agent_count  # normalized by agent count
            total_loss += (total_Contrastive_loss + total_L2_loss)
            if writer is not None:
                writer.add_scalar("Contrastive_loss", total_Contrastive_loss.cpu().detach().item(), training_step)
                writer.add_scalar("L2_loss", total_L2_loss.cpu().detach().item(), training_step)

        self.optimizer.zero_grad()  # clear previous gradients before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clipped_norm_value)  # after backward() before step()
        self.optimizer.step()
        return total_loss.cpu().detach().numpy()

    def train_target_network_hard(self):
        for target_param, param in zip(self.T_net.parameters(), self.net.parameters()):
            target_param.data.copy_(param.data)

    def _config_cuda(self):
        self.net.cuda()
        self.T_net.cuda()

    def _config_train_mode(self):
        self.net.train()  # set train mode

    def _config_evaluation_mode(self):
        self.net.eval()  # set evaluation mode

    def save_model(self, output):
        print("save_model() ...")
        torch.save(self.net.state_dict(), '{}-net.pkl'.format(output))

    def load_weights(self, output):
        print("load_weights() ...")
        self.net.load_state_dict(torch.load('{}-net.pkl'.format(output)))
        self.train_target_network_hard()


