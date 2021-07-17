import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

USE_CUDA = False  # torch.cuda.is_available()  # ==> without image input, using cuda is a bad choice
# besides, the NCC_VAE and MAAC are incompatible with cuda ... I do not fix it
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

torch.manual_seed(0)
if USE_CUDA:
    torch.cuda.manual_seed(0)


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)


class ActorIndependent(nn.Module):
    def __init__(self, args):
        super(ActorIndependent, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()   # do not use python dict {}
            # parameters for pre-processing observations
            parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            for j in range(self.args.hidden_layer_count):
                parameters_dict["fc" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

            # parameters for generating actions
            if self.args.env_name in ["routing6v4", "routing12v20", "routing24v128"]:
                action_dim = self.args.action_dim_list[i]
                if action_dim == 2:  # the flow will go to egress-C !or! egress-D
                    parameters_dict["action"] = nn.Linear(self.args.hidden_dim, 2)
                elif action_dim == 4:  # the flow will go to egress-C !and! egress-D
                    parameters_dict["action_toC"] = nn.Linear(self.args.hidden_dim, 2)
                    parameters_dict["action_toD"] = nn.Linear(self.args.hidden_dim, 2)
                elif action_dim == 6:  # the flow will go to egress-D !and! egress-E !and! egress-F
                    parameters_dict["action_toD"] = nn.Linear(self.args.hidden_dim, 2)
                    parameters_dict["action_toE"] = nn.Linear(self.args.hidden_dim, 2)
                    parameters_dict["action_toF"] = nn.Linear(self.args.hidden_dim, 2)
            else:
                raise ValueError("self.args.env_name is not defined! ...")
            self.parameters_all_agent.append(parameters_dict)

    def forward(self, observation_batch_list):
        action_list = []
        for i in range(self.args.agent_count):
            out = F.relu(self.parameters_all_agent[i]["fc_obs"](observation_batch_list[i]))

            for j in range(self.args.hidden_layer_count):
                out = F.relu(self.parameters_all_agent[i]["fc" + str(j)](out))

            if self.args.env_name in ["routing6v4", "routing12v20", "routing24v128"]:
                action_dim = self.args.action_dim_list[i]
                if action_dim == 2:  # the flow will go to egress-C !or! egress-D
                    action = F.softmax(self.parameters_all_agent[i]["action"](out), dim=1)  # softmax()
                elif action_dim == 4:  # the flow will go to egress-C !and! egress-D
                    action_toC = F.softmax(self.parameters_all_agent[i]["action_toC"](out), dim=1)
                    action_toD = F.softmax(self.parameters_all_agent[i]["action_toD"](out), dim=1)
                    action = torch.cat([action_toC, action_toD], dim=1)
                elif action_dim == 6:  # the flow will go to egress-D !and! egress-E !and! egress-F
                    action_toD = F.softmax(self.parameters_all_agent[i]["action_toD"](out), dim=1)
                    action_toE = F.softmax(self.parameters_all_agent[i]["action_toE"](out), dim=1)
                    action_toF = F.softmax(self.parameters_all_agent[i]["action_toF"](out), dim=1)
                    action = torch.cat([action_toD, action_toE, action_toF], dim=1)
            else:
                raise ValueError("self.args.env_name is not defined! ...")
            action_list.append(action)
        return action_list


class CriticBase(nn.Module):
    def __init__(self, args):
        super(CriticBase, self).__init__()
        self.args = args
        self._define_parameters()

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        pass

    def _define_parameters(self):
        self.parameters_all_agent = nn.ModuleList()  # do not use python list []
        for i in range(self.args.agent_count):
            parameters_dict = nn.ModuleDict()  # do not use python dict {}
            # parameters for pre-processing observations and actions
            parameters_dict["fc_obs"] = nn.Linear(self.args.observation_dim_list[i], self.args.hidden_dim)
            parameters_dict["fc_action"] = nn.Linear(self.args.action_dim_list[i], self.args.hidden_dim)

            # parameters for hidden layers
            self._define_parameters_for_hidden_layers(parameters_dict, i)

            # parameters for generating Qvalues
            parameters_dict["Qvalue"] = nn.Linear(self.args.hidden_dim, 1)
            self.parameters_all_agent.append(parameters_dict)

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        pass

    def forward(self, observation_batch_list, action_batch_list):
        # pre-process
        out_obs_list, out_action_list = [], []
        for i in range(self.args.agent_count):
            out_obs = F.relu(self.parameters_all_agent[i]["fc_obs"](observation_batch_list[i]))
            out_action = F.relu(self.parameters_all_agent[i]["fc_action"](action_batch_list[i]))
            out_obs_list.append(out_obs)
            out_action_list.append(out_action)

        # key part of difference MARL methods
        out_hidden_list = self._forward_of_hidden_layers(out_obs_list, out_action_list)
        if self.args.agent_name == "NCC_AC":
            out_hidden_list, C_hat_list, obs_hat_list, action_hat_list = out_hidden_list
        elif self.args.agent_name == "Contrastive":
            out_hidden_list, C_hat_list = out_hidden_list

        # post-process
        Qvalue_list = []
        for i in range(self.args.agent_count):
            Qvalue = self.parameters_all_agent[i]["Qvalue"](out_hidden_list[i])  # linear activation for Q-value
            Qvalue_list.append(Qvalue)

        if self.args.agent_name == "NCC_AC":
            return (Qvalue_list, C_hat_list, obs_hat_list, action_hat_list)
        elif self.args.agent_name == "Contrastive":
            return (Qvalue_list, C_hat_list)
        else:
            return Qvalue_list


class CriticIndependent(CriticBase):
    def __init__(self, args):
        super(CriticIndependent, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_cat"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        # due to fc_cat, hidden_layer_count needs -1
        for j in range(self.args.hidden_layer_count - 1):
            parameters_dict["fc" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        out_hidden_list = []
        for i in range(self.args.agent_count):
            out_oa = torch.cat([out_obs_list[i], out_action_list[i]], dim=1)
            out = F.relu(self.parameters_all_agent[i]["fc_cat"](out_oa))
            for j in range(self.args.hidden_layer_count - 1):
                out = F.relu(self.parameters_all_agent[i]["fc" + str(j)](out))
            out_hidden_list.append(out)
        return out_hidden_list


class CriticMADDPG(CriticBase):
    def __init__(self, args):
        super(CriticMADDPG, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_cat"] = nn.Linear(self.args.hidden_dim * 2 * self.args.agent_count, self.args.hidden_dim)
        # due to fc_cat, hidden_layer_count needs -1
        for j in range(self.args.hidden_layer_count - 1):
            parameters_dict["fc" + str(j)] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        out_hidden_list = []
        for i in range(self.args.agent_count):
            out_oa = torch.cat(out_obs_list + out_action_list, dim=1)
            out = F.relu(self.parameters_all_agent[i]["fc_cat"](out_oa))
            for j in range(self.args.hidden_layer_count - 1):
                out = F.relu(self.parameters_all_agent[i]["fc" + str(j)](out))
            out_hidden_list.append(out)
        return out_hidden_list


class CriticAttentionalMADDPG(CriticBase):
    def __init__(self, args):
        super(CriticAttentionalMADDPG, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        hidden_dim = self.args.hidden_dim
        head_dim = hidden_dim
        encoder_input_dim = hidden_dim * (self.args.agent_count + 1)
        decoder_input_dim = hidden_dim * (self.args.agent_count - 1)

        parameters_dict["fc_encoder_input"] = nn.Linear(encoder_input_dim, hidden_dim)
        for k in range(self.args.head_count):
            parameters_dict["fc_encoder_head" + str(k)] = nn.Linear(hidden_dim, head_dim)

        parameters_dict["fc_decoder_input"] = nn.Linear(decoder_input_dim, head_dim)

    def _global_attention(self, encoder_H, decoder_H):
        # encoder_H has a shape (source_vector_count, batch_size, hidden_dim)
        # decoder_H has a shape (batch_size, hidden_dim)
        # scores is based on "dot-product" function, it works well for the global attention
        temp_scores = torch.mul(encoder_H, decoder_H)  # (source_vector_count, batch_size, hidden_dim)
        scores = torch.sum(temp_scores, dim=2)  # (source_vector_count, batch_size)
        attention_weights = F.softmax(scores.permute(1, 0), dim=1)  # (batch_size, source_vector_count)
        attention_weights = torch.unsqueeze(attention_weights, dim=2)  # (batch_size, source_vector_count, 1)
        contextual_vector = torch.matmul(encoder_H.permute(1, 2, 0), attention_weights)  # (batch_size, hidden_dim, 1)
        contextual_vector = torch.squeeze(contextual_vector)  # (batch_size, hidden_dim)
        return contextual_vector

    # in fact, K-head module and attention module are integrated into one module
    def _attention_module(self, obs_list, action_list, agent_index):
        encoder_input_list = obs_list + [action_list[agent_index]]
        decoder_input_list = action_list[:agent_index] + action_list[agent_index + 1:]

        # generating a temp hidden layer "h" (the encoder part, refer the figure in our paper)
        encoder_input = torch.cat(encoder_input_list, dim=1)
        encoder_h = F.relu(self.parameters_all_agent[agent_index]["fc_encoder_input"](encoder_input))

        # generating action-conditional Q-value heads (i.e., the encoder part)
        encoder_head_list = []
        for k in range(self.args.head_count):
            encoder_head = F.relu(self.parameters_all_agent[agent_index]["fc_encoder_head" + str(k)](encoder_h))
            encoder_head_list.append(encoder_head)
        encoder_heads = torch.stack(encoder_head_list, dim=0)  # (head_count, batch_size, head_dim)

        # generating a temp hidden layer "H" (the decoder part, refer the figure in our paper)
        decoder_input = torch.cat(decoder_input_list, dim=1)
        decoder_H = F.relu(self.parameters_all_agent[agent_index]["fc_decoder_input"](decoder_input))

        # generating content vector (i.e., the decoder part)
        contextual_vector = self._global_attention(encoder_heads, decoder_H)  # (batch_size, head_dim)

        # contextual_vector need to be further transformed into 1-dimension Q-value
        # this will be done by the forward() function in CriticBase()

        return contextual_vector

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        out_hidden_list = []
        for i in range(self.args.agent_count):
            out = self._attention_module(out_obs_list, out_action_list, i)
            out_hidden_list.append(out)
        return out_hidden_list


class CriticVaeNCC(CriticBase):
    def __init__(self, args):
        super(CriticVaeNCC, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_gcn_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_gcn_action"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_A"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_meam"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_logstd"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_out_for_Qvalue"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        #
        parameters_dict["fc_decoder_H_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_h_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_obs"] = nn.Linear(self.args.hidden_dim, self.args.observation_dim_list[agent_index])
        parameters_dict["fc_decoder_H_action"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_h_action"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_decoder_action"] = nn.Linear(self.args.hidden_dim, self.args.action_dim_list[agent_index])

    # this is the same as _GCN_module() of SimpleNCC
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

    def _Cognition_module(self, Ho, Ha, agent_index):
        H = torch.cat([Ho, Ha], dim=1)
        # do not use shortcut to generate A (this is different from the description in paper)
        A = self.parameters_all_agent[agent_index]["fc_cognition_A"](H)
        C_mean = self.parameters_all_agent[agent_index]["fc_cognition_C_meam"](H)
        C_logstd = self.parameters_all_agent[agent_index]["fc_cognition_C_logstd"](H)
        C_hat = C_mean + torch.exp(C_logstd) * torch.normal(mean=0.0, std=1.0, size=C_logstd.shape)
        return A, C_hat

    def _Decoder(self, C_hat_list):
        obs_hat_list, action_hat_list = [], []
        for i in range(self.args.agent_count):
            recovered_Ho = F.relu(self.parameters_all_agent[i]["fc_decoder_H_obs"](C_hat_list[i]))
            recovered_ho = F.relu(self.parameters_all_agent[i]["fc_decoder_h_obs"](recovered_Ho))
            obs_hat = self.parameters_all_agent[i]["fc_decoder_obs"](recovered_ho)  # linear activation
            obs_hat_list.append(obs_hat)
            recovered_Ha = F.relu(self.parameters_all_agent[i]["fc_decoder_H_action"](C_hat_list[i]))
            recovered_ha = F.relu(self.parameters_all_agent[i]["fc_decoder_h_action"](recovered_Ha))
            action_hat = self.parameters_all_agent[i]["fc_decoder_action"](recovered_ha)  # linear activation
            action_hat_list.append(action_hat)
        return obs_hat_list, action_hat_list

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        # FC-module is equal to the pre-process in forward() function of CriticBase(), so we skip FC-Module
        out_hidden_list, C_hat_list = [], []
        for i in range(self.args.agent_count):
            Ho = self._GCN_module(out_obs_list, i, type="obs")
            Ha = self._GCN_module(out_action_list, i, type="action")
            A, C_hat = self._Cognition_module(Ho, Ha, i)
            # out = A + C_hat   ==> change to the following concat (this is different from the description in paper)
            AC = torch.cat([A, C_hat], dim=1)
            out = self.parameters_all_agent[i]["fc_out_for_Qvalue"](AC)
            out_hidden_list.append(out)
            C_hat_list.append(C_hat)
        obs_hat_list, action_hat_list = self._Decoder(C_hat_list)
        return (out_hidden_list, C_hat_list, obs_hat_list, action_hat_list)


class CriticContrastiveNCC(CriticBase):
    def __init__(self, args):
        super(CriticContrastiveNCC, self).__init__(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_gcn_obs"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_gcn_action"] = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        parameters_dict["fc_cognition_A"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_meam"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_cognition_C_logstd"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_out_for_Qvalue"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)

    # this is the same as _GCN_module() of SimpleNCC
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

    def _Cognition_module(self, Ho, Ha, agent_index):
        H = torch.cat([Ho, Ha], dim=1)
        # do not use shortcut to generate A (this is different from the description in paper)
        A = self.parameters_all_agent[agent_index]["fc_cognition_A"](H)
        C_mean = self.parameters_all_agent[agent_index]["fc_cognition_C_meam"](H)
        C_logstd = self.parameters_all_agent[agent_index]["fc_cognition_C_logstd"](H)
        C_hat = C_mean + torch.exp(C_logstd) * torch.normal(mean=0.0, std=1.0, size=C_logstd.shape)
        return A, C_hat

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        # FC-module is equal to the pre-process in forward() function of CriticBase(), so we skip FC-Module
        out_hidden_list, C_hat_list = [], []
        for i in range(self.args.agent_count):
            Ho = self._GCN_module(out_obs_list, i, type="obs")
            Ha = self._GCN_module(out_action_list, i, type="action")
            A, C_hat = self._Cognition_module(Ho, Ha, i)
            # out = A + C_hat ==> change to the following concat (this is different from the description in paper)
            AC = torch.cat([A, C_hat], dim=1)
            out = self.parameters_all_agent[i]["fc_out_for_Qvalue"](AC)
            out_hidden_list.append(out)
            C_hat_list.append(C_hat)
        return (out_hidden_list, C_hat_list)


# if adopt the !parameter_sharing! method, please define a class with shared parameters
# Multi-Actor-Attention-Critic (MAAC): Actor-Attention-Critic for Multi-Agent Reinforcement Learning (ICML19)
import numpy as np
class MultiHeadAttention:
    def __init__(self, args):
        self.all_Wq, self.all_Wk, self.all_Wv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()  # do not use []
        for _ in range(args.head_count):
            Wq = nn.Linear(args.hidden_dim, args.hidden_dim)
            Wk = nn.Linear(args.hidden_dim, args.hidden_dim)
            Wv = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.all_Wq.append(Wq)
            self.all_Wk.append(Wk)
            self.all_Wv.append(Wv)

    def forward(self, e_list, agent_index):
        # https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py#L124
        attention_head_list = []
        for Wq, Wk, Wv in zip(self.all_Wq, self.all_Wk, self.all_Wv):  # for each attention-head
            ei = e_list[agent_index]
            # query = torch.matmul(ei, Wq)  # (batch_size, hidden_dim)
            query = Wq(ei)

            # calculate attention across agents
            query = torch.unsqueeze(query, dim=1)  # (batch_size, 1, hidden_dim)
            # keys = [torch.matmul(ej, Wk) for j, ej in enumerate(e_list) if j != agent_index]
            keys = [Wk(ej) for j, ej in enumerate(e_list) if j != agent_index]
            keys = torch.stack(keys, dim=0)  # (agent_count-1==count_of_other_agents, batch_size, hidden_dim)
            keys = keys.permute(1, 2, 0)  # (batch_size, hidden_dim, agent_count-1)
            attend_logits = torch.matmul(query, keys)  # (batch_size, 1, agent_count-1)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_logits = attend_logits / np.sqrt(keys[0].shape[1])  # (batch_size, 1, agent_count-1)
            attend_weights = F.softmax(scaled_logits, dim=-1)  # (batch_size, 1, agent_count-1)

            # calculate "The contribution from other agents, xi, is a weighted sum of each agent's"
            # values = [F.leaky_relu(torch.matmul(ej, Wv)) for j, ej in enumerate(e_list) if j != agent_index]
            values = [F.leaky_relu(Wv(ej)) for j, ej in enumerate(e_list) if j != agent_index]
            # for values: h is an element-wise nonlinearity (we have used leaky ReLU).
            values = torch.stack(values, dim=0)  # (agent_count-1, batch_size, hidden_dim)
            values = values.permute(1, 2, 0)  # (batch_size, hidden_dim, agent_count-1)
            attention_head = torch.mul(values, attend_weights)  # (batch_size, hidden_dim, agent_count-1)
            attention_head = torch.sum(attention_head, dim=2)  # (batch_size, hidden_dim)
            attention_head_list.append(attention_head)
        # In this case, each head, using a separate set of parameters (Wk, Wq, Wv),
        # gives rise to an aggregated contribution from all other agents to the agent i and
        # we simply *concatenate* the contributions from all heads as a single vector.
        xi = torch.cat(attention_head_list, dim=1)  # (batch_size, hidden_dim*head_count_of_MHA)
        return xi

class CriticMAAC(CriticBase):
    def __init__(self, args):
        super(CriticMAAC, self).__init__(args)
        self.MHA = MultiHeadAttention(args)

    def _define_parameters_for_hidden_layers(self, parameters_dict, agent_index=None):
        parameters_dict["fc_cat_oa"] = nn.Linear(self.args.hidden_dim * 2, self.args.hidden_dim)
        parameters_dict["fc_cat_ex"] = nn.Linear(self.args.hidden_dim * (1+self.args.head_count), self.args.hidden_dim)

    def _forward_of_hidden_layers(self, out_obs_list, out_action_list):
        e_list = []
        for i in range(self.args.agent_count):
            # gi is a one-layer MLP embedding function. (for each individual agent)
            # https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py#L108  ==> concatenate
            out_oa = torch.cat([out_obs_list[i], out_action_list[i]], dim=1)
            e = F.relu(self.parameters_all_agent[i]["fc_cat_oa"](out_oa))
            e_list.append(e)

        # in MAAC, the MultiHeadAttention module (and its parameters) is shared among agents
        x_list = []
        for i in range(self.args.agent_count):
            x = self.MHA.forward(e_list, i)
            x_list.append(x)

        out_hidden_list = []
        for i in range(self.args.agent_count):
            # originally in MAAC, fi is a two-layer MLP. (for each individual agent)
            # here, I use one-layer MLP, as the post-process in forward() of CriticBase() is the same as the 2nd layer
            # https://github.com/shariqiqbal2810/MAAC/blob/master/utils/critics.py#L148  ==> concatenate
            out_ex = torch.cat([e_list[i], x_list[i]], dim=1)
            out = F.relu(self.parameters_all_agent[i]["fc_cat_ex"](out_ex))
            out_hidden_list.append(out)
        return out_hidden_list


class Agent(object):
    def __init__(self, args):
        self.args = args
        print("=" * 30, "create agent", self.args.agent_name)
        if self.args.agent_name == 'IND_AC':
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)  # target network
            self.critic = CriticIndependent(args)
            self.T_critic = CriticIndependent(args)
        elif self.args.agent_name == 'MADDPG':
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)
            self.critic = CriticMADDPG(args)
            self.T_critic = CriticMADDPG(args)
        elif self.args.agent_name == "ATT_MADDPG":
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)
            self.critic = CriticAttentionalMADDPG(args)
            self.T_critic = CriticAttentionalMADDPG(args)
        elif self.args.agent_name == 'NCC_AC':
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)
            self.critic = CriticVaeNCC(args)
            self.T_critic = CriticVaeNCC(args)
        elif self.args.agent_name == 'MAAC':
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)
            self.critic = CriticMAAC(args)
            self.T_critic = CriticMAAC(args)
        elif self.args.agent_name == 'Contrastive':
            self.actor = ActorIndependent(args)
            self.T_actor = ActorIndependent(args)
            self.critic = CriticContrastiveNCC(args)
            self.T_critic = CriticContrastiveNCC(args)
        else:
            raise ValueError('args.agent_name is not defined ...')

        self._init_necessary_info()

    def _init_necessary_info(self):
        # xavier-init main networks before training
        for m in self.actor.modules():  # will visit all modules recursively (including sub-sub-...-sub-modules)
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
        for m in self.critic.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

        # init target network before training
        self.train_target_network_hard()

        # set target network to evaluation mode
        self.T_actor.eval()
        self.T_critic.eval()

        # create optimizers
        self.MSEloss = nn.MSELoss(reduction="mean")
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

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

    def generate_action(self, observation_list):
        self._config_evaluation_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_list = self.actor(observation_list)
        if self.args.agent_name in ["NCC_AC", "Contrastive"]:
            Qvalue_list = self.critic(observation_list, action_list)
            _, C_hat_list, _, _ = Qvalue_list
            self.save_cognition_for_human_understanding([C_hat.cpu().detach().numpy() for C_hat in C_hat_list])
        return [action.cpu().detach().numpy() for action in action_list]

    def train_critic(self, observation_list, action_list, reward_list, next_observation_list, done_batch, writer=None, training_step=0):
        self._config_train_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]
        action_list = [to_tensor(action) for action in action_list]
        reward_list = [to_tensor(reward) for reward in reward_list]
        next_observation_list = [to_tensor(next_observation) for next_observation in next_observation_list]
        multiplier_batch = to_tensor(1.0 - done_batch)

        Qvalue_list = self.critic(observation_list, action_list)  # not T_net
        next_action_list = self.T_actor(next_observation_list)  # use T_net
        target_Qvalue_list = self.T_critic(next_observation_list, next_action_list)  # use T_net
        if self.args.agent_name == "NCC_AC":
            Qvalue_list, C_hat_list, obs_hat_list, action_hat_list = Qvalue_list
            target_Qvalue_list, _, _, _ = target_Qvalue_list
        elif self.args.agent_name == "Contrastive":
            Qvalue_list, C_hat_list = Qvalue_list
            target_Qvalue_list, _ = target_Qvalue_list

        total_loss = 0.0
        for i in range(self.args.agent_count):
            TDtarget_i = reward_list[i] + multiplier_batch * self.args.gamma * target_Qvalue_list[i]
            total_loss += self.MSEloss(Qvalue_list[i], TDtarget_i.cpu().detach())  # note the detach
        if writer is not None:
            writer.add_scalar("TD_loss", total_loss.cpu().detach().item(), training_step)

        if self.args.agent_name == "NCC_AC":
            total_KL_loss, total_L2_loss = 0.0, 0.0
            for i in range(self.args.agent_count):
                KL_loss = 0.0
                for j, value_j in enumerate(self.args.adj[i]):
                    if value_j == 1:  # agent j is a neighbor of agent i
                        KL_loss += self.KLDivLoss(F.log_softmax(C_hat_list[i], dim=1), F.softmax(C_hat_list[j], dim=1))
                        # KLDivLoss: the input given is expected to contain log-probabilities, target is probabilities
                        # print("KL_loss ==>", KL_loss)
                total_KL_loss += KL_loss / sum(self.args.adj[i])  # normalized by neighbor count
                total_L2_loss += self.MSEloss(obs_hat_list[i], observation_list[i])
                total_L2_loss += self.MSEloss(action_hat_list[i], action_list[i])
            total_KL_loss = (self.args.alpha_KL * total_KL_loss) / self.args.agent_count  # normalized by agent count
            total_L2_loss = (self.args.alpha_L2 * total_L2_loss) / self.args.agent_count  # normalized by agent count
            total_loss += (total_KL_loss + total_L2_loss)
            if writer is not None:
                writer.add_scalar("KL_loss", total_KL_loss.cpu().detach().item(), training_step)
                writer.add_scalar("L2_loss", total_L2_loss.cpu().detach().item(), training_step)
        if self.args.agent_name == "Contrastive":
            C_hat_tensor = torch.stack(C_hat_list).permute(1, 0, 2)  # (batch_size, num_agents, dim_hidden)
            for i in range(self.args.hidden_dim):
                print("C_hat_tensor_agent0_dim"+str(i), C_hat_tensor[0, 0, i].cpu().detach().item())
                print("C_hat_tensor_agent1_dim"+str(i), C_hat_tensor[0, 1, i].cpu().detach().item())
                writer.add_scalar("C_hat_tensor_agent0_dim"+str(i), C_hat_tensor[0, 0, i].cpu().detach().item(), training_step)
                writer.add_scalar("C_hat_tensor_agent1_dim"+str(i), C_hat_tensor[0, 1, i].cpu().detach().item(), training_step)
            C_hat_tensor_Transpose = C_hat_tensor.permute(0, 2, 1)  # (batch_size, dim_hidden, num_agents)
            bilinear_similarity = torch.bmm(C_hat_tensor, C_hat_tensor_Transpose)  # (batch_size, num_agents, num_agents)
            max_v = torch.max(bilinear_similarity, 2).values.detach()  # (batch_size, num_agents)
            max_v = max_v.unsqueeze(2).repeat(1, 1, self.args.agent_count)  # (batch_size, num_agents, num_agents)
            exp_v = torch.exp(bilinear_similarity - max_v)  # (batch_size, num_agents, num_agents)

            total_Contrastive_loss = 0.0
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
            total_Contrastive_loss = (self.args.alpha_CON * total_Contrastive_loss) / self.args.agent_count  # normalized by agent count
            total_loss += total_Contrastive_loss
            if writer is not None:
                writer.add_scalar("Contrastive_loss", total_Contrastive_loss.cpu().detach().item(), training_step)

        self.optimizer_critic.zero_grad()  # clear previous gradients before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clipped_norm_value)  # after bw() before step()
        self.optimizer_critic.step()
        return total_loss.cpu().detach().numpy()

    def train_actor(self, observation_list, writer=None, training_step=0):
        self._config_train_mode()
        observation_list = [to_tensor(observation) for observation in observation_list]

        action_list = self.actor(observation_list)  # not T_net
        Qvalue_list = self.critic(observation_list, action_list)  # not T_net
        if self.args.agent_name == "NCC_AC":
            Qvalue_list, _, _, _ = Qvalue_list
        elif self.args.agent_name == "Contrastive":
            Qvalue_list, _ = Qvalue_list

        total_loss = 0.0
        for i in range(self.args.agent_count):
            loss_i = -Qvalue_list[i].mean()  # negative Qvalue
            total_loss += loss_i
        total_loss /= self.args.agent_count  # normalized by agent count
        if writer is not None:
            writer.add_scalar("Actor_loss", total_loss.cpu().detach().item(), training_step)

        self.optimizer_actor.zero_grad()  # clear previous gradients before update
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clipped_norm_value)  # after bw() before step()
        self.optimizer_actor.step()
        return total_loss.cpu().detach().numpy()

    def train_target_network_soft(self):
        for target_param, param in zip(self.T_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)
        for target_param, param in zip(self.T_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def train_target_network_hard(self):
        for target_param, param in zip(self.T_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.T_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def _config_cuda(self):
        self.actor.cuda()
        self.T_actor.cuda()
        self.critic.cuda()
        self.T_critic.cuda()

    def _config_train_mode(self):
        self.actor.train()  # set train mode
        self.critic.train()

    def _config_evaluation_mode(self):
        self.actor.eval()  # set evaluation mode
        self.critic.eval()

    def save_model(self, output):
        print("save_model() ...")
        torch.save(self.actor.state_dict(), '{}-actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}-critic.pkl'.format(output))

    def load_weights(self, output):
        print("load_weights() ...")
        self.actor.load_state_dict(torch.load('{}-actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}-critic.pkl'.format(output)))
        # init target network before training
        self.train_target_network_hard()
