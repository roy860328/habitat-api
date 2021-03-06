#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.common.utils import CategoricalNet, Flatten

def _print_model_parameters(net):
    total_weight = 0
    for param in net.parameters():
        dim = 1
        for s in list(param.size()):
            dim *= s
        total_weight += dim
    print(total_weight)

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._setup_net(input_dim, output_dim)
    
    def _setup_net(self, feature_dim, actor_out_size):
        self.net = nn.Sequential(
                   nn.Linear(feature_dim, 64),
                   nn.ReLU(True),
                   nn.Linear(64, actor_out_size),
                   # nn.Sigmoid(),
        )
        self._init_weight()
        print(self.net)
        _print_model_parameters(self.net)

    def _init_weight(self):
        for layer in self.net:
            if isinstance(layer, (nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, feature):
        action = self.net(feature)
        return action

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self._setup_net(input_dim)
    
    def _setup_net(self, feature_dim):
        self.net = nn.Sequential(
                   nn.Linear(feature_dim, 64),
                   nn.ReLU(True),
                   nn.Linear(64, 1),
        )
        self._init_weight()
        print(self.net)
        _print_model_parameters(self.net)

    def _init_weight(self):
        for layer in self.net:
            if isinstance(layer, (nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, feature):
        value = self.net(feature)
        return value

# https://pytorch.org/docs/1.4.0/nn.html?highlight=lstm#torch.nn.LSTM
# https://pytorch.org/docs/1.4.0/nn.html?highlight=lstm#torch.nn.GRU
class RNN_encoder(nn.Module):
    """docstring for CNN_encoder"""

    def __init__(self, input_dim, hidden_dim, n_layer, drop_prob=0.2):
        '''
        Dict(depth:Box(256, 256, 1), pointgoal_with_gps_compass:Box(2,), rgb:Box(256, 256, 3))
        '''
        super().__init__()
        self._setup_net(input_dim, hidden_dim, n_layer, drop_prob)
    
    def _setup_net(self, input_dim, hidden_dim, n_layer, drop_prob):
        self.rnn = nn.GRU(input_dim, hidden_dim, n_layer, dropout=drop_prob)
        self._init_weight()
        print(self.rnn)
        _print_model_parameters(self.rnn)

    def _init_weight(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, cnn_feature, rnn_hidden_state, masks):
        cnn_feature = cnn_feature.unsqueeze(0)
        # print(cnn_feature.size())
        # print(rnn_hidden_state.size())
        rnn_hidden_state = masks * rnn_hidden_state
        out, rnn_hidden_state = self.rnn(cnn_feature, rnn_hidden_state)

        return out.squeeze(0), rnn_hidden_state

class CNN_encoder(nn.Module):
    """docstring for CNN_encoder"""

    def __init__(self, rgb_input, feature_dim):
        '''
        Dict(depth:Box(256, 256, 1), pointgoal_with_gps_compass:Box(2,), rgb:Box(256, 256, 3))
        '''
        super().__init__()
        self._setup_net(rgb_input, feature_dim)
    
    def _setup_net(self, rgb_input, feature_dim):
        self.cnn = nn.Sequential(
                   nn.Conv2d(in_channels=rgb_input, out_channels=32, kernel_size=3, stride=2),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
                   nn.ReLU(True),
                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2),
                   nn.ReLU(True),
        )
        self.flatten = nn.Sequential(
                   Flatten(),
                   nn.Linear(15*15*256, feature_dim),
                   # nn.ReLU(True),
        )

        self._init_weight()
        print(self.cnn)
        _print_model_parameters(self.cnn)

    def _init_weight(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
        for layer in self.flatten:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, obs):
        depth = obs["depth"]
        depth = depth.permute(0, 3, 1, 2)
        feature_map = self.cnn(depth)
        feature     = self.flatten(feature_map)
        return feature_map, feature

"""Splitnet"""

"""Splitnet visaul decoder"""

class CNN_decoder(nn.Module):
    """Splitnet visaul decoder"""

    def __init__(self, ):
        '''
        Dict(depth:Box(256, 256, 1), pointgoal_with_gps_compass:Box(2,), rgb:Box(256, 256, 3))
        '''
        super().__init__()
        self._setup_net()
    
    def _setup_net(self,):
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, output_padding=1),
            nn.Tanh(),
        )
        self._init_weight()
        print(self.decoder)
        _print_model_parameters(self.decoder)

    def _init_weight(self):
        for layer in self.decoder:
            if isinstance(layer, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, feature):
        depth_predict = self.decoder(feature)
        depth_predict = depth_predict.permute(0, 2, 3, 1)
        return depth_predict

""" Motion Auxiliary Tasks """

class ActionAuxiliary(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input : feature * 2
        output: action
        """
        super().__init__()
        self._setup_net(input_dim, output_dim)
    
    def _setup_net(self, feature_dim, action_out_size):
        self.predict_action = nn.Sequential(
                   nn.Linear(feature_dim, 64),
                   nn.ReLU(True),
                   nn.Linear(64, action_out_size),
                   # nn.Sigmoid(),
        )
        self._init_weight()

    def _init_weight(self):
        for layer in self.predict_action:
            if isinstance(layer, (nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    def forward(self, feature):
        """
        feature_t_1 : feature t-1
        feature_t_2 : feature t
        """
        feature = feature.squeeze(0)
        feature_t_1 = feature[:-1]
        feature_t_2 = feature[1:]
        cat_feature = torch.cat([feature_t_1, feature_t_2], dim=-1)
        action = self.predict_action(cat_feature)
        return action


class FeatureAuxiliary(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input : feature + action
        output: next feature
        """
        super().__init__()
        self._setup_net(input_dim, output_dim)
    
    def _setup_net(self, feature_dim, next_feature_dim):
        self.predict_next_feature = nn.Sequential(
                   nn.Linear(feature_dim, 64),
                   nn.ReLU(True),
                   nn.Linear(64, next_feature_dim),
                   # nn.Sigmoid(),
        )
        self._init_weight()

    def _init_weight(self):
        for layer in self.predict_next_feature:
            if isinstance(layer, (nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, feature, action):
        """
        input : feature + action
        output: next feature
        """
        feature = feature.squeeze(0)
        cat_input = torch.cat([feature[:-1], action[:-1]], dim=-1)
        next_feature = self.predict_next_feature(cat_input)
        return next_feature


"""Splitnet"""

class CustomCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

class PointNavBaselinePolicy_(nn.Module):
    def __init__(self, 
                 cnn_parameter, 
                 depth_decoder_parameter, 
                 rnn_parameter, 
                 actor_parameter, 
                 critic_parameter,
                 use_splitnet_auxiliary=False):
        super().__init__()
        self.cnn               = CNN_encoder(rgb_input=cnn_parameter["observation_spaces"].spaces["depth"].shape[2], 
                                             feature_dim=cnn_parameter["feature_dim"]
                                             )
        input_dim              = cnn_parameter["feature_dim"] + cnn_parameter["observation_spaces"].spaces["pointgoal_with_gps_compass"].shape[0]
        self.rnn               = RNN_encoder(input_dim=input_dim, 
                                             hidden_dim=rnn_parameter["hidden_dim"], 
                                             n_layer=rnn_parameter["n_layer"],
                                             )
        self.actor             = Actor(input_dim=rnn_parameter["hidden_dim"], 
                                       output_dim=actor_parameter["action_spaces"],
                                       )
        self.critic            = Critic(input_dim=rnn_parameter["hidden_dim"], 
                                        )
        
        self.use_splitnet_auxiliary = use_splitnet_auxiliary
        if use_splitnet_auxiliary:
            self.cnn_decoder       = CNN_decoder(
                                                 )
            self.action_auxiliary  = ActionAuxiliary(rnn_parameter["hidden_dim"]*2,
                                                     actor_parameter["action_spaces"]
                                                     )
            self.feature_auxiliary = FeatureAuxiliary(rnn_parameter["hidden_dim"] + actor_parameter["action_spaces"],
                                                      rnn_parameter["hidden_dim"],
                                                      )
    # 給rollout
    def act(self, obs, rnn_hidden_state, masks):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state, masks)

        distributions_logits = self.actor(feature)
        distributions        = CustomCategorical(logits=distributions_logits)
        action               = distributions.sample()
        actions_log_probs    = distributions.log_probs(action)
        value                = self.critic(feature)
        return actions_log_probs, action, value, distributions_logits, rnn_hidden_state

    # 給update取最後一次value
    def get_value(self, obs, rnn_hidden_state, masks):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state, masks)

        value   = self.critic(feature)
        return value

    # update用，計算Policy loss, value loss
    def evaluate_value(self, obs, action, rnn_hidden_state, masks):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state, masks)

        distributions_logits  = self.actor(feature)
        print(distributions_logits)
        print(torch.max(distributions_logits, 1)[1])
        distributions         = CustomCategorical(logits=distributions_logits)
        # print("distributions.entropy:", distributions.entropy())
        distributions_entropy = distributions.entropy().mean()
        actions_log_probs     = distributions.log_probs(action)
        value = self.critic(feature)
        return distributions_entropy, actions_log_probs, value

    def evaluate_auxiliary(self, obs, rnn_hidden_state, action):
        # cnn_decoder
        feature_map, _ = self.cnn(obs)
        depth_img = self.cnn_decoder(feature_map)
        # ActionAuxiliary
        action_predict = self.action_auxiliary(rnn_hidden_state)
        # FeatureAuxiliary
        next_feature_predict = self.feature_auxiliary(rnn_hidden_state, action)

        return depth_img, action_predict, next_feature_predict

    def _run_cnn_rnn(self, obs, rnn_hidden_state, masks):
        _, feature = self.cnn(obs)
        feature = self._cat_pointgoal_with_gps_compass(obs, feature)

        feature, rnn_hidden_state = self.rnn(feature, rnn_hidden_state, masks)
        return feature, rnn_hidden_state

    def _cat_pointgoal_with_gps_compass(self, obs, feature):
        return torch.cat([feature, obs["pointgoal_with_gps_compass"]], dim=-1)