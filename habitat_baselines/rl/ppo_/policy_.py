#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc

import numpy as np
import torch
import torch.nn as nn

from habitat_baselines.common.utils import CategoricalNet, Flatten

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._setup_net(input_dim, output_dim)
    
    def _setup_net(self, feature_dim, actor_out_size):
        self.net = nn.Sequential(
                   nn.Linear(feature_dim, 64),
                   nn.ReLU(True),
                   nn.Linear(64, actor_out_size),
                   nn.ReLU(True),
        )
        self._init_weight()
        print(self.net)

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

    def _init_weight(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, cnn_feature, rnn_hidden_state):
        cnn_feature = cnn_feature.unsqueeze(0)
        # print(cnn_feature.size())
        # print(rnn_hidden_state.size())
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
                   Flatten(),
                   nn.Linear(15*15*256, feature_dim),
                   nn.ReLU(True),
        )

        self._init_weight()
        print(self.cnn)

    def _init_weight(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, obs):
        depth = obs["depth"]
        depth = depth.permute(0, 3, 1, 2)
        feature = self.cnn(depth)
        return feature

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
    def __init__(self, cnn_parameter, rnn_parameter, actor_parameter, critic_parameter):
        super().__init__()
        self.cnn = CNN_encoder(rgb_input=cnn_parameter["observation_spaces"].spaces["depth"].shape[2], 
                               feature_dim=cnn_parameter["feature_dim"]
                               )
        input_dim = cnn_parameter["feature_dim"] + cnn_parameter["observation_spaces"].spaces["pointgoal_with_gps_compass"].shape[0]
        self.rnn = RNN_encoder(input_dim=input_dim, 
                               hidden_dim=rnn_parameter["hidden_dim"], 
                               n_layer=rnn_parameter["n_layer"],
                              )

        self.actor = Actor(input_dim=rnn_parameter["hidden_dim"], 
                           output_dim=actor_parameter["action_spaces"])
        self.critic = Critic(input_dim=rnn_parameter["hidden_dim"], 
                             )
    # 給rollout
    def act(self, obs, rnn_hidden_state):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state)

        distributions_logits = self.actor(feature)
        distributions        = CustomCategorical(logits=distributions_logits)
        action               = distributions.sample()
        actions_log_probs    = distributions.log_probs(action)
        value                = self.critic(feature)
        return actions_log_probs, action, value, distributions_logits, rnn_hidden_state

    # 給update取最後一次value
    def get_value(self, obs, rnn_hidden_state):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state)

        value   = self.critic(feature)
        return value

    # update用，計算Policy loss, value loss
    def evaluate_value(self, obs, action, rnn_hidden_state):
        feature, rnn_hidden_state = self._run_cnn_rnn(obs, rnn_hidden_state)

        distributions_logits  = self.actor(feature)
        distributions         = CustomCategorical(logits=distributions_logits)
        # print("distributions.entropy:", distributions.entropy())
        distributions_entropy = distributions.entropy().mean()
        actions_log_probs     = distributions.log_probs(action)
        value = self.critic(feature)
        return distributions_entropy, actions_log_probs, value

    def _run_cnn_rnn(self, obs, rnn_hidden_state):
        feature = self.cnn(obs)
        feature = self._cat_pointgoal_with_gps_compass(obs, feature)

        feature, rnn_hidden_state = self.rnn(feature, rnn_hidden_state)
        return feature, rnn_hidden_state

    def _cat_pointgoal_with_gps_compass(self, obs, feature):
        return torch.cat([feature, obs["pointgoal_with_gps_compass"]], dim=-1)