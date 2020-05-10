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
    def __init__(self, feature_dim, actor_out_size):
        super().__init__()
        self._setup_net(feature_dim, actor_out_size)
    
    def _setup_net(self, feature_dim, actor_out_size):
        self.net = nn.Sequential(
                   nn.Linear(feature_dim, actor_out_size),
        )

    def forward(self, feature):
        action = self.net(feature)
        return action

class Critic(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self._setup_net(feature_dim)
    
    def _setup_net(self, feature_dim):
        self.net = nn.Sequential(
                   nn.Linear(feature_dim, 1),
        )

    def forward(self, feature):
        value = self.net(feature)
        return value

class CNN_encoder(nn.Module):
    """docstring for CNN_encoder"""
    def __init__(self, rgb_input, feature_dim):
        super().__init__()
        self._setup_net(rgb_input, feature_dim)
    
    def _setup_net(self, rgb_input, feature_dim):
        self.cnn = nn.Sequential(
                   nn.Conv2d(in_channels=rgb_input, out_channels=32, kernel_size=3, stride=1),
                   nn.ReLU(True),
                   Flatten(),
                   nn.Linear(254*254*32, feature_dim),
        )
        print(self.cnn)

    def forward(self, x):
        (b, i, j, k) = x.shape
        x = x.unsqueeze(0).view(b, k, i, j)
        feature = self.cnn(x)
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
    def __init__(self, observation_spaces, feature_dim, action_spaces):
        super().__init__()
        self.cnn = CNN_encoder(
                                rgb_input=observation_spaces.spaces["rgb"].shape[2], 
                                feature_dim=feature_dim
                                )
        self.actor = Actor(feature_dim, actor_out_size=action_spaces)
        self.critic = Critic(feature_dim)
    # 給rollout
    def act(self, obs):
        feature = self.cnn(obs["rgb"])
        distributions_logits = self.actor(feature)
        distributions        = CustomCategorical(logits=distributions_logits)
        action               = distributions.sample()
        actions_log_probs    = distributions.log_probs(action)
        value                = self.critic(feature)
        return actions_log_probs, action, value, distributions_logits

    # 給update取最後一次value
    def get_value(self, obs):
        feature = self.cnn(obs["rgb"])
        value   = self.critic(feature)
        return value

    # update用，計算Policy loss, value loss
    def evaluate_value(self, obs, action):
        feature = self.cnn(obs["rgb"])
        distributions_logits  = self.actor(feature)
        distributions         = CustomCategorical(logits=distributions_logits)
        print("distributions.entropy:", distributions.entropy())
        distributions_entropy = distributions.entropy().mean()
        actions_log_probs     = distributions.log_probs(action)
        value = self.critic(feature)
        return distributions_entropy, actions_log_probs, value
