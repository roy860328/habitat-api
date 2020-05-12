#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from habitat_baselines.rl.ppo_ import PPO_, PointNavBaselinePolicy_


@baseline_registry.register_trainer(name="ppo_")
class PPOTrainer_(BaseRLTrainer):
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )
        self._setup_actor_critic()
        self.rollout = RolloutStorage(
                        self.config.RL.PPO.num_steps,
                        self.envs.num_envs,
                        self.envs.observation_spaces[0],
                        self.envs.action_spaces[0],
                        self.config.RL.PPO.hidden_size
                        )
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.rollout.to(self.device)

    def _setup_actor_critic(self):
        print(self.config)
        # len(observation_spaces)==NUM_PROCESSES==4?
        print(self.envs.observation_spaces[0])
        print(self.envs.action_spaces[0])
        self.actor_critic = PointNavBaselinePolicy_(
                            cnn_parameter={"observation_spaces":self.envs.observation_spaces[0], "feature_dim":50},
                            rnn_parameter={"intput_dim":0, "hidden_dim":self.config.RL.PPO.hidden_size, "n_layer":1},
                            actor_parameter={"action_spaces":4, },
                            critic_parameter={},
                            )
        self.agent = PPO_(self.actor_critic,
                          self.config, )

    def _collect_rollout_step(self, rewards_record, count):
        '''
        obss :   "rgb"
                 "depth"
                 "pointgoal_with_gps_compass"
        '''
        obs = {k:v[self.rollout.step] for k, v in self.rollout.observations.items()}
        hidden_states = self.rollout.recurrent_hidden_states[self.rollout.step]
        ### PASS DATA
        with torch.no_grad():
            (actions_log_probs, 
             actions, 
             value, 
             distributions, 
             rnn_hidden_state) = (self.actor_critic.act(obs, hidden_states))
            # print("distributions:", distributions)
            # print("actions_log_probs:", actions_log_probs)
            # print("actions:", actions)

        ### PASS ENV
        res = self.envs.step([action.item() for action in actions])

        ### Process
        observations, rewards, dones, infos = [list(x) for x in zip(*res)]
        observations = batch_obs(observations, self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(-1)
        dones   = torch.tensor(dones, dtype=torch.float, device=self.device)
        inverse_dones   = torch.tensor([[0] if done else [1] for done in dones], dtype=torch.float, device=self.device)
        self.rollout.insert(
                            observations,
                            rnn_hidden_state,
                            actions,
                            actions_log_probs,
                            value,
                            rewards,
                            inverse_dones,
                            )
        # print("rewards:", rewards)
        # print(infos)
        if rewards_record is None:
            rewards_record, count = rewards, dones
        else:
            rewards_record += rewards
            count = dones
        return rewards_record, count

    def _update_agent(self):
        with torch.no_grad():
            last_obs = {k:v[self.rollout.step] for k, v in self.rollout.observations.items()}
            hidden_states = self.rollout.recurrent_hidden_states[self.rollout.step]
            next_value = self.actor_critic.get_value(last_obs, hidden_states).detach()

        self.rollout.compute_returns(next_value, 
                                     self.config.RL.PPO.use_gae, 
                                     self.config.RL.PPO.gamma, 
                                     self.config.RL.PPO.tau)
        loss = self.agent.update(self.rollout)
        self.rollout.after_update()

        return loss

    def train(self) -> None:
        #### init
        obs = self.envs.reset()
        #### 先暫存一個 obs
        batch = batch_obs(obs, device=self.device)
        for sensor in self.rollout.observations:
            self.rollout.observations[sensor][0].copy_(batch[sensor])
        #### Para
        rewards_record = None
        count = None
        #### 開始訓練
        with TensorboardWriter(
            "tb_example", flush_secs=self.flush_secs
        ) as writer:
            for epoch in range(self.config.NUM_UPDATES):
                #### 蒐集rollout
                print("=== collect rollout ===")
                for step in range(self.config.RL.PPO.num_steps):
                    rewards_record, count = self._collect_rollout_step(rewards_record, count)
                #### 更新
                loss = self._update_agent()
                #### LOGGER
                writer.add_scalars(
                    "loss", loss, epoch*self.envs.num_envs
                )
                writer.add_scalar(
                    "reward", rewards_record.sum(), epoch*self.envs.num_envs
                )
                writer.add_scalar(
                    "count", count.sum(), epoch*self.envs.num_envs
                )
                print("rewards_record:", rewards_record.sum())
                print("count:", count.sum())
                rewards_record = None
                count = None
        self.envs.close()

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0, ) -> None:
        pass
