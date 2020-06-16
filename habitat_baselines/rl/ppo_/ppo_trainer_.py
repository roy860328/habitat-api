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
        self._setup_actor_critic_agent(self.config)
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

    def _setup_actor_critic_agent(self, ppo_cfg):
        print(self.config)
        # len(observation_spaces)==NUM_PROCESSES==4?
        print(self.envs.observation_spaces[0])
        print(self.envs.action_spaces[0])
        self.actor_critic = PointNavBaselinePolicy_(
                            cnn_parameter={"observation_spaces":self.envs.observation_spaces[0], "feature_dim":self.config.RL.PPO.cnn_output_size},
                            depth_decoder_parameter={},
                            rnn_parameter={"input_dim":0, "hidden_dim":self.config.RL.PPO.hidden_size, "n_layer":1},
                            actor_parameter={"action_spaces":4, },
                            critic_parameter={},
                            use_splitnet_auxiliary=self.config.RL.PPO.use_splitnet_auxiliary,
                            )
        self.agent = PPO_(self.actor_critic,
                          self.config,
                          )

    def _collect_rollout_step(self, current_episode_reward, running_episode_stats):
        '''
        obss :   "rgb"
                 "depth"
                 "pointgoal_with_gps_compass"
        '''
        ### PASS DATA
        with torch.no_grad():
            obs = {k:v[self.rollout.step] for k, v in self.rollout.observations.items()}
            hidden_states = self.rollout.recurrent_hidden_states[self.rollout.step]
            inverse_dones = self.rollout.masks[self.rollout.step]
            (actions_log_probs, 
             actions, 
             value, 
             distributions, 
             rnn_hidden_state) = (self.actor_critic.act(obs, hidden_states, inverse_dones))
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
        # print("dones:", dones)
        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - inverse_dones) * current_episode_reward
        running_episode_stats["count"] += 1 - inverse_dones
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - inverse_dones) * v

        current_episode_reward *= inverse_dones
        return

    def _save_checkpoint(self, checkpoint_name):
        checkpoint_dict = {"state_dict": self.agent.state_dict(),
                           "config"    : self.config,
                           }
        torch.save(checkpoint_dict, 
                   os.path.join(self.config.CHECKPOINT_FOLDER, checkpoint_name),
                   )
    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        return torch.load(checkpoint_path, *args, **kwargs)

    def _update_agent(self):
        with torch.no_grad():
            last_obs = {k:v[self.rollout.step] for k, v in self.rollout.observations.items()}
            hidden_states = self.rollout.recurrent_hidden_states[self.rollout.step]
            inverse_dones = self.rollout.masks[self.rollout.step]
            next_value = self.actor_critic.get_value(last_obs, 
                                                     hidden_states, 
                                                     inverse_dones).detach()

        self.rollout.compute_returns(next_value, 
                                     self.config.RL.PPO.use_gae, 
                                     self.config.RL.PPO.gamma, 
                                     self.config.RL.PPO.tau)
        loss, loss_auxiliary = self.agent.update(self.rollout)
        self.rollout.after_update()

        return loss, loss_auxiliary

    def train(self) -> None:
        #### init
        obs = self.envs.reset()
        #### 先暫存一個 obs
        batch = batch_obs(obs, device=self.device)
        for sensor in self.rollout.observations:
            self.rollout.observations[sensor][0].copy_(batch[sensor])
        #### Para
        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        #### PPO LOG PARA
        count_steps = 0
        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.config.RL.PPO.reward_window_size)
        )
        
        #### 開始訓練
        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for epoch in range(self.config.NUM_UPDATES):
                #### decay
                if self.config.RL.PPO.use_linear_lr_decay:
                    lr_scheduler.step()
                if (epoch+1) % self.config.CHECKPOINT_INTERVAL==0:
                    self.agent.entropy_coef = self.agent.entropy_coef*0.9
                print(self.agent.entropy_coef)

                #### 蒐集rollout
                print("=== collect rollout ===")
                for step in range(self.config.RL.PPO.num_steps):
                    self._collect_rollout_step(current_episode_reward, running_episode_stats)
                    count_steps += self.envs.num_envs
                #### 更新
                loss, loss_auxiliary = self._update_agent()
                #### LOGGER
                writer.add_scalars(
                    "loss", loss, epoch*self.envs.num_envs
                )
                writer.add_scalars(
                    "loss_auxiliary", loss_auxiliary, epoch*self.envs.num_envs
                )

                #### PPO LOG PARA
                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)
                #### PPO LOG PARA

                if epoch%(1) == 0:
                    print("count_steps:", count_steps)
                    print("deltas:", deltas)
                    print("metrics:", metrics)
                if epoch%self.config.CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(f"checkpoint.{epoch}.pth")

        self.envs.close()

    # 每個checkpoint call一次
    # TEST_EPISODE_COUNT 跑幾次EPISODE
    # 還沒結束就將observations存到rgb_frames中，並且做後處裡，top_down_map...
    # Dones之後，generate_video，存到disk跟tensorboard
    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter, checkpoint_index: int = 0, ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            1,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (_, 
                 actions, 
                 _, 
                 _, 
                 test_recurrent_hidden_states) = (self.actor_critic.act(batch, test_recurrent_hidden_states, not_done_masks))

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)
            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():

            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
