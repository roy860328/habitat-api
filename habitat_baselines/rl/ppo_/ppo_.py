#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

EPS_PPO = 1e-5


class PPO_(nn.Module):
    def __init__(self, actor_critic, config):
        super().__init__()
        self.actor_critic = actor_critic
        self.config = config
        self.optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
                                    lr=self.config.RL.PPO.lr,
                                    eps=self.config.RL.PPO.eps,)
        self.clip_param = self.config.RL.PPO.clip_param
        self.value_loss_coef = self.config.RL.PPO.value_loss_coef
        self.entropy_coef = self.config.RL.PPO.entropy_coef

        self.reconstruction_function = torch.nn.MSELoss()

    # mainly for self.actor_critic.act()
    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollout):
        advantages = rollout.returns - rollout.value_preds
        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollout):
        loss = 0
        advantages = self.get_advantages(rollout)

        print("=== UPDATE ===")

        for epoch in range(self.config.RL.PPO.ppo_epoch):
            print("=== epoch {} ===".format(epoch))
            data_generator = rollout.recurrent_generator(advantages, self.config.RL.PPO.num_mini_batch)
            for mini_batch in data_generator:
                (
                    observations_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = mini_batch
                # print(observations_batch["depth"].size())
                # print(recurrent_hidden_states_batch.size())
                # print(actions_batch.size())
                # print(return_batch.size())
                distributions_entropy, actions_log_probs, value = \
                    self.actor_critic.evaluate_value(observations_batch, 
                                                     actions_batch,
                                                     recurrent_hidden_states_batch,
                                                     masks_batch)
                depth_img = self.actor_critic.evaluate_decoder(observations_batch, 
                                                               )
                # print("return_batch:", return_batch)
                # print("value:", value)

                # actions_loss
                ratio = torch.exp(actions_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                actions_loss = -1 * torch.min(surr1, surr2).mean()
                print(ratio[0])
                print(surr1[0])

                # values_loss
                values_loss  = 0.5 * (return_batch - value).pow(2).mean()

                # depth img predict loss
                depth_img_loss = self.reconstruction_function(depth_img, 
                                                              observations_batch["depth"])

                # optimizer
                self.optimizer.zero_grad()
                # total_loss
                total_loss = (actions_loss
                              + values_loss * self.value_loss_coef 
                              - distributions_entropy * self.entropy_coef
                              + depth_img_loss)
                total_loss.backward()

                self.optimizer.step()

                loss += total_loss.item()
        print("loss:", loss)
        print("actions_loss:", actions_loss)
        print("values_loss:", values_loss)
        print("distributions_entropy:", distributions_entropy)
        print("depth_img_loss:", depth_img_loss)
        loss_dict = {"loss":loss,
                     "actions_loss":actions_loss,
                     "values_loss":values_loss,
                     "distributions_entropy":distributions_entropy.detach(),
                     "depth_img_loss":depth_img_loss,
                     }
        return loss_dict