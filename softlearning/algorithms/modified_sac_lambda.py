from collections import OrderedDict
import os
import traceback

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    """

    def __init__(
            self,
            policy,
            qfs,
            vf,
            target_qfs=None,

            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            lambda_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            use_grad_clip=True,
            use_huber_loss=False,

            optimizer_class=optim.Adam,
            beta_1=0.9,
            q_lambda=2.0,
            log_lambda=None,
            q_lambda_min=0.01,
            q_lambda_max=10.0,
            target_thresh=40.0,

            q_update_times=1,
            bc_reg_weight=0.0,
            rl_pg=False
    ):
        self.policy = policy
        self.qfs = qfs if isinstance(qfs, list) else [qfs]
        self.vf = vf

        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        if target_qfs is None:
            self.target_qfs = []
            for qf in self.qfs:
                self.target_qfs.append(qf.copy())
        else:
            self.target_qfs = target_qfs
        self.eval_statistics = None

        self.use_grad_clip = use_grad_clip
        self.use_huber_loss = use_huber_loss
        print("USE HUBER LOSS = ", self.use_huber_loss)

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.qfs_optimizer = []
        for qf in self.qfs:
            self.qfs_optimizer.append(optimizer_class(
                qf.parameters(),
                lr=qf_lr,
                betas=(beta_1, 0.999)
            ))

        if log_lambda is None:
            self.log_lambda = ptu.zeros(1, requires_grad=True)
        else:
            self.log_lambda = log_lambda
        self.lambda_optimizer = optimizer_class(
            [self.log_lambda],
            lr=lambda_lr,
            betas=(beta_1, 0.999)
        )
        self.lambda_min = q_lambda_min
        self.lambda_max = q_lambda_max
        print("Q LAMBDA MIN = ", self.lambda_min)
        print("Q LAMBDA MAX = ", self.lambda_max)
        print("LAMBDA LR = ", lambda_lr)

        self.target_thresh = target_thresh
        print("TARGET THRESH = ", target_thresh)

        self.q_update_times = q_update_times
        self.bc_reg_weight = bc_reg_weight
        self.use_rl_pg = int(rl_pg)
        if self.use_rl_pg:
            print("REINFORCE")

    def train_step(self, batch, **kwargs):
        training_mode = kwargs.get("training_mode", None)
        rewards = self.reward_scale * batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        discounted_rewards = self.reward_scale * batch['discounted_rewards']

        if training_mode is None or training_mode == "rl":
            """
            QF Loss
            """
            if self.q_update_times == -1:
                q_update_times = 1 + kwargs.get("extra_upd", 0)
            else:
                q_update_times = self.q_update_times if kwargs.get("extra_upd", False) else 1
            for _ in range(q_update_times):
                q_loss_mean, q_target_loss_mean, q_extra_loss_mean, lambda_loss, qfs_pred = \
                    self.critic_step(obs, actions, rewards, next_obs, terminals)

            """
            Policy Loss
            """
            policy_loss, log_pi, policy_mean, policy_log_std = self.actor_step(obs, actions)

            """
            Update networks
            """
            self._update_target_network()

            """
            Save some statistics for eval
            """
            if self.eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Reward Scale'] = self.reward_scale
                self.eval_statistics['QFs Loss'] = q_loss_mean
                self.eval_statistics['QFs Target Loss'] = q_target_loss_mean
                self.eval_statistics['QFs Extra Loss'] = q_extra_loss_mean
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
                self.eval_statistics['Lambda'] = ptu.get_numpy(self.log_lambda.exp())
                self.eval_statistics['Lambda Loss'] = np.mean(ptu.get_numpy(lambda_loss))
                for i, q_pred in enumerate(qfs_pred):
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q%d Predictions' % (i + 1),
                        ptu.get_numpy(q_pred),
                    ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

        elif training_mode == "bc" or training_mode == "pretrain":
            if training_mode != "pretrain" and self.use_rl_pg > 1:
                self.reinforce(obs, actions, discounted_rewards)
            else:
                self.mle_bc(obs, actions)
        else:
            raise ValueError("Invalid training mode")

    def critic_step(self, obs, actions, rewards, next_obs, terminals):
        with torch.no_grad():
            policy_outputs_ = self.policy(next_obs, return_log_prob=True)
            next_actions = policy_outputs_[0].detach()
            target_q_values = [qf(next_obs, next_actions) for qf in self.target_qfs]
            target_q_values = torch.cat(target_q_values, dim=-1)
            min_q_values = torch.min(target_q_values, dim=-1).values
            target_v_values = torch.unsqueeze(min_q_values, dim=-1)
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
        with torch.no_grad():
            _policy_outputs = self.policy(obs, return_log_prob=True)
            _actions = _policy_outputs[0].detach()
            _target_q_values = [qf(obs, _actions) for qf in self.target_qfs]
            _target_q_values = torch.cat(_target_q_values, dim=-1)
            _min_q_values = torch.min(_target_q_values, dim=-1).values
            _q_target = torch.unsqueeze(_min_q_values, dim=-1)

        qfs_pred = [qf(obs, actions) for qf in self.qfs]
        _qfs_pred = [qf(obs, _actions) for qf in self.qfs]

        if self.use_huber_loss:
            q_target_loss = [torch.mean(huber_loss(q_pred - q_target.detach())) for q_pred in qfs_pred]
            q_extra_loss = [
                self.log_lambda.exp() * torch.mean(
                    2 * huber_loss(_q_pred - _q_target.detach()) - self.target_thresh
                )
                for _q_pred in _qfs_pred
            ]
        else:
            q_target_loss = [0.5 * torch.mean((q_pred - q_target.detach()) ** 2) for q_pred in qfs_pred]
            q_extra_loss = [
                self.log_lambda.exp() * torch.mean(
                    (_q_pred - _q_target.detach()) ** 2 - self.target_thresh
                )
                for _q_pred in _qfs_pred
            ]

        q_target_loss_mean = 0.0
        q_extra_loss_mean = 0.0
        qfs_loss = []
        lambda_loss = ptu.zeros(1, requires_grad=True)
        for q_loss, _q_loss in zip(q_target_loss, q_extra_loss):
            q_target_loss_mean += np.mean(ptu.get_numpy(q_loss))
            q_extra_loss_mean += np.mean(ptu.get_numpy(_q_loss))
            qfs_loss.append(q_loss + _q_loss)
            lambda_loss = lambda_loss + _q_loss

        q_target_loss_mean /= len(qfs_loss)
        q_extra_loss_mean /= len(qfs_loss)
        q_loss_mean = q_target_loss_mean + q_extra_loss_mean
        lambda_loss = - lambda_loss / len(qfs_loss)

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward(retain_graph=True)
        # clip_gradient(self.lambda_optimizer)
        self.lambda_optimizer.step()
        self.log_lambda.data.clamp_(min=np.log(self.lambda_min), max=np.log(self.lambda_max))

        for i, opt in enumerate(self.qfs_optimizer):
            opt.zero_grad()
            qfs_loss[i].backward()
            clip_gradient(opt)
            opt.step()

        return q_loss_mean, q_target_loss_mean, q_extra_loss_mean, lambda_loss, qfs_pred

    def cql_critic_step(self, obs, actions, rewards, next_obs, terminals):
        q1_pred = self.qfs[0](obs, actions)
        q2_pred = self.qfs[-1](obs, actions)
        next_actions, _, _, next_log_pi = self.policy(next_obs, return_log_prob=True)
        _actions, _, _, _log_pi = self.policy(obs, return_log_prob=True)
        target_q = torch.min(self.target_qfs[0](next_obs, next_actions), self.target_qfs[-1](next_obs, next_actions))

    def actor_step(self, obs, actions):
        # Make sure policy accounts for squashing functions like tanh correctly!
        log_prob = self.policy.get_log_prob(obs, actions)
        mle_policy_loss = -1.0 * log_prob.mean()

        action_num = 10
        obs = obs.unsqueeze(1).repeat(1, action_num, 1).view(-1, obs.shape[1])

        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        qfs_new_actions = [qf(obs, new_actions) for qf in self.qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)

        policy_grad_loss = torch.mean(-qfs_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        bc_reg_loss = self.bc_reg_weight * mle_policy_loss
        policy_reg_loss = mean_reg_loss + std_reg_loss + bc_reg_loss
        policy_loss = policy_grad_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_gradient(self.policy_optimizer)
        self.policy_optimizer.step()

        return policy_loss, log_pi, policy_mean, policy_log_std

    def mle_bc(self, obs, acts):
        log_prob = self.policy.get_log_prob(obs, acts)
        policy_loss = -1.0 * log_prob.mean()

        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # squared_diff = (new_actions - acts) ** 2
        # policy_loss = torch.sum(squared_diff, dim=-1).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def reinforce(self, obs, acts, discounted_rewards):
        log_prob = self.policy.get_log_prob(obs, acts)
        policy_loss = (-log_prob * discounted_rewards).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_gradient(self.policy_optimizer)
        self.policy_optimizer.step()

    @property
    def networks(self):
        return [self.policy] + self.qfs + self.target_qfs

    def _update_target_network(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            ptu.soft_update_from_to(qf, target_qf, self.soft_target_tau)

    def get_snapshot(self):
        return dict(
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            policy=self.policy,
            log_lambda=self.log_lambda
        )

    def q_min(self, obs, actions):
        qfs_new_actions = [qf(obs, actions) for qf in self.qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)
        return qfs_new_actions

    def get_q_func(self):
        return self.q_min

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        print("END_EPOCH")
        self.eval_statistics = None


def clip_gradient(optimizer, grad_clip=0.5):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def huber_loss(x, delta=1):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * (x ** 2),
        delta * (torch.abs(x) - 0.5 * delta)
    )
