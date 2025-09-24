# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict, Counter

import verl.utils.torch_functional as verl_F
from verl.utils.pattern import catch_rethink_patterns


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

import torch

def find_optimal_split(tensor: torch.Tensor) -> int:
    """
    找到一个分割点，使得两组均值的差的绝对值最大。

    Args:
        tensor: 一个已排序的 PyTorch Tensor。

    Returns:
        最优分割点 n。
    """
    N = tensor.size(0)
    if N <= 1:
        return 0

    overall_mean = tensor.mean()
    max_diff = 0  # Initialize to a negative value
    optimal_n = N // 2

    if overall_mean == 0 or overall_mean == 1:
        return optimal_n, max_diff

    # for n in range(1, N):
    group1 = tensor[:optimal_n]
    group2 = tensor[optimal_n:]

    mean1 = group1.mean() if optimal_n > 0 else torch.tensor(0.0)  # Empty group 1
    mean2 = group2.mean() if optimal_n < N else torch.tensor(0.0)  # Empty group 2

    diff = torch.abs(mean1 - mean2)

        # if diff > max_diff:
        #     max_diff = diff
        #     optimal_n = n

    return optimal_n, max_diff


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def compute_dr_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6,
                                   max_length=8192):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / max_length
        scores = scores.unsqueeze(-1) * response_mask 

    return scores, scores


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor,
                                                           response_mask: torch.Tensor,
                                                           index: torch.Tensor,
                                                           epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask)

    return scores, scores


def compute_dr_length_on_mean_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   alpha: float = 1.0,
                                   beta: float = 0.0,
                                   _lambda: float = 1.0,
                                   epsilon: float = 1e-6,
                                   max_length=8192):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(response_length[i])
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

            
        for i in range(bsz):
            if l_id2ls[i]:
                scores[i] = ((_lambda * (alpha * scores[i] - id2mean_short[index[i]] + beta) + (1 - _lambda)* (scores[i] - id2mean_long[index[i]]))/ max_length)
            else:
                scores[i] = ((_lambda * (scores[i] - alpha * id2mean_long[index[i]] - beta) + (1 - _lambda) * (scores[i] - id2mean_short[index[i]])) / max_length)
            # if l_id2ls[i]:
            #     scores[i] = (alpha * scores[i] - id2mean_short[index[i]]) / max_length
            # else:
            #     scores[i] = (scores[i] - alpha * id2mean_long[index[i]]) / max_length

        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores


def compute_dr_entropy_per_token_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   entropy: torch.Tensor,
                                   index: torch.Tensor,
                                   alpha: float = 1.0,
                                   beta: float = 0.0,
                                   _lambda: float = 1.0,
                                   epsilon: float = 1e-6,
                                   max_length=8192
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)
    entropy = verl_F.masked_mean(entropy, response_mask, axis=-1)
    # entropy = entropy.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(entropy[i])
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]


        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # if l_id2ls[i]:
            #     scores[i] = ((scores[i] - alpha * id2mean_short[index[i]] + beta) / max_length)
            # else:
            #     scores[i] = ((alpha * scores[i] - id2mean_long[index[i]] - beta) / max_length)
            if l_id2ls[i]:
                scores[i] = ((_lambda * (scores[i] - alpha * id2mean_short[index[i]] + beta) + (1 - _lambda)* (scores[i] - id2mean_long[index[i]]))/ max_length)
            else:
                scores[i] = ((_lambda * (alpha * scores[i] - id2mean_long[index[i]] - beta) + (1 - _lambda) * (scores[i] - id2mean_short[index[i]])) / max_length)
        scores = scores.unsqueeze(-1) * response_mask


        
    return scores, scores

def cosine_annealing(epoch, total_epochs, max_lr=1e-3, min_lr=1e-5, warmup_epochs=5):
    """
    标准余弦退火学习率调度函数
    """
    if epoch < warmup_epochs:
        # 预热阶段：线性增加学习率
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # 余弦退火阶段
        epoch -= warmup_epochs
        total_epochs -= warmup_epochs
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs))

def cosine_annealing_with_restart(epoch, total_epochs, max_lr=1e-3, min_lr=1e-5, 
                                  warmup_epochs=5, cycles=3):
    """
    带重启的余弦退火学习率调度函数
    """
    if epoch < warmup_epochs:
        # 预热阶段：线性增加学习率
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        # 余弦退火阶段（带重启）
        epoch -= warmup_epochs
        total_epochs -= warmup_epochs
        cycle_length = total_epochs // cycles
        cycle = epoch // cycle_length
        cycle_epoch = epoch % cycle_length
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * cycle_epoch / cycle_length))

def compute_dr_entropy_per_token_cosine_with_restart_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   entropy: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   max_length=8192,
                                   step=0
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)
    entropy = verl_F.masked_mean(entropy, response_mask, axis=-1)
    # entropy = entropy.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}
    total_epochs=150
    max_lr=0.6
    min_lr=0
    warmup_epochs=30
    
    mean_of_all = cosine_annealing_with_restart(step, total_epochs=total_epochs, max_lr=max_lr, min_lr=min_lr, warmup_epochs=warmup_epochs)
    # init =  {'q7':0.25818, 'q1':0.08667}
    # init = 0.25818
    # mean_of_all = torch.clamp(((torch.mean(scores) - init) / (0.5 - init)) / 2,0,1) * 0.5 + 0.5 * (torch.mean(scores) - init) / (1.0 - init)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(entropy[i])
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]


        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):


            if l_id2ls[i]:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_short[index[i]]) + mean_of_all * (scores[i] - id2mean_long[index[i]]))/ max_length)
            else:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_long[index[i]]) + mean_of_all * (scores[i] - id2mean_short[index[i]])) / max_length)
        scores = scores.unsqueeze(-1) * response_mask


        
    return scores, scores


def compute_dr_entropy_per_token_cosine_with_restart_r_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   entropy: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   max_length=8192,
                                   step=0
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)
    entropy = verl_F.masked_mean(entropy, response_mask, axis=-1)
    # entropy = entropy.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}
    total_epochs=150
    max_lr=1
    min_lr=0.4
    warmup_epochs=30
    
    mean_of_all = 1 - cosine_annealing_with_restart(step, total_epochs=total_epochs, max_lr=max_lr, min_lr=min_lr, warmup_epochs=warmup_epochs)
    # init =  {'q7':0.25818, 'q1':0.08667}
    # init = 0.25818
    # mean_of_all = torch.clamp(((torch.mean(scores) - init) / (0.5 - init)) / 2,0,1) * 0.5 + 0.5 * (torch.mean(scores) - init) / (1.0 - init)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(entropy[i])
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]


        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):


            if l_id2ls[i]:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_short[index[i]]) + mean_of_all * (scores[i] - id2mean_long[index[i]]))/ max_length)
            else:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_long[index[i]]) + mean_of_all * (scores[i] - id2mean_short[index[i]])) / max_length)
        scores = scores.unsqueeze(-1) * response_mask


        
    return scores, scores


def compute_dr_entropy_per_token_annel_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   entropy: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   max_length=8192,
                                   step=0
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)
    entropy = verl_F.masked_mean(entropy, response_mask, axis=-1)
    # entropy = entropy.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}
    # mean_of_all = torch.mean(scores)
    mean_of_all = 1 - torch.mean(scores)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(entropy[i])
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]


        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):


            if l_id2ls[i]:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_short[index[i]]) + mean_of_all * (scores[i] - id2mean_long[index[i]]))/ max_length)
            else:
                scores[i] = (((1 - mean_of_all) * (scores[i] - id2mean_long[index[i]]) + mean_of_all * (scores[i] - id2mean_short[index[i]])) / max_length)
        scores = scores.unsqueeze(-1) * response_mask


        
    return scores, scores



def compute_dr_random_group_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   entropy: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   max_length=8192
                                   ):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    # response_length = response_mask.sum(-1).float()
    scores = token_level_rewards.sum(dim=-1)
    # entropy = verl_F.masked_sum(entropy, response_mask, axis=-1)
    # entropy = entropy.sum(dim=-1)

    id2score = defaultdict(list)
    id2length = defaultdict(list)
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)

    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2length[index[i]].append(i)
            id2score[index[i]].append(scores[i])
            id2l_id[index[i]].append(i)
        
        for idx in id2length:
            paired = sorted(zip(id2length[idx], id2score[idx], id2l_id[idx]), key=lambda x: x[0])
            id2score_sort = [score for (time, score, id) in paired]
            id2score_id = [id for (time, score, id) in paired] 

            n = len(paired)
            mid = n // 2
            # mid, diff = find_optimal_split(torch.tensor(id2score_sort))
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i >= mid
            # if n % 2 == 0:
            id2score_short[idx] = id2score_sort[:mid]
            id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if l_id2ls[i]:
                scores[i] = ((scores[i] - id2mean_short[index[i]]) / max_length)
            else:
                scores[i] = ((scores[i] - id2mean_long[index[i]]) / max_length)
        scores = scores.unsqueeze(-1) * response_mask
        
    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: np.ndarray,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.
    Args:
        loss_mat: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=None,
                        cliprange_low=None,
                        cliprange_high=None,
                        clip_ratio_c=3.0,
                        loss_agg_mode="token-mean"):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        clip_ratio_c: (float) default: 3.0
            The lower bound of the ratio for dual-clip PPO, See https://arxiv.org/pdf/1912.09729
        loss_agg_mode: (str) choices: "token-mean" / "seq-mean-token-sum" / "seq-mean-token-mean"
            "token-mean" is the default behavior        

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
        pg_clipfrac_lower: (float)
            the fraction of policy gradient loss being clipped when the advantage is negative
    """
    assert clip_ratio_c > 1.0, f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low,
                                           1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1,
                                    pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses2, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=response_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), response_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
