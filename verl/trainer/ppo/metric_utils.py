# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Metrics related to the PPO trainer.
"""

import torch
from typing import Any, Dict, List, Callable
import numpy as np
from verl import DataProto
from collections import Counter, defaultdict
import verl.utils.torch_functional as verl_F
from functools import partial


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def cal_freq(id2sum, samplenum):
    # 初始化一个长度为samplenum+1的列表，用于记录成功次数的频次
    successcounts = [0] * (samplenum + 1)
    
    # 遍历字典，统计每个成功次数的频次
    for success in id2sum.values():
        success = int(success)
        if success <= samplenum:
            successcounts[success] += 1
    
    # 计算频率，即频次除以总的关卡数
    totallevels = len(id2sum)
    successfrequencies = [count / totallevels for count in successcounts]
    
    return successfrequencies


def compute_level_percent(token_level_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   index: torch.Tensor):
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
    scores = token_level_rewards
    response_length = response_mask.sum(-1).float()

    id2score = defaultdict(list)
    id2length = defaultdict(list)

    id2sum = {}
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)
    id2mean = {}
    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}

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
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i > mid
            if n % 2 == 0:
                id2score_short[idx] = id2score_sort[:mid]
                id2score_long[idx] = id2score_sort[mid:]
            else:
                id2score_short[idx] = id2score_sort[:mid+1]
                id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)

                id2sum[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = (id2mean_short[idx] + id2mean_long[idx])/2

                if -1 in token_level_rewards:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx])) + n) / 2)
                else:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx]))))

            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
        values_tensor = torch.stack(list(id2mean_long.values()))
        mean_of_long = torch.mean(values_tensor)

        values_tensor = torch.stack(list(id2mean_short.values()))
        mean_of_short = torch.mean(values_tensor)
        freq = cal_freq(id2sum, n)
        
        
    return mean_of_long, mean_of_short, freq


def compute_entropy(token_level_rewards: torch.Tensor,
                    entropy: torch.Tensor,
                    response_mask: torch.Tensor,
                    index: torch.Tensor):
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
    scores = token_level_rewards
    entropy = verl_F.masked_sum(entropy, response_mask, axis=-1)
    response_length = entropy

    id2score = defaultdict(list)
    id2length = defaultdict(list)

    id2sum = {}
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)
    id2mean = {}
    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}

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
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i > mid
            if n % 2 == 0:
                id2score_short[idx] = id2score_sort[:mid]
                id2score_long[idx] = id2score_sort[mid:]
            else:
                id2score_short[idx] = id2score_sort[:mid+1]
                id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)

                id2sum[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = (id2mean_short[idx] + id2mean_long[idx])/2

                if -1 in token_level_rewards:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx])) + n) / 2)
                else:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx]))))

            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
        values_tensor = torch.stack(list(id2mean_long.values()))
        mean_of_long = torch.mean(values_tensor)

        values_tensor = torch.stack(list(id2mean_short.values()))
        mean_of_short = torch.mean(values_tensor)
        
        
    return mean_of_long, mean_of_short


def compute_entropy_per_token(token_level_rewards: torch.Tensor,
                    entropy: torch.Tensor,
                    response_mask: torch.Tensor,
                    index: torch.Tensor):
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
    scores = token_level_rewards
    entropy = verl_F.masked_mean(entropy, response_mask, axis=-1)
    response_length = entropy

    id2score = defaultdict(list)
    id2length = defaultdict(list)

    id2sum = {}
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)
    id2mean = {}
    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}

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
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i > mid
            if n % 2 == 0:
                id2score_short[idx] = id2score_sort[:mid]
                id2score_long[idx] = id2score_sort[mid:]
            else:
                id2score_short[idx] = id2score_sort[:mid+1]
                id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)

                id2sum[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = (id2mean_short[idx] + id2mean_long[idx])/2

                if -1 in token_level_rewards:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx])) + n) / 2)
                else:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx]))))

            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
        values_tensor = torch.stack(list(id2mean_long.values()))
        mean_of_long = torch.mean(values_tensor)

        values_tensor = torch.stack(list(id2mean_short.values()))
        mean_of_short = torch.mean(values_tensor)
        
        
    return mean_of_long, mean_of_short

def compute_rethink_pattern(token_level_rewards: torch.Tensor,
                    pattern_count: torch.Tensor,
                    response_mask: torch.Tensor,
                    index: torch.Tensor):
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
    scores = token_level_rewards
    response_length = pattern_count

    id2score = defaultdict(list)
    id2length = defaultdict(list)

    id2sum = {}
    l_id2ls= defaultdict(list)
    id2l_id = defaultdict(list)
    id2mean = {}
    id2score_long = defaultdict(list)
    id2score_short = defaultdict(list)
    id2mean_long = {}
    id2mean_short = {}

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
            for i, item in enumerate(id2score_id):
                l_id2ls[item] = i > mid
            if n % 2 == 0:
                id2score_short[idx] = id2score_sort[:mid]
                id2score_long[idx] = id2score_sort[mid:]
            else:
                id2score_short[idx] = id2score_sort[:mid+1]
                id2score_long[idx] = id2score_sort[mid:]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean_long[idx] = torch.tensor(0.0)
                id2mean_short[idx] = torch.tensor(0.0)
                id2mean[idx] = torch.tensor(0.0)

                id2sum[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean_long[idx] = torch.mean(torch.tensor(id2score_long[idx]))
                id2mean_short[idx] = torch.mean(torch.tensor(id2score_short[idx]))
                id2mean[idx] = (id2mean_short[idx] + id2mean_long[idx])/2

                if -1 in token_level_rewards:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx])) + n) / 2)
                else:
                    id2sum[idx] = int((torch.sum(torch.tensor(id2score[idx]))))

            else:
                raise ValueError(f"no score in prompt index: {idx}")
            
        values_tensor = torch.stack(list(id2mean_long.values()))
        mean_of_long = torch.mean(values_tensor)

        values_tensor = torch.stack(list(id2mean_short.values()))
        mean_of_short = torch.mean(values_tensor)
        
        
    return mean_of_long, mean_of_short

def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']


    verify = batch.batch['verify']
    backtrack = batch.batch['backtrack']
    subgoal = batch.batch['subgoal']
    total_rethink = verify.clone() + backtrack.clone() + subgoal.clone()



    max_response_length = batch.batch['responses'].shape[-1]
    entropy = batch.batch['entropy']
    log_probs = batch.batch['old_log_probs']

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']


    verify_per_token = np.mean([verify[i] / response_length[i] for i in range(len(verify))])
    backtrack_per_token = np.mean([backtrack[i] / response_length[i] for i in range(len(backtrack))])
    subgoal_per_token = np.mean([subgoal[i] / response_length[i] for i in range(len(subgoal))])
    total_rethink_per_token = np.mean([total_rethink[i] / response_length[i] for i in range(len(total_rethink))])

    indexs = batch.non_tensor_batch['uid']
    mean_long, mean_short, freq_lst = compute_level_percent(sequence_score, response_mask, indexs)
    mean_long_entropy, mean_short_entropy = compute_entropy(sequence_score, entropy, response_mask, indexs)
    mean_long_entropy_per_token, mean_short_entropy_per_token = compute_entropy_per_token(sequence_score, entropy, response_mask, indexs)
    mean_long_logprob, mean_short_logprob = compute_entropy(sequence_score, log_probs, response_mask, indexs)
    more_verify, less_verify = compute_rethink_pattern(sequence_score, verify, response_mask, indexs)
    more_subgoal, less_subgoal = compute_rethink_pattern(sequence_score, subgoal, response_mask, indexs)
    more_backtrack, less_backtrack = compute_rethink_pattern(sequence_score, backtrack, response_mask, indexs)
    more_total_rethink, less_total_rethink = compute_rethink_pattern(sequence_score, total_rethink, response_mask, indexs)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)
    # for i in range(bsz):


    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/mean_long':
            mean_long.detach().item(),
        'critic/score/mean_short':
            mean_short.detach().item(),
        'critic/score/mean_long_entropy':
            mean_long_entropy.detach().item(),
        'critic/score/mean_short_entropy':
            mean_short_entropy.detach().item(),
        'critic/score/mean_long_entropy_per_token':
            mean_long_entropy_per_token.detach().item(),
        'critic/score/mean_short_entropy_per_token':
            mean_short_entropy_per_token.detach().item(),
        'critic/score/mean_long_logprob':
            mean_long_logprob.detach().item(),
        'critic/score/mean_short_logprob':
            mean_short_logprob.detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        'rethink_core/verification/mean': torch.mean(verify).detach().item(),
        'rethink_core/backtrack/mean': torch.mean(backtrack).detach().item(),
        'rethink_core/subgoal_setting/mean': torch.mean(subgoal).detach().item(),
        'rethink_core/total_rethink/mean': torch.mean(total_rethink).detach().item(),

        'rethink_core/verification/token_mean': verify_per_token,
        'rethink_core/backtrack/token_mean': backtrack_per_token,
        'rethink_core/subgoal_setting/token_mean': subgoal_per_token,
        'rethink_core/total_rethink/token_mean': total_rethink_per_token,

        'rethink/train/verification/mean': torch.mean(verify).detach().item(),
        'rethink/train/verification/min': torch.min(verify).detach().item(),
        'rethink/train/verification/max': torch.max(verify).detach().item(),
        'rethink/train/verification/score_more': more_verify.detach().item(),
        'rethink/train/verification/score_less': less_verify.detach().item(),
        'rethink/train/backtrack/mean': torch.mean(backtrack).detach().item(),
        'rethink/train/backtrack/min': torch.min(backtrack).detach().item(),
        'rethink/train/backtrack/max': torch.max(backtrack).detach().item(),
        'rethink/train/backtrack/score_more': more_backtrack.detach().item(),
        'rethink/train/backtrack/score_less': less_backtrack.detach().item(),
        'rethink/train/subgoal_setting/mean': torch.mean(subgoal).detach().item(),
        'rethink/train/subgoal_setting/min': torch.min(subgoal).detach().item(),
        'rethink/train/subgoal_setting/max': torch.max(subgoal).detach().item(),
        'rethink/train/subgoal_setting/score_more': more_subgoal.detach().item(),
        'rethink/train/subgoal_setting/score_less': less_subgoal.detach().item(),
        'rethink/train/total_rethink/mean': torch.mean(total_rethink).detach().item(),
        'rethink/train/total_rethink/min': torch.min(total_rethink).detach().item(),
        'rethink/train/total_rethink/max': torch.max(total_rethink).detach().item(),
        'rethink/train/total_rethink/score_more': more_total_rethink.detach().item(),
        'rethink/train/total_rethink/score_less': less_total_rethink.detach().item(),
        
    }
    
    for i, freq in enumerate(freq_lst):
        metrics[f"hardness/level_{i}"] = freq
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw['step']
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        'perf/total_num_tokens': total_num_tokens,
        'perf/time_per_step': time,
        'perf/throughput': total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(data: list[Any],
                     subset_size: int,
                     reduce_fns: list[Callable[[np.ndarray], float]],
                     n_bootstrap: int = 1000,
                     seed: int = 42) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    # print(bootstrap_metric_lsts)
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str],
                               sample_inputs: list[str],
                               infos_dict: dict[str, list[Any]],
                               seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.
    
    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample
        
    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    (bon_mean, bon_std), (won_mean, won_std) = bootstrap_metric(data=var_vals,
                                                                                subset_size=n,
                                                                                reduce_fns=[np.max, np.min],
                                                                                seed=seed)
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        lst = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                            seed=seed)
                        maj_n_mean, maj_n_std = lst[0]
                        # print(lst)
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
