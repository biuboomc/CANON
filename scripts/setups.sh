
# loss_agg_mode="token-mean" # if GRPO/RLOO/R++/REMAX
# adv_estimator=rloo
# adv_estimator=grpo
# adv_estimator=reinforce_plus_plus
# adv_estimator=remax


loss_agg_mode="seq-mean-token-sum" # If dr_grpo and canon
adv_estimator=dr_grpo # baseline
adv_estimator=dr_random # r++
adv_estimator=dr_entropy_token_budget # based on entropy
adv_estimator=dr_length_on_mean # based on length
adv_estimator=dr_entropy_token_budget_annel # First-Inter-Later-Intra and First-Intra-Later-Inter
adv_estimator=dr_entropy_token_budget_cosine_restart # Cosin-First-Intra-Later-Inter
adv_estimator=dr_entropy_token_budget_cosine_restart_r # Cosin-First-Inter-Later-Intra


alpha=1.0 # alpha in Eq. 8
_lambda=1.0 # mu in Eq. 5
clip_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))

enable_filter_groups=False
train_prompt_bsz=512
train_prompt_mini_bsz=32
n_resp_per_prompt=16
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))


python -m verl.trainer.ppo.main_dapo \
    algorithm.adv_estimator=$adv_estimator \
    +algorithm.alpha=$alpha \
    +algorithm._lambda=$_lambda \
    data.train_files="" \
    data.val_files="" \
    data.gen_batch_size=${train_prompt_bsz} \
    +data.seed=42 \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation="left" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.clip_ratio_high=$clip_high \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +actor_rollout_ref.actor.max_response_length=$max_response_length \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger=["console","wandb","tensorboard"] \
    trainer.project_name="" \
    trainer.experiment_name="" \
    trainer.n_gpus_per_node=$PROC_PER_NODE \
    trainer.nnodes=$NODE_COUNT \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.default_local_dir="" \
    trainer.resume_mode=auto