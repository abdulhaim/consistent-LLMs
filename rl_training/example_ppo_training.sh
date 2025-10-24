#!/bin/bash

# Example PPO training pipeline, perhaps easier to run each command in order rather than editing this file
# edit relevant variables

conda activate openrlhf

# modify task flag variable with one of [Chatting, Education, Therapy]
# other tasks currently still need to be supported in jsonl_gen.py
# takes in files in training_data/in and outputs training jsons in training_data/out
python jsonl_gen.py --task=Chatting

# start a vllm server for calculating the reward
# edit port as needed, SHOULD BE THE SAME AS IN THE REWARD FUNCTION SCRIPT!
# edit CUDA_VISIBLE_DEVICES to choose GPUs to host on (PPO requires at least one other separate one)
CUDA_VISIBLE_DEVICES=5,6 nohup vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --port=8001 --tensor-parallel-size=2 --download_dir=./models > llama_reward_server.out &

# start a ray session to host PPO model training
# all models can be put on a single H200 gpu, might need more for H100 or A100
# dashboard port must match --address in below command, other ports can vary if necessary
CUDA_VISIBLE_DEVICES=4 ray start --head --node-ip-address 0.0.0.0 --dashboard-port=8270 --port=6383  --dashboard-agent-listen-port=52367 --num-gpus 1 --temp-dir=./tmp

 # working_dir might need to exist/have enough space but i'm not sure
# keeps all of the above models on the same gpu as opposed to gpus_per_node * num_node gpus per model
# use a custom reward
# pretrained model, can be specified as an actual file directory to saved checkpoints
# path to a py file with reward_func defined to use in calculating rewards
# checkpoints are ~15 gb
# directory with train.jsonl and test.jsonl
# save checkpoints every 10 steps. NOTE: i've run into model crashes in saving checkpoints so maybe remove if that is an issue
# max number of checkpoints to keep at a time (each quite large, 20-30 gb i think) (remove if not saving)
# remove if not saving
 # wandb key to monitor run stats

# 2 gpus, 1 for actor + ref, 1 for critic + vLLM
nohup ray job submit --address="http://127.0.0.1:8270" \
    --runtime-env-json='{"working_dir": "./openrlhf"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --colocate_all_models \
    --ref_reward_offload \
    --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
    --remote_rm_url ./reward_func_prompt.py \
    --save_path ./llama-8b-ppo-prompt \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data json@./training_data/out \
    --input_key in_text \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --max_ckpt_num 3 \
    --ckpt_path ./checkpoints/education/checkpoints/llama-8b-ppo-prompt \
    --save_hf_ckpt \
    --use_wandb ... > ppo_education.out &



# 1 H200 GPU
nohup ray job submit --address="http://127.0.0.1:8270" \
    --runtime-env-json='{"working_dir": "./openrlhf"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --colocate_all_models \
    --ref_reward_offload \
    --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
    --remote_rm_url ./reward_func_prompt.py \
    --save_path ./checkpoints/Chatting/llama-8b-ppo-prompt2 \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data json@./training_data/out \
    --input_key in_text \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --max_ckpt_num 3 \
    --ckpt_path ./checkpoints/Chatting/checkpoints/llama-8b-ppo-prompt2 \
    --save_hf_ckpt \
    --use_wandb ... > ppo.out &


# ppo
nohup ray job submit --address="http://127.0.0.1:8270" \
    --runtime-env-json='{"working_dir": "./openrlhf"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --colocate_all_models \
    --ref_reward_offload \
    --pretrain ./checkpoints/Chatting/llama3-8b-sft \
    --remote_rm_url ./reward_func_prompt.py \
    --save_path ./checkpoints/Chatting/llama-8b-sft-ppo-prompt \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data json@./training_data/out \
    --input_key in_text \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --max_ckpt_num 3 \
    --ckpt_path ./checkpoints/Chatting/checkpoints/llama-8b-sft-ppo-prompt \
    --save_hf_ckpt \
    --use_wandb ... > ppo_sft_chatting.out &

# sft therapy
nohup ray job submit --address="http://127.0.0.1:8270" \
    --runtime-env-json='{"working_dir": "./openrlhf"}' \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --colocate_all_models \
    --ref_reward_offload \
    --pretrain ./checkpoints/therapy/llama3-8b-sft \
    --remote_rm_url ./reward_func_prompt.py \
    --save_path ./checkpoints/therapy/llama-8b-sft-ppo-prompt \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --max_samples 100000 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --packing_samples \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data json@./training_data/out \
    --input_key in_text \
    --normalize_reward \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 10 \
    --max_ckpt_num 3 \
    --ckpt_path ./checkpoints/therapy/checkpoints/llama-8b-sft-ppo-prompt \
    --save_hf_ckpt \
    --use_wandb ... > ppo_sft_therapy.out &