#!/bin/bash

# Example KTO training pipeline, perhaps easier to run each command in order rather than editing this file
# edit relevant variables

conda activate openrlhf

# modify task flag variable with one of [Chatting, Education, Therapy]
# other tasks currently still need to be supported in jsonl_gen.py
# takes in files in training_data/in and outputs training jsons in training_data/out
python jsonl_gen.py --task=Chatting

# specify gpu numbers to host on after localhost:
  # around 15 gb
    # directory with train.jsonl and test.jsonl
    # wandb key to monitor run stats
nohup deepspeed --include localhost:1,2 --master_port 61000 --module openrlhf.cli.train_kto \
   --save_path ./checkpoints/education/llama3-8b-kto-prompt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset json@./consistency_LLMs/training_data/out \
   --input_key in_text \
   --output_key out_text \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --use_wandb ... > kto_education.out & 

# education sft kto
nohup deepspeed --include localhost:1,2 --master_port 61000 --module openrlhf.cli.train_kto \
   --save_path ./checkpoints/education/llama3-8b-kto-sft-prompt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoints/education/llama3-8b-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset json@./consistency_LLMs/training_data/out \
   --input_key in_text \
   --output_key out_text \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --use_wandb ... > kto_sft_education.out & 

# Chatting sft kto
nohup deepspeed --include localhost:5,6 --master_port 61000 --module openrlhf.cli.train_kto \
   --save_path ./checkpoints/Chatting/llama3-8b-kto-sft-prompt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoints/Chatting/llama3-8b-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset json@./consistency_LLMs/training_data/out \
   --input_key in_text \
   --output_key out_text \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --use_wandb ... > kto_sft_chatting.out & 


nohup deepspeed --include localhost:4,5,6 --master_port 61000 --module openrlhf.cli.train_kto \
   --save_path ./checkpoints/therapy/llama3-8b-kto-sft-prompt \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 255 \
   --micro_train_batch_size 1 \
   --pretrain ./checkpoints/therapy/llama3-8b-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --dataset json@./consistency_LLMs/training_data/out \
   --input_key in_text \
   --output_key out_text \
   --label_key score \
   --flash_attn \
   --beta 0.1 \
   --gradient_checkpointing \
   --use_wandb ... > kto_sft_therapy.out & 