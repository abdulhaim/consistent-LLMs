# Consistent-LLMs

<div align="left" style="line-height: 1;">
  <a href="" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2506.07468-b31b1b?logo=arxiv&logoColor=white"/>
  </a>
  <a href="https://github.com/abdulhaim/consistent-LLMs" target="_blank">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-consistent--LLMs-181717?logo=github"/>
  </a>
</div>

This repository accompanies our NeurIPS 2025 paper:  
**[Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning]()**  
by *Marwa Abdulhai, Ryan Cheng, Donovan Clay, Tim Althoff, Sergey Levine, Natasha Jaques.*

---

## Overview

**Consistent-LLMs** introduces a reinforcement learning framework to **measure and improve consistency** in large language models (LLMs).  
Our work identifies and addresses the challenge of *drift* — when LLMs lose track of a user’s identity, beliefs, or prior behavior over time.

We propose three complementary **consistency metrics**:

- **Prompt-to-Line Consistency** – measures faithfulness to the initial persona and prompt context.  
- **Line-to-Line Consistency** – quantifies coherence and continuity across dialogue turns.  
- **Q&A Consistency** – assesses stability of beliefs, self-identity, and memory across sessions.

These metrics are used as reward signals in multi-turn reinforcement learning, leading to LLMs that simulate more stable, human-like personas across different domains (education, mental health, conversation).

---

## Quick Start

### Installation

We recommend setting up a clean conda environment:

```bash
git clone https://github.com/abdulhaim/consistent-LLMs
cd consistent-LLMs
conda create --name consistent python=3.10
conda activate consistent
pip install -e .
pip install openrlhf[vllm]
```

### Measuring Consistency 
There are three notebooks: `chatting/chatting.ipynb`, `education/teaching.ipynb` and `therapy/patient-therapist.ipynb` that provide the pipeline to generate personas, dialogue, and measure consistency for the Chit-Chat, Teaching, and Mental Health Tasks.

### Training
Our multi-turn RL pipleine is based on OpenRLHF. To visit the original repo:  [GitHub Repo](https://github.com/OpenRLHF/OpenRLHF/tree/main).

Please find commands and hyper-parameters to run training in `rl_training`.

1. Either run ```python jsonl_gen.py --task=<task>``` for ```<task>``` ```Chatting```, ```Education```, or ```Therapy``` (does not matter the first time it's run) to create empty ```training_data/in``` and ```training_data/out``` folders within the ```rl_training``` directory, or create these folders manually.
2. Place conversation jsons of the conversations you would like to include in your training data in ```training_data/in```.
3. Run ```python jsonl_gen.py --task=<task>``` for ```<task>``` ```Chatting```, ```Education```, ```Therapy``` to pair the specific scenario prompts with the dialogues and conglomerate conversation data into the ```jsonl``` format needed by OpenRLHF. Local paths may need to be replaced by absolute paths in all of the following training scripts. 
4. (Optional) Train SFT with one of the scripts in ```example_sft.sh```. 
5. Train KTO with one of the scripts in ```example_kto.sh```. 
6. To train PPO, first start a vLLM instance on a separate GPU than is planned for PPO training. 

    a. ```reward_func_prompt.py``` has hyperparameters near the top of the file that may need to be changed depending on how the vLLM instance was started (e.g. port, model name). The ```chat_completion``` call can also be replaced by a call to an online hosted model if you do not wish to locally host a separate GPU instance for training.
    
    b.

