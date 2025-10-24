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

## Measuring Consistency 
There are three notebooks: `chatting/chatting.ipynb`, `education/teaching.ipynb` and `therapy/patient-therapist.ipynb` that provide the pipeline to generate personas, dialogue, and measure consistency for the Chit-Chat, Teaching, and Mental Health Tasks.

## Training
Our multi-turn RL pipleine is based on OpenRLHF. To visit the original repo:  [GitHub Repo](https://github.com/OpenRLHF/OpenRLHF/tree/main).

Please find commands and hyper-parameters to run training in `rl_training`

