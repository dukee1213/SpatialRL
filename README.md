# Spatial Multi-Agent Reinforcement Learning for Epidemic Control with Heterogeneous Risk Preference (code)

**(Paper title here)**  
**Authors:** (Your name, affiliation)

This repository contains the custom multi-agent reinforcement learning (MARL) environment and training scripts used in our study on epidemic-aware interaction and social utility optimization. The environment is implemented in Python on top of **agileRL 2.0.6** and **PettingZoo**, and the experiments correspond to the figures reported in the paper.

> **Note**: model checkpoints and large result spreadsheets are not included in this public repo. Please refer to the link below for pretrained models.

---

## Prerequisites (WSL2 + mini Conda)
First, set up WSL2 and a GPU-enabled PyTorch environment.

- WSL2 + GPU guide: [WSL2 GPU Setup](https://www.freecodecamp.org/news/how-to-setup-windows-machine-for-ml-dl-using-nvidia-graphics-card-cuda/)

- agileRL MADDPG tutorial (optional): [PettingZoo MADDPG tutorial](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/maddpg.html)

We assume you are inside WSL2 and have `conda` available.

## Necessary Setup

Create the env like this:
```bash
conda env create -f environment.yml
conda activate epi
```

Project layout
- `custom-environment/`
    - notebooks for training and ablations (`Train.ipynb`, `Train_As.ipynb`, etc.)
    - inference scripts (`Inference.py`, `Inference_As.py`)
    - `env/` — custom PettingZoo-like environment (`env_v1.py`, *rewards*, *disease cost*, *infection rate* infos.)
    - `models/` — local trained MADDPG checkpoints (not pushed)
    - `result/` — spreadsheets for figures

- `environment.yml`— compatible conda env deps.

Before you run the code:
```bash
cd custom-environment # project root
```

---

## Inference a Pretrained model
Basic CLI:
```bash
python3 Inference.py
```
Optionally specify a checkpoint:
```bash
python3 Inference.py --ckptpath models/MADDPG/MADDPG_trained_agent.pt
```
Pretrained [weights](https://drive.google.com/drive/folders/19BLmB27GP_XQCgeG8sFiW96_cyWIhMwJ?usp=sharing):
```bash
gdown --id 1mKkZu0Qe1PMNrO0D0Ni_eV15M2cRdlXq -O models/MADDPG/MADDPG_trained_agent.pt
```
If the pretrained model is no longer available, contact the author.

## Train your own models
Open and run `Train.ipynb`. Adjust *max_steps* as needed.
- We recommend ≥ 30k steps to stabilize training.
- On an RTX 4070, ~11 minutes for 10k steps (your mileage may vary).
- Run `Train_As.ipynb` for representative learning / parameter-space grid search (details in paper).

## Reproducing Figures
The RL engine is written in agileRL 2.0.6 and the environment in PettingZoo.
To inspect environment variables (rewards, disease cost, infection rate, timestep), read:
`custom-environment/env/env_v1.py`

To reproduce the paper’s figures:
- edit / run `Inference.py` or `Inference_As.py` (used for Fig. IV, Fig. V, etc.)
- or run the `Visualization.ipynb` using the data under `result/`

---
### Issues / future work
- Open an issue if you have questions.

- Please star the repo if you find it useful.

- Future: add Dockerfile, support for other multi-agent RL algorithms.