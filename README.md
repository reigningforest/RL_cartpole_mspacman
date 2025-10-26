# Reinforcement_learning_games

A small collection of simple reinforcement learning examples (Pong, CartPole) using Gymnasium and PyTorch.

## Requirements

- Python: 3.10 or later
- Package manager: conda or pip
- CUDA (for GPU): CUDA-enabled GPU with matching CUDA/cuDNN drivers for your PyTorch build
- **IMPORTANT**: If no GPU is found the scripts will NOT run.


## Installation

1. Create and activate a Python environment (conda example):

```bash
conda create -n rl_1 python=3.10 -y
conda activate rl_1
```

(Or using venv):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python requirements:

```bash
pip install -r requirements.txt
```

Note about Gymnasium extras and AutoROM:
- To use Atari environments you must install the `atari` extra for Gymnasium and install Atari ROMs. Gymnasium provides extras like `atari` and `classic-control` but does not provide an `accept-rom-license` extra. The ROM license/automatic download extra is provided by the AutoROM project. Install both as shown below.

## Atari ROM setup (step-by-step)

1. Install AutoROM (accepts ROM license via extra) and run it to download ROMs:

```bash
pip install "autorom[accept-rom-license]"
# or if autorom already installed, run:
AutoROM --accept-license
```

2. Verify ROMs are available (optional):

```bash
python -c "import ale_py.roms; print('Available ROMs:', len(ale_py.roms.get_all_rom_ids()))"
```

If you see Atari ROM ids (e.g. `'pong'`) then ALE is ready.

## GPU usage notes

- The scripts detect CUDA via `torch.cuda.is_available()` and move models/tensors to the selected device.
- How to choose GPU_ID: if you have multiple GPUs, set `GPU_ID = N` in the script (0-based index) or export `CUDA_VISIBLE_DEVICES` before running. Example:

```bash
# Use GPU 0 (if present) by environment variable
export CUDA_VISIBLE_DEVICES=0
python cartpole_v1_dqn.py
```

Tips:
- Ensure your PyTorch build matches your CUDA driver version. Use `torch.cuda.get_device_name(0)` to confirm device visibility.

## File structure (important files)

- `cartpole_v1_dqn.py` — CartPole training script
- `mspacman_v0_dqn.py` — Ms. Pacman training script
- `requirements.txt` — Python dependencies
- `plots/` — folder where training plots are saved

## How to run

Run each script in the background and log output using the nohup pattern used in this repo:

CartPole DQN:

```bash
nohup python -u cartpole_v1_dqn.py > cartpole_dqn_output.log 2>&1 &
```

Ms. Pacman DQN:

```bash
nohup python -u mspacman_v0_dqn.py > mspacman_dqn_output.log 2>&1 &
```

(Use `tail -f output.log` or the specific log files to watch progress.)