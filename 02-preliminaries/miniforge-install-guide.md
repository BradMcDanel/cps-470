# Miniforge Installation Guide

This guide covers installing Miniforge on your personal machine to match the lab environment.

## What is Miniforge?

Miniforge is a minimal conda installer that includes both `conda` and `mamba` commands. It uses conda-forge by default and avoids Anaconda licensing issues.

## Installation

### Download Miniforge

Visit: https://github.com/conda-forge/miniforge/releases/latest

Download the installer for your operating system:

**Linux/WSL:**
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
```

**macOS (Intel):**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh
```

**macOS (Apple Silicon):**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```

**Windows:**
Download and run: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe

### Configuration

After installation, initialize your shell:

```bash
mamba shell init --shell bash --root-prefix=~/miniforge3
```

Disable auto-activation of base environment:

```bash
conda config --set auto_activate_base false
```

Restart your terminal or run:

```bash
source ~/.bashrc
```

## Verify Installation

```bash
mamba --version
conda --version
```

## Create the CPS470 Course Environment

After installation, create the course environment:

```bash
mamba env create -f environment.yml
```

This creates a `cps470` environment with Python 3.11, PyTorch, NumPy, Matplotlib, and IPython. The environment will automatically detect and use GPUs when available (lab machines) or run on CPU (personal laptops).

**Activate the environment:**
```bash
conda activate cps470
```

**Verify installation:**
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Deactivate when done:**
```bash
conda deactivate
```

## Troubleshooting

**If you have an existing Anaconda/Miniconda installation:**

Your existing conda initialization in `~/.bashrc` may interfere. Either:
1. Remove the old conda initialization block from `~/.bashrc`, or
2. Uninstall the old conda/miniconda before installing Miniforge

**Verify you're using the correct installation:**
```bash
which conda
which mamba
```

These should point to `~/miniforge3/bin/` (or `~/miniforge3/condabin/`), not an older installation.
