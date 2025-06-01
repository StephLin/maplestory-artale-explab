# MapleStory Artale ExpLab

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg?style=flat-square)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Formatter](https://img.shields.io/badge/Python_Formatter-ruff-black?style=flat-square)](https://github.com/astral-sh/ruff)

**English** | [中文](README.zh.md)

A cross-platform Python tool that monitors MapleStory Artale player experience efficiency, HP, and MP consumption using pure computer vision techniques, allowing real-time assessment of player training efficiency.

![Demo](docs/images/demo.png)
<small>Figure. Display experience efficiency, estimated time to level up, and HP/MP consumption.</small>

## Features

- Based on computer vision technology, not involving script injection/tampering with game data.
- Cross-platform support (Windows, macOS).
- Almost no configuration (so far).

## Installation

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies and the execution environment.

1. Install uv (if not already installed):

```bash
pip install uv
```

2. Sync project dependencies:

```bash
uv sync
```

3. (Windows w/ Nvidia GPU, Optional) Install CUDA-based PyTorch:

Please follow the [PyTorch official website](https://pytorch.org/get-started/locally/) to install CUDA-based PyTorch.

```bash
# An example to install CUDA 12.8-based PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Another example to install CUDA 12.6-based PyTorch
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Usage

```bash
uv python main.py
```

## Contributing

Due to the author's busy full-time job, it may not be possible to respond to all requests immediately. However, please feel free to submit issues for discussions or pull requests for contributions.

## License

- In principle, follows the [BSD-3 License](./LICENSE).
- However, using this project for activities that may affect other players' gaming experience (e.g., bot farming) is not permitted.

## Declaration

This project uses [GitHub Copilot](https://github.com/features/copilot) (mostly with Gemini 2.5 Pro) to generate some of the code. The main implementation has been reviewed manually, but errors or improper implementations may still exist. If you find any issues, please feel free to submit an issue or pull request.

## TODO

See [#1](https://github.com/StephLin/maplestory-artale-explab/issues/1) for details.
