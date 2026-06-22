#!/usr/bin/env python3
"""Run LeRobot training with a WSL-friendlier PyTorch sharing strategy."""

from __future__ import annotations

import sys

import torch.multiprocessing as mp

from lerobot.scripts.lerobot_train import main


if __name__ == "__main__":
    mp.set_sharing_strategy("file_system")
    raise SystemExit(main())
