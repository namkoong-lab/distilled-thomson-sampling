#!/usr/bin/env python3
import torch


# pyre-ignore [19]
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
DTYPE: torch.dtype = torch.float
