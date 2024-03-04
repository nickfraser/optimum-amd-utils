
import pytest

import numpy as np

import torch

single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="At least one GPU required for the GPU tests"
)


multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Multiple GPUs required for the multi-GPU tests"
)


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")


def allclose(x, y):
    return np.allclose(x, y, rtol=1e-02, atol=1e-02, equal_nan=False)

def allexact(x, y):
    return np.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=False)
