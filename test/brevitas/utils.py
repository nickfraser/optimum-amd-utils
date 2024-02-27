
import pytest

import torch

single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="At least one GPU required for the GPU tests"
)


multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Multiple GPUs required for the multi-GPU tests"
)


def ptid2pathname(string):
    return string.replace("/", "-").replace(":", "-")
