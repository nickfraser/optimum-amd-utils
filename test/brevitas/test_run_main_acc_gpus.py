
import pytest

import torch

multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Multiple GPUs required for the multi-GPU tests"
)

@pytest.fixture()
def toggle_run_args_acc_gpus(toggle_run_args):
    args = toggle_run_args
    args.device = "auto"
    args.gpu_device_map = {i: 5.0e8 for i in range(torch.cuda.device_count())}
    return args


@pytest.fixture()
def recommended_run_args_acc_gpus(recommended_run_args):
    args = recommended_run_args
    args.device = "auto"
    args.gpu_device_map = {i: 5.0e8 for i in range(torch.cuda.device_count())}
    return args


@pytest.fixture()
def all_run_args_acc_gpus(all_run_args):
    args = all_run_args
    args.device = "auto"
    args.gpu_device_map = {i: 5.0e8 for i in range(torch.cuda.device_count())}
    return args


@pytest.mark.run
@pytest.mark.acc_gpus
@pytest.mark.opt
@pytest.mark.short
@multigpu
def test_toggle_opt_acc_gpus(run_main_test, toggle_run_args_acc_gpus):
    args = toggle_run_args_acc_gpus
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_gpus
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@multigpu
def test_recommended_opt_acc_gpus(run_main_test, recommended_run_args_acc_gpus):
    args = recommended_run_args_acc_gpus
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_gpus
@pytest.mark.opt
@pytest.mark.long
@multigpu
def test_all_opt_acc_gpus(run_main_test, all_run_args_acc_gpus):
    args = all_run_args_acc_gpus
    run_main_test(args)
