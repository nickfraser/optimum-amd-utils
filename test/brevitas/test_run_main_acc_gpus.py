
import pytest

import torch

from test.brevitas.utils import multigpu

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
def recommended_large_model_run_args_acc_gpus(recommended_run_args_acc_gpus):
    args = recommended_run_args_acc_gpus
    args.gpu_device_map = None
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


@pytest.mark.run
@pytest.mark.acc_gpus
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_small_models_acc_gpus(run_main_test, recommended_run_args_acc_gpus, small_models):
    args = recommended_run_args_acc_gpus
    args.model = small_models
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_gpus
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
def test_recommended_large_models_acc_gpus(run_main_test, recommended_large_model_run_args_acc_gpus, large_models):
    args = recommended_large_model_run_args_acc_gpus
    args.model = large_models
    run_main_test(args)
