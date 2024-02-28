
import pytest

import torch

from test.brevitas.utils import multigpu

@pytest.fixture()
def recommended_args_acc_gpus(recommended_args):
    args = recommended_args
    args.device = "auto"
    args.gpu_device_map = {i: 5.0e8 for i in range(torch.cuda.device_count())}
    return args


@pytest.fixture()
def recommended_large_model_args_acc_gpus(recommended_args):
    args = recommended_args
    args.device = "auto"
    args.gpu_device_map = None
    return args


@pytest.mark.ppl
@pytest.mark.acc_gpus
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@multigpu
def test_recommended_opt_acc_gpus(ppl_main_test, recommended_args_acc_gpus, default_model_with_ppl):
    args = recommended_args_acc_gpus
    ppl_main_test(args, default_model_with_ppl.float_ppl, default_model_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc_gpus
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
@multigpu
def test_recommended_small_models_acc_gpus(ppl_main_test, recommended_args_acc_gpus, small_models_with_ppl):
    args = recommended_args_acc_gpus
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc_gpus
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@pytest.mark.xfail(strict=False)
@multigpu
def test_recommended_large_models_acc_gpus(ppl_main_test, recommended_large_model_args_acc_gpus, large_models_with_ppl):
    args = recommended_large_model_args_acc_gpus
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl)
