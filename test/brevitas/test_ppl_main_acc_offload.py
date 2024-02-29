
import pytest

from test.brevitas.utils import single_gpu

@pytest.fixture()
def recommended_args_acc_offload(recommended_args):
    args = recommended_args
    args.device = "auto"
    args.gpu_device_map = {0: 5.0e8}
    return args


@pytest.fixture()
def recommended_large_model_args_acc_offload(recommended_args):
    args = recommended_args
    args.device = "auto"
    args.gpu_device_map = {0: calc_gpu_device_map(absolute_mem_margin=2.0 * 1e9, relative_mem_margin=0.3)[0]}
    return args


@pytest.mark.ppl
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_opt_acc_offload(ppl_main_test, recommended_args_acc_offload, default_model_with_ppl):
    args = recommended_args_acc_offload
    ppl_main_test(args, default_model_with_ppl.float_ppl, default_model_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc_offload
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_small_models_acc_offload(ppl_main_test, recommended_args_acc_offload, small_models_with_ppl):
    args = recommended_args_acc_offload
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc_offload
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@single_gpu
def test_recommended_large_models_acc_offload(ppl_main_test, recommended_large_model_args_acc_offload, large_models_with_ppl):
    args = recommended_large_model_args_acc_offload
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl)
