
import pytest

from test.brevitas.utils import single_gpu

@pytest.fixture()
def recommended_args_gpu(recommended_args):
    args = recommended_args
    args.device = "cuda:0"
    return args


@pytest.mark.ppl
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_opt_gpu(ppl_main_test, recommended_args_gpu, default_model_with_ppl):
    args = recommended_args_gpu
    ppl_main_test(args, default_model_with_ppl.float_ppl, default_model_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.gpu
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_small_models_gpu(ppl_main_test, recommended_args_gpu, small_models_with_ppl):
    args = recommended_args_gpu
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.gpu
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@pytest.mark.xfail(strict=False)
@single_gpu
def test_recommended_large_models_gpu(ppl_main_test, recommended_args_gpu, large_models_with_ppl):
    args = recommended_args_gpu
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl)
