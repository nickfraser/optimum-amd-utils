
import pytest

@pytest.fixture()
def recommended_args_acc(recommended_args):
    args = recommended_args
    args.device = "auto"
    return args


@pytest.mark.ppl
@pytest.mark.acc
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_acc(ppl_main_test, recommended_args_acc, default_model_with_ppl):
    args = recommended_args_acc
    ppl_main_test(args, default_model_with_ppl.float_ppl, default_model_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_small_models_acc(ppl_main_test, recommended_args_acc, small_models_with_ppl):
    args = recommended_args_acc
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl)


@pytest.mark.ppl
@pytest.mark.acc
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@pytest.mark.xfail(strict=False)
def test_recommended_large_models_acc(ppl_main_test, recommended_args_acc, large_models_with_ppl):
    args = recommended_args_acc
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl)
