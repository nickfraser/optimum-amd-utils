
import pytest

@pytest.fixture()
def recommended_args_cpu(recommended_args):
    args = recommended_args
    args.device = "cpu"
    return args


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_cpu(ppl_main_test, recommended_args_cpu):
    args = recommended_args_cpu
    ppl_main_test(args, 0.0, 0.0)


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_small_models_cpu(ppl_main_test, recommended_args_cpu, small_models_with_ppl):
    args = recommended_args_cpu
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl)


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
def test_recommended_large_models_cpu(ppl_main_test, recommended_args_cpu, large_models_with_ppl):
    args = recommended_args_cpu
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl)