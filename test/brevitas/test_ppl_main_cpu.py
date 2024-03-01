
import pytest

@pytest.fixture()
def recommended_args_cpu(recommended_args):
    args = recommended_args
    args.device = "cpu"
    return args


@pytest.mark.ppl
@pytest.mark.cpu
@pytest.mark.opt
@pytest.mark.very_long
@pytest.mark.recommended
def test_recommended_opt_cpu(ppl_main_test, recommended_args_cpu, default_model_with_ppl):
    args = recommended_args_cpu
    ppl_main_test(args, default_model_with_ppl.float_ppl, default_model_with_ppl.quant_ppl, default_model_with_ppl.onnx_ppl)


@pytest.mark.ppl
@pytest.mark.cpu
@pytest.mark.small_models
@pytest.mark.very_long
@pytest.mark.recommended
def test_recommended_small_models_cpu(ppl_main_test, recommended_args_cpu, small_models_with_ppl):
    args = recommended_args_cpu
    args.model = small_models_with_ppl.name
    ppl_main_test(args, small_models_with_ppl.float_ppl, small_models_with_ppl.quant_ppl, small_models_with_ppl.onnx_ppl)


@pytest.mark.ppl
@pytest.mark.cpu
@pytest.mark.large_models
@pytest.mark.very_long
@pytest.mark.recommended
def test_recommended_large_models_cpu(ppl_main_test, recommended_args_cpu, large_models_with_ppl):
    args = recommended_args_cpu
    args.model = large_models_with_ppl.name
    ppl_main_test(args, large_models_with_ppl.float_ppl, large_models_with_ppl.quant_ppl, large_models_with_ppl.onnx_ppl)
