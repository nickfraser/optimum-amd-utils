
from collections import OrderedDict

import pytest

import torch

from optimum_amd_utils.examples.quantize_llm import main

@pytest.fixture(scope="session")
def default_model():
    return "facebook/opt-125m"


@pytest.fixture(scope="session")
def default_small_opt(default_model):
    return default_small_opt


@pytest.fixture(scope="session")
def default_small_llama():
    return "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


@pytest.fixture(scope="session")
def default_small_mistral():
    return "hf-internal-testing/tiny-random-MistralForCausalLM"


@pytest.fixture()
def default_args(default_model):
    args = OrderedDict()
    args.model = default_model
    args.apply_gptq = False
    args.apply_weight_equalization = False
    args.apply_bias_correction = False
    args.activations_equalization = None # choices=[None, "cross_layer", "layerwise"]
    args.is_static = False
    args.seqlen = 128
    args.nsamples = 128
    args.device = "auto"
    args.onnx_output_path = "llm_quantized_onnx"
    args.gpu_device_map = None
    args.cpu_device_map = None
    return args


@pytest.fixture()
def default_run_args(default_args):
    args = default_args
    args.nsamples = 2
    args.seqlen = 2
    return args


@pytest.fixture()
def recommended_args(default_args):
    args = default_args
    args.apply_gptq = True
    args.activations_equalization = "layerwise"
    return args


@pytest.fixture()
def recommended_run_args(recommended_args):
    args = recommended_args
    args.nsamples = 2
    args.seqlen = 2
    return args


@pytest.fixture()
def recommended_llama_run_args(recommended_run_args, default_small_llama):
    args = recommended_run_args
    args.model = default_small_llama
    return args


@pytest.fixture()
def recommended_mistral_run_args(recommended_run_args, default_small_mistral):
    args = recommended_run_args
    args.model = default_small_mistral
    return args


@pytest.fixture(params=[False, True])
def apply_gptq(request):
    yield request.param


@pytest.fixture(params=[False, True])
def apply_weight_equalization(request):
    yield request.param


@pytest.fixture(params=[False, True])
def apply_bias_correction(request):
    yield request.param


@pytest.fixture(params=[None, "cross_layer", "layerwise"])
def activations_equalization(request):
    yield request.param


@pytest.fixture(params=[False, True])
def is_static(request):
    yield request.param


@pytest.fixture()
def all_run_args(default_run_args, apply_gptq, apply_weight_equalization, apply_bias_correction, activations_equalization, is_static):
    args = default_run_args
    args.apply_gptq = apply_gptq
    args.apply_weight_equalization = apply_weight_equalization
    args.apply_bias_correction = apply_bias_correction
    args.activations_equalization = activations_equalization
    args.is_static = is_static
    return args


@pytest.fixture(params=[
        [False, False, False, None, False],
        [True, False, False, None, False],
        [False, True, False, None, False],
        [False, False, True, None, False],
        [False, False, False, "cross_layer", False],
        [False, False, False, "layerwise", False],
        [False, False, False, None, True],
    ])
def toggle_run_args(default_run_args, request):
    args = default_run_args
    args.apply_gptq = request.param[0]
    args.apply_weight_equalization = request.param[1]
    args.apply_bias_correction = request.param[2]
    args.activations_equalization = request.param[3]
    args.is_static = request.param[4]
    yield args


@pytest.fixture(scope="session")
def run_main():
    def _run_main(args):
        return_val = main(args)
        return return_val
    return _run_main


@pytest.fixture(scope="session")
def run_main():
    def _run_main(args):
        return_val = main(args)
        return return_val
    return _run_main


@pytest.fixture(scope="session")
def run_main_test(run_main):
    def _run_main_test(args):
        return_val = run_main(args)
        assert type(return_val["float_perplexity"]) == torch.Tensor
        assert type(return_val["quant_perplexity"]) == torch.Tensor
    return _run_main_test
