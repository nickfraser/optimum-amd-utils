
from collections import OrderedDict
import pytest

import torch

from optimum_amd_utils.examples.quantize_llm import main


@pytest.fixture()
def default_args():
    args = OrderedDict()
    args.model = "facebook/opt-125m"
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
def run_all_args(default_run_args, apply_gptq, apply_weight_equalization, apply_bias_correction, activations_equalization, is_static):
    args = default_run_args
    args.apply_gptq = apply_gptq
    args.apply_weight_equalization = apply_weight_equalization
    args.apply_bias_correction = apply_bias_correction
    args.activations_equalization = activations_equalization
    args.is_static = is_static
    return args


@pytest.fixture(scope="session")
def run_main():
    def _run_main(args):
        return_val = main(args)
        return return_val
    return _run_main


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.opt
def test_opt_cpu(run_main, run_all_args):
    args = run_all_args
    args.device = "cpu"
    return_val = run_main(run_all_args)
    assert type(return_val["float_perplexity"]) == torch.Tensor
    assert type(return_val["quant_perplexity"]) == torch.Tensor
