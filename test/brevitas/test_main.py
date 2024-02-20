
from collections import OrderedDict
import pytest

import torch

from optimum_amd_utils.examples.quantize_llm import main


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def default_run_args(default_args):
    args = default_args
    args.nsamples = 2
    args.seqlen = 2
    return args


@pytest.fixture(scope="session")
def run_main():
    def _run_main(args):
        return_val = main(args)
        return return_val
    return _run_main


def test_opt(run_main, default_run_args):
    return_val = run_main(default_run_args)
    assert type(return_val["float_perplexity"]) == torch.Tensor
    assert type(return_val["quant_perplexity"]) == torch.Tensor
