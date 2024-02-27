
from collections import OrderedDict
import os
import shutil
from dataclasses import dataclass

import pytest

import torch

import onnx

from optimum_amd_utils.examples.quantize_llm import main

from test.brevitas.utils import ptid2pathname


@dataclass
class ModelAndPpl:
    name: str
    float_ppl: float
    quant_ppl: float
    onnx_ppl: float


@pytest.fixture(scope="session")
def default_model_with_ppl():
    return ModelAndPpl(
        name="facebook/opt-125m",
        float_ppl=69.3260,
        quant_ppl=80.5520,
        onnx_ppl=0.0,
    )


@pytest.fixture(scope="session")
def default_model(default_model_with_ppl):
    return default_model_with_ppl.name


@pytest.fixture(scope="session", params=[
    ModelAndPpl(
        name="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        float_ppl=0.0,
        quant_ppl=0.0,
        onnx_ppl=0.0,
    ),
    ModelAndPpl(
        name="openaccess-ai-collective/tiny-mistral",
        float_ppl=0.0,
        quant_ppl=0.0,
        onnx_ppl=0.0,
    ),
])
def small_models_with_ppl(request):
    yield request.param


@pytest.fixture(scope="session")
def small_models(small_model_with_ppl):
    return small_model_with_ppl.name


@pytest.fixture(scope="session", params=[
    ModelAndPpl(
        name="facebook/opt-1.3b",
        float_ppl=0.0,
        quant_ppl=0.0,
        onnx_ppl=0.0,
    ),
    ModelAndPpl(
        name="TheBloke/Llama-2-7B-fp16",
        float_ppl=0.0,
        quant_ppl=0.0,
        onnx_ppl=0.0,
    ),
    ModelAndPpl(
        name="mistralai/Mistral-7B-v0.1",
        float_ppl=0.0,
        quant_ppl=0.0,
        onnx_ppl=0.0,
    ),
])
def large_models_with_ppl(request):
    yield request.param


@pytest.fixture(scope="session")
def large_models(large_model_with_ppl):
    return large_model_with_ppl.name


@pytest.fixture()
def default_args(default_model, request):
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
    args.onnx_output_path = ptid2pathname(request.node.nodeid)
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
        onnx_model = onnx.load(os.path.join(args.onnx_output_path, "model.onnx"))
        shutil.rmtree(args.onnx_output_path)
    return _run_main_test


@pytest.fixture(scope="session")
def ppl_main_test(run_main):
    def _ppl_main_test(args, expected_float_ppl, expected_quant_ppl):
        return_val = run_main(args)
        shutil.rmtree(args.onnx_output_path)
        assert return_val["float_perplexity"] == return_val["quant_perplexity"]
    return _ppl_main_test
