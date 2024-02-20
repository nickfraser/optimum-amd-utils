
#from collections import OrderedDict
#import pytest
#
#from quantize_llm import main
#
#
#@pytest.fixture(scope="session")
#def default_args():
#    args = OrderedDict()
#    args.model = "facebook/opt-125m"
#    args.apply_gptq = False
#    args.apply_weight_equalization = False
#    args.apply_bias_correction = False
#    args.activations_equalization = None # choices=[None, "cross_layer", "layerwise"]
#    args.is_static = False
#    args.seqlen = 128
#    args.nsamples = 128
#    args.device = "auto"
#    args.onnx_output_path = llm_quantized_onnx
#    return args
#
#
#@pytest.fixture(scope="session")
#def default_run_args():
#    args = default_args()
#    args.nsamples = 2
#    args.seqlen = 2
#    return args
#
#
#@pytest.fixture(scope="session")
#def run_main(args):
#    def _run_main(args):
#        return_val = main(args)
#        return return_val
#    return _run_main
#
#
#def test_opt(run_main, default_run_args):
#    return_val = run_main(default_run_args)
#    assert type(return_val["float_perplexity"]) == float
#    assert type(return_val["quant_perplexity"]) == float

def test_sanity():
    assert True
