# Utils for Testing Optimum-AMDs Brevitas Example

A small collection of utils and a test suite for optimum-amd's Brevitas example.

## Prerequisites

The test suite has been tested with the following:
 - python==3.11
 - PyTorch==2.12
 - accelerate==main
 - transformers==`4b236aed7618d90546cd2e8797dab5b4a24c5dce`
 - optimum>=1.17.0
 - brevitas>=0.10.2
 - pytest==8.0.1
 - optimum-amd (See instructions below)

### Environment Setup

The easiest way to set up the environment is with miniforge as follows:

```bash
mamba env create -n oamdu -f conda/oamd_hf_main_pt2.1.2_minimal.yml
pip install -e /path/to/optimum-amd/
ln -s /path/to/optimum-amd/examples/quantization/brevitas/quantize_llm.py src/optimum_amd_utils/examples/
pip install -e .
```

### Running the Tests

The tests have many markers representing different PTQ algorithms and device targets.
To run a test suite which tests:
 - facebook/opt-125m
 - every PTQ algorithm individually
 - every device (cpu, gpu, accelerate without CPU offload, accelerate with CPU offload),

run the following:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -m "short and (cpu or gpu or acc or acc_offload)"
```

Alternatively, to run all the tests with the "recommended" settings, run:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -m "recommended"
```

For any failing tests, you can use it's name to run it individually, for example:

```
CUDA_VISIBLE_DEVICES=0 pytest "test/brevitas/test_main.py::test_toggle_opt_acc[run_toggle_args4]"
```
