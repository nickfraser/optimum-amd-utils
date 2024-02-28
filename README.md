# Utils for Testing Optimum-AMDs Brevitas Example

A small collection of utils and a test suite for optimum-amd's Brevitas example.

## Prerequisites

The test suite has been tested with the following:
 - python==3.11
 - PyTorch==2.12
 - accelerate==main
 - transformers==4.38.0
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
 - Small and large OPT, Llama & Mistral models
 - every PTQ algorithm individually
 - every device (cpu, gpu, accelerate without CPU offload, accelerate with CPU offload, accelerate with multiple GPUs),

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

#### Sanity Check Tests

A good set of tests to run to test that the main modes don't fail are:

```bash
CUDA_VISIBLE_DEVICES=0 pytest -m "short and run and (opt or small_models) and (cpu or gpu)"
CUDA_VISIBLE_DEVICES=1 pytest -m "short and run and (opt or small_models) and (acc or acc_offload)"
CUDA_VISIBLE_DEVICES=2 pytest -m "short and run and (opt or small_models) and recommended"
```

assuming you have 3 GPUs.

To run the mulit-GPU tests, a good set is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pytest -m "short and run and (opt or small_models) and recommended and acc_gpus"
```

Again, assuming you have 3 GPUs.

#### Deliverable Tests

The following will test all the models that we're supposed to deilver on multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 pytest -m "large_models and ppl and acc_gpus"
```

