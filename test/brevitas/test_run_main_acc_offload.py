
import pytest

from optimum.amd.brevitas.accelerate_utils import calc_gpu_device_map


single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="At least one GPU required for the GPU tests"
)


@pytest.fixture()
def toggle_run_args_acc_offload(toggle_run_args):
    args = toggle_run_args
    args.device = "auto"
    args.gpu_device_map = {0: 5.0e8}
    return args


@pytest.fixture()
def recommended_run_args_acc_offload(recommended_run_args):
    args = recommended_run_args
    args.device = "auto"
    args.gpu_device_map = {0: 5.0e8}
    return args


@pytest.fixture()
def recommended_large_model_run_args_acc_offload(recommended_run_args_acc_offload):
    args = recommended_run_args_acc_offload
    args.gpu_device_map = {0: calc_gpu_device_map(absolute_mem_margin=2.0 * 1e9, relative_mem_margin=0.3)[0]}
    return args


@pytest.fixture()
def all_run_args_acc_offload(all_run_args):
    args = all_run_args
    args.device = "auto"
    args.gpu_device_map = {0: 5.0e8}
    return args


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.short
@single_gpu
def test_toggle_opt_acc_offload(run_main_test, toggle_run_args_acc_offload):
    args = toggle_run_args_acc_offload
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_opt_acc_offload(run_main_test, recommended_run_args_acc_offload):
    args = recommended_run_args_acc_offload
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.long
@single_gpu
def test_all_opt_acc_offload(run_main_test, all_run_args_acc_offload):
    args = all_run_args_acc_offload
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_small_models_acc_offload(run_main_test, recommended_run_args_acc_offload, small_models):
    args = recommended_run_args_acc_offload
    args.model = small_models
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@single_gpu
def test_recommended_large_models_acc_offload(run_main_test, recommended_large_model_run_args_acc_offload, large_models):
    args = recommended_large_model_run_args_acc_offload
    args.model = large_models
    run_main_test(args)
