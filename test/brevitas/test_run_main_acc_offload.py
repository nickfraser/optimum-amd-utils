
import pytest

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
def all_run_args_acc_offload(all_run_args):
    args = all_run_args
    args.device = "auto"
    args.gpu_device_map = {0: 5.0e8}
    return args


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.short
def test_toggle_opt_acc_offload(run_main_test, toggle_run_args_acc_offload):
    args = toggle_run_args_acc_offload
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_acc_offload(run_main_test, recommended_run_args_acc_offload):
    args = recommended_run_args_acc_offload
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc_offload
@pytest.mark.opt
@pytest.mark.long
def test_all_opt_acc_offload(run_main_test, all_run_args_acc_offload):
    args = all_run_args_acc_offload
    run_main_test(args)
