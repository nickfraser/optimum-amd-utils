
import pytest

@pytest.fixture()
def toggle_run_args_gpu(toggle_run_args):
    args = toggle_run_args
    args.device = "cuda:0"
    return args


@pytest.fixture()
def recommended_run_args_gpu(recommended_run_args):
    args = recommended_run_args
    args.device = "cuda:0"
    return args


@pytest.fixture()
def all_run_args_gpu(all_run_args):
    args = all_run_args
    args.device = "cuda:0"
    return args


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.short
def test_toggle_opt_gpu(run_main_test, toggle_run_args_gpu):
    args = toggle_run_args_gpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_gpu(run_main_test, recommended_run_args_gpu):
    args = recommended_run_args_gpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.long
def test_all_opt_gpu(run_main_test, all_run_args_gpu):
    args = all_run_args_gpu
    run_main_test(args)
