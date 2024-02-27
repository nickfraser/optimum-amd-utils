
import pytest


single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 1, reason="At least one GPU required for the GPU tests"
)


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
@single_gpu
def test_toggle_opt_gpu(run_main_test, toggle_run_args_gpu):
    args = toggle_run_args_gpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_opt_gpu(run_main_test, recommended_run_args_gpu):
    args = recommended_run_args_gpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.opt
@pytest.mark.long
@single_gpu
def test_all_opt_gpu(run_main_test, all_run_args_gpu):
    args = all_run_args_gpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.small_models
@pytest.mark.short
@pytest.mark.recommended
@single_gpu
def test_recommended_small_models_gpu(run_main_test, recommended_run_args_gpu, small_models):
    args = recommended_run_args_gpu
    args.model = small_models
    run_main_test(args)


@pytest.mark.run
@pytest.mark.gpu
@pytest.mark.large_models
@pytest.mark.long
@pytest.mark.recommended
@pytest.mark.xfail(strict=False)
@single_gpu
def test_recommended_large_models_gpu(run_main_test, recommended_run_args_gpu, large_models):
    args = recommended_run_args_gpu
    args.model = large_models
    run_main_test(args)
