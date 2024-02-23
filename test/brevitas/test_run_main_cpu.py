
import pytest

@pytest.fixture()
def toggle_run_args_cpu(toggle_run_args):
    args = toggle_run_args
    args.device = "cpu"
    return args


@pytest.fixture()
def recommended_run_args_cpu(recommended_run_args):
    args = recommended_run_args
    args.device = "cpu"
    return args


@pytest.fixture()
def all_run_args_cpu(all_run_args):
    args = all_run_args
    args.device = "cpu"
    return args


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.opt
@pytest.mark.short
def test_toggle_opt_cpu(run_main_test, toggle_run_args_cpu):
    args = toggle_run_args_cpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_cpu(run_main_test, recommended_run_args_cpu):
    args = recommended_run_args_cpu
    run_main_test(args)


@pytest.mark.run
@pytest.mark.cpu
@pytest.mark.opt
@pytest.mark.long
def test_all_opt_cpu(run_main_test, all_run_args):
    args = all_run_args
    run_main_test(args)
