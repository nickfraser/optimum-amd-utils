
import pytest

@pytest.fixture()
def toggle_run_args_acc(toggle_run_args):
    args = toggle_run_args
    args.device = "auto"
    return args


@pytest.fixture()
def recommended_run_args_acc(recommended_run_args):
    args = recommended_run_args
    args.device = "auto"
    return args


@pytest.fixture()
def all_run_args_acc(all_run_args):
    args = all_run_args
    args.device = "auto"
    return args


@pytest.mark.run
@pytest.mark.acc
@pytest.mark.opt
@pytest.mark.short
def test_toggle_opt_acc(run_main_test, toggle_run_args_acc):
    args = toggle_run_args_acc
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc
@pytest.mark.opt
@pytest.mark.short
@pytest.mark.recommended
def test_recommended_opt_acc(run_main_test, recommended_run_args_acc):
    args = recommended_run_args_acc
    run_main_test(args)


@pytest.mark.run
@pytest.mark.acc
@pytest.mark.opt
@pytest.mark.long
def test_all_opt_acc(run_main_test, all_run_args_acc):
    args = all_run_args_acc
    run_main_test(args)
