import numpy as np
import pytest
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.optimizers import (
    AdamOptimizerConfig,
    FlatLRSchedule,
    InitiallyZeroLRSchedule,
    PolynomialLRSchedule,
)

LEARNING_RATES = [1.0, 0.1, 0.01]


@pytest.mark.parametrize("lr", LEARNING_RATES)
def test_flat_schedule(lr):
    obj = FlatLRSchedule(lr=lr)
    assert obj(1.0) == lr
    assert obj(0.5) == lr
    assert obj(0.0) == lr


@pytest.mark.parametrize("lr", LEARNING_RATES)
@pytest.mark.parametrize("power", [1, 2, 4])
def test_polynomial_schedule(lr, power):
    baseline = lr / 2
    obj = PolynomialLRSchedule(lr=lr, power=power, baseline=baseline)

    assert obj(1.0) == lr
    assert baseline < obj(0.5) and obj(0.5) < lr
    if power == 1.0:
        assert np.allclose(obj(0.5), (lr + baseline) / 2, atol=0.0)
    assert obj(0.0) == baseline


@pytest.mark.parametrize("lr", LEARNING_RATES)
@pytest.mark.parametrize("zero_proportion", [1e-6, 1 - 1e-6])
def test_initially_zero_schedule(lr, zero_proportion):
    baseline = lr / 2
    obj = InitiallyZeroLRSchedule(zero_proportion=zero_proportion, base=PolynomialLRSchedule(lr=lr, baseline=lr / 2, power=1))

    assert obj(1.0) == 0.0
    assert obj(1 - zero_proportion) == lr
    middle = 1e-8
    assert baseline < obj(middle) and obj(middle) < lr
    assert obj(0.0) == baseline


def test_adam_scale_batch_size():
    # Larger batch size implies smaller betas and larger LR
    from_bs = 32
    to_bs = 256

    opt = AdamOptimizerConfig(lr=FlatLRSchedule(0.003), eps=1e-4, betas=(0.9, 0.999))
    new_opt = opt.scale_batch_size(from_bs=from_bs, to_bs=to_bs)

    assert check_cast(FlatLRSchedule, new_opt.lr).lr == 0.008485281374238571
    assert new_opt.betas[0] == 0.4304672100000001
    assert new_opt.betas[1] == 0.992027944069944
