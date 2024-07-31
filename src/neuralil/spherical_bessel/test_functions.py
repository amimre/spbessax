import pytest
import jax.numpy as jnp
from jax import grad
from jax import config
config.update("jax_enable_x64", True)
import numpy as np

from scipy.special import spherical_jn

from .functions import create_j_l, _j_0, _j_1

MAX_TEST_ORDER = 40
TEST_ORDERS = [i for i in range(MAX_TEST_ORDER)]

r = jnp.linspace(1e-3, 50., num=1001)

# Test _j_0 function
def test_j_0():
    expected = jnp.sinc(r / jnp.pi)
    assert np.allclose(_j_0(r), expected)

# Test _j_1 function
def test_j_1():
    expected = (jnp.sinc(r / jnp.pi) - jnp.cos(r)) / r
    assert np.allclose(_j_1(r), expected)

# Test invalid order raises ValueError
def test_create_j_l_invalid_order():
    with pytest.raises(ValueError):
        create_j_l(-1)


## Test for real valued arguments

@pytest.mark.parametrize("order", TEST_ORDERS)
def test_create_j_l_single_order(order):
    func = create_j_l(order=order, dtype=jnp.float64, output_all=False)
    expected = spherical_jn(order, r)
    assert np.allclose(func(r), expected, atol=1e-8, rtol=1e-8)


def test_create_j_l_output_all():
    func = create_j_l(order=MAX_TEST_ORDER, dtype=jnp.float64, output_all=True)
    expected = []
    for order in range(MAX_TEST_ORDER+1):
        expected.append(spherical_jn(order, r))
    expected = np.array(expected).swapaxes(0, 1)
    result = func(r)
    assert np.allclose(func(r), expected, atol=1e-8, rtol=1e-8)
