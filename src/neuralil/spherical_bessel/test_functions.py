import pytest
import jax.numpy as jnp
from jax import grad
import numpy as np
from .functions import create_j_l, _j_0, _j_1, _calc_starting_order_Cai

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

