#!/usr/bin/env python
# Copyright 2019-2024 The NeuralIL and spbessax contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp


def _j_0(r: jnp.ndarray) -> jnp.ndarray:
    """Order-0 spherical Bessel function of the first kind.

    The evaluation is delegated to jax.numpy.sinc. The condition r >= 0 is
    not assessed.

    Args:
        r: The radial coordinate.

    Returns:
        The values of the order-0 spherical Bessel function of the first kind.
    """
    return jnp.sinc(r / jnp.pi)


def _j_1(r: jnp.ndarray) -> jnp.ndarray:
    """Order-1 spherical Bessel function of the first kind.

    The function is built on the basis of calls to existing jax.numpy code. The
    condition r >= 0 is not assessed.

    Args:
        r: The radial coordinate.

    Returns:
        The values of the order-1 spherical Bessel function of the first kind.
    """
    return (jnp.sinc(r / jnp.pi) - jnp.cos(r)) / r


def _calc_starting_order_Cai(order: int, r):
    """Use Cai's recipe to compute a starting point for his recurrence.

    When the method of Computer Physics Communications 182 (2011) 664,
    doi:10.1016/j.cpc.2010.11.019 is used to calculate j_l, an starting
    order is required for the descending recurrence. The original
    prescription works for any complex number, but this implementation
    only works for real arguments.

    Args:
        order: The order of the spherical Bessel function to be evaluated.
        r: A single value of the argument to the spherical bessel function.
    """
    cais_m = int(onp.floor(1.83 * r**0.91 + 9.))
    lower_bound = max(order + 1, cais_m)
    # This bound should depend on the kind of floating point used, but that
    # is not a critical point for our narrow use case.
    upper_bound = int(onp.floor(235. + 50. * onp.sqrt(r)))
    return min(lower_bound, upper_bound)


def create_j_l(order: int,
               output_all: bool = False,
               dtype: onp.dtype = onp.float32):
    """Generate the l-th spherical Bessel function of the first kind.

    This implementation is only intended for real arguments, and the focus is
    on compatibility with JAX. The calculation is handled by two different
    recurrence strategies, depending on whether the argument is less than or
    greater than the order.

    Args:
        order: The order of the spherical Bessel function, which cannot be
            negative.
        output_all: Whether to return all the values of the function up to the
            selected order or just the last one.
        dtype: A floating-point data type on which the function is intended to
           operate.

    Returns:
        A callable that takes radial coordinates and returns the values of
            the selected Bessel function.

    Raises:
        ValueError: if "order" is negative.
    """
    if order < 0:
        raise ValueError("the order of the function cannot be negative")
    elif order == 0:
        return _j_0
    elif order == 1:
        return ((lambda r: jnp.asarray([_j_0(r), _j_1(r)], dtype=dtype))
                if output_all else _j_1)

    def j_l_upward(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind with derivative.

        This implementation, intended for r>l, is based on ascendent recurrence
        and adapted from SciPy itself. The condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the spherical Bessel function of the first
            kind and the values of its derivative up to order l, in a tuple.
        """
        order_0 = _j_0(r)
        order_1 = _j_1(r)

        derivative_0 = jnp.vectorize(jax.grad(_j_0,
                                    holomorphic=jnp.iscomplexobj(dtype)))(r)
        derivative_1 = jnp.vectorize(jax.grad(_j_1,
                                    holomorphic=jnp.iscomplexobj(dtype)))(r)

        init = (order_0, order_1, 0)
        def loop_orders(carry, x):
            order_minus_2, order_minus_1, i = carry
            y = (2 * i + 3) / r * order_minus_1 - order_minus_2
            deriv = order_minus_1 - (i + 3) * y / r
            carry = (order_minus_1, y, i+1)
            return carry, (y, deriv)

        _, computed_orders = jax.lax.scan(loop_orders,
                                          init, xs=None, length=order-1,
                                          unroll=True)
        orders = jnp.concatenate((jnp.array([order_0, order_1]),
                                computed_orders[0]))
        derivatives = jnp.concatenate((jnp.array([derivative_0, derivative_1]),
                                    computed_orders[1]))

        return (orders, derivatives)

    starting_order = _calc_starting_order_Cai(order, order)
    # Cai's original prescription is 1e-305, which works for double
    # precision but causes huge problems with the derivative. eps**2 * r
    # seems to work well, since smaller r require a smaller value.
    initial_prefactor = onp.finfo(dtype).eps**2

    def j_l_Cai(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind with derivative.

        This implementation is based on the algorithm developed by L.-W. Cai
        and published in Computer Physics Communications 182 (2011) 664,
        doi:10.1016/j.cpc.2010.11.019 . Whereas Cai's focus is on complex
        numbers, this implementation is intended only for use with real
        arguments in JAX-based differentiable programs. Moreover, the potential
        arguments are assumed to be bounded from above by the order of the
        function.The condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the spherical Bessel function of the first
            kind and the values of its derivative up to order l, in a tuple.
        """
        plus_1 = jnp.zeros_like(r, dtype=dtype)
        temp = initial_prefactor * r * jnp.ones_like(r, dtype=dtype)

        init = (plus_1, temp, 0)
        def iterate_orders(carry, x):
            plus_1, temp, i = carry
            minus_1 = (2 * (starting_order - i) + 1) * temp / r - plus_1
            carry = (temp, minus_1, i+1)
            return carry, (minus_1, (starting_order-i-1) * minus_1 / r - temp)

        # Iterate from the starting order to get the desired accuracy
        (plus_1, minus_1, _), calculated_orders = (
            jax.lax.scan(iterate_orders,
                         init, xs=None, length=starting_order, unroll=True))

        # Flip since we interated from high to low order
        unnormalized = jnp.flip(calculated_orders[0][-(order+1):])
        unnormalized_derivative = jnp.flip(calculated_orders[1][-(order+1):])

        # Generally speaking, we use the explicit form of j_0 to obtain the
        # right normalization for our function, but close to the zeros of j_0
        # it is more convenient to use j_1 instead.
        order_0 = _j_0(r)
        order_1 = _j_1(r)
        prefactor = jnp.where(
            abs(order_0) >= abs(order_1),
            order_0 / jnp.asarray(minus_1, dtype=dtype),
            order_1 / jnp.asarray(plus_1, dtype=dtype)
        )
        normalized = prefactor * unnormalized
        normalized_derivative = prefactor *unnormalized_derivative

        return normalized, normalized_derivative

    @jax.custom_jvp
    @partial(jnp.vectorize,
             signature="()->(l)" if output_all else "()->()")
    def j_l(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind.

        This function just dispatches each argument to the right implementation
        based on whether r is lower than the order of the function or not. The
        condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the order-l spherical Bessel function of the first
                kind.
        """
        r = jnp.asarray(r, dtype=dtype)

        Cai_values = j_l_Cai(jnp.clip(r, a_max=order))[0]
        upward_l_values = j_l_upward(r)[0]
        condition = abs(r) < order                                              # TODO: not clear if this should be abs(r) or r.real
        selected_values = (condition * Cai_values +
                           (1. - condition) * upward_l_values)

        if not output_all:
            return selected_values[-1]
        return selected_values

    # Since the derivatives of the spherical Bessel functions are inexpensive
    # to calculate, it pays off to define a custom jvp rule for them.
    @j_l.defjvp
    def j_l_jvp(primals, tangents):
        r, = primals
        r_dot, = tangents
        Cai, Cai_dot = j_l_Cai(jnp.clip(r, a_max=order))
        upward, upward_dot = j_l_upward(r)
        condition = abs(r) < order                                              # TODO: not clear if this should be abs(r) or r.real
        primal_out = condition * Cai + (1. - condition) * upward
        tangent_out = (
            condition * Cai_dot + (1. - condition) * upward_dot
        ) * r_dot

        if not output_all:
            return primal_out[-1], tangent_out[-1]
        return primal_out, tangent_out

    return j_l


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy as sp
    import scipy.special

    ORDER = 10

    returned_function = create_j_l(ORDER, dtype=onp.float32)
    r = jnp.linspace(1e-3, 50., num=1001)
    value = returned_function(r)
    reference = sp.special.spherical_jn(ORDER, r)

    plt.figure()
    plt.plot(r, value, label="value")
    plt.plot(r, reference, label="reference")
    plt.axvline(x=ORDER, color="#666666", lw=2.)
    plt.ylim(reference.min(), reference.max())
    plt.xlabel("$r$")
    plt.ylabel(f"$j_{ORDER}$")
    plt.legend(loc="best")
    plt.tight_layout()

    derivative_function = jnp.vectorize(jax.grad(returned_function))
    derivative_value = derivative_function(r)
    derivative_reference = scipy.special.spherical_jn(ORDER, r, derivative=True)

    plt.figure()
    plt.plot(r, derivative_value, label="derivative")
    plt.plot(r, derivative_reference, label="reference")
    plt.axvline(x=ORDER, color="#666666", lw=2.)
    plt.ylim(derivative_reference.min(), derivative_reference.max())
    plt.xlabel("$r$")
    plt.ylabel(f"$j'_{ORDER}$")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.show()
