"""
This module contains classes defining random vairable distributions.

Single variable distributions inherit from ``Distribution`` and define
a ``draw()`` method. This method takes a numpy random generator and uses it
to draw one or more random values from the distribution.

Examples
--------

>>> from qdflow.util import distribution
>>> import numpy as np
>>> rng = np.random.default_rng()

>>> mean, stdev = 5, 2
>>> normal_dist = distribution.Normal(mean, stdev)
>>> normal_dist.draw(rng)
5.137    # random

Multiple values can be drawn at once via the size parameter:

>>> normal_dist.draw(rng, size=(2,3))
array([[3.825, 6.440, 4.821],
       [2.739, 5.512, 7.807]])    # random

Distributions can be combined together with each other, as well as with scalars
via basic operations +, -, *, /.

>>> dist_1 = distribution.Uniform(1,5)
>>> dist_2 = distribution.Uniform(3,7)
>>> combined_dist = 2 * dist_1 - dist_2
>>> combined_dist.draw(rng, size=4)
array([-2.311, 1.339, 4.067, 0.713])    # random

This module also provides a framework for multivariable distributions, via the
class ``CorrelatedDistribution``. After defining a correlated distribution,
a set of linked, dependent, single-variable distributions can be obtained with
the ``dependent_distributions()`` function.
Drawing from each of these distributions yields a set of random, correlated
values.

>>> normal_dist = distribution.Normal(5, 2)
>>> matrix = np.array([[-1], [2]])
>>> correlated_dist = distribution.MatrixCorrelated(matrix, [normal_dist])
>>> dist_1, dist_2 = correlated_dist.dependent_distributions()
>>> result_1 = dist_1.draw(rng, size=3)
>>> result_1
array([-5.813, -1.782, -6.021])    # random

>>> result_2 = dist_2.draw(rng, size=3)
>>> result_2
array([11.626, 3.564, 12.042])    # NOT random, correlated with result_1

``result_1`` and ``result_2`` are random, but dependent on each other.

>>> result_2 / -2
array([-5.813, -1.782, -6.021])    # equal to result_1
"""

from typing import overload, TypeVar, Generic, Any, Sequence
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import warnings

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)


class Distribution(ABC, Generic[T]):
    """
    An abstract class which defines a random distribution.

    Subclasses must implement the ``draw()`` function, which draws one or more
    values from the distribution.
    """

    @overload
    @abstractmethod
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    @abstractmethod
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    @abstractmethod
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        """
        Draws one or more samples from the distribution.

        Parameters
        ----------
        rng : np.random.Generator
            A random generator to use to draw samples.
        size : int | tuple[int] | None
            The number of samples or size of the output array.

        Returns
        -------
        T | ndarray
            One or more samples drawn from the distribution.
            If `size` is ``None``, a single value should be returned.
            Otherwise, an NDArray with shape `size` should be returned, where
            each element is independently drawn from the distribution.
        """
        pass

    def __str__(self) -> str:
        return repr(self)

    def __add__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return SumDistribution(self, dist)

    def __radd__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return SumDistribution(dist, self)

    def __neg__(self) -> "Distribution[T]":
        return NegationDistribution(self)

    def __sub__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return DifferenceDistribution(self, dist)

    def __rsub__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return DifferenceDistribution(dist, self)

    def __mul__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return ProductDistribution(self, dist)

    def __rmul__(self, dist: "Distribution[T]|T") -> "Distribution[T]":
        return ProductDistribution(dist, self)

    def __truediv__(self, dist: "Distribution[Any]|Any") -> "Distribution[Any]":
        return QuotientDistribution(self, dist)

    def __rtruediv__(self, dist: "Distribution[Any]|Any") -> "Distribution[Any]":
        return QuotientDistribution(dist, self)

    def abs(self) -> "Distribution[T]":
        """
        Returns a distribution defined by the absolute value of the value drawn
        from this distribution.
        """
        return AbsDistribution(self)


class SumDistribution(Distribution[T], Generic[T]):
    """
    A distribution defined by the sum of the values drawn from two other distributions.

    Attributes
    ----------
    dist_1, dist_2 : Distribution[T] | T
        The distributions to add together.
    """

    def __init__(self, dist_1: Distribution[T] | T, dist_2: Distribution[T] | T):
        self._dist_1 = dist_1
        self._dist_2 = dist_2
        if not (isinstance(dist_1, Distribution) or isinstance(dist_2, Distribution)):
            self._dist_1 = Delta(dist_1)

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        d1 = (
            self._dist_1.draw(rng, size)
            if isinstance(self._dist_1, Distribution)
            else self._dist_1
        )
        d2 = (
            self._dist_2.draw(rng, size)
            if isinstance(self._dist_2, Distribution)
            else self._dist_2
        )
        if hasattr(d1, "__add__") or hasattr(d2, "__radd__"):
            return d1 + d2  # type: ignore[operator]
        else:
            raise ValueError(
                "Addition not supported for %s and %s" % (repr(d1), repr(d2))
            )

    def __repr__(self) -> str:
        return "%s + %s" % (repr(self._dist_1), repr(self._dist_2))


class NegationDistribution(Distribution[T], Generic[T]):
    """
    A distribution defined by the negation of the value drawn from another distribution.

    Attributes
    ----------
    dist : Distribution[T]
        The distribution to negate.
    """

    def __init__(self, dist: Distribution[T]):
        self._dist = dist

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        d = self._dist.draw(rng, size)
        if hasattr(d, "__neg__"):
            return -d  # type: ignore[operator]
        else:
            raise ValueError("Negation not supported for %s" % (repr(d)))

    def __repr__(self) -> str:
        return "-(%s)" % (repr(self._dist))


class DifferenceDistribution(Distribution[T], Generic[T]):
    """
    A distribution defined by the difference of the values drawn from two other distributions.

    Attributes
    ----------
    dist_1, dist_2 : Distribution[T] | T
        The distributions from which take the difference ``dist_1 - dist_2``.
    """

    def __init__(self, dist_1: Distribution[T] | T, dist_2: Distribution[T] | T):
        self._dist_1 = dist_1
        self._dist_2 = dist_2
        if not (isinstance(dist_1, Distribution) or isinstance(dist_2, Distribution)):
            self._dist_1 = Delta(dist_1)

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        d1 = (
            self._dist_1.draw(rng, size)
            if isinstance(self._dist_1, Distribution)
            else self._dist_1
        )
        d2 = (
            self._dist_2.draw(rng, size)
            if isinstance(self._dist_2, Distribution)
            else self._dist_2
        )
        if hasattr(d1, "__sub__") or hasattr(d2, "__rsub__"):
            return d1 - d2  # type: ignore[operator]
        else:
            raise ValueError(
                "Subtraction not supported for %s and %s" % (repr(d1), repr(d2))
            )

    def __repr__(self) -> str:
        return "%s - (%s)" % (repr(self._dist_1), repr(self._dist_2))


class ProductDistribution(Distribution[T], Generic[T]):
    """
    A distribution defined by the product of the values drawn from two other distributions.

    Attributes
    ----------
    dist_1, dist_2 : Distribution[T] | T
        The distributions to multiply together.
    """

    def __init__(self, dist_1: Distribution[T] | T, dist_2: Distribution[T] | T):
        self._dist_1 = dist_1
        self._dist_2 = dist_2
        if not (isinstance(dist_1, Distribution) or isinstance(dist_2, Distribution)):
            self._dist_1 = Delta(dist_1)

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        d1 = (
            self._dist_1.draw(rng, size)
            if isinstance(self._dist_1, Distribution)
            else self._dist_1
        )
        d2 = (
            self._dist_2.draw(rng, size)
            if isinstance(self._dist_2, Distribution)
            else self._dist_2
        )
        if hasattr(d1, "__mul__") or hasattr(d2, "__rmul__"):
            return d1 * d2  # type: ignore[operator]
        else:
            raise ValueError(
                "Multiplication not supported for %s and %s" % (repr(d1), repr(d2))
            )

    def __repr__(self) -> str:
        return "(%s) * (%s)" % (repr(self._dist_1), repr(self._dist_2))


class QuotientDistribution(Distribution[Any]):
    """
    A distribution defined by the quotient of the values drawn from two other distributions.

    Attributes
    ----------
    dist_1, dist_2 : Distribution | Any
        The distributions from which take the quotient ``dist_1 / dist_2``.
    """

    def __init__(
        self, dist_1: Distribution[Any] | Any, dist_2: Distribution[Any] | Any
    ):
        self._dist_1 = dist_1
        self._dist_2 = dist_2
        if not (isinstance(dist_1, Distribution) or isinstance(dist_2, Distribution)):
            self._dist_1 = Delta(dist_1)

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> Any: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> Any | NDArray:
        d1 = (
            self._dist_1.draw(rng, size)
            if isinstance(self._dist_1, Distribution)
            else self._dist_1
        )
        d2 = (
            self._dist_2.draw(rng, size)
            if isinstance(self._dist_2, Distribution)
            else self._dist_2
        )
        if hasattr(d1, "__truediv__") or hasattr(d2, "__rtruediv__"):
            return d1 / d2  # type: ignore[operator]
        else:
            raise ValueError(
                "Division not supported for %s and %s" % (repr(d1), repr(d2))
            )

    def __repr__(self) -> str:
        return "(%s) / (%s)" % (repr(self._dist_1), repr(self._dist_2))


class AbsDistribution(Distribution[T], Generic[T]):
    """
    A distribution defined by the absolute value of the value drawn from
    another distribution.

    Attributes
    ----------
    dist : Distribution[T]
        The distribution to take the absolute value of.
    """

    def __init__(self, dist: Distribution[T]):
        self._dist = dist

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        return np.abs(self._dist.draw(rng, size))  # type: ignore

    def __repr__(self) -> str:
        return "(%s).abs()" % (repr(self._dist))


class Delta(Distribution[T], Generic[T]):
    """
    A delta-function distribution which always returns `value`.

    Attributes
    ----------
    value : T
        The value that the delta-function distribution should return.
    """

    def __init__(self, value: T):
        self._value = value

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        if size is None:
            return self._value
        else:
            return np.full(size, self._value)

    def __repr__(self) -> str:
        return "distribution.Delta(%s)" % (repr(self._value))


class Normal(Distribution[float]):
    """
    A normal distribution with the specified mean and standard deviation.

    Attributes
    ----------
    mean : float
        The mean of the normal distribution.
    stdev : float
        The standard deviation of the normal distribution.
    """

    def __init__(self, mean: float, stdev: float):
        self._mean = mean
        self._stdev = stdev

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> float: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray[np.float64]: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]:
        return rng.normal(self._mean, self._stdev, size=size)

    def __repr__(self) -> str:
        return "distribution.Normal(%s, %s)" % (repr(self._mean), repr(self._stdev))


class Uniform(Distribution[float]):
    """
    A uniform distribution with range ``[min, max)``.

    Attributes
    ----------
    min : float
        The left side (inclusive) of the interval to draw from.
    max : float
        The right side (exclusive) of the interval to draw from.
    """

    def __init__(self, min: float, max: float):
        self._min = min
        self._max = max

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> float: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray[np.float64]: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]:
        return rng.uniform(self._min, self._max, size=size)

    def __repr__(self) -> str:
        return "distribution.Uniform(%s, %s)" % (repr(self._min), repr(self._max))


class LogNormal(Distribution[float]):
    """
    A log-normal distribution defined by `mu` and `sigma`.

    Note that `mu` and `sigma` are the mean and standard deviation of the
    underlying normal distribution, not of the log-normal distribution itself.

    Attributes
    ----------
    mu : float
        The mean of the underlying normal distribution.
    sigma : float
        The standard deviation of the underlying normal distribution.
    """

    def __init__(self, mu: float, sigma: float):
        self._mu = mu
        self._sigma = sigma

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> float: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray[np.float64]: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]:
        return rng.lognormal(self._mu, self._sigma, size=size)

    def __repr__(self) -> str:
        return "distribution.LogNormal(%s, %s)" % (repr(self._mu), repr(self._sigma))


class LogUniform(Distribution[float]):
    """
    A log-uniform (reciprocal) distribution between `min` and `max`.

    Attributes
    ----------
    min : float
        The minimum value that can be drawn. Must be positive.
    max : float
        The maximum value that can be drawn. Must be positive.
    """

    def __init__(self, min: float, max: float):
        self._min = min
        self._max = max

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> float: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray[np.float64]: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> float | NDArray[np.float64]:
        return np.minimum(
            self._max,
            np.maximum(
                self._min,
                np.exp(rng.uniform(np.log(self._min), np.log(self._max), size=size)),
            ),
        )

    def __repr__(self) -> str:
        return "distribution.LogUniform(%s, %s)" % (repr(self._min), repr(self._max))


class Binary(Distribution[T], Generic[T]):
    """
    A binary (Bernoulli) distribution, which returns `success` with
    probability `p`, and `fail` otherwise.

    Attributes
    ----------
    p : float
        The probability `success` will be returned. Must be between 0 and 1.
    success : T
        The value to return with probabilty `p`.
    fail : T
        The value to return with probabilty ``1-p``.
    """

    def __init__(self, p: float, success: T, fail: T):
        self._p = p
        self._success = success
        self._fail = fail

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> T: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> T | NDArray:
        return np.where(rng.uniform(size=size) < self._p, self._success, self._fail)  # type: ignore

    def __repr__(self) -> str:
        return "distribution.Binary(%s, %s, %s)" % (
            repr(self._p),
            repr(self._success),
            repr(self._fail),
        )


class Discrete(Distribution[int]):
    """
    A uniform discrete distribution, which returns a value between
    `min` (inclusive) and `max` (exclusive).

    Attributes
    ----------
    min : int
        The minimum value that can be drawn. Default 0.
    max : int
        An upper bound (exclusive) to the values which can be drawn.
    """

    @overload
    def __init__(self, max: int): ...
    @overload
    def __init__(self, min: int, max: int): ...

    def __init__(self, a, b=None):
        if b is None:
            self._min = 0
            self._max = a
        else:
            self._min = a
            self._max = b

    @overload
    def draw(self, rng: np.random.Generator, size: None = ...) -> int: ...
    @overload
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...]
    ) -> NDArray[np.int_]: ...
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> int | NDArray[np.int_]:
        return self._min + rng.choice(self._max - self._min, size=size)

    def __repr__(self) -> str:
        return "distribution.Discrete(%s, %s)" % (repr(self._min), repr(self._max))


class DependentDistributionWarning(UserWarning):
    """
    A warning raised when ``CorrelatedDistribution.DependentDistribution.draw()``
    is called unexpectedly. This can occur if ``draw()`` is called from linked
    distributions with different ``size`` parameters, or if ``draw()`` is called
    twice from the same ``DependentDistribution`` without calling ``draw()`` on
    all other linked distributions.
    """

    pass


class CorrelatedDistribution(Distribution[NDArray[Any]], Generic[T]):
    """
    An abstract class which defines a random distribution in multiple,
    correlated variables.

    Subclasses must implement the ``draw()`` function, which draws one or
    more sets of values from the distribution.

    A set of linked, single-variable Distributions can be obtained via
    the ``dependent_distributions()`` function.
    """

    @property
    @abstractmethod
    def num_variables(self) -> int:
        """
        The number of variables this distribution returns.
        """
        pass

    @abstractmethod
    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> NDArray:
        """
        Draws one or more samples from the distribution.

        In this context, one sample refers to a set of correlated values.

        Parameters
        ----------
        rng : np.random.Generator
            A random generator to use to draw samples.
        size : int | tuple[int] | None
            The number of samples or size of the output array.

        Returns
        -------
        NDArray
            One or more samples drawn from the distribution.
            If `size` is ``None``, an array containing a single value for each
            variable should be returned.
            Otherwise, an array of shape ``(size, num_variables)`` should be
            returned, where elements with the same indeces corresponding to
            `size` are correlated.
        """
        pass

    class __DependentDistributionCore(Generic[U]):
        def __init__(self, dist: "CorrelatedDistribution[U]"):
            self._dist = dist
            self._values: NDArray | None = None
            self._should_redraw = np.full(dist.num_variables, True, dtype=np.bool_)
            self._old_size = np.array([0], dtype=np.int_)
            self._dep_dist = tuple(
                [
                    CorrelatedDistribution.DependentDistribution(self, i)
                    for i in range(self._dist.num_variables)
                ]
            )

        @overload
        def draw(self, index: int, rng: np.random.Generator, size: None) -> U: ...
        @overload
        def draw(
            self, index: int, rng: np.random.Generator, size: int | tuple[int, ...]
        ) -> NDArray: ...
        def draw(
            self,
            index: int,
            rng: np.random.Generator,
            size: int | tuple[int, ...] | None,
        ) -> U | NDArray:
            size_tup = (
                (1,) if size is None else ((size,) if isinstance(size, int) else size)
            )
            if (
                self._should_redraw[index]
                or self._values is None
                or len(self._old_size) != len(size_tup)
                or not np.all(self._old_size == np.array(size_tup))
            ):
                if not self._should_redraw[index]:
                    warnings.warn(
                        DependentDistributionWarning(
                            "draw() called on linked dependent distributions with differing size parameters."
                        )
                    )
                elif not np.all(self._should_redraw):
                    warnings.warn(
                        DependentDistributionWarning(
                            "draw() called twice on dependent distribution without drawing other correlated variables."
                        )
                    )
                self._values = self._dist.draw(rng, size_tup)
                self._should_redraw = np.full(
                    self._dist.num_variables, False, dtype=np.bool_
                )
                self._old_size = np.array(size_tup)
            self._should_redraw[index] = True
            size_slice: tuple[slice | int, ...] = (
                *((slice(None),) * len(size_tup)),
                index,
            )
            return (
                self._values[size_slice] if size is not None else self._values[0, index]
            )

        def dependent_distributions(
            self,
        ) -> tuple["CorrelatedDistribution.DependentDistribution[U]", ...]:
            return self._dep_dist

    class DependentDistribution(Distribution[U], Generic[U]):
        def __init__(
            self,
            core: "CorrelatedDistribution.__DependentDistributionCore[U]",
            index: int,
        ):
            self.__core = core
            self.__index = index

        @overload
        def draw(self, rng: np.random.Generator, size: None = ...) -> U: ...
        @overload
        def draw(
            self, rng: np.random.Generator, size: int | tuple[int, ...]
        ) -> NDArray: ...
        def draw(
            self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
        ) -> U | NDArray:
            return self.__core.draw(self.__index, rng, size)

        def __repr__(self) -> str:
            return "%s.dependent_distributions()[%s]" % (
                repr(self.__core._dist),
                repr(self.__index),
            )

        @property
        def dependent_distributions(
            self,
        ) -> tuple["CorrelatedDistribution.DependentDistribution[U]", ...]:
            return self.__core._dep_dist

    def dependent_distributions(
        self,
    ) -> tuple["CorrelatedDistribution.DependentDistribution[T]", ...]:
        """
        Returns a set of linked, single-variable Distributions.

        When ``draw()`` is called on one of these Distributions, values for all
        correlated variables are drawn, but only one variable is returned: the one
        corresponding to the ``Distribution`` on which ``draw()`` was called.
        Afterwards, calling ``draw()`` for the other Distributions in the set will
        return the other drawn values.

        If ``draw()`` is called a second time on one of the Distributions in a set
        before it has been called on all of the other Distributions in the set,
        then previously drawn values will be cleared, this call to ``draw()`` will
        be treated as if it were the first, and a warning will be given.

        Similarly, if ``draw()`` is called with a different value of ``size``
        for two Distributions in a set, then previously drawn values will be cleared,
        and the most recent call to ``draw()`` will be treated as if it were the
        first, and a warning will be given.
        """
        core = CorrelatedDistribution.__DependentDistributionCore(self)
        return core.dependent_distributions()


class FullyCorrelated(CorrelatedDistribution[T], Generic[T]):
    """
    A ``CorrelatedDistribution`` which returns `n` copies of the value drawn from
    a ``Distribution``.

    Attributes
    ----------
    dist : Distribution[T]
        The distribution to draw values from.
    n : int
        The number of variables. All variables will yield the same values as
        one another on a given draw.
    """

    def __init__(self, dist: Distribution[T], n: int):
        self._dist = dist
        self._n = n

    @property
    def num_variables(self) -> int:
        return self._n

    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> NDArray:
        if size is None:
            val = self._dist.draw(rng, size=None)
            return np.full(self._n, val)
        else:
            vals = self._dist.draw(rng, size=size)
            return np.repeat(np.expand_dims(vals, -1), self._n, axis=-1)

    def __repr__(self) -> str:
        return "distribution.FullyCorrelated(%s, %s)" % (repr(self._dist), repr(self._n))


class MatrixCorrelated(CorrelatedDistribution[T], Generic[T]):
    """
    A ``CorrelatedDistribution`` which draws values from one or more distributions,
    then returns variables given by linear combinatons of those values.

    Attributes
    ----------
    matrix : NDArray
        An array with shape ``(num_variables, len(distributions))``.
    distributions : Sequence[Distribution[T]]
        The distributions to draw independent values from.
    """

    def __init__(self, matrix: NDArray, distributions: Sequence[Distribution[T]]):
        self._matrix = matrix
        self._distributions = distributions

    @property
    def num_variables(self) -> int:
        return self._matrix.shape[0]

    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> NDArray:
        i_vals = np.array([d.draw(rng, size=size) for d in self._distributions])
        d_vals = np.tensordot(i_vals, self._matrix, axes=([0], [1]))
        return d_vals

    def __repr__(self) -> str:
        return "distribution.MatrixCorrelated(%s, %s)" % (
            repr(self._matrix),
            repr(self._distributions),
        )


class SphericallyCorrelated(CorrelatedDistribution[float]):
    """
    A ``CorrelatedDistribution`` which returns `n` variables drawn uniformly
    from the surface of an `n`-dimensional hypershphere with the given radius
    (or radius drawn from the given ``Distribution``).

    Attributes
    ----------
    n : int
        The number of variables.
    radius : float | Distribution[float]
        The radius of the hypersphere
        or a distribution from which to draw such radius.
    """

    def __init__(self, n: int, radius: float | Distribution[float] = 1):
        self._radius = radius
        self._n = n

    @property
    def num_variables(self) -> int:
        return self._n

    def draw(
        self, rng: np.random.Generator, size: int | tuple[int, ...] | None = None
    ) -> NDArray:
        if size is None:
            r = (
                self._radius.draw(rng, size=None)
                if isinstance(self._radius, Distribution)
                else self._radius
            )
            x = rng.normal(0, 1, size=self._n)
            x_norm = np.sqrt(np.sum(x**2))
            return r * x / x_norm
        else:
            r_arr = (
                self._radius.draw(rng, size=size)
                if isinstance(self._radius, Distribution)
                else np.full(size, self._radius)
            )
            x_size = ((size,) if isinstance(size, int) else tuple(size)) + (self._n,)
            x = rng.normal(0, 1, size=x_size)
            x_norm = np.expand_dims(np.sqrt(np.sum(x**2, axis=-1)), -1)
            return np.expand_dims(r_arr, -1) * x / x_norm

    def __repr__(self) -> str:
        return "distribution.SphericallyCorrelated(%s, %s)" % (
            repr(self._n),
            repr(self._radius),
        )
