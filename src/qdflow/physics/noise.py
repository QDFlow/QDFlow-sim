"""
This module contains classes and functions for adding noise to CSDs.

The main noise-adding functions are contained within the ``NoiseGenerator`` class.
Each ``NoiseGenerator`` object must be initialized with a ``NoiseParameters``
object which contains parameters defining the strength of each type of noise.

Often, a wide variety of noise types and strengths are desired in a dataset.
This is accomplished by using different ``NoiseGenerator`` objects, each initiated
by a different ``NoiseParameters`` object. Different ``NoiseParameters`` objects
can be obtained with the ``random_noise_params()`` function.

``random_noise_params()`` generates ``NoiseParameters`` based on a set of
Distributions cointained within the ``NoiseRandomizationParameters`` dataclass.
Thus there are two randomization steps:

``NoiseRandomizationParameters`` metaparameters -> ``NoiseParameters`` parameters
-> specific noise realization.

Examples
--------

>>> from qdflow.physics import noise
>>> from qdflow.util import distribution
>>> meta_params = noise.NoiseRandomizationParameters.default()
>>> meta_params.unint_dot_mag = distribution.Uniform(0,.05) # adjust meta_params here

>>> noise_params_1 = random_noise_params(meta_params)
>>> noise_params_2 = random_noise_params(meta_params)
>>> noise_gen_1 = NoiseGenerator(noise_params_1)
>>> noise_gen_2 = NoiseGenerator(noise_params_2)

>>> csd = np.load("csd_data.npy")
>>> noisy_csd_1a = noise_gen_1.calc_noisy_map(csd)
>>> noisy_csd_1b = noise_gen_1.calc_noisy_map(csd)
>>> noisy_csd_2 = noise_gen_2.calc_noisy_map(csd)

Here ``noisy_csd_1a`` and ``noisy_csd_1b`` will look very similar, since they
both have the same white noise strength, the same pink noise strength, the same
amount of latching noise, etc. They will not be exactly the same, as the exact
noise realizations will be different in each case.

However, ``noisy_csd_2`` will (likely) look significantly different, since it
is generated with a completely different white noise strength, pink noise strength,
amount of latching noise, etc.
"""

import numpy as np
import scipy  # type: ignore[import-untyped]
from numpy.typing import NDArray
from typing import Any, Self, TypeVar
import scipy.ndimage  # type: ignore[import-untyped]
import dataclasses
from dataclasses import dataclass, field
import copy

# import .util.distribution as distribution
from ..util.distribution import Distribution, LogNormal, LogUniform, Uniform, Normal
from .simulation import ThomasFermi, ThomasFermiOutput, is_transition
# from physics import simulation

T = TypeVar("T")


_rng:np.random.Generator = np.random.default_rng()


def set_rng_seed(seed):
    """
    Initializes a new random number generator with the given seed,
    used to generate random data.

    Parameters
    ----------
    seed : {int, array_like[int], SeedSequence, BitGenerator, Generator}
        The seed to use to initialize the random number generator.
    """
    global _rng
    _rng = np.random.default_rng(seed)


@dataclass(kw_only=True)
class NoiseParameters:
    """
    Set of parameters used to describe the various types and strengths of noise.

    Attributes
    ----------
    white_noise_magnitude : float
        Magnitude of the white noise to add to the data. The noise at each pixel
        is drawn from a Gaussian distribution with standard deviation `white_noise_magnitude`.
    pink_noise_magnitude : float
        Magnitude of the pink noise to add to the data. The noise at each pixel
        will have standard deviation `pink_noise_magnitude`, but will have 1/f correlation.
    telegraph_magnitude, telegraph_stdev : float
        The magnitude and standard deviation of the telegraph noise to add to the data.
        Each jump will add or subtract a constant drawn from a normal distribution
        with mean ``telegraph_magnitude/2`` and standard deviation ``telegraph_stdev/sqrt(2)``.
        This means that the total jump distance will have mean and standard
        deviation given by `telegraph_magnitude` and `telegraph_stdev`.
    telegraph_low_pixels, telegraph_high_pixels : float
        The average number of pixels before a jump from low to high (`telegraph_low_pixels`)
        or from high to low (`telegraph_high_pixels`) in the telegraph noise.
        Must be greater than or equal to 1.
    noise_axis : int
        The axis along which to add telegraph noise, latching, and sech blur.
    latching_pixels : float
        The average number of pixels by which to shift each line when applying
        latching noise.
    latching_positive : bool
        Whether to shift in the positive or negative direction when applying
        latching noise.
    sech_blur_width : float
        The width in pixels of the sech^2 blur.
    unint_dot_mag : float
        The strength of the unintended dot effects.
    unint_dot_spacing : ndarray[float] | None
        A vector (with length equal to the number of gates)
        normal to the unitended dot transition, with magnitude
        equal to the spacing between transitions.
        If ``None``, no unitended dot peaks will be applied.
    unint_dot_width : float
        The width of the unitended dot peaks.
    uint_dot_offset : float
        A value between 0 and 1 which defines by how much each unintended dot peak
        should be offset, relative to the norm of `unint_dot_spacing`.
    coulomb_peak_width : float | None
        The width of the sech^2 curve for applying coulomb peak effects.
        If ``None``, no Coulomb peak effects will be applied.
    coulomb_peak_offset : float
        A value between 0 and 1 which defines by how much each sech^2 peak
        should be offset applying coulomb peak effects, relative to the norm
        of `coulomb_peak_spacing`.
    coulomb_peak_spacing : float
        A value which determines how far apart each sech^2 peak should be
        when applying coulomb peak effects.
    sensor_gate_coupling : ndarray[float] | None
        A vector with length equal to the number of gates, giving the value
        of the sensor-gate coupling per pixel for each gate.
        If ``None``, no sensor-gate coupling will be applied.
    use_pink_noise_all_dims : bool
        Whether pink noise should be correlated in all dimensions (``True``), or
        only along ``noise_axis`` (``False``).
    """

    white_noise_magnitude: float = 0.0
    pink_noise_magnitude: float = 0.0
    telegraph_magnitude: float = 0.0
    telegraph_stdev: float = 0.0
    telegraph_low_pixels: float = 1.0
    telegraph_high_pixels: float = 1.0
    noise_axis: int = 0
    latching_pixels: float = 0.0
    latching_positive: bool = True
    sech_blur_width: float = 0.0
    unint_dot_magnitude: float = 0.0
    unint_dot_spacing: NDArray[np.float64] | None = None
    unint_dot_width: float = 0.0
    unint_dot_offset: float = 0.0
    coulomb_peak_spacing: float = 1.0
    coulomb_peak_offset: float = 0.0
    coulomb_peak_width: float | None = None
    sensor_gate_coupling: NDArray[np.float64] | None = None
    use_pink_noise_all_dims: bool = True

    def _get_unint_dot_spacing(self) -> NDArray[np.float64] | None:
        return self._unint_dot_spacing

    def _set_unint_dot_spacing(self, val: NDArray[np.float64] | None):
        self._unint_dot_spacing = (
            np.array(val, dtype=np.float64) if val is not None else None
        )

    def _get_sensor_gate_coupling(self) -> NDArray[np.float64] | None:
        return self._sensor_gate_coupling

    def _set_sensor_gate_coupling(self, val: NDArray[np.float64] | None):
        self._sensor_gate_coupling = (
            np.array(val, dtype=np.float64) if val is not None else None
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``NoiseParameters`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.
            Default values are set for keys not included in the dict.

        Returns
        -------
        NoiseParameters
            A new ``NoiseParameters`` object with the values specified by ``dict``.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``NoiseParameters`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``NoiseParameters`` object.
        """
        return dataclasses.asdict(self)

    def copy(self) -> Self:
        """
        Creates a copy of a ``NoiseParameters`` object.

        Returns
        -------
        NoiseParameters
            A new ``NoiseParameters`` object with the same attribute values as ``self``.
        """
        return dataclasses.replace(self)


NoiseParameters.sensor_gate_coupling = property(
    NoiseParameters._get_sensor_gate_coupling, NoiseParameters._set_sensor_gate_coupling
)  # type: ignore
NoiseParameters.unint_dot_spacing = property(
    NoiseParameters._get_unint_dot_spacing, NoiseParameters._set_unint_dot_spacing
)  # type: ignore


class NoiseGenerator:
    """
    Adds noise and other postprocessing to simulated quantum dot devices.

    The following types of noise are supported:
    White noise, Pink (1/f) noise, Telegraph noise, Latching, Coulomb peak,
    Sech blur, Unintended dot, Sensor-gate coupling.

    When initiating this class, a NoiseParameters object should be provided
    to define the strengths of each of the noise types.

    Parameters
    ----------
    noise_parameters : NoiseParameters | dict[str, Any]
        ``NoiseParameters`` object or dictionary that gives the parameters
        used to generate noise.
    rng : np.random.Generator
        Random number generator used to generate noise.
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters | dict[str, Any],
        rng: np.random.Generator | None = None,
    ):
        self.noise_parameters = (
            NoiseParameters.from_dict(noise_parameters)
            if isinstance(noise_parameters, dict)
            else noise_parameters.copy()
        )
        self.rng = rng if rng is not None else _rng

    def coulomb_peak(
        self, data_map: NDArray[np.float64], peak_center: float, peak_width: float
    ) -> NDArray[np.float64]:
        """
        Calculate sensor value from potential using a single sech^2 lineshape,
        which is valid in the weak coupling regime of dot.

        .. deprecated::
            Prefer ``high_coupling_coulomb_peak()`` instead.

        See: Beenakker, Phys. Rev. B 44, 1646.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to transform.
        peak_center, peak_width : float
            Parameters defining the shape of the sech function as follows:
            ``sech((data_map-peak_center) / peak_width) ** 2``

        Returns
        -------
        ndarray[float]
            `data_map` with sech^2 transformation applied.
        """
        return 1 / np.cosh((data_map - peak_center) / peak_width) ** 2

    def high_coupling_coulomb_peak(
        self,
        data_map: NDArray[np.float64],
        peak_offset: float,
        peak_width: float,
        peak_spacing: float,
    ) -> NDArray[np.float64]:
        """
        Calculate sensor value from potential using a series of sech^2 functions.

        Specifically, returns a sum over ``i`` of the following:
        ``sech((data_map - (i + peak_offset) * peak_spacing)) / peak_width) ** 2``

        See: Beenakker, Phys. Rev. B 44, 1646.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to transform.
        peak_offset : float
            A value between 0 and 1 which defines by how much each sech^2
            function should be offset, relative to `peak_spacing`.
        peak_width : float
            The width of the sech^2 functions.
        peak_spacing : float
            A value which determines how far apart each sech^2 peak should be.

        Returns
        -------
        ndarray[float]
            `data_map` with coulomb peak transformation applied.
        """

        pmax = int(np.ceil(np.max(data_map) / peak_spacing - peak_offset) + 1)
        pmin = int(np.floor(np.min(data_map) / peak_spacing - peak_offset) - 1)
        output = np.zeros(data_map.shape)
        for p_i in range(pmin, pmax + 1):
            output += (
                1
                / np.cosh((data_map - (p_i + peak_offset) * peak_spacing) / peak_width)
                ** 2
            )
        return output

    def white_noise(
        self, data_map: NDArray[np.float64], magnitude: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Adds white noise to `data_map`.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add noise to.
        magnitude : float | ndarray[float]
            The standard deviation of the gaussian distribution from which to
            draw the noise at each pixel.
            If an array is passed, it should have the same shape as `data_map`.

        Returns
        -------
        ndarray[float]
            `data_map` with white noise added to it.
        """
        return data_map + self.rng.normal(0, magnitude, data_map.shape)

    def pink_noise(
        self, data_map: NDArray[np.float64], magnitude: float | NDArray[np.float64],
        axis: int | None = None
    ) -> NDArray[np.float64]:
        """
        Adds pink (1/f) noise to `data_map`.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add noise to.
        magnitude : float | ndarray[float]
            The standard deviation of the noise at each pixel. Note that pink
            noise is self-correlated.
            If an array is passed, it should have the same shape as `data_map`.
        axis : int
            Which axis the pink noise should correlated in, or ``None`` if the
            pink noise should be correlated in all directions.
            
        Returns
        -------
        ndarray[float]
            `data_map` with pink noise added to it.
        """
        phases = self.rng.uniform(0, 2 * np.pi, data_map.shape)
        magnitudes = self.rng.normal(0, 1, data_map.shape)
        if axis is None:
            sq_list = [
                (np.minimum(np.arange(0, l), np.arange(l, 0, -1))) ** 2
                for l in data_map.shape
            ]
            f_factor = sq_list[0]
            for sql in sq_list[1:]:
                f_factor = np.add.outer(f_factor, sql)
            np.put(f_factor, [0] * len(data_map.shape), 1)
            f_factor = 1 / np.sqrt(f_factor)
            np.put(f_factor, [0] * len(data_map.shape), 0)
            f_factor_scale = np.sqrt(np.sum(f_factor**2))
            pink_noise = (
                np.real(np.fft.fftn(magnitudes * np.exp(phases * 1j) * f_factor))
                * np.sqrt(2) / f_factor_scale
            )
        else:
            l = data_map.shape[axis]
            f_factor = np.minimum(np.arange(0, l), np.arange(l, 0, -1))
            np.put(f_factor, 0, 1)
            f_factor = 1 / np.sqrt(f_factor)
            np.put(f_factor, 0, 0)
            ex_axes = tuple(list(range(0,axis))+list(range(axis+1,len(data_map.shape))))
            f_factor = np.expand_dims(f_factor, ex_axes)
            f_factor_scale = np.sqrt(np.sum(f_factor**2))
            pink_noise = (
                np.real(np.fft.fft(magnitudes * np.exp(phases * 1j) * f_factor, axis=axis))
                * np.sqrt(2) / f_factor_scale
            )
        return data_map + magnitude * pink_noise

    def telegraph_noise(
        self,
        data_map: NDArray[np.float64],
        magnitude: float | NDArray[np.float64],
        stdev: float | NDArray[np.float64],
        ave_low_pixels: float,
        ave_high_pixels: float,
        axis: int,
    ) -> NDArray[np.float64]:
        """
        Adds  `telegraph noise <en.wikipedia.org/wiki/burst_noise>`_ to `data_map`.

        Specifically, adds a constant value to a line of several continuous pixels,
        Then adds a different constant value to the next several pixels, etc.
        The number of pixels before each jump is drawn from a geometric distribution
        with mean given by `ave_low_pixels` or `ave_high_pixels`,
        and the constant value added is drawn from a normal distribution with
        mean +/- ``magnitude/2`` and standard deviation ``stdev/sqrt(2)``,
        with the sign alternating after each jump.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add noise to.
        magnitude, stdev : float | ndarray[float]
            The average magnitude and standard deviation of the telegraph noise.
            Each jump will add or subtract a constant drawn from a normal distribution
            with mean ``magnitude/2`` and standard deviation ``stdev/sqrt(2)``.
            This means that the total jump distance will have mean and standard
            deviation given by `magnitude` and `stdev`.
            If arrays are passed, they should have the same shape as `data_map`.
        ave_low_pixels, ave_high_pixels : float
            The average number of pixels before a jump from low to high (`ave_low_pixels`)
            or from high to low (`ave_high_pixels`). Must be greater than or equal to 1.
        axis : int
            Which axis the telegraph noise should be applied along.

        Returns
        -------
        ndarray[float]
            `data_map` with telegraph noise added to it.
        """
        output = np.array(data_map)
        low_p = 1 / max(ave_low_pixels, 1)
        high_p = 1 / max(ave_high_pixels, 1)
        ax_len:int = data_map.shape[axis]
        non_axis_shape = tuple(data_map.shape[:axis]) + tuple(
            data_map.shape[axis + 1 :]
        )
        start_low:NDArray[np.bool_] = self.rng.uniform(0,1,size=non_axis_shape) \
                          < ave_low_pixels / (ave_low_pixels + ave_high_pixels)
        for ind in np.ndindex(non_axis_shape):
            sd = (
                stdev[tuple(ind[:axis]) + (slice(None),) + tuple(ind[axis:])]
                if isinstance(stdev, np.ndarray)
                else stdev
            ) / np.sqrt(2)
            mag = (
                magnitude[tuple(ind[:axis]) + (slice(None),) + tuple(ind[axis:])]
                if isinstance(magnitude, np.ndarray)
                else magnitude
            ) / 2
            rand_arr = self.rng.uniform(0, 1, size=ax_len)
            norm_arr = self.rng.normal(0, sd, ax_len)
            norm_start = self.rng.normal(0, (sd[0] if isinstance(sd, np.ndarray) else sd))
            low_jump = rand_arr < low_p
            high_jump = rand_arr < high_p
            is_low = bool(start_low[ind])
            noise = np.zeros(ax_len)
            current_val = (-1 if is_low else 1) * (
                mag[0] if isinstance(mag, np.ndarray) else mag
            ) + norm_start
            for i in range(ax_len):
                noise[i] = current_val
                if is_low and low_jump[i]:
                    is_low = False
                    current_val = (
                        mag[i] if isinstance(mag, np.ndarray) else mag
                    ) + norm_arr[i]
                elif not is_low and high_jump[i]:
                    is_low = True
                    current_val = (
                        -(mag[i] if isinstance(mag, np.ndarray) else mag) + norm_arr[i]
                    )
            output[tuple(ind[:axis]) + (slice(None),) + tuple(ind[axis:])] += noise
        return output

    def line_shift(
        self,
        data_map: NDArray[np.float64],
        ave_pixels: float,
        axis: int,
        shift_positive: bool = True,
    ) -> NDArray[np.float64]:
        """
        Mimics latching effects by shifting each line in `data_map` by a random
        number of pixels along the direction of the line.

        Specifically, ``line_shift_output[i_0, i_1, ... , i_axis, ...]`` is given by
        ``data_map[i_0, i_1, ... , (i_axis - shift[i_0, ...]), ...]``,
        where ``shift[i_0, ...] + 1`` is drawn from a geometric distribution
        with mean ``ave_pixels + 1``, and does not depend on ``i_axis``.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to shift. Must be at least 2d.
        ave_pixels : float
            The average number of pixels by which to shift each line.
            Specifically, each line is shifted by ``x-1`` pixels, where ``x`` is
            drawn from a geometric distribution with mean ``ave_pixels + 1``.
        axis : int
            Which axis to shift lines along. Each line running parallel to `axis`
            will be shifted by some amount in the direction parallel to `axis`.
        shift_positive : bool
            Whether to shift in the positive or negative direction.
            If ``True``, each line will be shifted towards positive infinity.

        Returns
        -------
        ndarray[float]
            `data_map` with each line randomly shifted by some amount.
        """
        if len(data_map.shape) <= 1:
            return np.array(data_map)
        transpose_axes = list(range(len(data_map.shape)))
        transpose_axes[0] = axis
        transpose_axes[axis] = 0
        data_map_t = np.array(np.transpose(data_map, axes=transpose_axes))

        shift_all = np.minimum(self.rng.geometric(1 / (ave_pixels + 1), data_map_t.shape[1:]) - 1, data_map_t.shape[0])

        if shift_positive:
            for ind, shift in np.ndenumerate(shift_all):
                if shift != 0:
                    delta = (
                        data_map_t[(1,) + tuple(ind)] - data_map_t[(0,) + tuple(ind)]
                    )
                    d0 = data_map_t[(0,) + tuple(ind)]
                    data_map_t[(slice(shift, None),) + tuple(ind)] = data_map_t[
                        (slice(None, -shift),) + tuple(ind)
                    ]
                    data_map_t[(slice(None, shift),) + tuple(ind)] = np.linspace(
                        d0 - shift * delta, d0, shift, endpoint=False
                    )
        else:
            for ind, shift in np.ndenumerate(shift_all):
                if shift != 0:
                    delta = (
                        data_map_t[(-1,) + tuple(ind)] - data_map_t[(-2,) + tuple(ind)]
                    )
                    d0 = data_map_t[(-1,) + tuple(ind)]
                    data_map_t[(slice(None, -shift),) + tuple(ind)] = data_map_t[
                        (slice(shift, None),) + tuple(ind)
                    ]
                    data_map_t[(slice(-shift, None),) + tuple(ind)] = np.linspace(
                        d0 + delta, d0 + shift * delta, shift, endpoint=True
                    )

        return np.transpose(data_map_t, axes=transpose_axes)

    def latching_noise(
        self,
        data_map: NDArray[np.float64],
        excited_data: NDArray[np.float64],
        dot_charges: NDArray[np.int_],
        are_dots_combined: NDArray[np.bool_],
        ave_pixels: float,
        axis: int,
        shift_positive: bool = True,
    ) -> NDArray[np.float64]:
        """
        Adds latching effects by selecting data from ``excited_data`` for a few
        pixels after each transition.

        The number of pixels ``shift`` is determined by drawing from a geometric
        distribution with mean ``ave_pixels + 1`` and subtracting 1 from this
        result.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add latching noise to.
        excited_data : ndarray[float]
            An array with the same shape as ``data_map`` giving the sensor readout
            (or whatever data is being plotted) for an excited state.
            The excited state should be the previous stable charge state along the
            measurement axis.
        dot_charges : ndarray[int]
            An array with shape ``(*(data_map.shape), n_dots)`` giving the charge
            state for each dot at each pixel.
        are_dots_combined : ndarray[bool]
            An array with shape ``(*(data_map.shape), n_dots-1)`` which indicates
            whether the dots on each side of a barrier are combined together.
        ave_pixels : float
            The average number of pixels by which to shift each line.
            Specifically, each line is shifted by ``x-1`` pixels, where ``x`` is
            drawn from a geometric distribution with mean ``ave_pixels + 1``.
        axis : int
            Which axis to shift lines along. Each line running parallel to `axis`
            will be shifted by some amount in the direction parallel to `axis`.
        shift_positive : bool
            Whether to shift in the positive or negative direction.
            If ``True``, each line will be shifted towards positive infinity.

        Returns
        -------
        ndarray[float]
            `data_map` with several pixels after each transition selected from
            `excited_data`.
        """
        if len(data_map.shape) <= 1:
            return np.array(data_map)
        transpose_axes = list(range(len(data_map.shape)))
        transpose_axes[0] = axis
        transpose_axes[axis] = 0
        data_map_t = np.array(np.transpose(data_map, axes=transpose_axes))
        excited_data_t = np.transpose(excited_data, axes=transpose_axes)
        dot_charges_t = np.transpose(
            dot_charges, axes=(transpose_axes + [len(data_map.shape)])
        )
        are_dots_combined_t = np.transpose(
            are_dots_combined, axes=(transpose_axes + [len(data_map.shape)])
        )

        if not shift_positive:
            data_map_t = np.flip(data_map_t, axis=0)
            excited_data_t = np.flip(excited_data_t, axis=0)
            dot_charges_t = np.flip(dot_charges_t, axis=0)
            are_dots_combined_t = np.flip(are_dots_combined_t, axis=0)

        shift = self.rng.geometric(1 / (ave_pixels + 1), data_map_t.shape) - 1
        x_max = data_map_t.shape[0]

        for ind in np.ndindex(data_map_t.shape):
            if ind[0] > 0:
                if np.any(
                    is_transition(
                        dot_charges_t[ind],
                        are_dots_combined_t[ind],
                        dot_charges_t[(ind[0] - 1, *(ind[1:]))],
                        are_dots_combined_t[(ind[0] - 1, *(ind[1:]))],
                    )[0]
                ):
                    i_sh = (slice(ind[0], min(ind[0] + shift[ind], x_max)), *(ind[1:]))
                    data_map_t[i_sh] = excited_data_t[i_sh]  # type: ignore

        if not shift_positive:
            data_map_t = np.flip(data_map_t, axis=0)
        return np.transpose(data_map_t, axes=transpose_axes)

    def unint_dot_add(
        self,
        data_map: NDArray[np.float64],
        magnitude: float | NDArray[np.float64],
        spacing: NDArray[np.float64],
        width: float,
        offset: float,
        gate_data_matrix: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Add a series of transitions with quantum dot lineshapes to data.

        Specifically, for each pixel with coordinates ``x``, adds the following:
        ``tanh((dot(x, spacing)/|spacing| - (i + offset)*|spacing|) / width)``
        summed over integers ``i`` in range.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add noise to.
        magnitude : float | ndarray[float]
            The strength of the unintended dot effects.
            If an array is passed, it should have the same shape as `data_map`.
        spacing : ndarray[float]
            A vector (with length equal to the number of gates)
            normal to the unitended dot transition, with magnitude
            equal to the spacing between transitions.
        width : float
            The width in pixels of the unitended dot peaks.
        offset : float
            A value between 0 and 1 which defines by how much each unintended dot peak
            should be offset, relative to the norm of `unint_dot_spacing`.
        gate_data_matrix : ndarray[float] | None
            A matrix with shape ``(n_gates, len(data_map.shape))`` that indicates
            how each of the gates changes as one of the axes of `data_map` changes.
            By default, an identity matrix will be used -- this assumes
            ``len(spacing) == len(data_map.shape)``.

        Returns
        -------
        ndarray[float]
            `data_map` with unintended dot effects added.
        """
        spc = np.sqrt(np.sum(spacing**2))
        gdm = (
            gate_data_matrix
            if gate_data_matrix is not None
            else np.identity(len(data_map.shape))
        )
        phi_list = [
            np.dot(spacing, gdm[:, l]) * np.arange(data_map.shape[l]) / spc
            for l in range(len(data_map.shape))
        ]
        phi = phi_list[0]
        for pl in phi_list[1:]:
            phi = np.add.outer(phi, pl)
        pmax = int(np.ceil(np.max(phi) / spc - offset) + 1)
        pmin = int(np.floor(np.min(phi) / spc - offset) - 1)
        noise = np.zeros(data_map.shape)
        for p_i in range(pmin, pmax + 1):
            noise += np.tanh((phi - (p_i + offset) * spc) / width)
        return data_map + magnitude * noise

    def sech_blur(
        self, data_map: NDArray[np.float64], blur_width: float, noise_axis: int
    ) -> NDArray[np.float64]:
        """
        Blurs `datamap` by convoluting with sech^2 kernel.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to blur.
        blur_width : float
            The width in pixels of the blur.
            `data_map` is convolved with ``np.cosh(x/blur_width)**-2``

        Returns
        -------
        ndarray[float]
            `data_map` with sech blur applied.
        """
        conv_max = max(int(np.ceil(blur_width * 3)), 3)
        conv_elems = 2 * conv_max + 1
        conv = np.cosh(np.linspace(-conv_max, conv_max, conv_elems) / blur_width) ** -2
        conv = conv / np.sum(conv)
        if noise_axis >= len(data_map.shape):
            raise ValueError(
                "noise_axis must be less than the number of dimensions of data_map"
            )
        new_dims = list(range(0, noise_axis)) + list(
            range(noise_axis + 1, len(data_map.shape))
        )
        conv = np.expand_dims(conv, tuple(new_dims))
        return scipy.ndimage.convolve(data_map, conv, mode="nearest")

    def sensor_gate(
        self,
        data_map: NDArray[np.float64],
        sensor_gate_coupling: NDArray[np.float64],
        magnitude: float | NDArray[np.float64] = 1,
        gate_data_matrix: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        Add a gradient due to sensor-gate coupling to data.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to add noise to.
        sensor_gate_coupling : ndarray[float]
            A vector with length equal to the number of gates, giving the value
            of the sensor-gate coupling per pixel for each gate.
        magnitude : float | ndarray[float]
            The total sensor-gate coupling will be multiplied by `magnitude`.
            Usually, this value should be set to 1, and the magnitude encoded in
            `sensor_gate_coupling`.
            If an array is passed, it should have the same shape as `data_map`.
        gate_data_matrix : ndarray[float] | None
            A matrix with shape ``(n_gates, len(data_map.shape))`` that indicates
            how each of the gates changes as one of the axes of `data_map` changes.
            By default, an identity matrix will be used -- this assumes
            ``len(sensor_gate_coupling) == len(data_map.shape)``.

        Returns
        -------
        ndarray[float]
            `data_map` with sensor-gate coupling effects added.
        """
        gdm = (
            gate_data_matrix
            if gate_data_matrix is not None
            else np.identity(len(data_map.shape))
        )
        phi_list = [
            np.dot(sensor_gate_coupling, gdm[:, l]) * np.arange(data_map.shape[l])
            for l in range(len(data_map.shape))
        ]
        phi = phi_list[0]
        for pl in phi_list[1:]:
            phi = np.add.outer(phi, pl)
        return data_map + magnitude * phi

    def calc_noisy_map(
        self,
        data_map: NDArray[np.float64],
        latching_data: None
        | tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.bool_]] = None,
        gate_data_matrix: None | NDArray[np.float64] = None,
        *,
        noise_default=True,
        white_noise: bool | None = None,
        pink_noise: bool | None = None,
        coulomb_peak: bool | None = None,
        telegraph_noise: bool | None = None,
        latching: bool | None = None,
        unintended_dot: bool | None = None,
        sech_blur: bool | None = None,
        sensor_gate: bool | None = None,
    ) -> NDArray[np.float64]:
        """
        Adds noise to `data_map` with parameters specified by ``self.noiseParameters``.

        Parameters
        ----------
        data_map : ndarray[float]
            The data to apply noise to.
        latching_data : None or tuple[ndarray[float], ndarray[int], ndarray[bool]]
            Additional data used to add realistic latching effects.
            If this parameter is ``None``, latching will be simulated by shifting
            each line of `data_map` by a random amount.

            Alternatively, a tuple ``(excited_data, dot_charge, are_dots_combined)``
            can be supplied. Here ``dot_charge`` and ``are_dots_combined`` should
            give the charge state of the system at each pixel.
            ``excited_data`` should give the sensor readout (or whatever data
            `data_map` represents) for an excited state at each pixel.
            The excited state should be whichever the previous charge state was
            before the most recent transition.
        gate_data_matrix : ndarray[float] | None
            A matrix with shape ``(n_gates, len(data_map.shape))`` that indicates
            how each of the gates changes as one of the axes of `data_map` changes.
            By default, an identity matrix will be used -- this assumes
            ``len(unint_dot_spacing) == len(data_map.shape)`` and
            ``len(sensor_gate_coupling) == len(data_map.shape)``.
        noise_default : bool
            Whether to include all noise types (True) or no noise types (False)
            by default. Individual noise types can then be turned on or off
            via the parameters `white_noise`, `pink_noise`, etc.
        white_noise : bool | None
            Whether to include white noise.
        pink_noise : bool | None
            Whether to include pink noise.
        coulomb_peak : bool | None
            Whether to include coulomb peak effects.
        telegraph_noise : bool | None
            Whether to include telegraph noise.
        latching : bool | None
            Whether to include latching effects.
        unintended_dot : bool | None
            Whether to include unintended dot effects.
        sech_blur : bool | None
            Whether to include sech blur effects.
        sensor_gate : bool | None
            Whether to include sensor-gate coupling.

        Returns
        -------
        ndarray[float]
            `data_map` with various noise types added.
        """
        param = self.noise_parameters
        noisy_map = np.array(data_map)
        lt = latching if latching is not None else noise_default
        ud = unintended_dot if unintended_dot is not None else noise_default
        sb = sech_blur if sech_blur is not None else noise_default
        sg = sensor_gate if sensor_gate is not None else noise_default
        cp = coulomb_peak if coulomb_peak is not None else noise_default
        wn = white_noise if white_noise is not None else noise_default
        pn = pink_noise if pink_noise is not None else noise_default
        tn = telegraph_noise if telegraph_noise is not None else noise_default
        if lt:
            if latching_data is None:
                noisy_map = self.line_shift(
                    noisy_map,
                    param.latching_pixels,
                    param.noise_axis,
                    param.latching_positive,
                )
            else:
                noisy_map = self.latching_noise(
                    noisy_map,
                    latching_data[0],
                    latching_data[1],
                    latching_data[2],
                    param.latching_pixels,
                    param.noise_axis,
                    param.latching_positive,
                )
        if ud and param.unint_dot_spacing is not None:
            noisy_map = self.unint_dot_add(
                noisy_map,
                param.unint_dot_magnitude,
                param.unint_dot_spacing,
                param.unint_dot_width,
                param.unint_dot_offset,
                gate_data_matrix,
            )
        if sb:
            noisy_map = self.sech_blur(
                noisy_map, param.sech_blur_width, param.noise_axis
            )
        sgc = param.sensor_gate_coupling
        if sg and sgc is not None:
            noisy_map = self.sensor_gate(noisy_map, sgc, 1, gate_data_matrix)
        if wn:
            noisy_map = self.white_noise(noisy_map, param.white_noise_magnitude)
        if pn:
            if param.use_pink_noise_all_dims:
                noisy_map = self.pink_noise(noisy_map, param.pink_noise_magnitude, axis=None)
            else:
                noisy_map = self.pink_noise(noisy_map, param.pink_noise_magnitude, axis=param.noise_axis)
        if tn:
            noisy_map = self.telegraph_noise(
                noisy_map,
                param.telegraph_magnitude,
                param.telegraph_stdev,
                param.telegraph_low_pixels,
                param.telegraph_high_pixels,
                param.noise_axis,
            )
        if cp and param.coulomb_peak_width is not None:
            noisy_map = self.high_coupling_coulomb_peak(
                noisy_map,
                param.coulomb_peak_offset,
                param.coulomb_peak_width,
                param.coulomb_peak_spacing,
            )
        return noisy_map


@dataclass(kw_only=True)
class NoiseRandomization:
    """
    Meta-parameters used to determine how random ``NoiseParameters`` should
    be generated.

    All attributes should either be provided a single value
    (if no randmization is needed), or a ``randomize.Distribution`` object,
    from which the value will be drawn.

    Attributes
    ----------
    noise_axis : int
        The axis along which to add telegraph noise, latching, and sech blur.
    white_noise_magnitude : float | Distribution[float]
        Magnitude of the white noise to add to the data. The noise at each pixel
        is drawn from a Gaussian distribution with standard deviation `white_noise_magnitude`.
    pink_noise_magnitude : float | Distribution[float]
        Magnitude of the pink noise to add to the data. The noise at each pixel
        will have standard deviation `pink_noise_magnitude`, but will have 1/f correlation.
    telegraph_magnitude, telegraph_relative_stdev : float | Distribution[float]
        The magnitude and standard deviation of the telegraph noise to add to the data.
        Each jump will add or subtract a constant drawn from a normal distribution
        with mean ``telegraph_magnitude/2`` and standard deviation ``telegraph_stdev/sqrt(2)``.
        This means that the total jump distance will have mean and standard
        deviation given by `telegraph_magnitude` and ``telegraph_stdev``.
        ``telegraph_stdev`` is found by multiplying ``telegraph_relative_stdev``
        by `telegraph_magnitude`.
    telegraph_low_pixels, telegraph_high_pixels : float | Distribution[float]
        The average number of pixels before a jump from low to high (`telegraph_low_pixels`)
        or from high to low (`telegraph_high_pixels`) in the telegraph noise.
        Must be greater than or equal to 1.
    latching_pixels : float | Distribution[float]
        The average number of pixels by which to shift each line when applying
        latching noise.
    latching_positive : bool | Distribution[bool]
        Whether to shift in the positive or negative direction when applying
        latching noise.
    sech_blur_width : float | Distribution[float]
        The width in pixels of the sech^2 blur.
    unint_dot_mag : float
        The strength of the unintended dot effects.
    unint_dot_spacing : ndarray[float] | Distribution[float] | Distribution[ndarray] | None
        A vector (with length equal to the number of gates)
        normal to the unitended dot transition, with magnitude
        equal to the spacing between transitions.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
        If ``None``, no unitended dot peaks will be applied.
    unint_dot_width : float | Distribution[float]
        The width of the unitended dot peaks.
    uint_dot_offset : float | Distribution[float]
        A value between 0 and 1 which defines by how much each unintended dot peak
        should be offset, relative to the norm of `unint_dot_spacing`.
    coulomb_peak_width : float | Distribution[float] | None
        The width of the sech^2 curve for applying coulomb peak effects.
        If ``None``, no Coulomb peak effects will be applied.
    coulomb_peak_offset : float | Distribution[float]
        A value between 0 and 1 which defines by how much each sech^2 peak
        should be offset applying coulomb peak effects, relative to the norm
        of `coulomb_peak_spacing`.
    coulomb_peak_spacing : float | Distribution[float]
        A value which determines how far apart each sech^2 peak should be
        when applying coulomb peak effects.
    sensor_gate_coupling : ndarray[float] | Distribution[float] | Distribution[ndarray] | None
        A vector with length equal to the number of gates, giving the value
        of the sensor-gate coupling per pixel for each gate.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
        If ``None``, no sensor-gate coupling will be applied.
    use_pink_noise_all_dims : bool
        Whether pink noise should be correlated in all dimensions (``True``), or
        only along ``noise_axis`` (``False``).
    """

    n_gates: int = 2
    noise_axis: int = 0
    use_pink_noise_all_dims: bool = True

    latching_positive: bool | Distribution[bool] = True
    white_noise_magnitude: float | Distribution[float] = 0.0
    pink_noise_magnitude: float | Distribution[float] = 0.0
    telegraph_magnitude: float | Distribution[float] = 0.0
    telegraph_relative_stdev: float | Distribution[float] = 0.0
    telegraph_low_pixels: float | Distribution[float] = 1.0
    telegraph_high_pixels: float | Distribution[float] = 1.0
    latching_pixels: float | Distribution[float] = 0.0
    sech_blur_width: float | Distribution[float] = 0.0
    unint_dot_magnitude: float | Distribution[float] = 0.0
    unint_dot_spacing: (
        NDArray[np.float64]
        | Distribution[float]
        | Distribution[NDArray[np.float64]]
        | None
    ) = None
    unint_dot_offset: float | Distribution[float] = 0.0
    unint_dot_relative_width: float | Distribution[float] = 0.0
    coulomb_peak_offset: float | Distribution[float] = 0.0
    coulomb_peak_width: float | Distribution[float] | None = None
    coulomb_peak_spacing: float | Distribution[float] = 1.0
    sensor_gate_coupling: (
        NDArray[np.float64]
        | Distribution[float]
        | Distribution[NDArray[np.float64]]
        | None
    ) = None

    @classmethod
    def default(cls, q_positive: bool = False) -> Self:
        """
        Creates a new ``NoiseRandomization`` object with default values.

        Parameters
        ----------
        q_positive : bool
            ``True`` if ``phyics_parameters.q`` is positive, ``False`` otherwise.

        Returns
        -------
        NoiseRandomization
            A new ``NoiseRandomization`` object with default values.
        """
        sgc = -LogNormal(-3.5, 0.5) if q_positive else LogNormal(-3.5, 0.5)
        uim = Uniform(0.2, 0.25) if q_positive else Uniform(-0.25, -0.2)
        output = cls(
            noise_axis=0,
            n_gates=2,
            latching_positive=True,
            white_noise_magnitude=Uniform(0.08, 0.12),
            pink_noise_magnitude=Uniform(0.08, 0.12),
            telegraph_magnitude=Uniform(0.08, 0.12),
            telegraph_relative_stdev=Uniform(0, 0.3),
            telegraph_low_pixels=Normal(4, 1).abs(),
            telegraph_high_pixels=Normal(4, 1).abs(),
            latching_pixels=Normal(1, 0.3).abs(),
            sech_blur_width=Normal(0.7, 0.2).abs(),
            unint_dot_magnitude=uim,
            unint_dot_spacing=Normal(30, 10).abs(),
            unint_dot_offset=Uniform(0, 1),
            unint_dot_relative_width=Uniform(0.02, 0.03),
            coulomb_peak_offset=Uniform(0, 1),
            coulomb_peak_width=Normal(2.0, 0.3).abs(),
            coulomb_peak_spacing=Normal(8, 1).abs(),
            sensor_gate_coupling=sgc,
        )
        return output

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``NoiseRandomization`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        NoiseRandomization
            A new ``NoiseRandomization`` object with the values specified by ``dict``.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``NoiseRandomization`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``NoiseRandomization`` object.
        """
        memo:dict[int, Any] = {}
        output = {}
        for f in dataclasses.fields(NoiseRandomization):
            old_val = getattr(self, f.name)
            if id(old_val) in memo:
                output[f.name] = memo[id(old_val)]
            else:
                output[f.name] = copy.deepcopy(old_val, memo=memo)
        return output

    def copy(self) -> Self:
        """
        Creates a copy of a ``NoiseRandomization`` object.

        Returns
        -------
        CSDOutput
            A new ``NoiseRandomization`` object with the same attribute values as ``self``.
        """
        return copy.deepcopy(self)


def random_noise_params(
    randomization_params: NoiseRandomization, noise_scale_factor: float = 1.0
) -> NoiseParameters:
    """
    Generates a random set of noise parameters.

    Parameters
    ----------
    randomization_params : NoiseRandomization
        Meta-parameters which indicate how the ``NoiseParameters`` should be
        randomized.
    noise_scale_factor : float
        A float which can be adjusted to scale the overall amount of noise (default 1).

    Returns
    -------
    NoiseParameters
        The randomized set of noise parameters.
    """
    global _rng
    r_p = randomization_params
    noise = NoiseParameters()
    noise.noise_axis = r_p.noise_axis

    def draw(dist: T | Distribution[T], rng: np.random.Generator) -> T:
        if isinstance(dist, Distribution):
            return dist.draw(rng)
        else:
            return dist

    def multidraw(
        dist: NDArray | Distribution[Any] | Distribution[NDArray],
        n: int,
        rng: np.random.Generator,
    ) -> NDArray:
        if isinstance(dist, Distribution):
            a = dist.draw(rng)
            if isinstance(a, np.ndarray):
                return a
            else:
                a = np.array([a])
                if n == 1:
                    return a
                else:
                    a2 = dist.draw(rng, n - 1)
                    return np.concatenate([a, a2])
        else:
            return dist

    noise.white_noise_magnitude = noise_scale_factor * np.abs(
        draw(r_p.white_noise_magnitude, _rng)
    )
    noise.pink_noise_magnitude = noise_scale_factor * np.abs(
        draw(r_p.pink_noise_magnitude, _rng)
    )
    noise.telegraph_magnitude = noise_scale_factor * np.abs(
        draw(r_p.telegraph_magnitude, _rng)
    )
    noise.telegraph_stdev = noise.telegraph_magnitude * np.abs(
        draw(r_p.telegraph_relative_stdev, _rng)
    )
    noise.telegraph_low_pixels = 1 + np.abs(draw(r_p.telegraph_low_pixels, _rng) - 1)
    noise.telegraph_high_pixels = 1 + np.abs(draw(r_p.telegraph_high_pixels, _rng) - 1)
    noise.latching_pixels = noise_scale_factor * np.abs(draw(r_p.latching_pixels, _rng))
    noise.latching_positive = draw(r_p.latching_positive, _rng)
    noise.sech_blur_width = np.abs(draw(r_p.sech_blur_width, _rng))
    noise.unint_dot_magnitude = draw(r_p.unint_dot_magnitude, _rng)
    all_spacing = (
        multidraw(r_p.unint_dot_spacing, r_p.n_gates, _rng)
        if r_p.unint_dot_spacing is not None
        else None
    )
    noise.unint_dot_spacing = all_spacing
    noise.unint_dot_offset = draw(r_p.unint_dot_offset, _rng)
    noise.unint_dot_width = (
        np.sqrt(np.sum(all_spacing**2))
        * np.abs(draw(r_p.unint_dot_relative_width, _rng))
        if all_spacing is not None
        else 0.0
    )
    noise.coulomb_peak_spacing = np.abs(draw(r_p.coulomb_peak_spacing, _rng))
    noise.coulomb_peak_width = (
        np.abs(draw(r_p.coulomb_peak_width, _rng))
        if r_p.coulomb_peak_width is not None
        else None
    )
    noise.coulomb_peak_offset = draw(r_p.coulomb_peak_offset, _rng)
    noise.sensor_gate_coupling = (
        multidraw(r_p.sensor_gate_coupling, r_p.n_gates, _rng)
        if r_p.sensor_gate_coupling is not None
        else None
    )
    noise.use_pink_noise_all_dims = r_p.use_pink_noise_all_dims
    return noise
