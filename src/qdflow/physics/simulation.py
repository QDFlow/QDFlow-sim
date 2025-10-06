"""
This module defines the ``ThomasFermi`` class, which is responsible for
simulating a quantum dot nanowire and calculating the
particle (electron / hole) density, sensor readout, current, and state of the system.

Examples
--------

>>> from qdflow.physics import simulation
>>> from qdflow import generate
>>> phys_params = generate.default_physics(n_dots=2)
>>> tf_simulation = simulation.ThomasFermi(phys_params)

Get default set of physical parameters defining a double-dot device.
Define an instance of the ``ThomasFermi`` class with the
specified physical parameters.

>>> output = tf_simulation.run_calculations()
>>> output.island_charges
array([2, 2]) 

Perform the calculation. The results are returned in a dataclass.
In this example, the stable charge configuration has 2 electrons in each dot.
This result will vary if a different set of physical parameters ``phys_params``
are supplied to the ``ThomasFermi`` constructor.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, Self, overload
import warnings
import scipy  # type: ignore[import-untyped]
import scipy.special  # type: ignore[import-untyped]
import networkx  # type: ignore[import-untyped]
import itertools  # type: ignore[import-untyped]
import scipy.integrate  # type: ignore[import-untyped]
from numba import jit  # type: ignore[import-untyped]
import dataclasses
from dataclasses import dataclass, field


def calc_K_mat(x: NDArray[np.float64], K_0: float, sigma: float) -> NDArray[np.float64]:
    """
    Calculates the Coulomb interaction matrix.

    Parameters
    ----------
    x : ndarray[float]
        1D array containing the x-values (in nm) used in the simulation.
        These values should be evenly spaced, e.g., use
        ``x = np.linspace(x_min, x_max, num_points)``.
    K_0 : float
        Electron-electron Coulomb interaction strength (in meV * nm)
    sigma : float
        Softening parameter (in nm) for the el-el Coulomb interaction used to avoid
        divergence when x = x'. ``sigma`` should be on the scale of the width
        of the nanowire.

    Returns
    -------
    K_mat : ndarray[float]
        2D array, with shape ``(len(x), len(x))``,
        where ``K_mat[i, j]`` gives the value of the Coulomb interaction
        (in meV) between two particles at points ``x[i]`` and ``x[j]``.

    Notes
    -----
    This function returns:

    ``K_mat[x1, x2] = K_0 / sqrt((x1-x2)**2 + sigma**2)``.

    See eq. (2) in
    `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
    """
    dx = np.sqrt((x - np.expand_dims(x, -1)) ** 2 + sigma**2)
    K_matrix = K_0 / dx
    return K_matrix


@dataclass(kw_only=True)
class GateParameters:
    """
    Set of physical parameters defining a single gate.

    Attributes
    ----------
    mean : float
        The x-value (in nm) of the point along the nanowire where the gate
        creates the largest potential.
        This will be the x-value of the center of the gate.
    peak : float
        The peak value (in mV) of the potential along the nanowire due to the gate.
        This should be the value of the potential due to the gate at x equal to
        ``gate.mean``. Note that this is not the potential of the gate itself.
    rho : float
        The radius (in nm) of the cylindrical gate.
    h : float
        The distance (in nm) of the gate from the nanowire.
    screen : float
        The screening length (in nm) for the Coulomb interaction between the
        gate and the particles in the nanowire.
    """

    mean: float = 0
    peak: float = 0
    rho: float = 15
    h: float = 80
    screen: float = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``GateParameters`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.
            Default values are set for keys not included in the dict.

        Returns
        -------
        GateParameters
            A new ``GateParameters`` object with the values specified by `d`.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``GateParameters`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``GateParameters`` object.
        """
        return dataclasses.asdict(self)

    def copy(self) -> Self:
        """
        Creates a copy of a ``GateParameters`` object.

        Returns
        -------
        GateParameters
            A new ``GateParameters`` object with the same attribute values as ``self``.
        """
        return dataclasses.replace(self)



@dataclass(kw_only=True)
class PhysicsParameters:
    """
    Set of physical parameters of a quantum dot nanowire.

    Attributes
    ----------
    x : ndarray[float]
        1D array containing the x-values (in nm) used in the simulation.
        These values should be evenly spaced, e.g., use
        ``x = np.linspace(x_min, x_max, num_points)``.
    V : ndarray[float] | None
        1D array containing the total potential ``V(x)`` (in mV) from all of the gates
        at each of the x-values in `x`.
        If `V` is ``None``, it will be calculated using `gates`.
    q : float
        The charge of a particle, -1 for electrons, +1 for holes.
    gates : list[GateParameters]
        List of ``GateParameters`` defining the relevant parameters for each the gates.
    effective_peaks : ndarray[float] | None
        The effective gate peaks after inluding effects of charge induced from other gates.
        If `effective_peaks` is ``None``, it will be calculated via ``calc_effective_peaks(gates)``.
    K_mat : ndarray[float] | None
        2D array, with length in each dimension equal to ``len(x)``,
        where ``K_mat[i, j]`` gives the value of the Coulomb interaction
        (in meV) between two particles at points ``x[i]`` and ``x[j]``.
        If ``K_mat`` is ``None``, it will be calculated using
        ``calc_K_mat(x, K_0, sigma)`` when initializing a ``ThomasFermi`` object.
    K_0 : float
        Electron-electron Coulomb interaction strength (in meV * nm) used to
        calculate the effect of electrons on the sensors.
        Also used to calculate ``K_mat`` if it is not specified.
    sigma : float
        Softening parameter (in nm) for the el-el Coulomb interaction used to avoid
        divergence when x = x'. `sigma` should be on the scale of the width
        of the nanowire.
        If `K_mat` is specified, `sigma` will not be used.
    g0_dx_K_plus_1_inv : ndarray[float] | None
        Inverse of ``(g_0 * delta_x * K_mat + identity)``. If `g0_dx_K_plus_1_inv`
        is None, it will be automatically calculated when initializing a
        ``ThomasFermi`` object.
    mu : float
        The Fermi level (in meV), assumed to be equal for both leads.
    V_L : float
        The voltage (in mV) applied to left lead.
    V_R : float
        The voltage (in mV) applied to right lead.
    g_0 : float
        Coefficient of the density of states (in 1/(meV*nm) for 2D), see Eq. (1) of
        `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
    beta : float
        The inverse temperature ``1/(k_B T)`` (in 1/meV) used for self-consistent
        calculation of ``n(x)``, where ``k_B`` is the Boltzmann constant. See Eq. (1) of
        `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
    kT : float
        The temperature ``k_B T`` (in meV) used in the transport calculations,
        where ``k_B`` is the Boltzmann constant.
    c_k : float
        Coefficient (in meV*nm) that determines the kenetic energy of the
        Fermi sea on each island. See Eq. (5) of
        `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
    sensors : ndarray[float]
        Array with shape ``(n_sensors, 3)`` listing the positions ``(x, y, z)``
        (in nm) of the charge sensors, where ``x`` is the direction parrallel to
        the nanowire, and ``y`` is the direction parallel to the gates.
    screening_length : float
        The screening length (in nm) for the Coulomb interaction between the
        sensor and the particles in the nanowire.
    WKB_coef : float
        Coefficient (with units 1/(nm*sqrt(meV))) which goes in the exponent
        while calculating the WKB probability, setting the strength of WKB tunneling.
        WKB_coef should be equal to ``sqrt(2*m)/hbar``
        (converted to units of 1/(nm*sqrt(meV))), where ``m`` is the effective
        mass of a particle, and ``hbar`` is the reduced Planck's constant.
    barrier_current : float
        An arbitrary low current set to the device when in barrier mode.
    short_circuit_current : float
        An arbitrary high current value given to the device when in
        open / short circuit mode.
    v_F : float
        The fermi velocity (in nm/s).
    dot_regions : ndarray[float] | None
        Array with shape ``(n_dots, 2)``, where
        ``dot_regions[i,0] < x < dot_regions[i,1]`` defines the region used
        to determine the state of dot ``i``.
        If ``None``, automatically calculated from `gates` assuming an
        alternating pattern of barrier and plunger gates.
    """

    x: NDArray[np.float64] = field(
        default_factory=lambda: np.linspace(-300, 300, 151, endpoint=True)
    )
    V: NDArray[np.float64] | None = None
    q: float = -1
    gates: list[GateParameters] = field(
        default_factory=lambda: [
            GateParameters(mean=-200, peak=-7),
            GateParameters(mean=-100, peak=7),
            GateParameters(mean=0, peak=-5),
            GateParameters(mean=100, peak=7),
            GateParameters(mean=200, peak=-7),
        ]
    )
    effective_peaks: NDArray[np.float64] | None = None
    K_mat: NDArray[np.float64] | None = None
    K_0: float = 5
    sigma: float = 60
    g0_dx_K_plus_1_inv: NDArray[np.float64] | None = None
    mu: float = .5
    V_L: float = -1e-2
    V_R: float = 1e-2
    g_0: float = 0.0065
    beta: float = 100
    kT: float = 0.01
    c_k: float = 1.2
    sensors: NDArray[np.float64] = field(default_factory=lambda: np.array([[0, -250, 0]]))
    screening_length: float = 100
    WKB_coef: float = 0.089
    barrier_current: float = 1e-5
    short_circuit_current: float = 1e4
    v_F: float = 3.0e13
    dot_regions: NDArray[np.float64] | None = None

    def _get_x(self) -> NDArray[np.float64]:
        return self._x

    def _set_x(self, val: NDArray[np.float64]):
        self._x = np.array(val, dtype=np.float64)

    def _get_V(self) -> NDArray[np.float64] | None:
        return self._V

    def _set_V(self, val: NDArray[np.float64] | None):
        self._V = np.array(val, dtype=np.float64) if val is not None else None

    def _get_K_mat(self) -> NDArray[np.float64] | None:
        return self._K_mat

    def _set_K_mat(self, val: NDArray[np.float64] | None):
        self._K_mat = np.array(val, dtype=np.float64) if val is not None else None

    def _get_g0_dx_K_plus_1_inv(self) -> NDArray[np.float64] | None:
        return self._g0_dx_K_plus_1_inv

    def _set_g0_dx_K_plus_1_inv(self, val: NDArray[np.float64] | None):
        self._g0_dx_K_plus_1_inv = (
            np.array(val, dtype=np.float64) if val is not None else None
        )

    def _get_sensors(self) -> NDArray[np.float64]:
        return self._sensors

    def _set_sensors(self, val: NDArray[np.float64]):
        self._sensors = np.array(val, dtype=np.float64)

    def _get_gates(self) -> list[GateParameters]:
        return self._gates

    def _set_gates(self, val: list[GateParameters]):
        self._gates = [g.copy() for g in val]

    def _get_effective_peaks(self) -> NDArray[np.float64] | None:
        return self._effective_peaks

    def _set_effective_peaks(self, val: NDArray[np.float64] | None):
        self._effective_peaks = (
            np.array(val, dtype=np.float64) if val is not None else None
        )

    def _get_dot_regions(self) -> NDArray[np.float64] | None:
        return self._dot_regions

    def _set_dot_regions(self, val: NDArray[np.float64] | None):
        self._dot_regions = np.array(val, dtype=np.float64) if val is not None else None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``PhysicsParameters`` object from a ``dict`` of values.

        Attributes
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.
            Default values are set for keys not included in the dict.

        Returns
        -------
        PhysicsParameters
            A new ``PhysicsParameters`` object with the values specified by `d`.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                if k == "gates":
                    setattr(output, k, [GateParameters.from_dict(g) for g in v])
                else:
                    setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``PhysicsParameters`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``PhysicsParameters`` object.
        """
        return dataclasses.asdict(self)
        
    def copy(self) -> Self:
        """
        Creates a deep copy of a ``PhysicsParameters`` object.

        Returns
        -------
        PhysicsParameters
            A new ``PhysicsParameters`` object with the same attribute values as
            ``self``. This is a deep copy.
        """
        return dataclasses.replace(self)


PhysicsParameters.x = property(PhysicsParameters._get_x, PhysicsParameters._set_x)  # type: ignore
PhysicsParameters.V = property(PhysicsParameters._get_V, PhysicsParameters._set_V)  # type: ignore
PhysicsParameters.K_mat = property(
    PhysicsParameters._get_K_mat, PhysicsParameters._set_K_mat
)  # type: ignore
PhysicsParameters.g0_dx_K_plus_1_inv = property(
    PhysicsParameters._get_g0_dx_K_plus_1_inv, PhysicsParameters._set_g0_dx_K_plus_1_inv
)  # type: ignore
PhysicsParameters.sensors = property(
    PhysicsParameters._get_sensors, PhysicsParameters._set_sensors
)  # type: ignore
PhysicsParameters.gates = property(
    PhysicsParameters._get_gates, PhysicsParameters._set_gates
)  # type: ignore
PhysicsParameters.effective_peaks = property(
    PhysicsParameters._get_effective_peaks, PhysicsParameters._set_effective_peaks
)  # type: ignore
PhysicsParameters.dot_regions = property(
    PhysicsParameters._get_dot_regions, PhysicsParameters._set_dot_regions
)  # type: ignore


@dataclass(kw_only=True)
class NumericsParameters:
    """
    Set of options for numeric calculations.

    Attributes
    ----------
    calc_n_max_iterations_no_guess : int
        The maximum number of iterations to perfom, if no initial guess for n(x)
        is provided, when solving self-consistent equations to calculate the
        particle density n(x).
    calc_n_max_iterations_guess : int
        The maximum number of iterations to perfom, if an initial guess for n(x)
        is provided, when solving self-consistent equations to calculate the
        particle density n(x).
    calc_n_rel_tol : float
        The relative tolerance to accept a solution for the particle density
        n(x). The calculation will terminate when the difference ``delta_n``
        between successive iterations of n(x) is small enough that
        ``norm(delta_n)**2 < rel_tol**2 * norm(n) * norm(n_prev)``.
    calc_n_coulomb_steps : int
        The number of steps over which to turn on the Coulomb interaction
        when calculating the particle density n(x).
    calc_n_use_combination_method : bool
        Whether to use a combination of the previous 2 iterations when solving for n(x):
        ``n = (1 + g_0 * delta_x * K_mat)^-1 * (n + g_0 * delta_x * K_mat * n_prev)``
    island_relative_cutoff : float
        Cutoff for partitioning the nanowire into particle "islands". Regions
        where the value of n(x) is greater than ``relative_cutoff * max(n)``
        are considered islands, whereas regions where n(x) is smaller are
        considered barriers.
    island_min_occupancy : float
        Minimum particle occupancy to be considered an "island".
        If ``n`` integrated over a region is less than `island_min_occupancy`,
        it is considered part of a barrier.
    cap_model_matrix_softening : float
        Small value added to the denominator of the capacitance matrix formula
        to prevent blowup near zero charge states.
    stable_config_N_limit : int
        The algorithm will look for a stable configuration of particles which
        differs from the integral of n(x) over the island by at most
        ``N_limit``. This must be at least 1.
    count_transitions_sigma : float
        The minimum weight to accept as a transition when counting transitions.
    count_transitions_eps : float
        The maximum relative difference between incoming and outgoing weights
        to accept as a transition.
    create_graph_max_changes : int
        The maximum number of changes from the stable charge configuration
        to allow when creating the Markov graph.
    """

    calc_n_max_iterations_no_guess: int = 500
    calc_n_max_iterations_guess: int = 100
    calc_n_rel_tol: float = 1e-4
    calc_n_coulomb_steps: int = 1
    calc_n_use_combination_method: bool = True
    island_relative_cutoff: float = 1e-1
    island_min_occupancy: float = 1e-3
    cap_model_matrix_softening: float = 1e-6
    stable_config_N_limit: int = 1
    count_transitions_sigma: float = 1e-8
    count_transitions_eps: float = 1.95
    create_graph_max_changes: int = 2

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``NumericsParameters`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.
            Default values are set for keys not included in the dict.

        Returns
        -------
        NumericsParameters
            A new ``NumericsParameters`` object with the values specified by `d`.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``NumericsParameters`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``NumericsParameters`` object.
        """
        return dataclasses.asdict(self)

    def copy(self) -> Self:
        """
        Creates a copy of a ``NumericsParameters`` object.

        Returns
        -------
        NumericsParameters
            A new ``NumericsParameters`` object with the same attribute values as ``self``.
        """
        return dataclasses.replace(self)



@dataclass(kw_only=True)
class ThomasFermiOutput:
    """
    Output of Thomas Fermi calculations. Some attributes may be ``None``
    depending on which quantities are calculated.

    Attributes
    ----------

    island_charges : ndarray[int]
        ``self.island_charges``, an array with length equal to
        the number of islands representing the integer charge configuration
        which minimizes the capacitance energy.
    sensor : ndarray[float]
        ``self.sensor_output``, an array of floats with length equal to
        the number of sensors indicating the Coulomb potential at each sensor.
    are_dots_occupied : ndarray[bool]
        An array of booleans, one for each dot, indicating whether each dot is occupied.
    are_dots_combined : ndarray[bool]
        An array of booleans, one for each internal barrier,
        indicating whether the dots on each side are combined together
        (i.e. the barrier is too low).
        ``len(are_dots_combined)`` should always equal ``len(are_dots_occupied) - 1``.
    dot_charges : ndarray[int]
        An array of integers, one for each dot, indicating the total number
        of charges in each dot. In the case of combined dots, the
        total number of charges will be entered in the left-most dot,
        with the other dots padded with zeros.
    converged : bool
        Whether the calculation of n(x) properly converged.
    inv_cap_matrix : ndarray[float]
        The inverse capacitance matrix ``E_ij``.
    current : float
        ``self.current``, the current running through the wire.
    graph_charge : tuple[int, ...]
        ``self.graph_charge``, a tuple of integers with length equal to
        the number of islands, representing the charge configuration
        which has the highest weight in the steady-state of the Markov graph.
    transition_count : int
        ``self.trans_count``, the number of transitions.
    n : ndarray[float]
        ``self.n``, the particle density n(x), (in 1/nm).
    """

    island_charges: NDArray[np.int_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int_)
    )
    sensor: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    are_dots_occupied: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    are_dots_combined: NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    dot_charges: NDArray[np.int_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int_)
    )
    converged: bool = False
    inv_cap_matrix: NDArray[np.float64] | None = None
    current: float | None = None
    graph_charge: tuple[int, ...] | None = None
    transition_count: int | None = None
    n: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    def _get_island_charges(self) -> NDArray[np.int_]:
        return self._island_charges
    def _set_island_charges(self, val:NDArray[np.int_]):
        self._island_charges = np.array(val, dtype=np.int_)

    def _get_sensor(self) -> NDArray[np.float64]:
        return self._sensor
    def _set_sensor(self, val:NDArray[np.float64]):
        self._sensor = np.array(val, dtype=np.float64)

    def _get_are_dots_occupied(self) -> NDArray[np.bool_]:
        return self._are_dots_occupied
    def _set_are_dots_occupied(self, val:NDArray[np.bool_]):
        self._are_dots_occupied = np.array(val, dtype=np.bool_)

    def _get_are_dots_combined(self) -> NDArray[np.bool_]:
        return self._are_dots_combined
    def _set_are_dots_combined(self, val:NDArray[np.bool_]):
        self._are_dots_combined = np.array(val, dtype=np.bool_)

    def _get_dot_charges(self) -> NDArray[np.int_]:
        return self._dot_charges
    def _set_dot_charges(self, val:NDArray[np.int_]):
        self._dot_charges = np.array(val, dtype=np.int_)

    def _get_inv_cap_matrix(self) -> NDArray[np.float64]|None:
        return self._inv_cap_matrix
    def _set_inv_cap_matrix(self, val:NDArray[np.float64]|None):
        self._inv_cap_matrix = np.array(val, dtype=np.float64) if val is not None else None

    def _get_n(self) -> NDArray[np.float64]|None:
        return self._n
    def _set_n(self, val:NDArray[np.float64]|None):
        self._n = np.array(val, dtype=np.float64) if val is not None else None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """
        Creates a new ``ThomasFermiOutput`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.
            Values of keys not included in the dict are ``None``.

        Returns
        -------
        ThomasFermiOutput
            A new ``ThomasFermiOutput`` object with the values specified by `d`.
        """
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the ``ThomasFermiOutput`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``ThomasFermiOutput`` object.
        """
        return dataclasses.asdict(self)

    def copy(self) -> Self:
        """
        Creates a copy of a ``ThomasFermiOutput`` object.

        Returns
        -------
        ThomasFermiOutput
            A new ``ThomasFermiOutput`` object with the same attribute values as ``self``.
        """
        return dataclasses.replace(self)


ThomasFermiOutput.island_charges = property(ThomasFermiOutput._get_island_charges, ThomasFermiOutput._set_island_charges) # type: ignore
ThomasFermiOutput.sensor = property(ThomasFermiOutput._get_sensor, ThomasFermiOutput._set_sensor) # type: ignore
ThomasFermiOutput.are_dots_occupied = property(ThomasFermiOutput._get_are_dots_occupied, ThomasFermiOutput._set_are_dots_occupied) # type: ignore
ThomasFermiOutput.are_dots_combined = property(ThomasFermiOutput._get_are_dots_combined, ThomasFermiOutput._set_are_dots_combined) # type: ignore
ThomasFermiOutput.dot_charges = property(ThomasFermiOutput._get_dot_charges, ThomasFermiOutput._set_dot_charges) # type: ignore
ThomasFermiOutput.inv_cap_matrix = property(ThomasFermiOutput._get_inv_cap_matrix, ThomasFermiOutput._set_inv_cap_matrix) # type: ignore
ThomasFermiOutput.n = property(ThomasFermiOutput._get_n, ThomasFermiOutput._set_n) # type: ignore


@overload
def calc_V_gate(
    gate_params: GateParameters,
    x: float,
    y: float,
    z: float,
    effective_peak: float | None = ...,
) -> float: ...
@overload
def calc_V_gate(
    gate_params: GateParameters,
    x: NDArray[np.float64],
    y: float | NDArray[np.float64],
    z: float | NDArray[np.float64],
    effective_peak: float | None = ...,
) -> NDArray[np.float64]: ...
@overload
def calc_V_gate(
    gate_params: GateParameters,
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
    z: float | NDArray[np.float64],
    effective_peak: float | None = ...,
) -> float | NDArray[np.float64]: ...


def calc_V_gate(gate_params: GateParameters, x, y, z, effective_peak=None):
    """
    Calculates the potential at a given set of points due to the given gate.

    This model assumes the gate behaves like a infinite cylindrical conductor of
    radius ``rho``, centered at ``x = mean``, ``z = -h``, with axis parallel
    to the y-axis, and a screening length of ``screen``.

    Parameters
    ----------
    gate_params : thomas_fermi.GateParameters
        The set of parameters defining the gate
    x, y, z : float or ndarray[float]
        The x, y, or z-values (in nm) of the points for which to calculate the potential.
        `x` refers to the direction parallel to the nanowire.
        `y` refers to the direction parallel to the central axis of the cylinder
        defining each of the gates.
    effective_peak : float | None
        If not ``None``, this value will be used instead of ``gate_params.peak``.

    Returns
    -------
    float or ndarray[float]
        The (approximate) potential at each of the input points due to the gate.
    """
    peak = effective_peak if effective_peak is not None else gate_params.peak
    mean = gate_params.mean
    h = gate_params.h
    screen = gate_params.screen

    xb, yb, zb = np.broadcast_arrays(x, y, z)
    s = np.sqrt((xb - mean) ** 2 + (zb + h) ** 2)

    v = peak * scipy.special.kn(0, s / screen) / scipy.special.kn(0, h / screen)

    if hasattr(x, "__len__") or hasattr(y, "__len__") or hasattr(z, "__len__"):
        return v
    else:
        return v.item()


def calc_effective_peaks(gate_param_list: list[GateParameters]) -> NDArray[np.float64]:
    """
    Calculates the effective potential at each gate of a set of gates.

    If more than one gate are in close proximity, the field from each gate will
    induce charges on the others. This can be modeled by using an effective
    potential ``peak`` rather than the true ``peak``.


    Parameters
    ----------
    gate_param_list : list[GateParameters]
        The set of parameters defining each of the gates

    Returns
    -------
    ndarray[float]
        An array containing the effective potential peak of each gate.
    """
    gates = gate_param_list
    n_gates = len(gates)
    c_mat = np.zeros((n_gates, n_gates), dtype=np.float64)
    for i in range(n_gates):
        for j in range(n_gates):
            if i == j:
                c_mat[i][j] = 1
            else:
                c_mat[i][j] = calc_V_gate(
                    gates[j], gates[i].mean, 0, -gates[i].h, 1
                ) / calc_V_gate(
                    gates[j], gates[j].mean, 0, gates[j].rho - gates[j].h, 1
                )
    v_peak = np.array([g.peak for g in gates])
    return np.dot(np.linalg.inv(c_mat), v_peak)


@overload
def calc_V(
    gate_param_list: list[GateParameters],
    x: float,
    y: float,
    z: float,
    effective_peaks: None | NDArray[np.float64] = ...,
) -> float: ...
@overload
def calc_V(
    gate_param_list: list[GateParameters],
    x: NDArray[np.float64],
    y: float | NDArray[np.float64],
    z: float | NDArray[np.float64],
    effective_peaks: None | NDArray[np.float64] = ...,
) -> NDArray[np.float64]: ...
@overload
def calc_V(
    gate_param_list: list[GateParameters],
    x: float | NDArray[np.float64],
    y: float | NDArray[np.float64],
    z: float | NDArray[np.float64],
    effective_peaks: None | NDArray[np.float64] = ...,
) -> float | NDArray[np.float64]: ...


def calc_V(gate_param_list: list[GateParameters], x, y, z, effective_peaks=None):
    """
    Calculates the potential at a given set of points due to a list of gates.

    Parameters
    ----------
    gate_param_list : list[thomas_fermi.GateParameters]
        The parameters defining each of the gates
    x, y, z : float or ndarray[float]
        The x, y, or z-values (in nm) of the points for which to calculate the potential
    effective_peaks : None | ndarray[float]
        An array of coefficents to multiply the ``peak`` of each gate to correct
        for induced charge from the other gates.
        If ``None``, it will be calculated automatically via ``calc_effective_peaks()``,
        but it can be supplied here to avoid repeating the calculation every time
        ``calc_V()`` is called.

    Returns
    -------
    float or ndarray[float]
        The (approximate) potential at each of the input points due to the gate.
    """
    v_eff = (
        effective_peaks
        if effective_peaks is not None
        else calc_effective_peaks(gate_param_list)
    )
    return np.sum(
        np.array(
            [calc_V_gate(g, x, y, z, v) for (g, v) in zip(gate_param_list, v_eff)]
        ),
        axis=0,
    )


class ConvergenceWarning(UserWarning):
    """
    A warning raised when the Thomas Fermi calculation of n(x)
    does not properly converge.
    """

    pass



class ThomasFermi:
    """
    Thomas-Fermi simulation of a quantum dot nanowire.

    Given a specific set of physical parameters, this class can calculate
    the particle density, sensor readout, current, state of the system, etc.

    To use this class, initiate a ``ThomasFermi`` instance with the appropriate
    ``PhysicsParameters`` and (optionally) ``NumericsParameters``.
    Then use the ``run_calculations()`` function, which will perform all
    necessary calculations and return relavant quantities as an instance of
    the ``ThomasFermiOutput`` dataclass.


    Parameters
    ----------
    physics : PhysicsParameters | dict[str, Any]
        ``PhysicsParameters`` object or dictionary with the relevant
        physical parameter names and values.
    numerics : NumericsParameters | dict[str, Any], optional
        ``NumericsParameters`` object or dictionary with names and values
        for options for numeric calculations.

    See Also
    --------
    PhysicsParameters : Physical device parameters
    NumericsParameters : Numeric options
    """

    def __init__(
        self,
        physics: PhysicsParameters | dict[str, Any],
        numerics: NumericsParameters | dict[str, Any] | None = None,
    ):
        self.physics: PhysicsParameters = (
            PhysicsParameters.from_dict(physics)
            if isinstance(physics, dict)
            else physics.copy()
        )
        """
        PhysicsParameters
            ``PhysicsParameters`` with the physical device parameters
        """

        # calculate effective_peaks if not already specified
        self.effective_peaks: NDArray[np.float64] = (
            calc_effective_peaks(self.physics.gates)
            if self.physics.effective_peaks is None
            else self.physics.effective_peaks
        )
        """
        ndarray[float]
            1D array containing the effective peaks of each gate after including
            the effects of induced charge from the other gates.
        """

        # calculate V if not already specified
        self.V: NDArray[np.float64] = (
            calc_V(self.physics.gates, self.physics.x, 0, 0, self.effective_peaks)
            if self.physics.V is None
            else self.physics.V
        )
        """
        ndarray[float]
            1D array containing the total potential ``V(x)`` (in mV) from all of the gates
            at each of the x-values in `x`.
        """

        # calculate K_mat if not already specified
        self.K_mat: NDArray[np.float64] = (
            calc_K_mat(self.physics.x, self.physics.K_0, self.physics.sigma)
            if self.physics.K_mat is None
            else self.physics.K_mat
        )
        """
        ndarray[float]
            2D array, with length in each dimension equal to ``len(x)``,
            where ``K_mat[i, j]`` gives the value of the Coulomb interaction
            (in meV) between two particles at points ``x[i]`` and ``x[j]``.
        """

        self.numerics: NumericsParameters = (
            NumericsParameters() if numerics is None else (
                numerics.copy()
                if isinstance(numerics, NumericsParameters)
                else NumericsParameters.from_dict(numerics)
            )
        )
        """
        dict[str, Any]
            Dictionary with names and values for options for numeric calculations
        """

        self.delta_x: float = (self.physics.x[-1] - self.physics.x[0]) / (
            len(self.physics.x) - 1
        )
        """
        float
            The spacing (in nm) between successive x-values of the simulation.
        """

        # calculate g0_dx_K_plus_1_inv if not already specified
        self.g0_dx_K_plus_1_inv: NDArray[np.floating[Any]] = np.array([])
        if self.numerics.calc_n_use_combination_method:
            self.g0_dx_K_plus_1_inv = (
                np.linalg.inv(
                    self.physics.g_0 * self.delta_x * self.K_mat
                    + np.identity(len(self.K_mat))
                )
                if self.physics.g0_dx_K_plus_1_inv is None
                else self.physics.g0_dx_K_plus_1_inv
            )

        self.n: NDArray[np.float64]
        """
        ndarray[float]
            The particle density n(x), (in 1/nm), found by solving the set of
            self-consistent equations, eqs. (1) & (2) of
            `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
            This array has the same length as ``physics.x``.
        """

        self.converged: bool = False
        """
        bool
            Whether or not the calculation of ``self.n`` properly converged
        """

        self.phi: NDArray[np.float64]
        """
        ndarray[float]
            The electron-electron Coulomb potential, equal to ``dot(K_mat, n) * delta_x``.
            This array has the same length as ``physics.x``.
        """

        self.qV_TF: NDArray[np.float64]
        """
        ndarray[float]
            The Thomas-Fermi potential, equal to ``q*V + dot(K_mat,n) * delta_x``
        """

        self.islands: NDArray[np.int_]
        """
        ndarray[int]
            An array of islands, where each island is a length-2 integer array
            ``[begin_index, end_index + 1]``.
            These islands correspond to segments of self.n that are above
            a certain cutoff value; however, segments bordering the left or
            right endpoints of ``physics.x`` are not included.
        """

        self.all_islands: NDArray[np.int_]
        """
        ndarray[int]
            An array of islands, where each island is a length-2 integer array
            ``[begin_index, end_index + 1]``.
            These islands correspond to segments of self.n that are above
            a certain cutoff value. This is similar to self.islands, but
            includes segments bordering the left or right endpoints of
            ``physics.x``.
        """

        self.barriers: NDArray[np.int_]
        """
        ndarray[int]
            An array of barriers, where each barrier is a length-2 integer array
            ``[begin_index, end_index + 1]``.
            These barriers correspond to segments of self.n that are below
            a certain cutoff value.
        """

        self.tranmission_coef: NDArray[np.float64]
        """
        ndarray[float]
            An array with length equal to ``len(self.barriers)``,
            where each entry is the WKB transmission probability for the corresponding
            barrier.
        """

        self.p_WKB: NDArray[np.float64]
        """
        ndarray[float]
            An array with length equal to ``len(self.barriers)``,
            where each entry is the WKB transmission probability for the corresponding
            barrier multiplied by the average attempt rate of the islands
            directly to the left and right of the barrier.
            The attempt rate of an island is defined to be
            ``v_F / 2 / (x_width_of_island)``,
            and is a measure of how frequently a particle in the island
            collides with a given wall of the island and attempts to tunnel
            out.
        """

        self.charges: NDArray[np.float64]
        """
        ndarray[float]
            An array with length equal to ``len(self.islands)``,
            where each value in the array is equal to the total number of
            particles on the corresponding island, found by integrating
            ``self.n`` over the region defined by the island.
        """

        self.charge_centers: NDArray[np.float64]
        """
        ndarray[float]
            An array with length equal to ``len(self.islands)``,
            where each value in the array is equal to the "center of mass"
            for the charges on the corresponding island, found by integrating
            ``x * n(x)`` over the region defined by the island.
        """

        self.inv_cap_matrix: NDArray[np.float64]
        """
        ndarray[float]
            A 2d array representing the inverse capacitance matrix ``E_ij`` given by Eq. (5) of
            `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.
        """

        self.island_charges: NDArray[np.int_]
        """
        ndarray[int]
            An array with length equal to ``len(self.islands)`` representing
            the integer charge configuration which minimizes the capacitance 
            energy ``calc_cap_energy(self, self.island_charges)``.
        """

        self.G: networkx.DiGraph
        """
        networkx.DiGraph
            A Markov graph where nodes are tuples of integers, with 1 int per
            island representing the number of particles on each island.
            The graph is a directed graph with edges that have weights equal to
            ``calc_wieght(self, u, v)``, which are the transition rates between
            different charge configurations.
        """

        self.start_node: tuple[int, ...]
        """
        tuple[int, ...]
            The starting node for the graph ``self.G``,
            equal to ``tuple(self.island_charges)``.
        """

        self.dist: NDArray[np.float64]
        """
        ndarray[float]
            An array with length equal to ``len(self.G)`` representing
            the steady-state of the Markov graph.
        """

        self.graph_charge: tuple[int, ...]
        """
        tuple[int, ...]
            Tuple of integers with length equal to the number of islands,
            representing the node of ``self.G`` which has the highest
            probability in ``self.dist``.
        """

        self.current: float
        """
        float
            The current running through the wire, defined to be the sum over all
            tunneling events of: (rate at which the tunneling event occurs) *
            (``dist`` probability of initial node of tunneling event) *
            (change in number of particles in the Left sink)
        """

        self.sensor_output: NDArray[np.float64]
        """
        ndarray[float]
            An array of floats with length equal to ``len(physics.sensors)``
            indicating the Coulomb potential readout at each sensor.
        """

        self.are_dots_occupied: NDArray[np.bool_]
        """
        ndarray[bool]
            An array of booleans, one for each dot,
            indicating whether each dot is occupied.
        """

        self.are_dots_combined: NDArray[np.bool_]
        """
        ndarray[bool]
            An array of booleans, one for each internal barrier,
            indicating whether the dots on each side are combined together
            (i.e. the barrier is too low).
            ``len(self.are_dots_combined)`` will always equal ``len(self.are_dots_occupied) - 1``.
        """

        self.dot_charges: NDArray[np.int_]
        """
        dot_charges : ndarray[int]
            An array of integers, one for each dot, indicating the total number
            of charges in each dot. In the case of combined dots, the
            total number of charges will be entered in the left-most dot,
            with the other dots padded with zeros.
        """

        self.trans_count: int
        """
        int
            The number of graph nodes adjacent to ``self.start_node`` such that
            the incoming and outgoing weights between the nodes and
            ``self.start_node`` are equal. Never greater than 2.
        """

        self.is_short_circuit: bool
        """
        bool
            ``True`` if the system is in a short circuit state, i.e. if n(x) is
            large over the entire length on the nanwire.
        """

    @staticmethod
    @jit(nopython=True, cache=True)
    def calc_n_numba(
        n_0: NDArray[np.float64],
        qV: NDArray[np.float64],
        K_mat: NDArray[np.float64],
        g_0: float,
        beta: float,
        mu: float,
        delta_x: float,
        rel_tol: float,
        coulomb_steps: int,
        use_n_guess: bool,
        max_iterations: int,
        use_combination_method: bool,
        g0_dx_K_plus_1_inv: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
        """
        Calculates the particle density n(x) using numba.

        Parameters
        ----------
        n_0 : ndarray[float]
            Initial guess (in 1/nm) for the particle density n(x).
            Pass ``np.zeros(len(qV))`` if no initial guess is desired.
        qV : ndarray[float]
            The potential V(x) formed by the gates, multiplied by the particle
            charge q. (``qV`` has units of meV).
        K_mat : ndarray[float]
            2D array, with length in each dimension equal to ``len(x)``,
            where ``K_mat[i, j]`` gives the value of the Coulomb interaction
            (in meV) between two particles at points ``x[i]`` and ``x[j]``.
        g_0 : float
            Coefficient of the density of states (in 1/(meV*nm) for 2D).
        beta : float
            The inverse temperature :math:`1/(k_B T)` (in 1/meV).
        mu : float
            The Fermi level (in meV).
        delta_x:
            The spacing (in nm) between successive x-values.
        rel_tol : float
            The relative tolerance to accept a solution.
            The calculation will terminate when the difference ``delta_n``
            between successive iterations of n(x) is small enough that
            ``norm(delta_n)**2 < rel_tol**2 * norm(n) * norm(n_prev)``.
        coulomb_steps : int
            The number of steps over which to turn on the Coulomb interaction.
        use_n_guess : bool
            If True, the Coulomb potential will be applied all at once.
            If false, it will be turned on slowly over a series of steps.
        max_iterations : int
            The maximum number of iterations to perfom.
        use_combination_method : bool
            Whether to use a combination of the previous 2 iterations when solving for n(x):
            ``n = (1 + g_0 * delta_x * K_mat)^-1 * (n + g_0 * delta_x * K_mat * n_prev)``
        g0_dx_K_plus_1_inv : ndarray[float]
            Inverse of ``(g_0 * delta_x * K_mat + identity)``.

        Returns
        -------

        n : ndarray[float]
            The calculated particle density (in 1/nm).
        phi : ndarray[float]
            The calculated value (in meV) of phi(x),
            given by ``np.dot(K_mat, n) * delta_x``.
        converged : bool
            Whether the process converged to the required tolerance within
            the specified number of iterations.
        """

        def polylog_f_numba(x):
            y = beta * x
            z = np.exp(np.where(y <= 0, y, 0))  # exp(y), only used when y <= 0
            zinv = np.exp(np.where(y > 0, -y, 0))  # exp(-y), only used when y > 0
            return (g_0 / beta) * np.where(y > 0, y + np.log(1 + zinv), np.log(1 + z))

        n = n_0
        n_prev = n
        n_prev_prev = n
        phi = np.dot(K_mat, n) * delta_x
        last_i_averaged = 0
        converged = False
        if use_combination_method:
            mu_pos = not np.all(mu - qV <= 0)
            t_calced = False
            t_scale = 0.9

        for i in range(max_iterations):
            # average last 2 values of n if oscillating between two values
            if (
                (not use_combination_method)
                and (i - last_i_averaged >= 3)
                and (np.linalg.norm(n - n_prev) > np.linalg.norm(n - n_prev_prev))
            ):
                n = (n + n_prev) / 2
                last_i_averaged = i
            # turn on the Coulomb over some number of steps
            # only if a guess for phi was not provided
            if not use_n_guess and i < coulomb_steps:
                phi = ((i + 1) / coulomb_steps) * np.dot(K_mat, n) * delta_x
            else:
                phi = np.dot(K_mat, n) * delta_x
            n_prev_prev = n_prev
            n_prev = n
            mqvp = mu - qV - phi
            n = polylog_f_numba(mqvp)
            n_norm = np.linalg.norm(n)
            np_norm = np.linalg.norm(n_prev)

            if use_combination_method:
                if mu_pos:
                    if n_norm > np_norm or np.any(mqvp > 0):
                        n = np.dot(
                            g0_dx_K_plus_1_inv,
                            (n + g_0 * delta_x * np.dot(K_mat, n_prev)),
                        )
                    else:
                        if not t_calced:
                            t_calced = True
                            t_sum = np.sum(
                                polylog_f_numba(
                                    np.dot(
                                        g0_dx_K_plus_1_inv,
                                        np.where(mu - qV > 0, mu - qV, 0),
                                    )
                                )
                            )
                        n_sum = np.sum(n_prev)
                        if n_sum <= t_sum:
                            n = np.dot(
                                g0_dx_K_plus_1_inv,
                                (n + g_0 * delta_x * np.dot(K_mat, n_prev)),
                            )
                        else:
                            n = (
                                t_scale * t_sum / n_sum * n_prev
                                + (1 - t_scale * t_sum / n_sum) * n
                            )
                    n = np.where(n > 0, n, 0)
                    n_norm = np.linalg.norm(n)

            if (i > coulomb_steps or use_n_guess) and (
                np.linalg.norm(n - n_prev) ** 2 < (rel_tol) ** 2 * n_norm * np_norm
            ):
                converged = True
                break

        n = np.real(n)

        return n, phi, converged

    def _calc_n(self, n_guess: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """
        Calculates the particle density n(x) using a ThomasFermi model.
        Sets ``self.n`` equal to the result.

        Parameters
        ----------
        n_guess : ndarray[float], optional
            Initial guess (in 1/nm) for the particle density n(x).
            If ``None``, an array of zeros will be used.

        Returns
        -------
        ndarray[float]
            ``self.n``, the calculated particle density (in 1/nm).

        Raises
        ------
        ConvergenceWarning
            If the process does not converge to the required tolerance within
            the specified number of iterations, given by ``self.numerics.calc_n_rel_tol``
            and ``self.numerics.calc_n_max_iterations_guess`` or 
            ``self.numerics.calc_n_max_iterations_no_guess``.
        """
        qV = self.V * self.physics.q
        K_mat = self.K_mat
        g_0 = self.physics.g_0
        beta = self.physics.beta
        mu = self.physics.mu
        delta_x = self.delta_x
        g0_dx_K_plus_1_inv = self.g0_dx_K_plus_1_inv

        rel_tol = self.numerics.calc_n_rel_tol
        coulomb_steps = self.numerics.calc_n_coulomb_steps
        max_iterations = (self.numerics.calc_n_max_iterations_no_guess
                if n_guess is None else self.numerics.calc_n_max_iterations_guess)
        use_combination_method = self.numerics.calc_n_use_combination_method

        # provide option to use a guess for n, phi to speed up computation
        if n_guess is not None:
            use_n_guess = True
            n = n_guess
        else:
            use_n_guess = False
            n = np.zeros(len(qV))

        # calculate
        self.n, self.phi, converged = ThomasFermi.calc_n_numba(
            n,
            qV,
            K_mat,
            g_0,
            beta,
            mu,
            delta_x,
            rel_tol,
            coulomb_steps,
            use_n_guess,
            max_iterations,
            use_combination_method,
            g0_dx_K_plus_1_inv,
        )
        if not converged:
            warnings.warn(
                "ThomasFermi.calc_n() failed to converge.", ConvergenceWarning
            )
        self.converged = converged
        return self.n

    def _calc_qV_TF(self) -> NDArray[np.float64]:
        """
        Calculates the Thomas-Fermi potential (in meV),
        equal to ``q*V + dot(K_mat,n)*delta_x``.

        Sets ``self.qV_TF`` equal to the result.

        Returns
        -------
        ndarray[float]
            ``self.qV_TF``, the Thomas-Fermi potential (in meV),
            equal to ``q*V + dot(K_mat,n)*delta_x``.
        """
        self.qV_TF = self.physics.q * self.V + np.dot(self.K_mat, self.n) * self.delta_x
        return self.qV_TF

    def _calc_islands_and_barriers(self) -> NDArray[np.int_]:
        """
        Separates n(x) into islands of charge.

        Islands / barriers are defined by dividing the x-axis into segments
        for which n(x) is above / below the cutoff value
        ``numerics.island_relative_cutoff * max(n)``.
        Islands are defined to be the segments where n(x) is above the cutoff,
        and barriers are the segments where n(x) is below the cutoff.
        However, segments bordering the left or right endpoints of
        ``physics.x`` are not included as islands.

        This function sets ``self.islands`` equal to the list of islands,
        sets ``self.all_islands`` equal to the list of islands including
        the left and right sides, and sets ``self.barriers`` equal to the list
        of calculated barriers.

        Returns
        -------

        ndarray[int]
            ``self.islands``, an array of islands, where each island is
            a length-2 integer array: ``[begin_index, end_index + 1]``.
        """

        n = self.n
        relative_cutoff = self.numerics.island_relative_cutoff

        # n_eps sets the scale below which an island is assumed to end
        # adaptive eps makes the calculation more robust
        n_eps = relative_cutoff * np.max(n)

        n_chop = np.concatenate(
            ([0], np.array([1 if x > n_eps else 0 for x in n]), [0])
        )
        n_chop_bar = np.concatenate(
            ([0], np.array([0 if x > n_eps else 1 for x in n]), [0])
        )
        # replace non-zero elements by 1
        n_diff = np.abs(np.diff(n_chop))
        islands = np.where(n_diff == 1)[0].reshape(-1, 2)

        for isl in islands:
            if (
                np.sum(n[isl[0] : isl[1]]) * self.delta_x
                < self.numerics.island_min_occupancy
            ):
                for i in range(isl[0], isl[1]):
                    n_chop[i + 1] = 0
                    n_chop_bar[i + 1] = 1

        n_diff = np.abs(np.diff(n_chop))
        islands = np.where(n_diff == 1)[0].reshape(-1, 2)

        n_diff = np.abs(np.diff(n_chop_bar))
        barriers = np.where(n_diff == 1)[0].reshape(-1, 2)

        self.all_islands = islands

        # certain edge cases handled here
        if len(islands) == 0:
            # The system is a complete barrier with no islands, n = 0
            self.is_short_circuit = False
        elif len(islands) == 1 and islands[0][0] == 0 and islands[0][1] == (len(n)):
            # short-circuit condition with electron density all over
            islands = islands[1:]
            self.is_short_circuit = True
        else:
            self.is_short_circuit = False
        # if left and right leads are present, they are popped off since
        #    they do not form quantum dot islands
        if len(islands) > 0 and islands[0][0] == 0:
            islands = islands[1:]
        if len(islands) > 0 and islands[-1][1] == (len(n)):
            islands = islands[:-1]

        self.islands = islands
        self.barriers = barriers

        return self.islands

    def _calc_WKB_prob(self) -> NDArray[np.float64]:
        """
        Calculates the transition rates between islands.

        For each barrier, this function calculates the WKB probability of transmission through
        the barrier and multiplies that by the average attempt rate of the islands
        on either side. If there is only one adjacent island (this happens for
        the leftmost and rightmost barriers), then the attempt rate for that
        one island is used.

        The attempt rate of an island is defined to be:
        ``v_F / 2 / (x_width_of_island)``,
        and is a measure of how frequently an particle in the island
        collides with a given wall of the island and attempts to tunnel out.

        Sets ``self.p_WKB`` equal to the result.

        Additionally, this function sets ``self.transmission_coef`` to the
        transmission coefficients.

        Returns
        -------

        ndarray[float]
            ``self.p_WKB``, an array with length equal to the number of
            barriers, where each entry is the transition probability across
            the corresponding barrier. Returns ``[physics.WKB_prob]``
            if there are no islands.
        """

        self.tranmission_coef = np.zeros(len(self.barriers))
        self.p_WKB = np.zeros(len(self.barriers))

        # if there is a short circuit, then a default rate is set
        if self.is_short_circuit:
            return self.p_WKB

        x = self.physics.x
        qV_TF = self.qV_TF
        mu = self.physics.mu
        WKB_coef = self.physics.WKB_coef

        # in order to handle negative values near the boundaries, I have put in abs
        k = WKB_coef * np.sqrt(np.abs(qV_TF - mu))
        for i in range(len(self.barriers)):
            bar_start = self.barriers[i][0]
            bar_end = self.barriers[i][1]
            prob = np.exp(
                -2 * scipy.integrate.simpson(k[bar_start:bar_end], x[bar_start:bar_end])
            )

            self.tranmission_coef[i] = prob

        # calculate attempt rates only if islands are present
        if len(self.islands) >= 1:
            attempt_rate = []
            for i in range(len(self.islands)):
                island_start = self.islands[i][0]
                island_end = self.islands[i][1]
                # classical round trip time
                attempt_time = (
                    2 * (island_end - island_start) * self.delta_x / self.physics.v_F
                )
                rate = 1 / attempt_time
                attempt_rate.append(rate)
            # For each island, there are two barriers
            # The attempt rate is both tunneling probabilities of the two barriers
            #      is set equal to island formed by the barriers
            # For a barrier enclosed by two islands, the average attempt rate
            #     is used of the two islands
            attempt_rate_vec_left = np.array([attempt_rate[0]] + attempt_rate)
            attempt_rate_vec_right = np.array(attempt_rate + [attempt_rate[-1]])
            attempt_rate_vec = 0.5 * (attempt_rate_vec_left + attempt_rate_vec_right)
            self.p_WKB = attempt_rate_vec * self.tranmission_coef

        return self.p_WKB

    def _calc_charges(self) -> NDArray[np.float64]:
        """
        Calculates the total charge for each island by summing
        ``self.n`` over the region defined by each island.

        Sets ``self.charges`` equal to the result.

        Returns
        -------
        ndarray[float]
            ``self.charges``, an array with length equal to the number of
            islands, where each entry of the array is the total induced charge
            on the corresponding island. (Technically particle number,
            does not take into account ``q``).
        """
        charges = np.zeros(len(self.islands), dtype=np.float64)
        for i, isl in enumerate(self.islands):
            charges[i] = np.sum(self.n[isl[0] : isl[1]]) * self.delta_x
        self.charges = charges
        return self.charges

    def _calc_charge_centers(self) -> NDArray[np.float64]:
        """
        Calculates the center of each charge island by summing ``n * x`` over
        the region defined by each island and dividing by the total charge
        (this is similar to a center of mass calculation).

        Sets ``self.charge_centers`` equal to the result.

        Returns
        -------
        ndarray[float]
            ``self.charge_centers``, an array of charges with length equal to
            ``len[islands]`` found by summing ``n * x`` over the region defined
            by each island and dividing by the total charge.
        """
        charge_centers = np.zeros(len(self.islands), dtype=np.float64)
        for i, isl in enumerate(self.islands):
            charge_centers[i] = np.sum(
                self.n[isl[0] : isl[1]] * self.physics.x[isl[0] : isl[1]]
            ) / np.sum(self.n[isl[0] : isl[1]])
        self.charge_centers = charge_centers
        return self.charge_centers

    def _calc_inv_cap_matrix(self) -> NDArray[np.float64]:
        """
        Calculates the inverse capacitance matrix for the system.
        See eq. (5) of `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_.

        Sets ``self.inv_cap_matrix`` equal to the result.

        Returns
        -------
        energy_matrix : ndarray[float]
            The energy matrix (in meV) in the capacitance model (proportional to
            the inverse capacitance matrix). This is computed by taking the
            Coulomb potential energy between n(x) on island ``i`` and n(x)
            on island ``j``, and adding the kenetic energy of the island if
            ``i==j``.

        Notes
        -----
        This function differs from eq. (5) of
        `J. Zwolak et al. PLoS ONE 13(10): e0205844. <https://doi.org/10.1371/journal.pone.0205844>`_
        in that it includes a factor of 0.5 in the second term.
        """
        capacitance_matrix_softening = self.numerics.cap_model_matrix_softening

        islands = self.islands
        n = self.n

        # list of charge densities for islands
        n_list = []
        for isl in islands:
            n_island = np.zeros(len(n))
            n_island[isl[0] : isl[1]] = n[isl[0] : isl[1]]
            n_list.append(n_island)

        c_k = self.physics.c_k
        Z = self.charges
        K_mat = self.K_mat
        delta_x = self.delta_x
        delta_x_sq = delta_x * delta_x

        def cap_func(i, j):
            energy = 0.0
            if i == j:
                energy += c_k * np.sum(n_list[i] * n_list[i]) * delta_x
            energy += 0.5 * np.dot(np.dot(n_list[i].T, K_mat), n_list[j]) * delta_x_sq
            return energy

        # capacitance_matrix_softening = 1e-6 added to prevent blowup
        # of the capacitance matrix near zero charge states.
        energy_matrix = np.array(
            [
                (1.0 / ((Z[i] * Z[j]) + capacitance_matrix_softening)) * cap_func(i, j)
                for i in range(len(n_list))
                for j in range(len(n_list))
            ]
        ).reshape((len(n_list), len(n_list)))

        self.inv_cap_matrix = energy_matrix

        return self.inv_cap_matrix

    def _calc_cap_energy(self, N_vec: NDArray[np.float64]) -> float:
        """
        Calculates the capacitance energy (in meV) of a given charge configuration.

        Parameters
        ----------
        N_vec : ndarray[float]
            The charge configuration array. A 1d array with length equal to the
            number of islands specifying the number of charges on each island.

        Returns
        -------
        float
            The capacitance energy (in meV) of the given charge configuration, given by
            ``(N_vec - Z) E (N_vec - Z)^T``, where ``E == self.inv_cap_matrix``.
        """
        N_vec = np.array(N_vec)
        inv_cap_matrix = self.inv_cap_matrix
        charges = self.charges
        return np.dot(np.dot((N_vec - charges), inv_cap_matrix), (N_vec - charges).T)

    def _calc_stable_config(self) -> NDArray[np.int_]:
        """
        Calculates stable charge configuration, the configuration ``Q`` which
        minimizes ``self.calc_cap_energy(Q)``.

        Returns
        -------
        ndarray[int]
            The charge configuration array which minimizes the capacitance energy.
            A 1d array with length equal to the number of islands specifying the
            number of charges on each island.
        """
        Z = self.charges

        N_int = [int(np.floor(x)) for x in Z]
        N_limit = self.numerics.stable_config_N_limit

        dN_list = [range(max(0, x - N_limit + 1), x + N_limit + 1, 1) for x in N_int]
        N_list = list(itertools.product(*dN_list))

        energy_table = [self._calc_cap_energy(np.array(x)) for x in N_list]
        min_energy = min(energy_table)
        island_charges = N_list[energy_table.index(min_energy)]
        self.island_charges = np.array(island_charges, dtype=int)

        return self.island_charges

    def _fermi(self, E: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Calculates the fermi distribution at a given energy (or array of
        energies), with ``kT`` given by ``self.physics``.

        Parameters
        ----------
        E : float or ndarray[float]
            The energy / energies (in meV) for which to evaluate the
            fermi distribution.

        Returns
        -------
        float or ndarray[float]
            A scalar or ndarray with the same shape as ``E`` with values equal
            to the fermi distribution at ``E``.
        """
        kT = self.physics.kT
        return scipy.special.expit(-E / kT)

    # note: This function is unused
    def _dfermi(self, E: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """
        Calculates the derivative (in 1/meV) of the fermi distribution at a given
        energy (or array of energies), with ``kT`` given by ``self.physics``.

        Parameters
        ----------
        E : float or ndarray[float]
            The energy / energies (in meV) for which to evaluate the
            derivative of the fermi distribution.

        Returns
        -------
        float or ndarray[float]
            A scalar or ndarray with the same shape as ``E`` with values equal
            to the derivative of the fermi distribution at ``E`` (in 1/meV).
        """
        kT = self.physics.kT
        dfermi = scipy.special.expit(-E / kT) ** 2 * (-1 / kT) * np.exp(E / kT)
        return dfermi

    def _calc_weight(self, u: NDArray[np.int_], v: NDArray[np.int_]) -> float:
        """
        Calculates the transition rate from one configuration to another.

        Parameters
        ----------
        u, v : ndarray[int]
            The initial and final charge configuration arrays. These are 1d
            arrays with length equal to the number of islands specifying the
            number of particles on each island.

        Returns
        -------
        float
            The transition rate from ``u`` to ``v``, given by
            ``(self.p_WKB) * fermi(delta_E)``, where ``delta_E`` is the energy
            difference between ``u`` and ``v`` (including the energy of the
            particle escaping into the left / right bath if applicable).
        """
        diff = list(np.array(v) - np.array(u))
        nonzero_diff = np.nonzero(diff)[0]
        num_isls = len(self.islands)

        # check to make sure u -> v is a valid transition
        if len(nonzero_diff) == 0:
            return 0.0
        idx = nonzero_diff[0]
        if len(nonzero_diff) == 1:
            if idx > 0 and idx < num_isls - 1:
                return 0.0
        elif len(nonzero_diff) == 2:
            if nonzero_diff[1] - nonzero_diff[0] != 1:
                return 0.0
            if diff[idx + 1] + diff[idx] != 0:
                return 0.0
        else:  # len(nonzero_diff) > 2
            return 0.0
        if diff[idx] != 1 and diff[idx] != -1:
            return 0.0

        E_u = self._calc_cap_energy(np.array(u))
        E_v = self._calc_cap_energy(np.array(v))

        mu_L = self.physics.V_L * self.physics.q
        mu_R = self.physics.V_R * self.physics.q

        weight = 0.0

        if num_isls == 1:
            if diff[0] == 1:
                weight = self.p_WKB[0] * self._fermi(E_v - E_u - mu_L) \
                       + self.p_WKB[1] * self._fermi(E_v - E_u - mu_R)
            elif diff[0] == -1:
                weight = self.p_WKB[0] * (1 - self._fermi(E_u - E_v - mu_L)) \
                       + self.p_WKB[1] * (1 - self._fermi(E_u - E_v - mu_R))

        # Multi-island handling
        elif num_isls > 1:
            # Electrons coming or leaving from edges
            if idx == 0 and diff[idx] == 1:
                weight = self.p_WKB[0] * self._fermi(E_v - E_u - mu_L)
            elif idx == num_isls - 1 and diff[idx] == 1:
                weight = self.p_WKB[-1] * self._fermi(E_v - E_u - mu_R)
            elif idx == 0 and diff[idx] == -1:
                weight = self.p_WKB[0] * (1 - self._fermi(E_u - E_v - mu_L))
            elif idx == num_isls - 1 and diff[idx] == -1:
                weight = self.p_WKB[-1] * (1 - self._fermi(E_u - E_v - mu_R))
            # This gets interdot transitions
            elif diff[idx] == 1 and diff[idx + 1] == -1:
                weight = self.p_WKB[idx + 1] * self._fermi(E_v - E_u)
            elif diff[idx] == -1 and diff[idx + 1] == 1:
                weight = self.p_WKB[idx + 1] * self._fermi(E_v - E_u)

        return weight

    def _create_graph(self) -> networkx.DiGraph:
        """
        Creates the Markov graph of charge configurations.

        Nodes are tuples of integers representing the number of particles on
        each island. The graph is a directed graph with weights equal to:
        ``self.calc_wieght(u, v)``.
        All nodes in the graph differ from the least-energy configuration
        by a number of particles no more than
        ``self.numerics.create_graph_max_changes``.

        This method sets ``self.start_node`` equal to ``tuple(self.island_charges)``,
        and sets ``self.G`` equal to the resulting graph.

        Returns
        -------
        networkx.DiGraph
            ``self.G``, a graph with tuple[int, ...] nodes representing possible
            charge configurations. This graph is a Markov graph where edges
            represent the transition rates between possible configurations.
        """
        max_changes = self.numerics.create_graph_max_changes

        self.G = networkx.DiGraph()

        self.start_node = tuple(self.island_charges)

        num_isls = len(self.islands)

        if num_isls == 0:
            self.G.add_node(())
            return self.G

        # any deltas are allowed such that sum(abs(delta)) <= max_changes
        def calc_deltas(num_isl, max_ch):
            n = num_isl
            m = max_ch
            tpls = []

            def add_tuples(l):
                # add tuple(l) and copies with any combinations of signs changed
                nz = np.nonzero(l)[0]
                l2 = np.array(l)
                for p in itertools.product([1, -1], repeat=len(nz)):
                    np.put(l2, nz, p)
                    tpls.append(tuple(l2 * l))

            # calculate nonnegative length-n tuples that add up to m or less
            c = [0] * n
            sum = 0
            d = n - 1
            add_tuples(c)
            while d >= 0:
                if sum < m:
                    c[d] += 1
                    sum += 1
                    d = n - 1
                    add_tuples(c)
                else:
                    for i in range(d, n):
                        sum -= c[i]
                        c[i] = 0
                    d -= 1
            return tpls

        deltas = calc_deltas(num_isls, max_changes)

        # add nodes to graph
        for delta in deltas:
            node = np.array(self.start_node) + delta
            # Check for negative num of electrons
            if np.all(node >= 0):
                self.G.add_node(tuple(node))

        for x in list(self.G.nodes()):
            for y in list(self.G.nodes()):
                if x != y:
                    self.G.add_edge(x, y, weight=self._calc_weight(x, y))
        return self.G

    def _calc_stable_dist(self) -> NDArray[np.float64]:
        """
        Calculates the steady-state of the Markov graph.

        Returns
        -------
        ndarray[float]
            An array with length equal to len(G) representing the steady-state
            of the Markov graph.
        """
        # Adjacency matrix, caution not the Markov matrix
        a_mat = networkx.to_numpy_array(self.G)

        M = a_mat.T - np.diag(np.sum(a_mat, axis=1))

        M_solver = np.append(M[:-1, :], [np.ones(M.shape[0])]).reshape(M.shape)
        b = np.zeros(M.shape[0])
        b[-1] = 1

        # fix if one node is disconnected from graph
        for i in range(len(a_mat)):
            if np.sum(a_mat[i,:]) == 0 and np.sum(a_mat[:,i]) == 0 and (list(self.G.nodes)[i] != self.start_node):
                M_solver[i,i] = 1
                M_solver[-1,i] = 0

        self.dist = np.linalg.solve(M_solver, b)


        return self.dist

    def _calc_graph_charge(self):
        """
        Calculates the average charge configuration of the steady-state of the
        Markov graph.
        Sets ``self.graph_charge`` equal to the result.
        """
        if len(self.islands) == 0:
            graph_charge = ()
        else: 
            max_index = np.argmax(self.dist)
            graph_charge = list(self.G.nodes())[max_index]
        self.graph_charge = graph_charge
        return self.graph_charge

    def _calc_graph_current(self) -> float:
        """
        Calculates the current running through the wire, from the Markov graph.
        Sets ``self.current`` equal to the result.

        Returns
        -------
        float
            ``self.current``, the current running through the wire,
            defined to be the sum over all tunneling events of:
            (rate at which the tunneling event occurs) *
            (``dist`` probability of initial node of tunneling event) *
            (change in number of particles in the Left sink).
            A positive value indicates particles traveling from left to right.
        """
        mu_L = self.physics.V_L * self.physics.q
        current = 0.0
        if len(self.islands) == 0:
            current = (self.physics.short_circuit_current if self.is_short_circuit \
                       else self.physics.barrier_current) 
        elif len(self.islands) == 1:
            for u in list(self.G.nodes()):
                plus = tuple(np.array(u) + 1)
                minus = tuple(np.array(u) - 1)
                index_u = list(self.G.nodes()).index(u)
                E_u = self._calc_cap_energy(np.array(u))
                if plus in self.G:
                    E_plus = self._calc_cap_energy(np.array(plus))
                    gamma_plus = self.p_WKB[0] * self._fermi(E_plus - E_u - mu_L)
                    current += self.dist[index_u] * gamma_plus
                if minus in self.G:
                    E_minus = self._calc_cap_energy(np.array(minus))
                    gamma_minus = self.p_WKB[0] * self._fermi(E_minus + mu_L - E_u)
                    current -= self.dist[index_u] * gamma_minus
        else:  # len(self.islands) >= 2
            for u in list(self.G.nodes()):
                plus = tuple(
                    np.array(u) + np.pad(np.array([1]), (0, len(u) - 1), "constant")
                )
                minus = tuple(
                    np.array(u) - np.pad(np.array([1]), (0, len(u) - 1), "constant")
                )
                index_u = list(self.G.nodes()).index(u)
                if plus in self.G:
                    gamma_plus = self.G[u][plus]["weight"]
                    current += self.dist[index_u] * gamma_plus
                if minus in self.G:
                    gamma_minus = self.G[u][minus]["weight"]
                    current -= self.dist[index_u] * gamma_minus
        self.current = current
        return self.current

    def _calc_current(self):
        """
        Calculates the current running through the wire.
        Sets ``self.current`` equal to the result.
        This function takes care of a few edge cases, and otherwise calls
        ``self.calc_graph_current()``

        Returns
        -------
        float
            ``self.current``, the current running through the wire,
            defined to be the sum over all tunneling events of:
            (rate at which the tunneling event occurs) *
            (``dist`` probability of initial node of tunneling event) *
            (change in number of particles in the Left sink).
            A positive value indicates particles traveling from left to right.
        """
        if self.is_short_circuit:
            current = self.physics.short_circuit_current
        elif len(self.islands) == 0:
            current = self.physics.barrier_current
        else:
            self._create_graph()
            self._calc_stable_dist()
            current = self._calc_graph_current()
        self.current = current
        return self.current

    def _count_transitions(self) -> int:
        """
        Calculates the number of transitions, defined to be the number of
        graph nodes adjacent to ``self.start_node`` such that
        the incoming and outgoing weights between the nodes and
        ``self.start_node`` are (close to) equal.

        Sets ``self.trans_count`` equal to the result.

        Returns
        -------
        int
            ``self.trans_count``, the number of graph nodes adjacent to
            ``self.start_node`` such that the incoming and outgoing weights between
            the nodes and ``self.start_node`` are (close to) equal.
        """        
        eps = self.numerics.count_transitions_eps

        # Minimum weight to accept as a transition
        sigma = self.numerics.count_transitions_sigma

        # Only calculate transitions if we know there are islands
        if len(self.islands) == 0:
            self.trans_count = 0
            return self.trans_count
        else:
            trans_count = 0
            # Call most probable node the charge config (as in calc_graph_charge())
            stable_node = list(self.G.nodes)[np.argmax(self.dist)]

            # Loop through all neighbors of stable node
            for nbr in self.G[stable_node]:
                # If the weights of in/out edges are similar, then
                #    call it a transition
                w_in = self.G[nbr][stable_node]["weight"]
                w_out = self.G[stable_node][nbr]["weight"]

                # Calculate difference relative to weight size
                if (
                    (w_in > sigma or w_out > sigma)
                    and abs(w_in - w_out) / (0.5 * (w_in + w_out)) < eps
                ):
                    trans_count += 1
            self.trans_count = trans_count
            return self.trans_count

    def _sensor_from_island_charges(
        self, island_charges: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calculates the output of the charge sensors if the system were to be
        prepared to a given charge state.

        Parameters
        ----------
        island_charges : ndarray[float]
            An array indicating the number of charges on each island.

        Returns
        -------
        ndarray[float]
            An array indicating the potential readout for each sensor in ``physics.sensors``.
        """

        def calc_single_sensor(pos):
            (sx, sy, sz) = pos
            ss = np.sqrt(sy**2 + sz**2)
            sensor_scale = np.exp(-ss / self.physics.screening_length) / ss
            output = 0
            # calculate field at sensor due to charge islands
            for i in range(len(self.islands)):
                n_i = self.n[self.islands[i, 0] : self.islands[i, 1]]
                n_sum = np.sum(n_i)
                x_i = self.physics.x[self.islands[i, 0] : self.islands[i, 1]]
                r = np.sqrt((sx - x_i) ** 2 + sy**2 + sz**2)
                output += (
                    (self.physics.q * island_charges[i]) / sensor_scale
                    * np.sum(n_i * np.exp(-r / self.physics.screening_length) / r) / n_sum
                )
            return output

        sensor_output = np.zeros(len(self.physics.sensors), dtype=np.float64)
        for i, pos in enumerate(self.physics.sensors):
            sensor_output[i] = calc_single_sensor(pos)
        return sensor_output

    def _calc_sensor(self) -> NDArray[np.float64]:
        """
        Calculates the output of the charge sensors by summing the Coulomb
        potential contributions from the charge islands, evaluated at
        each sensor location.

        Sets ``self.sensor_output`` equal to the result.

        Returns
        -------
        ndarray[float]
            ``self.sensor_output``, an array indicating the Coulomb potential
            readout for each sensor in ``physics.sensors``.
        """
        self.sensor_output = self._sensor_from_island_charges(
            np.array(self.island_charges, dtype=np.float64)
        )
        return self.sensor_output

    def _calc_dot_states(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.int_]]:
        """
        Calculates whether each quantum dot in the system is occupied or combined,
        and matches the charges from ``self.island_charges`` to their respective dots.

        This function relies on the expected regions where dots can be formed to
        be defined via ``self.physics.dot_regions``. (If regions are not defined,
        defaults will be generated assuming an alternating pattern of barrier
        and plunger gates).

        There are several edge cases which can occur with a poor potential setup
        or a poor choice of ``dot_regions``. They are handled as follows:

        If an island with at least one charge lies between two dot regions,
        its charge will be added to the region which it is closer to.

        If an island with at least one charge lies to the far left or far right
        of all dot regions, it will be ignored.

        If a region has multiple islands (with charges) overlapping it, and these
        islands don't overlap other regions, then charges from all the islands
        will be added together as if they were a single dot.

        If a region has multiple islands (with charges) overlapping it, and at
        least one of these islands overlap one or more other regions,
        then that island will not be included as part of the region unless
        it overlaps the region by a greater area than any other island, or unless
        it is left with no other region to be associated with.

        Returns
        -------
        are_dots_occupied : ndarray[bool]
            An array of booleans, one for each dot,
            indicating whether each dot is occupied.
        are_dots_combined : ndarray[bool]
            An array of booleans, one for each internal barrier,
            indicating whether the dots on each side are combined together
            (i.e. the barrier is too low).
            ``len(are_dots_combined)`` will always equal ``len(are_dots_occupied) - 1``.
            ``are_dots_combined[i]`` is always false if either ``are_dots_occupied[i]``
            or ``are_dots_occupied[i+1]`` is false. If both are true, ``are_dots_combined[i]``
            may be true or false depending on the barrier height.
        dot_charges : ndarray[int]
            An array of integers, one for each dot, indicating the total number
            of charges in each dot. In the case of combined dots, the
            total number of charges will be entered in the left-most dot,
            with the other dots padded with zeros.
        """
        x = self.physics.x
        if self.physics.dot_regions is not None:
            dot_regions = self.physics.dot_regions
            n_dots = len(dot_regions)
        else:
            g_means = [g.mean for g in self.physics.gates]
            n_dots = (len(g_means) - 1) // 2
            # Regions are defined as running from:
            # (the midpoint between the plunger gate and the barrier gate to the left)
            # to (the midpoint between the plunger gate and the barrier gate to the right).
            # This leaves some space around each barrier gate that is not in any region.
            dot_regions = np.zeros((n_dots, 2))
            for i in range(n_dots):
                dot_regions[i] = [
                    (g_means[2 * i] + g_means[2 * i + 1]) / 2,
                    (g_means[2 * i + 1] + g_means[2 * i + 2]) / 2,
                ]

        are_dots_occupied = np.zeros(n_dots, dtype=np.bool_)  # False
        are_dots_combined = np.zeros(max(n_dots - 1, 0), dtype=np.bool_)  # False
        dot_charges = np.zeros(n_dots, dtype=np.int_)

        # only consider occupied islands
        full_isl_ind = np.nonzero(self.island_charges)[0]
        full_islands = np.take(self.islands, full_isl_ind, axis=0)
        full_charge = np.take(self.island_charges, full_isl_ind)
        full_center = np.take(self.charge_centers, full_isl_ind)
        n_full_isl = len(full_isl_ind)

        # check which islands overlap which dots
        overlaps = np.zeros((n_dots, n_full_isl), dtype=np.float64)
        for d in range(n_dots):
            dot_start = dot_regions[d, 0]
            dot_end = dot_regions[d, 1]
            for i in range(n_full_isl):
                isl_start = x[full_islands[i, 0]]
                isl_end = x[full_islands[i, 1] - 1]
                if dot_start < isl_end and dot_end > isl_start:
                    overlaps[d, i] = min(isl_end, dot_end) - max(isl_start, dot_start)
        is_overlap = overlaps > 0
        is_matched = is_overlap.copy()
        # if dot has multiple islands, and one of those islands spans multiple dots,
        # remove that island overlap unless it has the biggest overlap with the dot
        for d in range(n_dots):
            if np.sum(is_overlap[d, :]) > 1:
                for i in range(n_full_isl):
                    if np.sum(is_overlap[:, i]) > 1:
                        for i2 in range(n_full_isl):
                            if i != i2 and overlaps[d, i2] > overlaps[d, i]:
                                is_matched[d, i] = False
        # if an island matches with no dot due to the previous step,
        # match it with its largest overlap
        for i in range(n_full_isl):
            if np.sum(is_matched[:, i]) == 0 and np.sum(is_overlap[:, i]) > 1:
                d_match = np.argmax(overlaps[:, i])
                is_matched[d_match, i] = True
        # if an island matches no dot, match it with the closest dot
        # as long as it is between the left and rightmost regions
        dot_mids = (dot_regions[:, 0] + dot_regions[:, 1]) / 2
        for i in range(n_full_isl):
            if np.sum(is_overlap[:, i]) == 0:
                center = full_center[i]
                if center > dot_regions[0, 0] and center < dot_regions[-1, 1]:
                    d_match = np.argmin(np.abs(dot_mids - center))
                    is_matched[d_match, i] = True

        combined_dot = 0
        combined_isl = np.array([], dtype=np.int_)
        for d in range(n_dots):
            matches = np.nonzero(is_matched[d, :])[0]
            is_combined = False  # whether dot d is combined with the previous dots
            if len(matches) == 0:
                are_dots_occupied[d] = False
            else:
                are_dots_occupied[d] = True
                for i in matches:
                    if i in combined_isl:
                        is_combined = True
            if is_combined:
                dot_charges[d] = 0
                are_dots_combined[d - 1] = True
                for i in matches:
                    if i not in combined_isl:
                        combined_isl = np.append(combined_isl, i)
            else:
                if d != 0:
                    dot_charges[combined_dot] = np.sum(
                        np.take(full_charge, combined_isl)
                    )
                    are_dots_combined[d - 1] = False
                combined_dot = d
                combined_isl = matches
        dot_charges[combined_dot] = np.sum(np.take(full_charge, combined_isl))

        self.are_dots_occupied = are_dots_occupied
        self.are_dots_combined = are_dots_combined
        self.dot_charges = dot_charges
        return self.are_dots_occupied, self.are_dots_combined, self.dot_charges

    def _island_charges_from_charge_state(
        self, dot_charges: NDArray[np.int_], are_dots_combined: NDArray[np.bool_]
    ) -> NDArray[np.int_]:
        """
        Calculates how many charges would be on each island if the system were
        prepared to a particular charge state.

        In particular, this function attempts to match ``dot_charges`` (which
        records the charges as a function of which dot they are a part of) with
        ``self.islands``.

        Parameters
        ----------
        dot_charges : ndarray[int]
            An array of integers, one for each dot, indicating the total number
            of charges in each dot. In the case of combined dots, the
            total number of charges should be entered in the left-most dot,
            with the other dots padded with zeros.
        are_dots_combined : ndarray[bool]
            An array of booleans, one for each internal barrier,
            indicating whether the dots on each side are combined together
            (i.e. the barrier is too low).
            ``len(are_dots_combined)`` should equal ``len(are_dots_occupied) - 1``.

        Returns
        -------
        island_charges : ndarray[int]
            An array of integers, one for each island, indicating the total number
            of charges in each island.
        """
        x = self.physics.x
        islands = self.islands
        n_isl = len(islands)

        if self.physics.dot_regions is not None:
            dot_regions = self.physics.dot_regions
            n_dots = len(dot_regions)
        else:
            g_means = [g.mean for g in self.physics.gates]
            n_dots = (len(g_means) - 1) // 2
            # Regions are defined as running from:
            # (the midpoint between the plunger gate and the barrier gate to the left)
            # to (the midpoint between the plunger gate and the barrier gate to the right).
            # This leaves some space around each barrier gate that is not in any region.
            dot_regions = np.zeros((n_dots, 2))
            for i in range(n_dots):
                dot_regions[i] = [
                    (g_means[2 * i] + g_means[2 * i + 1]) / 2,
                    (g_means[2 * i + 1] + g_means[2 * i + 2]) / 2,
                ]

        island_charges = np.zeros(len(self.islands), dtype=np.int_)
        if n_isl == 0:
            return island_charges

        # group combined dots together
        combined_dot_regions = []
        combined_dot_charges = []
        new_dot = True
        for d in range(n_dots):
            if new_dot:
                charges = 0
                left = dot_regions[d][0]
                new_dot = False
            charges += dot_charges[d]
            if (d == n_dots - 1) or (not are_dots_combined[d]):
                new_dot = True
                combined_dot_charges.append(charges)
                combined_dot_regions.append([left, dot_regions[d][1]])
        c_dot_regions = np.array(combined_dot_regions)
        c_dot_charges = np.array(combined_dot_charges)
        n_c_dots = len(c_dot_regions)

        # check which islands overlap which combined dots
        is_overlap = np.full((n_c_dots, n_isl), False, dtype=np.bool_)
        for d in range(n_c_dots):
            dot_start = c_dot_regions[d, 0]
            dot_end = c_dot_regions[d, 1]
            for i in range(n_isl):
                isl_start = x[islands[i, 0]]
                isl_end = x[islands[i, 1] - 1]
                if dot_start < isl_end and dot_end > isl_start:
                    is_overlap[d, i] = True

        c_dot_mid = (c_dot_regions[:, 0] + c_dot_regions[:, 1]) / 2
        isl_mid = np.array(
            [(x[islands[i, 0]] + x[islands[i, 1] - 1]) / 2 for i in range(n_isl)],
            dtype=np.float64,
        )
        c_dot_regions_idx = np.zeros(c_dot_regions.shape, dtype=np.int_)
        for d in range(n_c_dots):
            c_dot_regions_idx[d, 0] = (
                np.argmin(
                    np.where(c_dot_regions[d, 0] >= x, c_dot_regions[d, 0] - x, np.inf)
                )
                if c_dot_regions[d, 0] > x[0]
                else 0
            )
            c_dot_regions_idx[d, 1] = (
                1 + np.argmin(
                    np.where(c_dot_regions[d, 1] <= x, x - c_dot_regions[d, 1], np.inf)
                )
                if c_dot_regions[d, 1] < x[-1]
                else len(x)
            )

        for d in range(n_c_dots):
            overlaps = np.nonzero(is_overlap[d])[0]
            if len(overlaps) == 0:
                isl_match = np.argmin(np.abs(c_dot_mid[d] - isl_mid))
                island_charges[isl_match] += c_dot_charges[d]
            elif len(overlaps) == 1:
                island_charges[overlaps[0]] += c_dot_charges[d]
            else:
                overlap_weight = np.array([
                    np.sum(self.n[
                        max(islands[i, 0], c_dot_regions_idx[d, 0]) : \
                        min(islands[i, 1], c_dot_regions_idx[d, 1])
                    ]) for i in overlaps
                ], dtype=np.float64)
                if np.sum(overlap_weight) == 0:
                    overlap_weight = np.ones(overlap_weight.shape)
                overlap_weight = (
                    c_dot_charges[d] * overlap_weight / np.sum(overlap_weight)
                )
                overlap_charges = np.array(np.around(overlap_weight), dtype=np.int_)
                if np.sum(overlap_charges) < c_dot_charges[d]:
                    for a in np.argsort(overlap_charges - overlap_weight)[
                        0 : (c_dot_charges[d] - np.sum(overlap_charges))
                    ]:
                        overlap_charges[a] += 1
                while np.sum(overlap_charges) > c_dot_charges[d]:
                    over = np.sum(overlap_charges) - c_dot_charges[d]
                    for a in np.argsort(overlap_weight - overlap_charges):
                        if over > 0 and overlap_charges[a] > 0:
                            overlap_charges[a] -= 1
                            over -= 1
                for i in range(len(overlaps)):
                    island_charges[overlaps[i]] += overlap_charges[i]

        return island_charges

    def sensor_from_charge_state(
        self, dot_charges: NDArray[np.int_], are_dots_combined: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        Calculates what the sensor output would be if the system were prepared
        to a particular charge state.

        ``run_calculations()`` must be run on this ``ThomasFermi`` instance
        before using this function.

        Parameters
        ----------
        dot_charges : ndarray[int]
            An array of integers, one for each dot, indicating the total number
            of charges in each dot. In the case of combined dots, the
            total number of charges should be entered in the left-most dot,
            with the other dots padded with zeros.
        are_dots_combined : ndarray[bool]
            An array of booleans, one for each internal barrier,
            indicating whether the dots on each side are combined together
            (i.e. the barrier is too low).
            ``len(are_dots_combined)`` should equal ``len(are_dots_occupied) - 1``.

        Returns
        -------
        sensor : ndarray[float]
            An array of floats, one for each sensor, indicating the voltage
            at each sensor.
        """
        return self._sensor_from_island_charges(
            np.array(
                self._island_charges_from_charge_state(dot_charges, are_dots_combined),
                dtype=np.float64,
            )
        )

    def run_calculations(
        self,
        *,
        inc_inv_cap_matrix: bool = False,
        inc_curr: bool = False,
        inc_trans: bool = False,
        n_guess: NDArray[np.float64] | None = None,
    ) -> ThomasFermiOutput:
        """
        Run the simulation, and return a dict with the relevant results.

        Parameters
        ----------
        inc_inv_cap_matrix : bool, optional
            Whether to include the capacitance model and WKB probabilities in
            the output
        inc_curr : bool, optional
            Whether to perform current calculations and include the results in
            the output
        inc_trans : bool, optional
            Whether to perform current calculations and include the transition
            count in the output
        n_guess : ndarray[float], optional
            Initial guess (in 1/nm) for the particle density n(x).
            If ``None``, an array of zeros will be used.

        Returns
        -------
        output : ThomasFermiOutput
            A ``ThomasFermiOutput`` object with the relevant results of the calculation.
        """
        output = ThomasFermiOutput()

        self._calc_n(n_guess=n_guess)
        self._calc_qV_TF()
        self._calc_islands_and_barriers()
        self._calc_WKB_prob()
        self._calc_charges()
        self._calc_charge_centers()
        self._calc_inv_cap_matrix()
        self._calc_stable_config()
        self._calc_dot_states()
        self._calc_sensor()

        output.island_charges = np.array(self.island_charges)
        output.sensor = np.array(self.sensor_output)
        output.are_dots_occupied = self.are_dots_occupied
        output.are_dots_combined = self.are_dots_combined
        output.dot_charges = self.dot_charges
        output.converged = self.converged

        if inc_curr:
            self._calc_current()
            self._calc_graph_charge()
            output.current = self.current

        output.n = np.array(self.n)

        if not inc_curr and inc_trans:
            self._create_graph()
            self._calc_stable_dist()
        if inc_trans:
            self._count_transitions()
            output.transition_count = self.trans_count

        if inc_inv_cap_matrix:
            output.inv_cap_matrix = np.array(self.inv_cap_matrix)
        
        return output


def is_transition(
    dot_charges_1: NDArray[np.int_],
    are_dots_combined_1: NDArray[np.bool_],
    dot_charges_2: NDArray[np.int_],
    are_dots_combined_2: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Determines whether a transition occurs between two points based off their
    charge states.

    Parameters
    ----------
    dot_charges_1 : ndarray[int]
        An array with length ``n_dots`` indicating how many electrons are in
        each dot for the first point.
        In the case of combined dots, the total number of charges should be
        entered in the left-most slot, with the other slots padded with zeros.
    are_dots_combined_1 : ndarray[bool]
        An array with length ``n_dots-1``, indicating whether the dots on either
        side of each barrier are combined together.
    dot_charges_2 : ndarray[int]
        The dot charges for the second point.
    are_dots_combined_2 : ndarray[bool]
        Whether the dots are combined together for the second point.

    Returns
    -------
    is_transition : ndarray[bool]
        An array with length ``n_dots`` indicating whether a transition is
        present in a particular dot. A transition occurs at a particular pixel
        and dot if the number of charges in that dot differ in any adjecent pixels.
    is_transition_combined : ndarray[bool]
        An array with length ``n_dots-1`` indicating whether there is a
        transition in a combined dot on either side of a particular barrier.
    """
    n_dots = len(dot_charges_1)
    is_trans = np.full(n_dots, False, dtype=np.bool_)
    is_trans_combined = np.full(n_dots - 1, False, dtype=np.bool_)
    d_end = 0
    while d_end < n_dots:
        d_start = d_end
        while d_end < n_dots - 1 and (
            are_dots_combined_1[d_end] or are_dots_combined_2[d_end]
        ):
            d_end += 1
        if np.sum(dot_charges_1[d_start : d_end + 1]) != np.sum(
            dot_charges_2[d_start : d_end + 1]
        ):
            is_trans_combined[d_start:d_end] = True
            is_trans[d_start : d_end + 1] = True
        d_end += 1
    return is_trans, is_trans_combined
