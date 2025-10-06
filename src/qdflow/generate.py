'''
This module contains classes and functions for generating CSDs and ray data.

Data generation is done with the ``calc_csd()``, calc_2d_csd(), and
``calc_rays()`` functions. These functions input a ``PhysicsParameters`` object,
as well as the desired resolution and voltages to sweep. The functions then
run the physics simulation once for each point in the desired range, and return
the resulting data in a single instance of the ``CSDOutput`` or ``RaysOutput``
dataclasses.

Often, diagrams with a wide variety of features are desired in a dataset.
This is accomplished by randomizing the values of the ``PhysicsParameters``
object for each diagram. ``random_physics()`` provides a convenient way of
performing this randomization, and the distributions from which each physics
parameter should be drawn can be set via the ``PhysicsRandomization`` dataclass. 

Examples
--------

>>> from qdflow import generate
>>> from qdflow.util import distribution
>>> phys_rand = generate.PhysicsRandomization.default()
>>> phys_rand.mu = distribution.Uniform(0, 1.2) # adjust parameter ranges here

>>> n_devices = 10
>>> phys_params = generate.random_physics(phys_rand, n_devices)

This will generate a list of 10 random sets of device parameters.

>>> import numpy as np
>>> phys = phys_params[0]

>>> # Set ranges and resolution of plunger gate sweeps
>>> V_x = np.linspace(2., 16., 100)
>>> V_y = np.linspace(2., 16., 100)

>>> csd = generate.calc_2d_csd(phys, V_x, V_y)

``csd`` is a single instance of the ``CSDOutput`` dataclass, which contains the
results of the calculations. The sensor readout can be obtained as a numpy array
by using ``csd.sensor[:, :, sensor_num]``, where ``sensor_num`` is the index of
the desired sensor (0 if just one sensor).
'''

import numpy as np
from typing import Any, Self, overload, ClassVar, TypeVar
from numpy.typing import NDArray
from .physics import simulation as simulation
from .util import distribution as distribution
import dataclasses
from dataclasses import dataclass, field
import copy

T = TypeVar('T')


_rng = np.random.default_rng()

def set_rng_seed(seed):
    '''
    Initializes a new random number generator with the given seed,
    used to generate random data.

    Parameters
    ----------
    seed : {int, array_like[int], SeedSequence, BitGenerator, Generator}
        The seed to use to initialize the random number generator.
    '''
    global _rng
    _rng = np.random.default_rng(seed)



@dataclass(kw_only=True)
class CSDOutput:
    '''
    Output of charge stability diagram calculations. Some attributes may be ``None``
    depending on which quantities are calculated.

    Attributes
    ----------
    physics : PhysicsParameters
        The set of physics parameters used in the simulation. 
    V_x, V_y : ndarray[float]
        Arrays of voltage values along the x- and y-axes, which together define
        the coordinates of each of the pixels.
    x_gate, y_gate : int
        The indeces of the dots whose gate voltages are plotted on the
        x- or y-axes.
    V_gates : ndarray[float]
        An array of length ``n_dots`` giving the voltages of each of the plunger
        gates. This is relevant only for plunger gates whose voltages remain
        constant over the whole diagram, and it contains values of the constant
        voltages. For the two plunger gates corresponding to the x- and y-axes
        (``x_gate`` and ``y_gate``), the value of ``V_gates`` is arbitrary.
    sensor : ndarray[float]
        An array with shape ``(len(V_x), len(V_y), n_sensors)`` giving the
        Coulomb potential at each point at a specific sensor.
    are_dots_occupied : ndarray[bool]
        An array with shape ``(len(V_x), len(V_y), n_dots)``, indicating whether
        each dot is occupied at each pixel in the diagram.
    are_dots_combined : ndarray[bool]
        An array with shape ``(len(V_x), len(V_y), n_dots-1)``, indicating
        at each pixel in the diagram, whether the dots on each side of an
        internal barrier are combined together (i.e. the barrier is too low).
    dot_charges : ndarray[int]
        An array with shape ``(len(V_x), len(V_y), n_dots)``, indicating the
        total number of charges in each dot at each pixel in the diagram.
        In the case of combined dots, the total number of charges will be
        entered in the left-most spot, with the other spots padded with zeros.
    converged : ndarray[bool] | None
        An array with shape ``(len(V_x), len(V_y))``, indicating whether the
        calculation of n(x) properly converged at each pixel in the diagram.
    dot_transitions : ndarray[bool] | None
        An array with shape ``(len(V_x), len(V_y), n_dots)``, indicating whether
        a transition in each dot occurs at each pisel in the diagram.
    are_transitions_combined : ndarray[bool] | None
        An array with shape ``(len(V_x), len(V_y), n_dots-1)``, indicating at
        each pixel in the diagram, whether a transition occurs on a combined dot
        comprised of dots on either side of each internal barrier.
    excited_sensor : ndarray[float] | None
        An array with shape ``(len(V_x), len(V_y), n_sensors)`` giving the
        Coulomb potential at an excited state at each point at a specific sensor.
    '''
    physics:simulation.PhysicsParameters=field(default_factory=lambda:simulation.PhysicsParameters())
    V_x:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    V_y:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    x_gate:int=0
    y_gate:int=0
    V_gates:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    sensor:NDArray[np.float32]=field(default_factory=lambda:np.zeros(0, dtype=np.float32))
    are_dots_occupied:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    are_dots_combined:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    dot_charges:NDArray[np.int_]=field(default_factory=lambda:np.zeros(0, dtype=np.int_))
    converged:NDArray[np.bool_]|None=None
    dot_transitions:NDArray[np.bool_]|None=None
    are_transitions_combined:NDArray[np.bool_]|None=None
    excited_sensor:NDArray[np.float32]|None=None
    current:NDArray[np.float32]|None=None
    
    def _get_physics(self) -> simulation.PhysicsParameters:
        return self._physics
    def _set_physics(self, val:simulation.PhysicsParameters):
        self._physics = val.copy()

    def _get_V_x(self) -> NDArray[np.float64]:
        return self._V_x
    def _set_V_x(self, val:NDArray[np.float64]):
        self._V_x = np.array(val, dtype=np.float64)

    def _get_V_y(self) -> NDArray[np.float64]:
        return self._V_y
    def _set_V_y(self, val:NDArray[np.float64]):
        self._V_y = np.array(val, dtype=np.float64)

    def _get_V_gates(self) -> NDArray[np.float64]:
        return self._V_gates
    def _set_V_gates(self, val:NDArray[np.float64]):
        self._V_gates = np.array(val, dtype=np.float64)

    def _get_sensor(self) -> NDArray[np.float32]:
        return self._sensor
    def _set_sensor(self, val:NDArray[np.float32]):
        self._sensor = np.array(val, dtype=np.float32)

    def _get_converged(self) -> NDArray[np.bool_]|None:
        return self._converged
    def _set_converged(self, val:NDArray[np.bool_]|None):
        self._converged = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_dots_occupied(self) -> NDArray[np.bool_]:
        return self._are_dots_occupied
    def _set_are_dots_occupied(self, val:NDArray[np.bool_]):
        self._are_dots_occupied = np.array(val, dtype=np.bool_)

    def _get_are_dots_combined(self) -> NDArray[np.bool_]:
        return self._are_dots_combined
    def _set_are_dots_combined(self, val:NDArray[np.bool_]):
        self._are_dots_combined = np.array(val, dtype=np.bool_)

    def _get_dot_charges(self) -> NDArray[np.int_]:
        return self._dot_states
    def _set_dot_charges(self, val:NDArray[np.int_]):
        self._dot_states = np.array(val, dtype=np.int_)

    def _get_dot_transitions(self) -> NDArray[np.bool_]|None:
        return self._dot_transitions
    def _set_dot_transitions(self, val:NDArray[np.bool_]|None):
        self._dot_transitions = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_transitions_combined(self) -> NDArray[np.bool_]|None:
        return self._are_transitions_combined
    def _set_are_transitions_combined(self, val:NDArray[np.bool_]|None):
        self._are_transitions_combined = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_excited_sensor(self) -> NDArray[np.float32]|None:
        return self._excited_sensor
    def _set_excited_sensor(self, val:NDArray[np.float32]|None):
        self._excited_sensor = np.array(val, dtype=np.float32) if val is not None else None

    def _get_current(self) -> NDArray[np.float32]|None:
        return self._current
    def _set_current(self, val:NDArray[np.float32]|None):
        self._current = np.array(val, dtype=np.float32) if val is not None else None


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``CSDOutput`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        CSDOutput
            A new ``CSDOutput`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                if k == "physics":
                    setattr(output, k, simulation.PhysicsParameters.from_dict(v))
                else:
                    setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``CSDOutput`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``CSDOutput`` object.
        '''
        return dataclasses.asdict(self)
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``CSDOutput`` object.

        Returns
        -------
        CSDOutput
            A new ``CSDOutput`` object with the same attribute values as ``self``.
        '''
        return dataclasses.replace(self)

CSDOutput.physics = property(CSDOutput._get_physics, CSDOutput._set_physics) # type: ignore
CSDOutput.V_x = property(CSDOutput._get_V_x, CSDOutput._set_V_x) # type: ignore
CSDOutput.V_y = property(CSDOutput._get_V_y, CSDOutput._set_V_y) # type: ignore
CSDOutput.V_gates = property(CSDOutput._get_V_gates, CSDOutput._set_V_gates) # type: ignore
CSDOutput.sensor = property(CSDOutput._get_sensor, CSDOutput._set_sensor) # type: ignore
CSDOutput.converged = property(CSDOutput._get_converged, CSDOutput._set_converged) # type: ignore
CSDOutput.are_dots_occupied = property(CSDOutput._get_are_dots_occupied, CSDOutput._set_are_dots_occupied) # type: ignore
CSDOutput.are_dots_combined = property(CSDOutput._get_are_dots_combined, CSDOutput._set_are_dots_combined) # type: ignore
CSDOutput.dot_charges = property(CSDOutput._get_dot_charges, CSDOutput._set_dot_charges) # type: ignore
CSDOutput.dot_transitions = property(CSDOutput._get_dot_transitions, CSDOutput._set_dot_transitions) # type: ignore
CSDOutput.are_transitions_combined = property(CSDOutput._get_are_transitions_combined, CSDOutput._set_are_transitions_combined) # type: ignore
CSDOutput.excited_sensor = property(CSDOutput._get_excited_sensor, CSDOutput._set_excited_sensor) # type: ignore
CSDOutput.current = property(CSDOutput._get_current, CSDOutput._set_current) # type: ignore



@dataclass(kw_only=True)
class PhysicsRandomization:
    '''
    Meta-parameters used to determine how random ``PhysicsParameters`` should
    be generated.

    Several attributes will not be randomized, and will be passed directly
    to the generated ``PhysicsParameters`` object.

    All other attributes should either be provided a single value
    (if no randmization is needed), or a ``distribution.Distribution`` object,
    from which the value will be drawn.
    
    Attributes
    ----------
    num_x_points : int
        The resolution of the x-axis. This value is not randomized.
    num_dots : int
        The number of dots. This value is not randomized.
    barrier_current : float
        An arbitrary low current set to the device when in barrier mode.
        This value is not randomized.
    short_circuit_current : float
        An arbitrary high current value given to the device when in
        open / short circuit mode. This value is not randomized.
    num_sensors : int
        The number of sensors to include. This value is not randomized.
    multiply_gates_by_q : bool
        Whether to multiply `barrier_peak`, `plunger_peak`, `barrier_peak_variations`,
        `plunger_peak_variations`, and `external_barrier_peak_variations` by `q`,
        changing the sign if ``q == -1``. Default True.
    dot_spacing : float | Distribution[float]
        The average distance (in nm) between dots.
    x_margins : float | Distribution[float]
        The length (in nm) of the nanowire to model on either end of the system.
        The total length of the nanowire will be:
        ``2 * (x_margins) + (num_dots - 1) * (dot_spacing)``.
    gate_x_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the total number of gates (barrier and plunger),
        giving the offset of the x-coordinate of each gate (in nm) relative to
        their positions if they were evenly spaced with spacing ``dot_spacing``.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    q : float | Distribution[float]
        The charge of a particle, -1 for electrons, +1 for holes.
    K_0 : float | Distribution[float]
        The electron-electron Coulomb interaction strength (in meV * nm)
    sigma : float | Distribution[float]
        The softening parameter (in nm) for the el-el Coulomb interaction used
        to avoid divergence when x = x'. `sigma` should be on the scale of
        the width of the nanowire.
    mu : float | Distribution[float]
        The Fermi level (in meV)
    g_0 : float | Distribution[float]
        The coefficient of the density of states
    V_L : float | Distribution[float]
        The voltage applied to left lead (in mV).
    V_R : float | Distribution[float]
        The voltage applied to right lead (in mV).
    beta : float | Distribution[float]
        The inverse temperature ``1/(k_B T)`` used to calculate ``n(x)``.
    kT : float | Distribution[float]
        The temperature ``(k_B T)`` used in the transport calculations.
    c_k : float | Distribution[float]
        The coefficient (in meV*nm) that determines the kenetic energy of the
        Fermi sea on each island.
    screening_length : float | Distribution[float]
        The screening length (in nm) for the Coulomb interaction.
    rho : float | Distribution[float]
        The radius (in nm) of the cylindrical gates.
    h : float | Distribution[float]
        The distance (in nm) of the gates from the nanowire.
    rho_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the total number of gates (barrier and plunger),
        giving a correction (in nm) which will be added to ``rho`` for each gate.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    h_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the total number of gates (barrier and plunger),
        giving a correction (in nm) which will be added to ``h`` for each gate.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    plunger_peak : float | Distribution[float]
        The peak value (in mV) of the potential at the nanowire due to the
        plunger gates.
    plunger_peak_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the number of plunger gates, giving a
        correction (in mV) which will be added to ``plunger_peak`` for each gate.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    barrier_peak : float | Distribution[float]
        The peak value (in mV) of the potential at the nanowire due to the
        barrier gates.
    external_barrier_peak_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length 2, giving a correction (in mV) which will be added
        to ``barrier_peak`` for each external barrier gate.
        If a float distribution is provided, an ndarray of the size 2
        will be generated by drawing from the distribution twice.
    internal_barrier_peak_variations : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the number of internal barrier gates, giving
        a correction (in mV) which will be added to ``barrier_peak`` for each gate.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    sensor_y : float | Distribution[float]
        The average y-coordinate (in nm) of the sensors.
    sensor_y_variation : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the total number of sensors, giving a
        correction (in nm) which will be added to ``sensor_y`` for each sensor.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    sensor_x_variation : float | ndarray[float] | Distribution[float] | Distribution[ndarray]
        An array with length equal to the total number of sensors, giving the
        offset of the x-coordinate of each sensor (in nm) relative to their
        positions if they were evenly spaced along the nanoire.
        If a float distribution is provided, an ndarray of the appropriate size
        will be generated by repeatedly drawing from the distribution.
    WKB_coef : float | Distribution[float]
        Coefficient (with units 1/(nm*sqrt(meV))) which goes in the exponent
        while calculating the WKB probability, setting the strength of WKB tunneling.
        WKB_coef should be equal to ``sqrt(2*m)/hbar``
        (converted to units of 1/(nm*sqrt(meV))), where ``m`` is the effective
        mass of a particle, and ``hbar`` is the reduced Planck's constant.
    v_F : float | Distribution[float]
        The fermi velocity (in nm/s).
    '''
    num_x_points:int=151
    num_dots:int=2
    barrier_current:float=1e-5
    short_circuit_current:float=1e4
    num_sensors:int=1
    multiply_gates_by_q:bool=True

    dot_spacing:float|distribution.Distribution[float]=200
    x_margins:float|distribution.Distribution[float]=200
    gate_x_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    q:float|distribution.Distribution[float]=-1
    K_0:float|distribution.Distribution[float]=5
    sigma:float|distribution.Distribution[float]=60
    mu:float|distribution.Distribution[float]=.5
    g_0:float|distribution.Distribution[float]=.0065
    V_L:float|distribution.Distribution[float]=-.01
    V_R:float|distribution.Distribution[float]=.01
    beta:float|distribution.Distribution[float]=100
    kT:float|distribution.Distribution[float]=.01
    c_k:float|distribution.Distribution[float]=1.2
    screening_length:float|distribution.Distribution[float]=100
    rho:float|distribution.Distribution[float]=15
    h:float|distribution.Distribution[float]=80
    rho_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    h_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    plunger_peak:float|distribution.Distribution[float]=-7
    plunger_peak_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    external_barrier_peak_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=2.
    barrier_peak:float|distribution.Distribution[float]=5.
    internal_barrier_peak_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    sensor_y:float|distribution.Distribution[float]=-250
    sensor_y_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    sensor_x_variations:float|distribution.Distribution[float]|distribution.Distribution[NDArray[np.float64]]|NDArray[np.float64]=0
    WKB_coef:float|distribution.Distribution[float]=.089
    v_F:float|distribution.Distribution[float]=3.0e13
    

    @classmethod
    def default(cls) -> Self:
        '''
        Creates a new ``PhysicsRandomization`` object with default values.

        Returns
        -------
        PhysicsRandomization
            A new ``PhysicsRandomization`` object with default values.
        '''
        output = cls(
            num_x_points=151,
            num_dots=2,
            barrier_current=1e-5,
            short_circuit_current=1e4,
            num_sensors=1,
            multiply_gates_by_q=True,
            dot_spacing=200,
            x_margins=200,
            gate_x_variations=0,
            q=-1,
            K_0=distribution.LogUniform(.5, 50),
            sigma=distribution.Uniform(40, 80),
            mu=.5,
            g_0=distribution.LogUniform(.0055, .008),
            V_L=distribution.Uniform(-.02, .02),
            V_R=distribution.Uniform(-.02, .02),
            beta=distribution.LogUniform(10, 1000),
            kT=distribution.LogUniform(.001, .1),
            c_k=distribution.LogUniform(.25, 6),
            screening_length=distribution.LogUniform(75, 150),
            rho=distribution.Uniform(10, 20),
            h=distribution.Normal(80, 5).abs(),
            rho_variations=0,
            h_variations=distribution.Normal(0,.15),
            plunger_peak=0,
            plunger_peak_variations=distribution.Uniform(-12,-2),
            external_barrier_peak_variations=distribution.Uniform(.5, 3.5),
            barrier_peak=5.,
            internal_barrier_peak_variations=distribution.Uniform(-1.5, 1.5),
            sensor_y=-250,
            sensor_y_variations=distribution.Uniform(-.5, .5),
            sensor_x_variations=distribution.Uniform(-.35, .35),
            WKB_coef=.089,
            v_F=3.0e13
        )
        return output


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``PhysicsRandomization`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        PhysicsRandomization
            A new ``PhysicsRandomization`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``PhysicsRandomization`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``PhysicsRandomization`` object.
        '''
        memo:dict[int, Any] = {}
        output = {}
        for f in dataclasses.fields(PhysicsRandomization):
            old_val = getattr(self, f.name)
            if id(old_val) in memo:
                output[f.name] = memo[id(old_val)]
            else:
                output[f.name] = copy.deepcopy(old_val, memo=memo)
        return output
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``PhysicsRandomization`` object.

        Returns
        -------
        CSDOutput
            A new ``PhysicsRandomization`` object with the same attribute values as ``self``.
        '''
        return copy.deepcopy(self)


def default_physics(n_dots:int=2) -> simulation.PhysicsParameters:
    '''
    Creates a new ``PhysicsParameters`` object initialized to a set of default values.

    Parameters
    ----------
    n_dots : int
        The number of dots in the device to model.

    Returns
    -------
    simulation.PhysicsParameters
        A default set of physics parameters.
    '''
    dot_spacing = 200
    x_points = 51 + 50 * n_dots
    x_size = 200 + dot_spacing * n_dots
    x = np.linspace(-x_size/2, x_size/2, x_points, endpoint=True)

    physics = simulation.PhysicsParameters(
        x=x, q=-1, K_0=5, sigma=60, mu=.5, g_0=.0065, V_L=-.01, V_R=.01,
        beta=100, kT=.01, c_k=1.2, screening_length=100, WKB_coef=.089,
        v_F=3.0e13, barrier_current=1e-5, short_circuit_current=1e4
    )
    
    def gate_peak(i):
        if i == 0 or i == 2*n_dots:
            return -7
        elif i % 2 == 0:
            return -5
        else:
            return 7
    gates = [simulation.GateParameters(mean=(i-n_dots)*dot_spacing/2, 
                    peak=gate_peak(i), rho=15, h=80, screen=100)
             for i in range(2*n_dots+1)]
    physics.gates = gates
    physics.sensors=np.array([[0, -250, 0]])
    return physics


@overload
def random_physics(randomization_params:PhysicsRandomization, num_physics:int) -> list[simulation.PhysicsParameters]: ...
@overload
def random_physics(randomization_params:PhysicsRandomization, num_physics:None=...) -> simulation.PhysicsParameters: ...

def random_physics(randomization_params:PhysicsRandomization, num_physics:int|None=None) \
                  -> simulation.PhysicsParameters|list[simulation.PhysicsParameters]:
    '''
    Creates a randomized set of physics parameters describing a QD device.

    Parameters
    ----------
    randomization_params : PhysicsRandomization
        Meta-parameters which indicate how the ``PhysicsParameters`` should be
        randomized.
    num_physics : int
        The number of ``PhysicsParameters`` sets to generate. 

    Returns
    -------
    simulation.PhysicsParameters
        The randomized set of physics parameters.
    '''
    global _rng
    r_p = randomization_params
    n_phys = 1 if num_physics is None else num_physics
    output = []

    def draw(dist:T|distribution.Distribution[T], rng:np.random.Generator) -> T:
        if isinstance(dist, distribution.Distribution):
            return dist.draw(rng)
        else:
            return dist
    
    def multidraw(
        dist: float | NDArray | distribution.Distribution[Any] | distribution.Distribution[NDArray],
        n: int,
        rng: np.random.Generator,
    ) -> NDArray:
        if isinstance(dist, distribution.Distribution):
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
        elif isinstance(dist, np.ndarray):
            return dist
        else:
            return np.full(n, dist)

    for phys_i in range(n_phys):
        n_dots = r_p.num_dots
        physics = default_physics(n_dots)
        physics.barrier_current = r_p.barrier_current
        physics.short_circuit_current = r_p.short_circuit_current
        dot_spacing = np.abs(draw(r_p.dot_spacing, _rng))
        x_margins = np.abs(draw(r_p.x_margins, _rng))
        x_len = 2 * x_margins + (n_dots-1) * dot_spacing
        physics.x = np.linspace(-x_len/2, x_len/2, r_p.num_x_points, endpoint=True)
        q = draw(r_p.q, _rng)
        physics.q = q
        physics.K_0 = np.abs(draw(r_p.K_0, _rng))
        physics.sigma = np.abs(draw(r_p.sigma, _rng))
        physics.mu = draw(r_p.mu, _rng)
        physics.g_0 = np.abs(draw(r_p.g_0, _rng))
        c_k = np.abs(draw(r_p.c_k, _rng))
        physics.c_k = c_k
        scr = np.abs(draw(r_p.screening_length, _rng))
        physics.screening_length = scr
        physics.WKB_coef = np.abs(draw(r_p.WKB_coef, _rng))
        physics.v_F = np.abs(draw(r_p.v_F, _rng))
        physics.V_R = draw(r_p.V_R, _rng)
        physics.V_L = draw(r_p.V_L, _rng)
        physics.kT = np.abs(draw(r_p.kT, _rng))
        physics.beta = np.abs(draw(r_p.beta, _rng))
        h = np.abs(draw(r_p.h, _rng))
        rho = np.abs(draw(r_p.rho, _rng))
        q_scale = q if r_p.multiply_gates_by_q else 1
        gates = [simulation.GateParameters(screen=scr) for i in range(2*n_dots+1)]
        h_var = multidraw(r_p.h_variations, 2*n_dots+1, _rng)
        rho_var = multidraw(r_p.rho_variations, 2*n_dots+1, _rng)
        gate_x_var = multidraw(r_p.gate_x_variations, 2*n_dots+1, _rng)
        for i in range(2*n_dots+1):
            gates[i].h = np.abs(h + h_var[i])
            gates[i].rho = np.abs(rho + rho_var[i])
            gates[i].mean = (i-n_dots)*dot_spacing/2 + gate_x_var[i]
        bar_peak = draw(r_p.barrier_peak, _rng)
        pl_peak = draw(r_p.plunger_peak, _rng)
        bar_peak_var = multidraw(r_p.internal_barrier_peak_variations, n_dots-1, _rng)
        ex_bar_peak_var = multidraw(r_p.external_barrier_peak_variations, 2, _rng)
        pl_peak_var = multidraw(r_p.plunger_peak_variations, n_dots, _rng)
        gates[0].peak = (bar_peak + ex_bar_peak_var[0]) * q_scale
        gates[2*n_dots].peak = (bar_peak + ex_bar_peak_var[1]) * q_scale
        for i, g in enumerate(range(1, 2*n_dots, 2)):
            gates[g].peak = (pl_peak + pl_peak_var[i]) * q_scale
        for i, g in enumerate(range(2, 2*n_dots, 2)):
            gates[g].peak = (bar_peak + bar_peak_var[i]) * q_scale
        physics.gates = gates
        n_sens = r_p.num_sensors
        sensor_y = draw(r_p.sensor_y, _rng)
        sensor_x_var = multidraw(r_p.sensor_x_variations, n_sens, _rng)
        sensor_y_var = multidraw(r_p.sensor_y_variations, n_sens, _rng)
        sensors = np.zeros((n_sens, 3), dtype=np.float64)
        for i in range(n_sens):
            sensors[i] = (((i+1)*x_len/(n_sens+1)-x_len/2 + sensor_x_var[i],
                        sensor_y + sensor_y_var[i], 0))
        physics.sensors = sensors
        output.append(physics)
    return output[0] if num_physics is None else output


def calc_csd(n_dots:int, physics:simulation.PhysicsParameters,
             V_x:NDArray[np.float64], V_y:NDArray[np.float64],
             V_gates:NDArray[np.float64], x_dot:int, y_dot:int,
             numerics:simulation.NumericsParameters|None=None,
             include_excited:bool=True, include_converged:bool=False,
             include_current:bool=False) -> CSDOutput:
    '''
    Calculates a charge-stability diagram, varying plunger voltages on
    2 dots and keeping all other gates constant.

    Parameters
    ----------
    n_dots : int
        The number of dots in the device.
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    V_x, V_y : ndarray[float]
        The possible x- and y-coordinates of the pixels in the diagram.
    V_gates : ndarray[float]
        An array of length `n_dots` giving the voltages of each of the plunger
        gates. This is relevant only for plunger gates whose voltages remain
        constant over the whole diagram, and it contains values of the constant
        voltages. For the two plunger gates corresponding to the x- and y-axes
        (`x_dot` and `y_dot`), the value of `V_gates` is arbitrary.
    x_dot, y_dot : int
        Integers between 0 and (n_dots - 1) inclusive, denoting the indeces of
        the dots whose gate voltages are plotted on the x- or y-axes.
    numerics : NumericsParameters | None
        The numeric parameters to be used during the simulation.
    include_excited : bool
        Whether to include excited state data for applying latching effects.
    include_converged : bool
        Whether to include data about whether the simulation properly converged
        at each pixel. 

    Returns
    -------
    CSDOutput
        A ``CSDOutput`` object wrapping the results of the computation.
    '''
    # make deep copy of physics, since gates will be modified
    phys = physics.copy()
    
    phys.K_mat = simulation.calc_K_mat(phys.x, phys.K_0, phys.sigma)
    phys.g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0*(phys.x[1]-phys.x[0])*phys.K_mat + np.identity(len(phys.x)))
    
    for d, v in enumerate(V_gates):
        phys.gates[2*d+1].peak = v

    N_v_x = len(V_x)
    N_v_y = len(V_y)

    csd_out = CSDOutput(physics=physics, V_x=V_x, V_y=V_y,
                        x_gate=x_dot, y_gate=y_dot, V_gates=V_gates,
                        sensor=np.zeros((N_v_x, N_v_y, len(phys.sensors)), dtype=np.float32),
                        are_dots_occupied=np.full((N_v_x, N_v_y, n_dots), False, dtype=np.bool_),
                        are_dots_combined=np.full((N_v_x, N_v_y, n_dots-1), False, dtype=np.bool_),
                        dot_charges=np.zeros((N_v_x, N_v_y, n_dots), dtype=np.int_),
                        converged=None, excited_sensor=None, current=None)
    if include_converged:
        csd_out.converged = np.full((N_v_x, N_v_y), False, dtype=np.bool_)
    if include_excited:
        csd_out.excited_sensor = np.zeros((N_v_x, N_v_y, len(phys.sensors)), dtype=np.float32)
    if include_current:
        csd_out.current = np.zeros((N_v_x, N_v_y), dtype=np.float32)

    dot_charge = np.zeros(n_dots, dtype=np.int_)
    are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    ex_dot_charge = np.zeros(n_dots, dtype=np.int_)
    ex_are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    n_guess_prev = None
    for j in range(N_v_y):
        n_guess = n_guess_prev
        for i in range(N_v_x):
            phys.gates[2*x_dot+1].peak = V_x[i]
            phys.gates[2*y_dot+1].peak = V_y[j]
            eff_peaks = simulation.calc_effective_peaks(phys.gates)
            phys.effective_peaks = eff_peaks
            V = simulation.calc_V(phys.gates, phys.x, 0, 0, eff_peaks) 
            phys.V = V
            tf = simulation.ThomasFermi(phys, numerics=numerics)
            tf_out = tf.run_calculations(n_guess=n_guess, inc_curr=include_current)
            n_guess = tf.n
            csd_out.are_dots_occupied[i,j,:] = tf_out.are_dots_occupied
            csd_out.are_dots_combined[i,j,:] = tf_out.are_dots_combined
            csd_out.dot_charges[i,j,:] = tf_out.dot_charges
            csd_out.sensor[i,j,:] = tf_out.sensor
            if csd_out.converged is not None:
                csd_out.converged[i,j] = tf_out.converged
            if i == 0:
                n_guess_prev = n_guess
                if include_excited:
                    dot_charge = tf_out.dot_charges
                    are_dot_combined = tf_out.are_dots_combined
                    ex_dot_charge = dot_charge
                    ex_are_dot_combined = are_dot_combined
            if include_excited and csd_out.excited_sensor is not None:
                if np.any(simulation.is_transition(dot_charge, are_dot_combined,
                            tf_out.dot_charges, tf_out.are_dots_combined)[0]):
                    ex_dot_charge = dot_charge
                    ex_are_dot_combined = are_dot_combined
                dot_charge = tf_out.dot_charges
                are_dot_combined = tf_out.are_dots_combined    
                csd_out.excited_sensor[i,j,:] = tf.sensor_from_charge_state(ex_dot_charge, ex_are_dot_combined)
            if include_current and csd_out.current is not None:
                csd_out.current[i,j] = tf_out.current

    return csd_out
    

def calc_2d_csd(physics:simulation.PhysicsParameters,
                V_x:NDArray[np.float64], V_y:NDArray[np.float64],
                numerics:simulation.NumericsParameters|None=None,
                include_excited:bool=True, include_converged=False,
                include_current=True) -> CSDOutput:
    '''
    Calculates a charge-stability diagram for the case where there are only
    two dots.

    Parameters
    ----------
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    V_x, V_y : ndarray[float]
        The possible x- and y-coordinates of the pixels in the diagram.
    numerics : NumericsParameters | None
        The numeric parameters to be used during the simulation.
    include_excited : bool
        Whether to include excited state data for applying latching effects.
    include_converged : bool
        Whether to include data about whether the simulation properly converged
        at each pixel. 
    
    Returns
    -------
    CSDOutput
        A ``CSDOutput`` object wrapping the results of the computation.
    '''
    return calc_csd(2, physics, V_x, V_y, np.array([0,0]), 0, 1, numerics=numerics,
                    include_excited=include_excited, include_converged=include_converged,
                    include_current=include_current)


def calc_transitions(dot_charges:NDArray[np.int_], are_dots_combined:NDArray[np.bool_]) \
                    -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    '''
    Calculates the locations and types of transitions.

    A transition is defined to be present at a pixel if it has a charge state
    that varies from any of its adjacent neighbors.

    Parameters
    ----------
    dot_charges : ndarray[int]
        An array with shape ``(csd_x, [...,] n_dots)``
        indicating how many electrons are in each dot. 
        In the case of combined dots, the total number of charges should be
        entered in the left-most slot, with the other slots padded with zeros.
    are_dots_combined : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots-1)``, 
        indicating whether the dots on either side of each barrier are combined
        together.
    
    Returns
    -------
    is_transition : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots)`` indicating
        whether a transition is present in a particular dot. A transition occurs
        at a particular pixel and dot if the number of charges in that dot differ
        in any adjecent pixels. 
    is_transition_combined : ndarray[bool]
        An array with shape ``(csd_x, [...,] n_dots-1)``
        indicating whether there is a transition in a combined dot on either
        side of a particular barrier.  
    '''
    is_transition = np.full(dot_charges.shape, False, dtype=np.bool_)
    is_transition_combined = np.full(are_dots_combined.shape, False, dtype=np.bool_)
    for p in np.ndindex(dot_charges.shape[:-1]):
        neighbors = []
        for i in range(len(p)):
            if p[i] > 0:
                pl = list(p)
                pl[i] -= 1
                neighbors.append(tuple(pl))
            if p[i] < dot_charges.shape[i] - 1:
                pl = list(p)
                pl[i] += 1
                neighbors.append(tuple(pl))
        for nei in neighbors:
            is_tr, is_tr_com = simulation.is_transition(dot_charges[p], are_dots_combined[p], dot_charges[nei], are_dots_combined[nei])
            is_transition_combined[p] = np.logical_or(is_transition_combined[p], is_tr_com)
            is_transition[p] = np.logical_or(is_transition[p], is_tr)
    return is_transition, is_transition_combined



@dataclass(kw_only=True)
class RaysOutput:
    '''
    Output of ray data calculations. Some attributes may be ``None``
    depending on which quantities are calculated.

    Attributes
    ----------
    physics : PhysicsParameters
        The set of physics parameters used in the simulation.
    centers : ndarray[float]
        An array with shape ``(n_centers, n_dots)`` indicating the points from
        which rays should start.
    rays : ndarray[float]
        An array with shape ``(n_rays, n_dots)`` indicating the direction and
        length of from each ray that extends from a single center point.
    resolution : int
        The number of points per ray to simulate.
    sensor : ndarray[float]
        An array with shape ``(n_centers, n_rays, resolution, n_sensors)``
        giving the Coulomb potential at each point at a specific sensor.
    are_dots_occupied : ndarray[bool]
        An array with shape ``(n_centers, n_rays, resolution, n_dots)``, indicating whether
        each dot is occupied at each point.
    are_dots_combined : ndarray[bool]
        An array with shape ``(n_centers, n_rays, resolution, n_dots-1)``, indicating
        at each point, whether the dots on each side of an
        internal barrier are combined together (i.e. the barrier is too low).
    dot_charges : ndarray[int]
        An array with shape ``(n_centers, n_rays, resolution, n_dots)``, indicating the
        total number of charges in each dot at each point.
        In the case of combined dots, the total number of charges will be
        entered in the left-most spot, with the other spots padded with zeros.
    converged : ndarray[bool] | None
        An array with shape ``(n_centers, n_rays, resolution)``, indicating whether the
        calculation of n(x) properly converged at each point.
    dot_transitions : ndarray[bool] | None
        An array with shape ``(n_centers, n_rays, resolution, n_dots)``, indicating whether
        a transition in each dot occurs at each pisel in the diagram.
    are_transitions_combined : ndarray[bool] | None
        An array with shape ``(n_centers, n_rays, resolution, n_dots-1)``, indicating at
        each point, whether a transition occurs on a combined dot
        comprised of dots on either side of each internal barrier.
    excited_sensor : ndarray[float] | None
        An array with shape ``(n_centers, n_rays, resolution, n_sensors)`` giving the
        Coulomb potential at an excited state at each point at a specific sensor.
    '''
    physics:simulation.PhysicsParameters=field(default_factory=lambda:simulation.PhysicsParameters())
    centers:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    rays:NDArray[np.float64]=field(default_factory=lambda:np.zeros(0, dtype=np.float64))
    resolution:int=0 # must be at least 2
    sensor:NDArray[np.float32]=field(default_factory=lambda:np.zeros(0, dtype=np.float32))
    are_dots_occupied:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    are_dots_combined:NDArray[np.bool_]=field(default_factory=lambda:np.zeros(0, dtype=np.bool_))
    dot_charges:NDArray[np.int_]=field(default_factory=lambda:np.zeros(0, dtype=np.int_))
    converged:NDArray[np.bool_]|None=None
    dot_transitions:NDArray[np.bool_]|None=None
    are_transitions_combined:NDArray[np.bool_]|None=None
    excited_sensor:NDArray[np.float32]|None=None
    current:NDArray[np.float32]|None=None
    
    def _get_physics(self) -> simulation.PhysicsParameters:
        return self._physics
    def _set_physics(self, val:simulation.PhysicsParameters):
        self._physics = val.copy()

    def _get_centers(self) -> NDArray[np.float64]:
        return self._centers
    def _set_centers(self, val:NDArray[np.float64]):
        self._centers = np.array(val, dtype=np.float64)

    def _get_rays(self) -> NDArray[np.float64]:
        return self._rays
    def _set_rays(self, val:NDArray[np.float64]):
        self._rays = np.array(val, dtype=np.float64)

    def _get_sensor(self) -> NDArray[np.float32]:
        return self._sensor
    def _set_sensor(self, val:NDArray[np.float32]):
        self._sensor = np.array(val, dtype=np.float32)

    def _get_converged(self) -> NDArray[np.bool_]|None:
        return self._converged
    def _set_converged(self, val:NDArray[np.bool_]|None):
        self._converged = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_dots_occupied(self) -> NDArray[np.bool_]:
        return self._are_dots_occupied
    def _set_are_dots_occupied(self, val:NDArray[np.bool_]):
        self._are_dots_occupied = np.array(val, dtype=np.bool_)

    def _get_are_dots_combined(self) -> NDArray[np.bool_]:
        return self._are_dots_combined
    def _set_are_dots_combined(self, val:NDArray[np.bool_]):
        self._are_dots_combined = np.array(val, dtype=np.bool_)

    def _get_dot_charges(self) -> NDArray[np.int_]:
        return self._dot_states
    def _set_dot_charges(self, val:NDArray[np.int_]):
        self._dot_states = np.array(val, dtype=np.int_)

    def _get_dot_transitions(self) -> NDArray[np.bool_]|None:
        return self._dot_transitions
    def _set_dot_transitions(self, val:NDArray[np.bool_]|None):
        self._dot_transitions = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_are_transitions_combined(self) -> NDArray[np.bool_]|None:
        return self._are_transitions_combined
    def _set_are_transitions_combined(self, val:NDArray[np.bool_]|None):
        self._are_transitions_combined = np.array(val, dtype=np.bool_) if val is not None else None

    def _get_excited_sensor(self) -> NDArray[np.float32]|None:
        return self._excited_sensor
    def _set_excited_sensor(self, val:NDArray[np.float32]|None):
        self._excited_sensor = np.array(val, dtype=np.float32) if val is not None else None

    def _get_current(self) -> NDArray[np.float32]|None:
        return self._current
    def _set_current(self, val:NDArray[np.float32]|None):
        self._current = np.array(val, dtype=np.float32) if val is not None else None


    @classmethod
    def from_dict(cls, d:dict[str, Any]) -> Self:
        '''
        Creates a new ``RaysOutput`` object from a ``dict`` of values.

        Parameters
        ----------
        d : dict[str, Any]
            A dict with keys corresponding to any of this class's attributes.

        Returns
        -------
        RaysOutput
            A new ``RaysOutput`` object with the values specified by ``dict``.
        '''
        output = cls()
        for k, v in d.items():
            if hasattr(output, k):
                if k == "physics":
                    setattr(output, k, simulation.PhysicsParameters.from_dict(v))
                else:
                    setattr(output, k, v)
        return output
    

    def to_dict(self) -> dict[str, Any]:
        '''
        Converts the ``RaysOutput`` object to a ``dict``.

        Returns
        -------
        dict[str, Any]
            A dict with values specified by the ``RaysOutput`` object.
        '''
        return dataclasses.asdict(self)
    

    def copy(self) -> Self:
        '''
        Creates a copy of a ``RaysOutput`` object.

        Returns
        -------
        RaysOutput
            A new ``RaysOutput`` object with the same attribute values as ``self``.
        '''
        return dataclasses.replace(self)

RaysOutput.physics = property(RaysOutput._get_physics, RaysOutput._set_physics) # type: ignore
RaysOutput.centers = property(RaysOutput._get_centers, RaysOutput._set_centers) # type: ignore
RaysOutput.rays = property(RaysOutput._get_rays, RaysOutput._set_rays) # type: ignore
RaysOutput.sensor = property(RaysOutput._get_sensor, RaysOutput._set_sensor) # type: ignore
RaysOutput.converged = property(RaysOutput._get_converged, RaysOutput._set_converged) # type: ignore
RaysOutput.are_dots_occupied = property(RaysOutput._get_are_dots_occupied, RaysOutput._set_are_dots_occupied) # type: ignore
RaysOutput.are_dots_combined = property(RaysOutput._get_are_dots_combined, RaysOutput._set_are_dots_combined) # type: ignore
RaysOutput.dot_charges = property(RaysOutput._get_dot_charges, RaysOutput._set_dot_charges) # type: ignore
RaysOutput.dot_transitions = property(RaysOutput._get_dot_transitions, RaysOutput._set_dot_transitions) # type: ignore
RaysOutput.are_transitions_combined = property(RaysOutput._get_are_transitions_combined, RaysOutput._set_are_transitions_combined) # type: ignore
RaysOutput.excited_sensor = property(RaysOutput._get_excited_sensor, RaysOutput._set_excited_sensor) # type: ignore
CSDOutput.current = property(CSDOutput._get_current, CSDOutput._set_current) # type: ignore



def calc_rays(physics:simulation.PhysicsParameters, centers:NDArray[np.float64],
              rays:NDArray[np.float64], resolution:int,
              numerics:simulation.NumericsParameters|None=None,
              include_excited:bool=False, include_converged=False,
             include_current:bool=False) -> RaysOutput:
    '''
    Calculates ray data, varying multiple plunger voltages at once to move along
    an arbitrary ray in voltage space.

    Parameters
    ----------
    physics : PhysicsParameters
        The physical parameters of the device to simulate.
    centers : ndarray[float]
        An array with shape ``(n_centers, n_dots)`` indicating the points from
        which rays should start.
    rays : ndarray[float]
        An array with shape ``(n_rays, n_dots)`` indicating the direction and
        length of from each ray that extends from a single center point.
    resolution : int
        The number of points per ray to simulate.
    numerics : NumericsParameters | None
        The numeric parameters to be used during the simulation.
    include_excited : bool
        Whether to include excited state data for applying latching effects.
    include_converged : bool
        Whether to include data about whether the simulation properly converged
        at each pixel. 

    Returns
    -------
    RaysOutput
        A ``RaysOutput`` object wrapping the results of the computation.
    '''
    # make deep copy of physics, since gates will be modified
    phys = physics.copy()
    
    phys.K_mat = simulation.calc_K_mat(phys.x, phys.K_0, phys.sigma)
    phys.g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0*(phys.x[1]-phys.x[0])*phys.K_mat + np.identity(len(phys.x)))
    
    n_dots = centers.shape[1]
    n_centers = centers.shape[0]
    n_rays = rays.shape[0]

    rays_out = RaysOutput(physics=physics, centers=centers, rays=rays, resolution=resolution,
                        sensor=np.zeros((n_centers, n_rays, resolution, len(phys.sensors)), dtype=np.float32),
                        are_dots_occupied=np.full((n_centers, n_rays, resolution, n_dots), False, dtype=np.bool_),
                        are_dots_combined=np.full((n_centers, n_rays, resolution, n_dots-1), False, dtype=np.bool_),
                        dot_charges=np.zeros((n_centers, n_rays, resolution, n_dots), dtype=np.int_),
                        converged=None, excited_sensor=None, current=None)
    if include_converged:
        rays_out.converged = np.full((n_centers, n_rays, resolution), False, dtype=np.bool_)
    if include_excited:
        rays_out.excited_sensor = np.zeros((n_centers, n_rays, resolution, len(phys.sensors)), dtype=np.float32)
    if include_current:
        rays_out.current = np.zeros((n_centers, n_rays, resolution), dtype=np.float32)

    dot_charge = np.zeros(n_dots, dtype=np.int_)
    are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)
    ex_dot_charge = np.zeros(n_dots, dtype=np.int_)
    ex_are_dot_combined = np.zeros(n_dots-1, dtype=np.bool_)

    for c_i in range(n_centers):
        n_guess_center = None
        for r_i in range(n_rays):
            n_guess = n_guess_center
            for i in range(resolution):
                if i == 0 and r_i != 0:
                    rays_out.are_dots_occupied[c_i,r_i,0,:] = rays_out.are_dots_occupied[c_i,0,0,:]
                    rays_out.are_dots_combined[c_i,r_i,0,:] = rays_out.are_dots_combined[c_i,0,0,:]
                    rays_out.dot_charges[c_i,r_i,0,:] = rays_out.dot_charges[c_i,0,0,:]
                    rays_out.sensor[c_i,r_i,0,:] = rays_out.sensor[c_i,0,0,:]
                    if rays_out.converged is not None:
                        rays_out.converged[c_i,r_i,0] = rays_out.converged[c_i,0,0]
                    if include_excited:
                            dot_charge = rays_out.dot_charges[c_i,0,0,:]
                            are_dot_combined = rays_out.are_dots_combined[c_i,0,0,:]
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                    if include_excited and rays_out.excited_sensor is not None:
                        rays_out.excited_sensor[c_i,r_i,0,:] = rays_out.excited_sensor[c_i,0,0,:]
                else:
                    pnt = centers[c_i] + i/(resolution-1) * rays[r_i]
                    for d_i in range(n_dots):
                        phys.gates[2*d_i+1].peak = pnt[d_i]
                    eff_peaks = simulation.calc_effective_peaks(phys.gates)
                    phys.effective_peaks = eff_peaks
                    V = simulation.calc_V(phys.gates, phys.x, 0, 0, eff_peaks) 
                    phys.V = V
                    tf = simulation.ThomasFermi(phys, numerics=numerics)
                    tf_out = tf.run_calculations(n_guess=n_guess, inc_curr=include_current)
                    n_guess = tf.n
                    rays_out.are_dots_occupied[c_i,r_i,i,:] = tf_out.are_dots_occupied
                    rays_out.are_dots_combined[c_i,r_i,i,:] = tf_out.are_dots_combined
                    rays_out.dot_charges[c_i,r_i,i,:] = tf_out.dot_charges
                    rays_out.sensor[c_i,r_i,i,:] = tf_out.sensor
                    if rays_out.converged is not None:
                        rays_out.converged[c_i,r_i,i] = tf_out.converged
                    if i == 0:
                        n_guess_center = n_guess
                        if include_excited:
                            dot_charge = tf_out.dot_charges
                            are_dot_combined = tf_out.are_dots_combined
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                    if include_excited and rays_out.excited_sensor is not None:
                        if np.any(simulation.is_transition(dot_charge, are_dot_combined,
                                    tf_out.dot_charges, tf_out.are_dots_combined)[0]):
                            ex_dot_charge = dot_charge
                            ex_are_dot_combined = are_dot_combined
                        dot_charge = tf_out.dot_charges
                        are_dot_combined = tf_out.are_dots_combined    
                        rays_out.excited_sensor[c_i,r_i,i,:] = tf.sensor_from_charge_state(ex_dot_charge, ex_are_dot_combined)
                    if include_current and rays_out.current is not None:
                        rays_out.current[c_i,r_i,i] = tf_out.current
                        
    return rays_out
