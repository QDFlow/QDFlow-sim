import numpy as np
import pytest
from qdflow import generate
from qdflow.physics import simulation
from qdflow.util import distribution

# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:
    
    @staticmethod
    def test_CSDOutput():
        d = {
            "physics": {"mu":1.23, "sigma":20},
            "V_x": np.linspace(-1, 1, 3),
            "x_gate": 1,
            "excited_sensor": np.ones((3,4,2)),
            "dot_transitions": None,
            "extraneous_key": 12345
        }
        csd = generate.CSDOutput.from_dict(d)
        assert csd.V_x.shape == (3,)
        assert np.all(csd.V_x == d["V_x"])
        assert csd.V_x is not d["V_x"]
        assert csd.x_gate == 1
        assert csd.dot_transitions is None
        assert isinstance(csd.physics, simulation.PhysicsParameters)
        assert np.isclose(csd.physics.mu, 1.23)
        assert csd.converged is None
        assert not hasattr(csd, "extraneous_key")
        
        # check for deep copy
        csd_copy = csd.copy()
        assert csd.x_gate == csd_copy.x_gate
        assert csd_copy.V_x.shape == csd.V_x.shape
        assert np.all(csd_copy.V_x == csd.V_x)
        assert csd_copy.V_x is not csd.V_x
        assert csd_copy is not csd
        assert csd_copy.physics.mu == csd.physics.mu
        assert csd_copy.physics is not csd.physics
        
        to_d = csd.to_dict()
        assert to_d["V_x"].shape == d["V_x"].shape
        assert np.all(to_d["V_x"] == d["V_x"])
        assert to_d["V_x"] is not d["V_x"]
        assert isinstance(to_d["physics"], dict)
        assert "mu" in to_d["physics"]

    @staticmethod
    def test_RaysOutput():
        d = {
            "physics": {"mu":1.23, "sigma":20},
            "centers": np.array([[1,2],[3,4],[5,6]]),
            "resolution": 3,
            "excited_sensor": np.ones((3,2,3,1)),
            "dot_transitions": None,
            "extraneous_key": 12345
        }
        ro = generate.RaysOutput.from_dict(d)
        assert ro.centers.shape == (3,2)
        assert np.all(ro.centers == d["centers"])
        assert ro.centers is not d["centers"]
        assert ro.resolution == 3
        assert ro.dot_transitions is None
        assert isinstance(ro.physics, simulation.PhysicsParameters)
        assert np.isclose(ro.physics.mu, 1.23)
        assert ro.converged is None
        assert not hasattr(ro, "extraneous_key")
        
        # check for deep copy
        ro_copy = ro.copy()
        assert ro.resolution == ro_copy.resolution
        assert ro_copy.centers.shape == ro.centers.shape
        assert np.all(ro_copy.centers == ro.centers)
        assert ro_copy.centers is not ro.centers
        assert ro_copy is not ro
        assert ro_copy.physics.mu == ro.physics.mu
        assert ro_copy.physics is not ro.physics
        
        to_d = ro.to_dict()
        assert to_d["centers"].shape == d["centers"].shape
        assert np.all(to_d["centers"] == d["centers"])
        assert to_d["centers"] is not d["centers"]
        assert isinstance(to_d["physics"], dict)
        assert "mu" in to_d["physics"]
    
    @staticmethod
    def test_PhysicsRandomization():
        corr = distribution.FullyCorrelated(distribution.Uniform(0,1), 2).dependent_distributions()
        d = {
            "num_x_points":20,
            "mu":1.23,
            "sigma":distribution.Delta(20),
            "V_L":corr[0],
            "V_R":corr[1],
            "extraneous_key": 12345
        }
        rand = generate.PhysicsRandomization.from_dict(d)
        assert rand.mu == 1.23
        assert rand.num_x_points == 20
        assert rand.sigma is d["sigma"]
        assert rand.V_L is corr[0]
        assert rand.c_k == generate.PhysicsRandomization().c_k
        assert not hasattr(rand, "extraneous_key")
        
        # check deep copy
        rand_copy = rand.copy()
        assert rand_copy.mu == rand.mu
        assert rand_copy is not rand
        assert rand_copy.sigma._value == rand.sigma._value
        assert rand_copy.sigma is not rand.sigma
        assert rand_copy.V_L is not rand.V_L
        assert rand_copy.V_R is not rand.V_R
        assert rand_copy.V_L.dependent_distributions[0] is rand_copy.V_L
        assert rand_copy.V_L.dependent_distributions[1] is rand_copy.V_R
        assert rand_copy.V_R.dependent_distributions[0] is rand_copy.V_L

        to_d = rand.to_dict()
        assert to_d is not d
        assert to_d["mu"] == d["mu"]
        assert to_d["sigma"]._value == d["sigma"]._value
        assert to_d["sigma"] is not d["sigma"]
        assert to_d["V_L"] is not rand.V_L
        assert to_d["V_R"] is not rand.V_R
        assert to_d["V_L"].dependent_distributions[0] is to_d["V_L"]
        assert to_d["V_L"].dependent_distributions[1] is to_d["V_R"]
        assert to_d["V_R"].dependent_distributions[0] is to_d["V_L"]

        rand_def1 = generate.PhysicsRandomization.default()
        rand_def2 = generate.PhysicsRandomization.default()
        assert rand_def1 is not rand_def2
        assert rand_def1.num_x_points == rand_def2.num_x_points


# ---------------------
# Test Module Functions
# ---------------------

class TestModuleFunctions:

    @staticmethod
    def test_default_physics():
        phys = generate.default_physics()
        assert isinstance(phys, simulation.PhysicsParameters)
        assert len(phys.gates) == 5
        phys = generate.default_physics(n_dots=3)
        assert isinstance(phys, simulation.PhysicsParameters)
        assert len(phys.gates) == 7

    @staticmethod
    def test_random_physics():
        generate.set_rng_seed(456)
        rand = generate.PhysicsRandomization()
        rand.mu = distribution.Uniform(2,3)
        rand.multiply_gates_by_q = False
        rand.plunger_peak = 1
        rand.plunger_peak_variations = 1
        rand.barrier_peak = 1
        rand.external_barrier_peak_variations = distribution.FullyCorrelated(distribution.Uniform(3,4), 2)
        rand.internal_barrier_peak_variations = distribution.Uniform(5,6)
        rand.rho = 15
        rand.rho_variations = distribution.Uniform(5,20)
        rand.h = 60
        rand.h_variations = np.array([0,10,20,30,40])
        phys = generate.random_physics(rand)
        assert isinstance(phys, simulation.PhysicsParameters)
        assert phys.mu >= 2 and phys.mu <= 3
        assert np.isclose(phys.gates[1].peak, 2)
        assert np.isclose(phys.gates[3].peak, 2)
        assert np.isclose(phys.gates[0].peak, phys.gates[-1].peak)
        assert phys.gates[0].peak >= 4 and phys.gates[0].peak <= 5
        assert phys.gates[2].peak >= 6 and phys.gates[2].peak <= 7
        g_rho = np.array([g.rho for g in phys.gates])
        assert len(g_rho) == 5
        assert np.all((g_rho >= 20) & (g_rho <= 35))
        g_h = np.array([g.h for g in phys.gates])
        assert len(g_rho) == 5
        assert np.allclose(g_h, [60,70,80,90,100])

        phys_list = generate.random_physics(rand, 10)
        assert isinstance(phys_list, list)
        assert len(phys_list) == 10
        assert isinstance(phys_list[0], simulation.PhysicsParameters)

    @staticmethod
    def test_calc_csd():
        n_dots = 3
        phys = generate.default_physics(n_dots)
        phys.mu = 1.23
        csd = generate.calc_csd(n_dots, phys, np.array([4,6,8,10]), np.array([5,7]),
                np.array([0,6,0]), 0, 2, include_excited=True, include_converged=True, include_current=True)
        assert csd.V_x.shape == (4,)
        assert np.allclose(csd.V_x, [4,6,8,10])
        assert csd.V_y.shape == (2,)
        assert np.allclose(csd.V_y, [5,7])
        assert csd.V_gates.shape == (3,)
        assert np.allclose(csd.V_gates, [0,6,0])
        assert csd.x_gate == 0
        assert csd.y_gate == 2
        assert csd.excited_sensor is not None
        assert csd.excited_sensor.shape == (4,2,len(phys.sensors))
        assert csd.converged is not None
        assert csd.converged.shape == (4,2)
        assert csd.sensor.shape == (4,2,len(phys.sensors))
        assert csd.are_dots_occupied.shape == (4,2,3)
        assert csd.are_dots_combined.shape == (4,2,2)
        assert csd.dot_charges.shape == (4,2,3)
        assert csd.physics.mu == 1.23
        assert csd.current is not None
        assert csd.current.shape == (4,2)
        
    @staticmethod
    def test_calc_2d_csd():
        phys = generate.default_physics(n_dots=2)
        phys.mu = 1.23
        csd = generate.calc_2d_csd(phys, np.array([4,6,8,10]), np.array([5,7]),
                                include_excited=False, include_converged=False)
        assert csd.V_x.shape == (4,)
        assert np.allclose(csd.V_x, [4,6,8,10])
        assert csd.V_y.shape == (2,)
        assert np.allclose(csd.V_y, [5,7])
        assert csd.V_gates.shape == (2,)
        assert csd.x_gate == 0
        assert csd.y_gate == 1
        assert csd.excited_sensor is None
        assert csd.converged is None
        assert csd.sensor.shape == (4,2,len(phys.sensors))
        assert csd.are_dots_occupied.shape == (4,2,2)
        assert csd.are_dots_combined.shape == (4,2,1)
        assert csd.dot_charges.shape == (4,2,2)
        assert csd.physics.mu == 1.23
        
    @staticmethod
    def test_calc_rays():
        phys = generate.default_physics(3)
        phys.mu = 1.23
        centers = np.array([[3,4,5],[6,8,10],[4,8,6],[7,5,9],[10,6,3]])
        rays = np.array([[-1,1,1],[1,-1,1],[1,1,-1]])
        resolution = 2
        ro = generate.calc_rays(phys, centers, rays, resolution,
                                include_excited=True, include_converged=True, include_current=True)
        assert ro.centers.shape == (5,3)
        assert np.allclose(ro.centers, centers)
        assert ro.rays.shape == (3,3)
        assert np.allclose(ro.rays, rays)
        assert ro.resolution == resolution
        assert ro.excited_sensor is not None
        assert ro.excited_sensor.shape == (5,3,resolution,len(phys.sensors))
        assert ro.converged is not None
        assert ro.converged.shape == (5,3,resolution)
        assert ro.sensor.shape == (5,3,resolution,len(phys.sensors))
        assert ro.are_dots_occupied.shape == (5,3,resolution,3)
        assert ro.are_dots_combined.shape == (5,3,resolution,2)
        assert ro.dot_charges.shape == (5,3,resolution,3)
        assert ro.physics.mu == 1.23
        assert ro.current is not None
        assert ro.current.shape == (5,3,resolution)

    @staticmethod
    def test_calc_transitions():
        dot_charges = np.array([[[2,2],[2,3]],[[3,2],[5,0]]])
        are_dots_combined = np.array([[[False],[False]],[[False],[True]]])
        tr, tr_comb = generate.calc_transitions(dot_charges, are_dots_combined)
        assert tr.shape == (2,2,2)
        assert tr_comb.shape == (2,2,1)
        assert np.all(tr == [[[True,True],[False,True]],[[True,False],[False,False]]])
        assert np.all(tr_comb == [[[False],[False]],[[False],[False]]])

        dot_charges = np.array([[2,3],[2,3],[3,2],[3,2],[5,0],[5,0],[6,0],[6,0]])
        are_dots_combined = np.array([[False],[False],[False],[False],[True],[True],[True],[True]])
        tr, tr_comb = generate.calc_transitions(dot_charges, are_dots_combined)
        assert tr.shape == (8,2)
        assert tr_comb.shape == (8,1)
        assert np.all(tr == [[False, False], [True, True], [True, True], [False, False],
                             [False, False], [True, True], [True, True], [False, False]])
        assert np.all(tr_comb == [[False], [False], [False], [False],
                                  [False], [True], [True], [False]])
