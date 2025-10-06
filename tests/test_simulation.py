import numpy as np
import pytest
from qdflow.physics import simulation
import warnings
import networkx # type: ignore[import-untyped]

# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:

    @staticmethod
    def test_GateParameters():
        d = {"mean": 1.0, "peak": 2.0, "rho": 3.0, "h": 4.0, "screen": 5.0}
        gate = simulation.GateParameters.from_dict(d)
        for k, v in d.items():
            assert getattr(gate, k) == v        
        gate_copy = gate.copy()
        assert gate_copy.mean == gate.mean
        assert gate is not gate_copy
        to_d = gate.to_dict()
        for k, v in d.items():
            assert to_d[k] == v
        d2 = {"mean": 1.0, "peak": 2.0, "extraneous_key": 12345}
        gate2 = simulation.GateParameters.from_dict(d2)
        assert gate2.rho == simulation.GateParameters().rho
        assert not hasattr(gate2, "extraneous_key")

    @staticmethod
    def test_PhysicsParameters():
        d = {
            "x": np.linspace(-1, 1, 3),
            "V": None,
            "q": -1,
            "gates": [
                {"mean": 1, "peak": 2, "rho": 3, "h": 4, "screen": 5},
                {"mean": 6, "peak": 7, "rho": 8, "h": 9, "screen": 10},
            ],
            "K_0": 10.0,
            "sigma": 2.0,
            "extraneous_key": 12345
        }
        phys = simulation.PhysicsParameters.from_dict(d)
        assert phys.x.shape == (3,)
        assert np.all(phys.x == d["x"])
        assert phys.x is not d["x"]
        assert phys.q == -1
        assert phys.V == None
        assert isinstance(phys.gates[0], simulation.GateParameters)
        assert phys.g_0 == simulation.PhysicsParameters().g_0
        assert not hasattr(phys, "extraneous_key")
        
        # check for deep copy
        phys_copy = phys.copy()
        assert phys.q == phys_copy.q
        assert phys_copy.x.shape == phys.x.shape
        assert np.all(phys_copy.x == phys.x)
        assert phys_copy.x is not phys.x
        assert phys_copy is not phys
        assert phys_copy.gates[1].mean == phys.gates[1].mean
        assert phys_copy.gates[1] is not phys.gates[1]
        
        to_d = phys.to_dict()
        assert to_d["x"].shape == d["x"].shape
        assert np.all(to_d["x"] == d["x"])
        assert to_d["x"] is not d["x"]
        assert isinstance(to_d["gates"][0], dict)
        assert "peak" in to_d["gates"][0]
        
    @staticmethod
    def test_NumericsParameters():
        d = {
            "calc_n_max_iterations_no_guess": 1,
            "calc_n_max_iterations_guess": 2,
            "calc_n_rel_tol": .3,
            "calc_n_coulomb_steps": 4,
            "calc_n_use_combination_method": True,
            "island_relative_cutoff": .5,
            "island_min_occupancy": .6,
            "cap_model_matrix_softening": .7,
            "stable_config_N_limit": 8,
            "count_transitions_sigma": .9,
            "count_transitions_eps": .11,
            "create_graph_max_changes": 12
        }
        numer = simulation.NumericsParameters.from_dict(d)
        for k, v in d.items():
            assert getattr(numer, k) == v        
        numer_copy = numer.copy()
        assert numer_copy.calc_n_rel_tol == numer.calc_n_rel_tol
        assert numer is not numer_copy
        to_d = numer.to_dict()
        for k, v in d.items():
            assert to_d[k] == v
        d2 = {"island_relative_cutoff": 1.0, "extraneous_key": 12345}
        numer2 = simulation.NumericsParameters.from_dict(d2)
        assert numer2.calc_n_rel_tol == simulation.NumericsParameters().calc_n_rel_tol
        assert not hasattr(numer2, "extraneous_key")

    @staticmethod
    def test_ThomasFermiOutput():
        d = {
            "sensor": np.array([0.1, 0.2]),
            "are_dots_occupied": np.array([True, False]),
            "converged": True,
            "n": np.array([0.0, 1.0]),
            "graph_charge": None,
            "extraneous_key": 12345
        }

        out = simulation.ThomasFermiOutput.from_dict(d)
        assert out.n.shape == d["n"].shape
        assert np.all(out.n == d["n"])
        assert out.n is not d["n"]
        assert out.converged == True
        assert out.graph_charge == None
        assert out.are_dots_combined.shape == simulation.ThomasFermiOutput().are_dots_combined.shape
        assert np.all(out.are_dots_combined == simulation.ThomasFermiOutput().are_dots_combined)
        assert not hasattr(out, "extraneous_key")
        
        # check for deep copy
        out_copy = out.copy()
        assert out_copy is not out
        assert out.converged == out_copy.converged
        assert out_copy.n.shape == out.n.shape
        assert np.all(out_copy.n == out.n)
        assert out_copy.n is not out.n
        
        to_d = out.to_dict()
        assert to_d["n"].shape == d["n"].shape
        assert np.all(to_d["n"] == d["n"])
        assert to_d["n"] is not d["n"]

# ---------------------------
# Test module-level functions
# ---------------------------

class TestModuleFunctions:

    @staticmethod
    def test_calc_K_mat():
        x = np.linspace(-5, 5, 7)
        K_0 = 2.0
        sigma = 1.0
        K_mat = simulation.calc_K_mat(x, K_0, sigma)
        assert K_mat.shape == (7, 7)
        assert np.allclose(np.diag(K_mat), K_0 / sigma)
        assert np.allclose(K_mat, K_mat.T)
        assert np.allclose(K_mat[:-1,:-1], K_mat[1:,1:])
        assert np.all(K_mat[0,:-1] > K_mat[0,1:])

    @staticmethod
    def test_calc_V_gate():
        gate1 = simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40)
        val1 = simulation.calc_V_gate(gate1, 0, -7, 7)
        val1b = simulation.calc_V_gate(gate1, 15, 0, 0)
        arr = simulation.calc_V_gate(gate1, np.array([0, 1, 2, 3]), -7, 7)
        assert isinstance(val1, float)
        assert np.isclose(val1b, 1.3)
        assert arr.shape == (4,)
        assert np.isclose(val1, arr[0])
        gate2 = simulation.GateParameters(mean=15, peak=2.6, rho=10, h=20, screen=40)
        val2 = simulation.calc_V_gate(gate2, 0, -7, 7)
        assert np.isclose(val2, 2 * val1)
        gate3 = simulation.GateParameters(mean=115, peak=1.3, rho=10, h=20, screen=40)
        val3 = simulation.calc_V_gate(gate3, 100, -7, 7)
        assert np.isclose(val3, val1)

    @staticmethod
    def test_calc_effective_peaks():
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40)
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (1,)
        assert np.isclose(eff_peaks[0], 1.3)
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=1.3, rho=10, h=20, screen=40),            
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (2,)
        assert eff_peaks[0] < 1.3
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=-1.3, rho=10, h=20, screen=40),            
        ]
        eff_peaks = simulation.calc_effective_peaks(gates)
        assert eff_peaks.shape == (2,)
        assert eff_peaks[0] > 1.3

    @staticmethod
    def test_calc_V():
        gates = [
            simulation.GateParameters(mean=15, peak=1.3, rho=10, h=20, screen=40),
            simulation.GateParameters(mean=65, peak=-1.3, rho=10, h=20, screen=40),            
        ]
        x = np.linspace(0, 100, 21)
        v = simulation.calc_V(gates, x, 0, 0)
        assert v.shape == (21,)
        assert v[13] < v[3]
        assert np.isclose(v[8], 0)

    @staticmethod
    def test_is_transition():
        dc1 = np.array([1, 0])
        adc1 = np.array([False])
        dc2 = np.array([2, 0])
        adc2 = np.array([False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (2,)
        assert is_tr_comb.shape == (1,)
        assert np.all(is_tr == [True, False])
        assert np.all(is_tr_comb == [False])

        dc1 = np.array([1, 2, 3])
        adc1 = np.array([False, False])
        dc2 = np.array([2, 1, 3])
        adc2 = np.array([False, False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (3,)
        assert is_tr_comb.shape == (2,)
        assert np.all(is_tr == [True, True, False])
        assert np.all(is_tr_comb == [False, False])

        dc1 = np.array([2, 2])
        adc1 = np.array([False])
        dc2 = np.array([4, 0])
        adc2 = np.array([True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (2,)
        assert is_tr_comb.shape == (1,)
        assert np.all(is_tr == [False, False])
        assert np.all(is_tr_comb == [False])

        dc1 = np.array([2, 2])
        adc1 = np.array([False])
        dc2 = np.array([5, 0])
        adc2 = np.array([True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (2,)
        assert is_tr_comb.shape == (1,)
        assert np.all(is_tr == [True, True])
        assert np.all(is_tr_comb == [True])

        dc1 = np.array([4, 0, 2])
        adc1 = np.array([True, False])
        dc2 = np.array([2, 4, 0])
        adc2 = np.array([False, True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (3,)
        assert is_tr_comb.shape == (2,)
        assert np.all(is_tr == [False, False, False])
        assert np.all(is_tr_comb == [False, False])

        dc1 = np.array([4, 0, 1])
        adc1 = np.array([True, False])
        dc2 = np.array([2, 4, 0])
        adc2 = np.array([False, True])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (3,)
        assert is_tr_comb.shape == (2,)
        assert np.all(is_tr == [True, True, True])
        assert np.all(is_tr_comb == [True, True])

        dc1 = np.array([4, 0, 2])
        adc1 = np.array([True, False])
        dc2 = np.array([5, 0, 1])
        adc2 = np.array([True, False])
        is_tr, is_tr_comb = simulation.is_transition(dc1, adc1, dc2, adc2)
        assert is_tr.shape == (3,)
        assert is_tr_comb.shape == (2,)
        assert np.all(is_tr == [True, True, True])
        assert np.all(is_tr_comb == [True, False])

# ---------------------------
# Test ThomasFermi simulation
# ---------------------------

class TestThomasFermi:

    @staticmethod
    def test_calc_n():
        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        assert n.shape == (len(x),)
        assert np.all(n >= 0)

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, K_0=0)
        V = simulation.calc_V(phys.gates, x, 0, 0)
        phys.V = V
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        de = phys.mu - phys.q * phys.V
        n_exact = (phys.g_0 / phys.beta) * np.where(de > 0,
            phys.beta * de + np.log(1 + np.exp(phys.beta * np.where(de > 0, -de, 0))),
            np.log(1 + np.exp(phys.beta * np.where(de <= 0, de, 0)))
        )
        assert n.shape == n_exact.shape
        assert np.allclose(n, n_exact)

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, K_0=10, g_0=.007, beta=1, mu=-20)
        V = 4 * phys.q * np.cos(2*np.pi*x/250)
        phys.V = V
        tf = simulation.ThomasFermi(phys)
        n = tf._calc_n()
        assert n.shape == (len(x),)
        assert np.all(n >= 0)
        assert np.all(n < 1e-5)

        x = np.linspace(-300,300,51)
        numer = simulation.NumericsParameters(calc_n_max_iterations_guess=2)
        phys = simulation.PhysicsParameters(x=x, K_0=0)
        V = simulation.calc_V(phys.gates, x, 0, 0)
        phys.V = V
        tf = simulation.ThomasFermi(phys, numer)
        de = phys.mu - phys.q * phys.V
        n_exact = (phys.g_0 / phys.beta) * np.where(de > 0,
            phys.beta * de + np.log(1 + np.exp(phys.beta * np.where(de > 0, -de, 0))),
            np.log(1 + np.exp(phys.beta * np.where(de <= 0, de, 0)))
        )
        n = tf._calc_n(n_guess=n_exact)
        assert tf.converged

    @staticmethod
    def test_calc_n_convergence_warning():
        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, g_0=.01, K_0=1000, sigma=20, beta=100)
        numer = simulation.NumericsParameters(calc_n_max_iterations_no_guess=50,
                                              calc_n_rel_tol=1e-4)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = tf._calc_n()
            assert len(w) == 1
            assert issubclass(w[0].category, simulation.ConvergenceWarning)
            assert tf.converged == False

        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, g_0=.001, K_0=1, sigma=20, beta=100)
        numer = simulation.NumericsParameters(calc_n_max_iterations_no_guess=200,
                                              calc_n_rel_tol=1e-4)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            n = tf._calc_n()
            assert len(w) == 0
            assert tf.converged == True

    @staticmethod
    def test_calc_islands_and_barriers():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        numer = simulation.NumericsParameters(island_relative_cutoff=.1)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.n = np.array([.0050, .01,  .02,  .033, 2.4, 5.5, 6,   3.7,  2.8,  1.9,
                         .010,  .011, .012, 2.3,  4.4, 4.5, 2.6, .017, .008, .009])
        tf._calc_islands_and_barriers()
        isl = tf.islands
        bar = tf.barriers
        assert isl.shape == (2,2)
        assert bar.shape == (3,2)
        assert np.all(isl == [[4,10], [13,17]])
        assert np.all(bar == [[0,4], [10,13], [17,20]])
        assert tf.is_short_circuit == False

        x = np.linspace(-10,10,20)
        phys = simulation.PhysicsParameters(x=x)
        numer = simulation.NumericsParameters(island_relative_cutoff=.1, island_min_occupancy=.01)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.n = 1e-4 * np.array([.0050, .01,  .02,  .033, 2.4, 5.5, 6,   3.7,  2.8,  1.9,
                                .010,  .011, .012, 2.3,  4.4, 4.5, 2.6, .017, .008, .009])
        tf._calc_islands_and_barriers()
        isl = tf.islands
        bar = tf.barriers
        assert bar.shape == (1,2)
        assert len(isl) == 0
        assert np.all(bar == [[0,20]])
        assert tf.is_short_circuit == False
        
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        numer = simulation.NumericsParameters(island_relative_cutoff=.1, island_min_occupancy=.01)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.n = np.array([1.0, 1.1, 1.2,  1.3,  1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                        1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19])
        tf._calc_islands_and_barriers()
        isl = tf.islands
        bar = tf.barriers
        assert len(isl) == 0
        assert len(bar) == 0
        assert tf.is_short_circuit == True
        
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        numer = simulation.NumericsParameters(island_relative_cutoff=.1, island_min_occupancy=.01)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.n = np.array([1.0, 1.1, 1.2,  1.3,  1.4, 0.05, 0.06, 0.07, 0.08, 0.09,
                        0.010, 0.011, 0.012, 0.013, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19])
        tf._calc_islands_and_barriers()
        isl = tf.islands
        bar = tf.barriers
        all_isl = tf.all_islands
        assert len(isl) == 0
        assert all_isl.shape == (2,2)
        assert bar.shape == (1,2)
        assert np.all(bar == [[5,14]])
        assert np.all(all_isl == [[0,5],[14,20]])
        assert tf.is_short_circuit == False

    @staticmethod
    def test_calc_charges():
        x = np.linspace(-300,300,20)
        delta_x = 600 / (20-1)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01,  .01, 2, 3, 4,  5,  2,  2,
                         .01, .01, .01, 2,  4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        ch = tf._calc_charges()
        assert ch.shape == (2,)
        assert np.allclose(ch, delta_x * np.array([18, 12]))
        
    @staticmethod
    def test_calc_charge_centers():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01, .01, 20, 7, 1, 3, 3, 1,
                         .01, .01, .01, 2, 4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        tf._calc_charges()
        chc = tf._calc_charge_centers()
        assert chc.shape == (2,)
        assert np.allclose(chc, [x[5], (x[14]+x[15])/2])

    @staticmethod
    def test_calc_inv_cap_matrix():
        x = np.linspace(-300,300,20)
        phys = simulation.PhysicsParameters(x=x)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.01, .01, .01, .01, 20, 7, 1, 3, 3, 1,
                         .01, .01, .01, 2, 4, 4, 2, .01, .01, .01])
        tf.islands = np.array([[4,10], [13,17]], dtype=np.int_)
        tf._calc_charges()
        icm = tf._calc_inv_cap_matrix()
        assert icm.shape == (2,2)
        assert np.allclose(icm, icm.T)
        assert np.all(icm >= 0)

    @staticmethod
    def test_calc_cap_energy():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.inv_cap_matrix = np.array([[20,10], [10,30]])
        tf.charges = np.array([2.4, 3.4])
        nvec1 = np.array([2,3])
        e1 = tf._calc_cap_energy(nvec1)
        nvec2 = np.array([2,4])
        e2 = tf._calc_cap_energy(nvec2)
        nvec3 = np.array([3,3])
        e3 = tf._calc_cap_energy(nvec3)
        assert np.isclose(e1, 11.2)
        assert np.isclose(e2, 9.2)
        assert np.isclose(e3, 7.2)

    @staticmethod
    def test_calc_stable_config():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.inv_cap_matrix = np.array([[20,10], [10,30]])
        tf.charges = np.array([2.4, 3.4])
        sc = tf._calc_stable_config()
        assert sc.shape == (2,)
        assert np.all(sc == [3,3])

    @staticmethod
    def test_fermi():
        phys = simulation.PhysicsParameters(kT=.1)
        tf = simulation.ThomasFermi(phys)
        val = tf._fermi(0)
        arr = tf._fermi(np.array([-.1,0,.1]))
        assert isinstance(val, float)
        assert arr.shape == (3,)
        assert np.allclose(arr, [np.e/(1+np.e), .5, 1/(1+np.e)])

    @staticmethod
    def test_dfermi():
        phys = simulation.PhysicsParameters(kT=.1)
        tf = simulation.ThomasFermi(phys)
        val = tf._dfermi(0)
        e_arr = np.array([-.1,0,.1])
        arr = tf._dfermi(e_arr)
        a1 = tf._fermi(e_arr)
        eps = 1e-5
        a2 = tf._fermi(e_arr + eps)
        approx = (a2 - a1) / eps
        assert isinstance(val, float)
        assert arr.shape == (3,)
        assert np.allclose(arr, approx, rtol=.01)

    @staticmethod
    def test_calc_qV_TF():
        x = np.linspace(-300,300,51)
        phys = simulation.PhysicsParameters(x=x, K_0=0, q=-1)
        V = simulation.calc_V(phys.gates, x, 0, 0)
        phys.V = V
        tf = simulation.ThomasFermi(phys)
        de = phys.mu - phys.q * phys.V
        tf.n = (phys.g_0 / phys.beta) * np.where(de > 0,
            phys.beta * de + np.log(1 + np.exp(phys.beta * np.where(de > 0, -de, 0))),
            np.log(1 + np.exp(phys.beta * np.where(de <= 0, de, 0)))
        )
        qvtf = tf._calc_qV_TF()
        assert qvtf.shape == V.shape
        assert np.allclose(qvtf, phys.q * phys.V)

        x = np.array([-1,0,1])
        K_mat = np.array([[3,2,1],[2,3,2],[1,2,3]])
        V = np.array([30,20,10])
        n = np.array([5,2,1])
        phys = simulation.PhysicsParameters(x=x, K_mat=K_mat, q=-1, V=V)
        tf = simulation.ThomasFermi(phys)
        tf.n = n
        qvtf = tf._calc_qV_TF()
        assert qvtf.shape == V.shape
        assert np.allclose(qvtf, [-10, -2, 2])        

    @staticmethod
    def test_calc_WKB_prob():
        x = np.linspace(-300,300,21)
        dx = 600 / (21-1)
        V = -np.cos(2*np.pi*x/300)
        phys = simulation.PhysicsParameters(x=x, q=-1, K_0=0, V=V, mu=0)
        tf = simulation.ThomasFermi(phys)
        tf.qV_TF = -V
        tf.islands = np.array([[3,8], [13,18]], dtype=np.int_)
        tf.barriers = np.array([[0,3], [8,13], [18,21]], dtype=np.int_)
        tf.is_short_circuit = False
        up_bound = phys.v_F / 2 / dx / 5
        wkb = tf._calc_WKB_prob()
        assert wkb.shape == (3,)
        assert np.all(wkb >= 0)
        assert np.all(wkb <= up_bound)

        x = np.linspace(-300,300,21)
        dx = 600 / (21-1)
        V = np.ones(x.shape)
        phys = simulation.PhysicsParameters(x=x, q=-1, K_0=0, V=V, mu=0)
        tf = simulation.ThomasFermi(phys)
        tf.qV_TF = -V
        tf.islands = np.array([], dtype=np.int_)
        tf.barriers = np.array([], dtype=np.int_)
        tf.is_short_circuit = True
        wkb = tf._calc_WKB_prob()
        assert len(wkb) == 0
        
    @staticmethod
    def test_calc_weight():
        phys = simulation.PhysicsParameters(V_L=-10, V_R=10, q=-1, mu=0, kT=20)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[3,8], [13,18], [23,28]], dtype=np.int_)
        tf.barriers = np.array([[0,3], [8,13], [18,23], [28,31]], dtype=np.int_)
        tf.is_short_circuit = False
        tf.p_WKB = np.array([1,2,3,4])
        tf.inv_cap_matrix = np.array([[20,10,3], [10,30,7], [3,7,15]])
        tf.charges = np.array([2.4, 3.4, 1.8])

        w = tf._calc_weight(np.array([2,3,4]), np.array([2,3,4]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([3,3,3]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([1,2,4]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,5,2]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([4,3,4]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,4,4]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([3,2,3]))
        assert w == 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([1,5,3]))
        assert w == 0

        w = tf._calc_weight(np.array([2,3,4]), np.array([1,3,4]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([3,3,4]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,3,3]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,3,5]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([1,4,4]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([3,2,4]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,4,3]))
        assert w > 0
        w = tf._calc_weight(np.array([2,3,4]), np.array([2,2,5]))
        assert w > 0

        phys = simulation.PhysicsParameters(V_L=-10, V_R=10, q=-1, mu=0, kT=20)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[13,18]], dtype=np.int_)
        tf.barriers = np.array([[0,13], [18,31]], dtype=np.int_)
        tf.is_short_circuit = False
        tf.p_WKB = np.array([1,2])
        tf.inv_cap_matrix = np.array([[20]])
        tf.charges = np.array([1.8])

        w = tf._calc_weight(np.array([2]), np.array([2]))
        assert w == 0
        w = tf._calc_weight(np.array([2]), np.array([4]))
        assert w == 0
        w = tf._calc_weight(np.array([2]), np.array([0]))
        assert w == 0
        w = tf._calc_weight(np.array([2]), np.array([3]))
        assert w > 0
        w = tf._calc_weight(np.array([2]), np.array([1]))
        assert w > 0

    @staticmethod
    def test_create_graph():
        phys = simulation.PhysicsParameters(V_L=-10, V_R=10, q=-1, mu=0, kT=1)
        numer = simulation.NumericsParameters(create_graph_max_changes=2)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([[3,8], [13,18], [23,28]], dtype=np.int_)
        tf.barriers = np.array([[0,3], [8,13], [18,23], [28,31]], dtype=np.int_)
        tf.p_WKB = np.array([1,2,3,4])
        tf.inv_cap_matrix = np.array([[20,10,3], [10,30,7], [3,7,15]])
        tf.charges = np.array([2.4, 3.4, 0.3])
        tf.island_charges = np.array([2, 3, 0], dtype=np.int_)
        G = tf._create_graph()
        nodes = G.nodes
        assert (2,3,0) in nodes
        assert (2,3,1) in nodes
        assert (2,5,0) in nodes
        assert (1,2,0) in nodes
        assert (3,2,0) in nodes
        assert (2,3,-1) not in nodes
        assert (5,3,0) not in nodes
        assert (3,2,1) not in nodes

        phys = simulation.PhysicsParameters(V_L=-10, V_R=10, q=-1, mu=0, kT=1)
        numer = simulation.NumericsParameters(create_graph_max_changes=2)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([], dtype=np.int_)
        tf.barriers = np.array([[0,31]], dtype=np.int_)
        tf.p_WKB = np.array([1])
        tf.inv_cap_matrix = np.array([])
        tf.charges = np.array([])
        tf.island_charges = np.array([], dtype=np.int_)
        G = tf._create_graph()
        nodes = G.nodes
        assert () in nodes
        assert len(nodes) == 1
        
    @staticmethod
    def test_calc_stable_dist():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        G = networkx.DiGraph()
        G.add_node((0,))
        G.add_node((1,))
        G.add_edge((0,), (1,), weight=1)
        G.add_edge((1,), (0,), weight=2)
        tf.G = G
        dist = tf._calc_stable_dist()
        assert dist.shape == (2,)
        assert np.isclose(dist[0], 2*dist[1])
        assert np.isclose(dist[0]+dist[1], 1)

    @staticmethod
    def test_calc_graph_charge():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[10,20]])
        G = networkx.DiGraph()
        G.add_node((0,))
        G.add_node((1,))
        G.add_edge((0,), (1,), weight=1)
        G.add_edge((1,), (0,), weight=3)
        tf.G = G
        tf.dist = np.array([.75,.25])
        gc = tf._calc_graph_charge()
        assert gc == (0,)

        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([])
        G = networkx.DiGraph()
        G.add_node(())
        tf.G = G
        tf.dist = np.array([1.])
        gc = tf._calc_graph_charge()
        assert gc == ()
        
    @staticmethod
    def test_calc_graph_current():
        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[10,20], [30,40]])
        G = networkx.DiGraph()
        G.add_node((0,1))
        G.add_node((1,0))
        G.add_node((0,0))
        G.add_edge((1,0), (0,0), weight=1)
        G.add_edge((0,0), (1,0), weight=3)
        G.add_edge((0,1), (1,0), weight=1)
        G.add_edge((1,0), (0,1), weight=3)
        G.add_edge((0,1), (0,0), weight=3)
        G.add_edge((0,0), (0,1), weight=1)
        tf.G = G
        tf.dist = np.array([1,1,1])/3
        tf.inv_cap_matrix = np.array([[10,5], [5,10]])
        tf.charges = np.array([.4, .4])
        tf.p_WKB = np.array([.5,.5,.5])
        gc = tf._calc_graph_current()
        assert np.isclose(gc, 2/3)

        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([])
        G = networkx.DiGraph()
        G.add_node(())
        tf.G = G
        tf.dist = np.array([1.])
        tf.inv_cap_matrix = np.array([])
        tf.charges = np.array([])
        tf.p_WKB = np.array([.5])
        tf.is_short_circuit = False
        gc = tf._calc_graph_current()
        assert np.isclose(gc, phys.barrier_current)

        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[10,20]])
        G = networkx.DiGraph()
        G.add_node((0,))
        G.add_node((1,))
        G.add_edge((0,), (1,), weight=1)
        G.add_edge((1,), (0,), weight=1)
        tf.G = G
        tf.dist = np.array([1,1])/2
        tf.inv_cap_matrix = np.array([[10]])
        tf.charges = np.array([.5])
        tf.p_WKB = np.array([.5,.5])
        gc = tf._calc_graph_current()
        assert np.isclose(gc, 0, atol=1e-8)

    @staticmethod
    def test_calc_current():
        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        numer = simulation.NumericsParameters(create_graph_max_changes=1)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([[3,8], [13,18]], dtype=np.int_)
        tf.barriers = np.array([[0,3], [8,13], [18,21]], dtype=np.int_)
        tf.is_short_circuit = False
        tf.p_WKB = np.array([1,1,1])
        tf.inv_cap_matrix = np.array([[1,0], [0,1]])
        tf.charges = np.array([.2, .2])
        tf.island_charges = np.array([0, 0], dtype=np.int_)
        cur = tf._calc_current()
        assert np.isclose(cur, 0., atol=1e-6)

        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        numer = simulation.NumericsParameters(create_graph_max_changes=1)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([], dtype=np.int_)
        tf.barriers = np.array([[0,21]], dtype=np.int_)
        tf.is_short_circuit = False
        tf.p_WKB = np.array([1])
        tf.inv_cap_matrix = np.array([])
        tf.charges = np.array([])
        tf.island_charges = np.array([], dtype=np.int_)
        cur = tf._calc_current()
        assert np.isclose(cur, phys.barrier_current)

        phys = simulation.PhysicsParameters(q=-1, V_L=0, V_R=0, mu=0, kT=1)
        numer = simulation.NumericsParameters(create_graph_max_changes=1)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([], dtype=np.int_)
        tf.barriers = np.array([], dtype=np.int_)
        tf.is_short_circuit = True
        tf.p_WKB = np.array([])
        tf.inv_cap_matrix = np.array([])
        tf.charges = np.array([])
        tf.island_charges = np.array([], dtype=np.int_)
        cur = tf._calc_current()
        assert np.isclose(cur, phys.short_circuit_current)

    @staticmethod
    def test_count_transitions():
        phys = simulation.PhysicsParameters()
        numer = simulation.NumericsParameters(count_transitions_sigma=.01, count_transitions_eps=.5)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([[10,20]])
        G = networkx.DiGraph()
        G.add_node((0,))
        G.add_node((1,))
        G.add_node((2,))
        G.add_edge((0,), (1,), weight=.6)
        G.add_edge((1,), (0,), weight=.61)
        G.add_edge((1,), (2,), weight=2.3)
        G.add_edge((2,), (1,), weight=2.7)
        tf.G = G
        tf.dist = np.array([.2,.7,.1])
        tr = tf._count_transitions()
        assert tr == 2
        numer = simulation.NumericsParameters(count_transitions_sigma=.01, count_transitions_eps=.05)
        tf.numerics = numer
        tr = tf._count_transitions()
        assert tr == 1
        numer = simulation.NumericsParameters(count_transitions_sigma=1., count_transitions_eps=.5)
        tf.numerics = numer
        tr = tf._count_transitions()
        assert tr == 1
        numer = simulation.NumericsParameters(count_transitions_sigma=1., count_transitions_eps=.05)
        tf.numerics = numer
        tr = tf._count_transitions()
        assert tr == 0

        phys = simulation.PhysicsParameters()
        numer = simulation.NumericsParameters(count_transitions_sigma=.01, count_transitions_eps=.5)
        tf = simulation.ThomasFermi(phys, numerics=numer)
        tf.islands = np.array([])
        G = networkx.DiGraph()
        G.add_node(())
        tf.G = G
        tf.dist = np.array([1.])
        tr = tf._count_transitions()
        assert tr == 0

    @staticmethod
    def test_sensor_from_island_charges():
        x = np.linspace(-50,50,11)
        sensors = np.array([[-30,-50,0], [30,-50,0]])
        phys = simulation.PhysicsParameters(x=x, q=-1, screening_length=50, sensors=sensors)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.1,3,6,9,.1,.01,.1,.1,18,.1,.1])
        tf.islands = np.array([[1,4],[8,9]])
        sens = tf._sensor_from_island_charges(np.array([0,1]))
        assert sens.shape == (2,)
        assert np.isclose(sens[1], -1)
        assert sens[0] > -1 and sens[0] < 0
        sens = tf._sensor_from_island_charges(np.array([1,0]))
        assert sens.shape == (2,)
        assert sens[0] > -1
        assert sens[1] > sens[0] and sens[1] < 0

    @staticmethod
    def test_calc_sensor():
        x = np.linspace(-50,50,11)
        sensors = np.array([[-30,-50,0], [30,-50,0]])
        phys = simulation.PhysicsParameters(x=x, q=-1, screening_length=50, sensors=sensors)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.1,3,6,9,.1,.01,.1,.1,18,.1,.1])
        tf.islands = np.array([[1,4],[8,9]])
        tf.island_charges = np.array([1,1])
        sens = tf._calc_sensor()
        assert sens.shape == (2,)
        assert sens[1] < sens[0]

    @staticmethod
    def test_calc_dot_states():
        x = np.linspace(-50,50,101)
        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-35,-8],[12,22]])+50
        tf.island_charges = np.array([3,2])
        tf.charge_centers = np.array([-21,16])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [3,2])
        assert np.all(comb == [False])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-25,17]])+50
        tf.island_charges = np.array([6])
        tf.charge_centers = np.array([-4])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [6,0])
        assert np.all(comb == [True])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-35,-21], [2,12]])+50
        tf.island_charges = np.array([0, 1])
        tf.charge_centers = np.array([-27, 7])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [0,1])
        assert np.all(comb == [False])
        assert np.all(oc == [False, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-45,-35], [-3,7]])+50
        tf.island_charges = np.array([2, 1])
        tf.charge_centers = np.array([-40, 2])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [0,1])
        assert np.all(comb == [False])
        assert np.all(oc == [False, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-25,-17], [-3,17], [24,38]])+50
        tf.island_charges = np.array([1,2,3])
        tf.charge_centers = np.array([-21, 7, 36])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [1,5])
        assert np.all(comb == [False])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-40, -27], [-23,37]])+50
        tf.island_charges = np.array([1,3])
        tf.charge_centers = np.array([-34, 6])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [4,0])
        assert np.all(comb == [True])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-33, -16], [-13,37]])+50
        tf.island_charges = np.array([1,3])
        tf.charge_centers = np.array([-25, 12])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [1,3])
        assert np.all(comb == [False])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-33, -16], [-13,17], [19,38]])+50
        tf.island_charges = np.array([1,3,2])
        tf.charge_centers = np.array([-24, 2, 29])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [1,5])
        assert np.all(comb == [False])
        assert np.all(oc == [True, True])

        tf = simulation.ThomasFermi(phys)
        tf.islands = np.array([[-33, 21], [24,37]])+50
        tf.island_charges = np.array([3,1])
        tf.charge_centers = np.array([-6, 31])
        oc, comb, ch = tf._calc_dot_states()
        assert oc.shape == (2,)
        assert comb.shape == (1,)
        assert ch.shape == (2,)
        assert np.all(ch == [4,0])
        assert np.all(comb == [True])
        assert np.all(oc == [True, True])

    @staticmethod
    def test_island_charges_from_charge_state():
        x = np.linspace(-50,50,101)
        gates = [simulation.GateParameters(mean=(20*i-40)) for i in range(5)]
        phys = simulation.PhysicsParameters(x=x, gates=gates, dot_regions=None)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-45,-35],[-25,-15],[-5,5],[15,25],[35,45]])+50
        ic = tf._island_charges_from_charge_state(np.array([2,3]), np.array([False]))
        assert ic.shape == (5,)
        assert np.all(ic == [0,2,0,3,0])
        
        x = np.linspace(-50,50,101)
        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-25,-7],[13,38]])+50
        ic = tf._island_charges_from_charge_state(np.array([2,3]), np.array([False]))
        assert ic.shape == (2,)
        assert np.all(ic == [2,3])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-25,38]])+50
        ic = tf._island_charges_from_charge_state(np.array([5,0]), np.array([True]))
        assert ic.shape == (1,)
        assert np.all(ic == [5])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-33,-23],[-19,-7],[11,31]])+50
        ic = tf._island_charges_from_charge_state(np.array([5,0]), np.array([False]))
        assert ic.shape == (3,)
        assert np.all(ic == [2,3,0])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([])+50
        ic = tf._island_charges_from_charge_state(np.array([2,3]), np.array([False]))
        assert len(ic) == 0
    
        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-32,-16],[-11,11],[14,32]])+50
        ic = tf._island_charges_from_charge_state(np.array([1,0]), np.array([True]))
        assert ic.shape == (3,)
        assert np.all(ic == [0,1,0])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[-32,-16],[-11,11],[14,32]])+50
        ic = tf._island_charges_from_charge_state(np.array([2,0]), np.array([True]))
        assert ic.shape == (3,)
        assert np.all(ic == [0,1,1])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.zeros(101)
        tf.islands = np.array([[-32,-26],[-21,21],[24,32]])+50
        ic = tf._island_charges_from_charge_state(np.array([3,0]), np.array([True]))
        assert ic.shape == (3,)
        assert np.all(ic == [1,1,1])

        dot_regions = np.array([[-30,-10],[10,30]])
        phys = simulation.PhysicsParameters(x=x, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.ones(101)
        tf.islands = np.array([[6,28]])+50
        ic = tf._island_charges_from_charge_state(np.array([0,2]), np.array([False]))
        assert ic.shape == (1,)
        assert np.all(ic == [2])


    @staticmethod
    def test_sensor_from_charge_state():
        x = np.linspace(-5,5,11)
        dot_regions = np.array([[-3,-1],[1,3]])
        sensors = np.array([[-3,-5,0], [3,-5,0]])
        phys = simulation.PhysicsParameters(x=x, q=-1, screening_length=5,
                                    sensors=sensors, dot_regions=dot_regions)
        tf = simulation.ThomasFermi(phys)
        tf.n = np.array([.1,3,6,9,.1,.01,.1,.1,13,.1,.1])
        tf.islands = np.array([[1,4],[8,9]])
        sens = tf.sensor_from_charge_state(np.array([3,2]), np.array([False]))
        assert sens.shape == (2,)
        assert np.all(sens < 0) and np.all(sens > -5)
  
    @staticmethod
    def test_run_calculations():
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        out = tf.run_calculations(inc_trans=True)
        assert out.transition_count is not None
        assert out.current is None
        assert out.inv_cap_matrix is None
        phys = simulation.PhysicsParameters()
        tf = simulation.ThomasFermi(phys)
        out = tf.run_calculations(inc_curr=True, inc_inv_cap_matrix=True)
        assert out.transition_count is None
        assert out.current is not None
        assert out.inv_cap_matrix is not None
        
