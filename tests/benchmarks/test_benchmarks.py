import numpy as np
import pytest
from qdflow.physics import simulation
from qdflow import generate
from qdflow.util import distribution



def varied_physics(n, n_dots):
    generate.set_rng_seed(42)
    phys_rand = generate.PhysicsRandomization.default()
    phys_rand.num_dots = n_dots
    phys_rand.gate_x_variations = distribution.Normal(0,5)
    phys_rand.K_0 = distribution.LogUniform(.3, 70)
    phys_rand.sigma = distribution.Uniform(40, 80) + distribution.Uniform(-20, 20)
    phys_rand.mu = -1.6
    phys_rand.g_0 = distribution.LogUniform(.005, .009)
    phys_rand.beta = distribution.LogUniform(10, 1000) * distribution.LogUniform(.1, 10)
    phys_rand.c_k = distribution.LogUniform(.25, 6) * distribution.LogUniform(.5, 2)
    phys_rand.screening_length = distribution.LogUniform(75, 150) * distribution.LogUniform(.7, 1.4)
    phys_rand.rho = distribution.Uniform(10, 20)
    phys_rand.h = distribution.Normal(80, 8).abs()
    phys_rand.h_variations = distribution.Normal(0,3.5)
    phys_rand.plunger_peak_variations = distribution.Uniform(-20,0)
    phys_rand.barrier_peak = 3.
    phys_rand.external_barrier_peak_variations = distribution.Uniform(.5, 3.5)
    phys_rand.internal_barrier_peak_variations = distribution.Uniform(-.6, 1.6) + distribution.Uniform(-1, 1)
    phys_rand.sensor_y = 0
    phys_rand.sensor_y_variations = -distribution.LogUniform(100, 400)
    phys_rand.num_sensors = 3
    phys_rand.sensor_x_variations = distribution.MatrixCorrelated(np.array([[1,0,0],[0,1,0],[0,0,1]]),
                                    [110-(distribution.LogUniform(4,80)*distribution.LogUniform(1,2)),
                                    distribution.Uniform(-40,40),
                                    (-110)+(distribution.LogUniform(4,80)*distribution.LogUniform(1,2))])
    phys_params = generate.random_physics(phys_rand, n)
    return phys_params

# -----------------------
# Benchmark solving for n
# -----------------------

class TestCalcNBenchmark:
    
    @staticmethod
    @pytest.mark.parametrize("n_dots", [2,3,4,5,6,8,10,12,14,16,18,20,22,24])
    def test_n_benchmark(benchmark, n_dots):
        n_rounds = 10000
        counter = 0
        phys_list = varied_physics(n_rounds, n_dots)
        numer = simulation.NumericsParameters(calc_n_rel_tol=1e-3, calc_n_max_iterations_no_guess=1000)
        
        def setup():
            nonlocal counter
            nonlocal phys_list
            nonlocal numer
            phys = phys_list[counter]
            counter += 1
            return (phys, numer), {}
        
        def wrapper(phys, numer):
            x = phys.x
            V = simulation.calc_V(phys.gates, x, 0, 0)
            K_mat = simulation.calc_K_mat(x, phys.K_0, phys.sigma)
            delta_x = (phys.x[-1] - phys.x[0]) / (len(phys.x) - 1)
            g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0 * delta_x * K_mat + np.identity(len(K_mat)))
            n, phi, converged = simulation.ThomasFermi.calc_n(phys, numer, V, K_mat, g0_dx_K_plus_1_inv)
        
        benchmark.pedantic(wrapper, setup=setup, rounds=n_rounds)

    @staticmethod
    @pytest.mark.parametrize("n_dots", [2,3,4,5,6,8,10,12,14,16,18,20,22,24])
    def test_cap_model_benchmark(benchmark, n_dots):
        n_rounds = 10000
        counter = 0
        phys_list = varied_physics(n_rounds, n_dots)
        numer = simulation.NumericsParameters(calc_n_rel_tol=1e-3, calc_n_max_iterations_no_guess=1000)
        
        def setup():
            nonlocal counter
            nonlocal phys_list
            nonlocal numer
            phys = phys_list[counter]
            counter += 1
            x = phys.x
            V = simulation.calc_V(phys.gates, x, 0, 0)
            K_mat = simulation.calc_K_mat(x, phys.K_0, phys.sigma)
            delta_x = (phys.x[-1] - phys.x[0]) / (len(phys.x) - 1)
            g0_dx_K_plus_1_inv = np.linalg.inv(phys.g_0 * delta_x * K_mat + np.identity(len(K_mat)))
            n, phi, converged = simulation.ThomasFermi.calc_n(phys, numer, V, K_mat, g0_dx_K_plus_1_inv)
            return (phys, numer, V, K_mat, n), {}
        
        def wrapper(physics, numerics, V, K_mat, n):
            qV_TF = simulation.ThomasFermi.calc_qV_TF(physics, V, K_mat, n)
            islands, barriers, all_islands, is_short_circuit = \
                        simulation.ThomasFermi.calc_islands_and_barriers(physics, numerics, n)
            charges = simulation.ThomasFermi.calc_charges(physics, n, islands)
            charge_centers = simulation.ThomasFermi.calc_charge_centers(physics, n, islands)
            energy_matrix = simulation.ThomasFermi.calc_energy_matrix(physics, numerics,
                        K_mat, n, islands, charges)
            integer_charges = simulation.ThomasFermi.calc_integer_charges(numerics, energy_matrix, charges)
            are_dots_occupied, are_dots_combined, dot_charges = \
                        simulation.ThomasFermi.calc_dot_states(physics, islands,
                        integer_charges, charge_centers)
            sensor_output = simulation.ThomasFermi.sensor_from_charge_state(physics, n,
                        islands, dot_charges, are_dots_combined)

        benchmark.pedantic(wrapper, setup=setup, rounds=n_rounds)

