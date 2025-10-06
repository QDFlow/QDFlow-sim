import numpy as np
import pytest
from qdflow.physics import noise
from qdflow.util import distribution


# ----------------
# Test Dataclasses
# ----------------

class TestDataclasses:
    
    @staticmethod
    def test_NoiseParameters():
        d = {
            "noise_axis": 1,
            "sech_blur_width": 1.5,
            "unint_dot_spacing": np.array([1,2,3]),
            "extraneous_key": 12345
        }
        params = noise.NoiseParameters.from_dict(d)
        assert params.unint_dot_spacing.shape == (3,)
        assert np.all(params.unint_dot_spacing == d["unint_dot_spacing"])
        assert params.unint_dot_spacing is not d["unint_dot_spacing"]
        assert np.isclose(params.sech_blur_width, 1.5)
        assert params.coulomb_peak_width is None
        assert not hasattr(params, "extraneous_key")
        
        # check for deep copy
        params_copy = params.copy()
        assert params.sech_blur_width == params_copy.sech_blur_width
        assert params_copy.unint_dot_spacing.shape == params.unint_dot_spacing.shape
        assert np.all(params_copy.unint_dot_spacing == params.unint_dot_spacing)
        assert params_copy.unint_dot_spacing is not params.unint_dot_spacing
        assert params_copy is not params
        
        to_d = params.to_dict()
        assert to_d["unint_dot_spacing"].shape == d["unint_dot_spacing"].shape
        assert np.all(to_d["unint_dot_spacing"] == d["unint_dot_spacing"])
        assert to_d["unint_dot_spacing"] is not d["unint_dot_spacing"]

    @staticmethod
    def test_NoiseRandomization():
        corr = distribution.FullyCorrelated(distribution.Uniform(1,5), 2).dependent_distributions()
        d = {
            "white_noise_magnitude":.3,
            "pink_noise_magnitude":distribution.Delta(.4),
            "telegraph_low_pixels":corr[0],
            "telegraph_high_pixels":corr[1],
            "extraneous_key": 12345
        }
        rand = noise.NoiseRandomization.from_dict(d)
        assert rand.white_noise_magnitude == .3
        assert rand.pink_noise_magnitude is d["pink_noise_magnitude"]
        assert rand.telegraph_low_pixels is corr[0]
        assert rand.sech_blur_width == noise.NoiseRandomization().sech_blur_width
        assert not hasattr(rand, "extraneous_key")
        
        # check deep copy
        rand_copy = rand.copy()
        assert rand_copy.white_noise_magnitude == rand.white_noise_magnitude
        assert rand_copy is not rand
        assert rand_copy.pink_noise_magnitude._value == rand.pink_noise_magnitude._value
        assert rand_copy.pink_noise_magnitude is not rand.pink_noise_magnitude
        assert rand_copy.telegraph_low_pixels is not rand.telegraph_low_pixels
        assert rand_copy.telegraph_high_pixels is not rand.telegraph_high_pixels
        assert rand_copy.telegraph_low_pixels.dependent_distributions[0] is rand_copy.telegraph_low_pixels
        assert rand_copy.telegraph_low_pixels.dependent_distributions[1] is rand_copy.telegraph_high_pixels
        assert rand_copy.telegraph_high_pixels.dependent_distributions[0] is rand_copy.telegraph_low_pixels

        to_d = rand.to_dict()
        assert to_d is not d
        assert to_d["white_noise_magnitude"] == d["white_noise_magnitude"]
        assert to_d["pink_noise_magnitude"]._value == d["pink_noise_magnitude"]._value
        assert to_d["pink_noise_magnitude"] is not d["pink_noise_magnitude"]
        assert to_d["telegraph_low_pixels"] is not rand.telegraph_low_pixels
        assert to_d["telegraph_high_pixels"] is not rand.telegraph_high_pixels
        assert to_d["telegraph_low_pixels"].dependent_distributions[0] is to_d["telegraph_low_pixels"]
        assert to_d["telegraph_low_pixels"].dependent_distributions[1] is to_d["telegraph_high_pixels"]
        assert to_d["telegraph_high_pixels"].dependent_distributions[0] is to_d["telegraph_low_pixels"]

        rand_def1 = noise.NoiseRandomization.default()
        rand_def2 = noise.NoiseRandomization.default()
        assert rand_def1 is not rand_def2
        assert rand_def1.n_gates == rand_def2.n_gates

# ------------------------
# Test random_noise_params
# ------------------------

def test_random_noise_params():
    noise.set_rng_seed(456)
    rand = noise.NoiseRandomization.default()
    rand.n_gates = 3
    rand.white_noise_magnitude = distribution.Uniform(.2,.3)
    rand.pink_noise_magnitude = .4
    rand.unint_dot_spacing = distribution.FullyCorrelated(distribution.Uniform(3,7), 3)
    rand.sensor_gate_coupling = distribution.Uniform(.1,.3)
    params = noise.random_noise_params(rand)
    assert isinstance(params, noise.NoiseParameters)
    assert params.white_noise_magnitude >= .2 and params.white_noise_magnitude <= .3
    assert np.isclose(params.pink_noise_magnitude, .4)
    assert params.unint_dot_spacing.shape == (3,)
    assert np.all((params.unint_dot_spacing >= 3) & (params.unint_dot_spacing <= 7))
    assert np.allclose(params.unint_dot_spacing, params.unint_dot_spacing[0])
    assert params.sensor_gate_coupling.shape == (3,)
    assert np.all((params.sensor_gate_coupling >= .1) & (params.sensor_gate_coupling <= .3))
    
    rand = noise.NoiseRandomization.default()
    rand.n_gates = 3
    rand.coulomb_peak_width = None
    rand.unint_dot_spacing = None
    rand.sensor_gate_coupling = np.array([.1,.2,.3])
    params = noise.random_noise_params(rand)
    assert isinstance(params, noise.NoiseParameters)
    assert params.coulomb_peak_width is None
    assert params.unint_dot_spacing is None
    assert params.sensor_gate_coupling.shape == (3,)
    assert np.allclose(params.sensor_gate_coupling, [.1,.2,.3])

    rand = noise.NoiseRandomization.default()
    rand.n_gates = 1
    rand.unint_dot_spacing = distribution.FullyCorrelated(distribution.Uniform(3,7), 1)
    rand.sensor_gate_coupling = distribution.Uniform(.1,.3)
    params = noise.random_noise_params(rand)
    assert isinstance(params, noise.NoiseParameters)
    assert params.unint_dot_spacing.shape == (1,)
    assert np.all((params.unint_dot_spacing >= 3) & (params.unint_dot_spacing <= 7))
    assert params.sensor_gate_coupling.shape == (1,)
    assert np.all((params.sensor_gate_coupling >= .1) & (params.sensor_gate_coupling <= .3))
    
# -------------------
# Test NoiseGenerator
# -------------------

class TestNoiseGenerator:

    @staticmethod
    def test_white_noise():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.white_noise(data, 1)
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.white_noise(data, 0)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, data)

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.white_noise(data, 1)
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

    @staticmethod
    def test_pink_noise():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.pink_noise(data, 1, axis=None)
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.pink_noise(data, 1, axis=0)
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.pink_noise(data, 0, axis=None)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, data)

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.pink_noise(data, 1, axis=None)
        assert noisy.shape == data.shape
        assert not np.allclose(noisy, data)

    @staticmethod
    def test_telegraph_noise():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.telegraph_noise(data, 7., 0, 5, 5, axis=0)
        assert noisy.shape == data.shape
        assert np.allclose(np.abs(noisy - data), 3.5)

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.telegraph_noise(data, 7., 0, 5, 5, axis=0)
        assert noisy.shape == data.shape
        assert np.allclose(np.abs(noisy - data), 3.5)

    @staticmethod
    def test_line_shift():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.line_shift(data, 1, axis=1, shift_positive=True)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, np.round(noisy))
        assert np.all(noisy < 30)
        assert np.all(noisy[1:,:] > -1)

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.line_shift(data, 1, axis=1, shift_positive=False)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, np.round(noisy))
        assert np.all(noisy > -1)
        assert np.all(noisy[:-1,:] < 30)

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.line_shift(data, 1, axis=0, shift_positive=True)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, data)

    @staticmethod
    def test_latching_noise():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        excited = np.array(np.arange(30,60).reshape((5,6)), dtype=np.float64)
        dot_charges = np.concatenate([np.full((5,6,1), 1, dtype=np.int_), 
                np.concatenate([np.full((2,6,1), 1, dtype=np.int_), np.full((3,6,1), 2, dtype=np.int_)], axis=0)
        ], axis=2)
        are_dots_combined = np.full((5,6,1), False, dtype=np.bool_)
        noisy = ng.latching_noise(data, excited, dot_charges, are_dots_combined, 1, axis=0, shift_positive=True)
        assert noisy.shape == data.shape
        assert np.allclose(noisy[0:2,:], data[0:2,:])
        assert np.all(np.isclose(noisy, data) | np.isclose(noisy, excited))

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        excited = np.array(np.arange(30,60).reshape((5,6)), dtype=np.float64)
        dot_charges = np.concatenate([np.full((5,6,1), 1, dtype=np.int_), 
                np.concatenate([np.full((2,6,1), 1, dtype=np.int_), np.full((3,6,1), 2, dtype=np.int_)], axis=0)
        ], axis=2)
        are_dots_combined = np.full((5,6,1), False, dtype=np.bool_)
        noisy = ng.latching_noise(data, excited, dot_charges, are_dots_combined, 1, axis=0, shift_positive=False)
        assert noisy.shape == data.shape
        assert np.allclose(noisy[2:,:], data[2:,:])
        assert np.all(np.isclose(noisy, data) | np.isclose(noisy, excited))

        data = np.array(np.arange(30), dtype=np.float64)
        excited = np.array(np.arange(30,60), dtype=np.float64)
        dot_charges = np.concatenate([np.full((30,1), 1, dtype=np.int_), 
                np.concatenate([np.full((15,1), 1, dtype=np.int_), np.full((15,1), 2, dtype=np.int_)], axis=0)
        ], axis=1)
        are_dots_combined = np.full((30,1), False, dtype=np.bool_)
        noisy = ng.latching_noise(data, excited, dot_charges, are_dots_combined, 1, axis=0, shift_positive=True)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, data)

    @staticmethod
    def test_unint_dot_add():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.unint_dot_add(data, 50, np.array([5, 5]), 1, .5)
        assert noisy.shape == data.shape
        assert (noisy[4,5] - noisy[0,0]) > 29+75 and (noisy[4,5] - noisy[0,0]) < 29+125

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.unint_dot_add(data, 50, np.array([7, 7, 7]), .1, .5,
                                 gate_data_matrix=np.array([[1,0],[0,1],[1,1]]))
        assert noisy.shape == data.shape
        assert (noisy[4,5] - noisy[0,0]) > 29+75 and (noisy[4,5] - noisy[0,0]) < 29+125

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = noisy = ng.unint_dot_add(data, 50, np.array([15]), 1, .5)
        assert noisy.shape == data.shape
        assert (noisy[29] - noisy[15]) > 14+75 and (noisy[29] - noisy[15]) < 14+125
        assert (noisy[15] - noisy[0]) > 15+75 and (noisy[15] - noisy[0]) < 15+125

    @staticmethod
    def test_sech_blur():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array([[20,20,20,20,20,20,20],[1,0,1,0,1,0,1],[20,20,20,20,20,20,20]], dtype=np.float64)
        noisy = ng.sech_blur(data, 3, noise_axis=1)
        assert noisy.shape == data.shape
        assert noisy[1,3] > .25
        assert noisy[1,3] < 1

        data = np.array([[20,20,20,20,20,20,20],[1,0,1,0,1,0,1],[20,20,20,20,20,20,20]], dtype=np.float64)
        noisy = ng.sech_blur(data, 3, noise_axis=0)
        assert noisy.shape == data.shape
        assert noisy[1,3] > 1
        assert noisy[1,3] < 20

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.sech_blur(data, 3, noise_axis=0)
        assert noisy.shape == data.shape
        assert np.isclose(noisy[15], data[15])
        assert not np.isclose(noisy[2], data[2])
        assert not np.isclose(noisy[28], data[28])

        error = False
        data = np.array([[20,20,20,20,20,20,20],[1,0,1,0,1,0,1],[20,20,20,20,20,20,20]], dtype=np.float64)
        try:
            noisy = ng.sech_blur(data, 3, noise_axis=4)
        except ValueError:
            error = True
        assert error

    @staticmethod
    def test_sensor_gate():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.ones((5,6))
        noisy = ng.sensor_gate(data, np.array([1.7, 3]), 1)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, 1+np.add.outer(1.7*np.arange(5), 3*np.arange(6)))

        data = np.ones((5,6))
        noisy = ng.sensor_gate(data, np.array([1.7, 3, -.5]), 1, gate_data_matrix=np.array([[1,0],[0,1],[1,1]]))
        assert noisy.shape == data.shape
        assert np.allclose(noisy, 1+np.add.outer(1.2*np.arange(5), 2.5*np.arange(6)))

        data = np.ones(30)
        noisy = noisy = ng.sensor_gate(data, np.array([1.7]), 1)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, 1+1.7*np.arange(30))

    @staticmethod
    def test_coulomb_peak():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.coulomb_peak(data, 15, 5)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0) & (noisy <= 1))

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.coulomb_peak(data, 14.5, 5)
        noisy2 = ng.coulomb_peak(data, 14.5, 10)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0) & (noisy <= 1))
        assert np.allclose(noisy, np.flip(noisy))
        assert noisy2.shape == data.shape
        assert np.all(noisy2 > noisy)

    @staticmethod
    def test_high_coupling_coulomb_peak():
        noise.set_rng_seed(456)
        ng = noise.NoiseGenerator(noise.NoiseParameters())
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.high_coupling_coulomb_peak(data, .5, 5, 10)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0))

        data = np.array(np.arange(30), dtype=np.float64)
        noisy = ng.high_coupling_coulomb_peak(data, .5, 3, 10)
        noisy2 = ng.high_coupling_coulomb_peak(data, 0, 3, 10)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0))
        assert noisy2.shape == data.shape
        assert np.all((noisy2 >= 0))
        assert np.all(noisy2[0:3] > noisy[0:3])
        assert np.all(noisy2[3:8] < noisy[3:8])
        assert np.all(noisy2[8:13] > noisy[8:13])
        assert np.all(noisy2[13:18] < noisy[13:18])
        assert np.all(noisy2[18:23] > noisy[18:23])
        assert np.all(noisy2[23:28] < noisy[23:28])
        assert np.all(noisy2[28:30] > noisy[28:30])


    @staticmethod
    def test_calc_noisy_map():
        noise.set_rng_seed(456)
        params = noise.NoiseParameters(
            white_noise_magnitude=1.,
            pink_noise_magnitude=1.,
            telegraph_magnitude=1.,
            telegraph_stdev=.3,
            telegraph_low_pixels=3,
            telegraph_high_pixels=3,
            noise_axis=0,
            latching_pixels=3,
            latching_positive=True,
            sech_blur_width=1,
            unint_dot_magnitude=1.,
            unint_dot_spacing=np.array([5,5]),
            unint_dot_width=1.,
            unint_dot_offset=.5,
            coulomb_peak_spacing=12,
            coulomb_peak_offset=.5,
            coulomb_peak_width=5,
            sensor_gate_coupling=np.array([-.3,-.7]),
            use_pink_noise_all_dims=False
        )
        ng = noise.NoiseGenerator(params)
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        excited = np.array(np.arange(30,60).reshape((5,6)), dtype=np.float64)
        dot_charges = np.concatenate([np.full((5,6,1), 1, dtype=np.int_), 
                np.concatenate([np.full((2,6,1), 1, dtype=np.int_), np.full((3,6,1), 2, dtype=np.int_)], axis=0)
        ], axis=2)
        are_dots_combined = np.full((5,6,1), False, dtype=np.bool_)
        noisy = ng.calc_noisy_map(data, latching_data=(excited, dot_charges, are_dots_combined), noise_default=True)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0) & (noisy <= 1.5))

        params = noise.NoiseParameters(
            white_noise_magnitude=1.,
            pink_noise_magnitude=1.,
            telegraph_magnitude=1.,
            telegraph_stdev=.3,
            telegraph_low_pixels=3,
            telegraph_high_pixels=3,
            noise_axis=0,
            latching_pixels=3,
            latching_positive=True,
            sech_blur_width=1,
            unint_dot_magnitude=1.,
            unint_dot_spacing=np.array([5,5]),
            unint_dot_width=1.,
            unint_dot_offset=.5,
            coulomb_peak_spacing=12,
            coulomb_peak_offset=.5,
            coulomb_peak_width=5,
            sensor_gate_coupling=np.array([-.3,-.7]),
            use_pink_noise_all_dims=True
        )
        ng = noise.NoiseGenerator(params)
        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.calc_noisy_map(data, noise_default=True)
        assert noisy.shape == data.shape
        assert np.all((noisy >= 0) & (noisy <= 1.5))

        data = np.array(np.arange(30).reshape((5,6)), dtype=np.float64)
        noisy = ng.calc_noisy_map(data, noise_default=False)
        assert noisy.shape == data.shape
        assert np.allclose(noisy, data)
