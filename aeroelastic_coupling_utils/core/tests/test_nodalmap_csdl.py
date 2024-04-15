import numpy as np
import csdl_alpha as csdl
from aeroelastic_coupling_utils.core.nodalmap_csdl import NodalMap

np.random.seed(1)

##test csdl nodalmap comprehensive all
def test_nodalmap_csdl_randominputs_gaussian():

    in_shape_test = (5, 3)
    out_shape_test = (8, 3)
    weight_eps_test_val = 3.

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # create test model that wraps the NodalMap model
    recorder = csdl.Recorder()
    recorder.start()
    test_model_Gaussian = NodalMap(weight_eps=weight_eps_test_val, weight_func_name='Gaussian', weight_to_be_normalized=True)
    # construct mesh csdl variables 
    solid_mesh_csdl = csdl.Variable(value=rng_solid_mesh)
    fluid_mesh_csdl = csdl.Variable(value=rng_fluid_mesh)
    # run Gaussian function test model
    nodal_map = test_model_Gaussian.evaluate(solid_mesh_csdl, fluid_mesh_csdl)
    recorder.stop()
    recorder.execute()

    nodalmap = nodal_map.value  # extract nodal map array

    nodalmap_row_sums = nodalmap.sum(axis=1)

    # assert that the row sums are all almost equal to 1
    np.testing.assert_almost_equal(
        nodalmap_row_sums, 
        np.ones((in_shape_test[0],)),
        decimal = 12,
    )

