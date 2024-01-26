from csdl import Model
from python_csdl_backend import Simulator
import csdl
import numpy as np

from aeroelastic_coupling_utils.utils.weightfunctions_csdl import WeightFunctions
from aeroelastic_coupling_utils.utils.distancecalculation_csdl import DistanceCalculation
from aeroelastic_coupling_utils.utils.weightnormalization_csdl import WeightNormalization

class NodalMap(Model):
    def initialize(self):
        # define constant model parameters
        self.parameters.declare('mesh_in_name', types=str, default="mesh_in")
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('mesh_out_name', types=str, default="mesh_out")
        self.parameters.declare('out_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
        self.parameters.declare('Weight_func_name', types=str, default='Gaussian')
        self.parameters.declare('Weight_eps_name', types=str, default='weight_eps')

    def define(self):
        Weight_func_name = self.parameters['Weight_func_name']
        Weight_eps_name = self.parameters['Weight_eps_name']
        # Weight_eps = self.parameters['Weight_eps']

        mesh_in_name = self.parameters['mesh_in_name']
        in_shape = self.parameters['in_shape']
        mesh_out_name = self.parameters['mesh_out_name']
        out_shape = self.parameters['out_shape']
        out_name = self.parameters['out_name']

        weight_eps = self.declare_variable(Weight_eps_name, shape=(1,))
        in_mesh = self.declare_variable(mesh_in_name, shape=in_shape)
        out_mesh = self.declare_variable(mesh_out_name, shape=out_shape)
        column_scaling_vector = self.declare_variable('column_scaling_vector', shape=(in_shape[0],), val=np.ones((in_shape[0],)))

        # compute distances between in_mesh and out_mesh
        self.add(DistanceCalculation(mesh_in_name=mesh_in_name_test, in_shape=in_shape_test, 
                            mesh_out_name=mesh_out_name_test, out_shape=out_shape_test, 
                            out_name='distance_array'), name='Distancecalc_model')

        # we register the distance array so it can be passed to the WeightFunctions objects
        self.declare_variable('distance_array', shape=(in_shape[0], out_shape[0]))

        self.add(WeightFunctions(Weight_func_name=Weight_func_name, Weight_eps_name=Weight_eps_name, 
                                      in_name='distance_array', in_shape=tuple([in_shape[0], out_shape[0]]), 
                                      out_name='dist_weights_uncorrected'), name='Weightfunction_comp')
        # declare the output of the WeightFunctions model for further use
        dist_weights_uncorrected = self.declare_variable('dist_weights_uncorrected', shape=(in_shape[0], out_shape[0]))

        self.add(WeightNormalization(weight_array_in_name='dist_weights_uncorrected', weight_array_in_shape=(in_shape[0], out_shape[0]),
                                     column_scaling_vec_name='column_scaling_vector', out_name=out_name, out_shape=(out_shape[0], in_shape[0])))
        dist_weights = self.declare_variable(out_name, shape=(out_shape[0], in_shape[0]))

        # self.register_output(out_name, dist_weights) 

if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    mesh_in_name_test = 'solid_mesh'
    in_shape_test = (5, 3)
    mesh_out_name_test = 'fluid_mesh'
    out_shape_test = (8, 3)
    out_name_test = 'projection_map'
    weight_eps_test_val = 3.

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # create test model that wraps the NodalMap model
    test_model = Model()
    test_model.add(NodalMap(mesh_in_name=mesh_in_name_test, in_shape=in_shape_test, 
                            mesh_out_name=mesh_out_name_test, out_shape=out_shape_test, 
                            out_name=out_name_test), name='nodalmap_model')

    test_model.create_input('solid_mesh_input', val=rng_solid_mesh)
    test_model.create_input('fluid_mesh_input', val=rng_fluid_mesh)
    test_model.create_input('weight_eps', val=weight_eps_test_val)
    test_model.connect('solid_mesh_input', 'nodalmap_model.{}'.format(mesh_in_name_test))
    test_model.connect('fluid_mesh_input', 'nodalmap_model.{}'.format(mesh_out_name_test))
    test_model.connect('weight_eps', 'nodalmap_model.weight_eps')
    nodal_map_test_sim = Simulator(test_model)

    nodal_map_test_sim.run()

    print("Projection map:")
    print(nodal_map_test_sim[out_name_test])
    print("Weight function parameter: {}".format(nodal_map_test_sim['weight_eps']))
