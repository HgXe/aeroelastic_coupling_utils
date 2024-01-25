from csdl import Model
from python_csdl_backend import Simulator
import csdl
import numpy as np

from aeroelastic_coupling_utils.utils.weightfunctions_csdl import WeightFunctions

class NodalMap(Model):
    def initialize(self):
        # define constant model parameters
        self.parameters.declare('mesh_in_name', types=str, default="mesh_in")
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('mesh_out_name', types=str, default="mesh_out")
        self.parameters.declare('out_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
        # self.parameters.declare('Weight_eps', types=float, default=0.01)
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

        weight_eps = self.create_input(Weight_eps_name, shape=(1,), val=0.1)
        in_mesh = self.declare_variable('solid_mesh', shape=in_shape, val=np.ones(in_shape))
        out_mesh = self.declare_variable(mesh_out_name, shape=out_shape, val=np.ones(out_shape))
        column_scaling_vector = self.declare_variable('column_scaling_vector', shape=(in_shape[0],), val=np.ones((in_shape[0],)))

        # compute distances between in_mesh and out_mesh
        # first we expand both meshes to fit the same shape
        in_mesh_exp = csdl.expand(in_mesh, tuple([in_shape[0], out_shape[0], in_shape[1]]), 'ik->ijk')
        out_mesh_exp = csdl.expand(out_mesh, tuple([in_shape[0], out_shape[0], out_shape[1]]), 'jk->ijk')
        # then we subtract their coordinates
        mesh_dist_exp = in_mesh_exp - out_mesh_exp
        # lastly we compute the 2-norm along the last axis of mesh_dist_exp
        distance_array = csdl.pnorm(mesh_dist_exp, 2, axis=2)

        # pass the (pointwise) mesh distances to the WeightFunctions class and re-declare the result
        # TODO: pass Weight_eps to WeightFunctions 
        # self.register_output('mesh_dist_arr', mesh_dist)
        # self.register_output('Weight_eps', Weight_eps)
        self.add(WeightFunctions(Weight_func_name=Weight_func_name, Weight_eps_name=Weight_eps_name, 
                                      in_name='distance_array', in_shape=tuple([in_shape[0], out_shape[0]]), 
                                      out_name='dist_weights_uncorrected'), name='Weightfunction_comp',
                                      promotes=[Weight_eps_name, 'distance_array', 'dist_weights_uncorrected'])
        # self.connect('Weightfunction_comp.mesh_dist_arr', 'mesh_dist')
        # self.connect('Weightfunction_comp.{}'.format(Weight_eps_name), Weight_eps_name)
        dist_weights_uncorrected = self.declare_variable('dist_weights_uncorrected', shape=(in_shape[0], out_shape[0]))

        # next we apply the column scaling vector
        dist_weights_scaled = csdl.einsum(dist_weights_uncorrected, column_scaling_vector, subscripts='ij,i->ji')
        # and compute the columnwise sum, before expanding the result
        dist_weights_scaled_colsums = csdl.sum(dist_weights_scaled, axes=(1,))
        dist_weights_scaled_colsums_exp = csdl.expand(dist_weights_scaled_colsums, tuple([out_shape[0], in_shape[0]]), 'i->ij')
        # lastly we divide the scaled distance weights by the column sums
        dist_weights = dist_weights_scaled / dist_weights_scaled_colsums_exp

        self.register_output(out_name, dist_weights) 


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    mesh_in_name_test = 'solid_mesh'
    in_shape_test = (5, 3)
    mesh_out_name_test = 'fluid_mesh'
    out_shape_test = (8, 3)
    out_name_test = 'projection_map'

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    nodal_map_test = NodalMap(mesh_in_name=mesh_in_name_test, in_shape=in_shape_test, mesh_out_name=mesh_out_name_test, out_shape=out_shape_test, out_name=out_name_test)
    nodal_map_test_sim = Simulator(nodal_map_test)

    # input solid and fluid meshes
    nodal_map_test_sim[mesh_in_name_test] = rng_solid_mesh
    nodal_map_test_sim[mesh_out_name_test] = rng_fluid_mesh

    nodal_map_test_sim.run()

    print(nodal_map_test_sim[out_name_test])
    # print(nodal_map_test_sim['Weight_eps_name'])
    print(nodal_map_test_sim['weight_eps'])
    print(nodal_map_test_sim['Weightfunction_comp.weight_eps'])