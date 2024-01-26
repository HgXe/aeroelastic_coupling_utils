from csdl import Model
from python_csdl_backend import Simulator
import csdl

class DistanceCalculation(Model):
    def initialize(self):
        self.parameters.declare('mesh_in_name', types=str, default="mesh_in")
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('mesh_out_name', types=str, default="mesh_out")
        self.parameters.declare('out_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
    
    def define(self):
        mesh_in_name = self.parameters['mesh_in_name']
        in_shape = self.parameters['in_shape']
        mesh_out_name = self.parameters['mesh_out_name']
        out_shape = self.parameters['out_shape']
        out_name = self.parameters['out_name']

        in_mesh = self.declare_variable(mesh_in_name, shape=in_shape)
        out_mesh = self.declare_variable(mesh_out_name, shape=out_shape)

        # Compute distances between in_mesh and out_mesh
        # first we expand both meshes to fit the same shape
        in_mesh_exp = csdl.expand(in_mesh, tuple([in_shape[0], out_shape[0], in_shape[1]]), 'ik->ijk')
        out_mesh_exp = csdl.expand(out_mesh, tuple([in_shape[0], out_shape[0], out_shape[1]]), 'jk->ijk')
        # then we subtract their coordinates
        mesh_dist_exp = in_mesh_exp - out_mesh_exp
        # lastly we compute the 2-norm along the last axis of mesh_dist_exp
        distance_array = csdl.pnorm(mesh_dist_exp, 2, axis=2)

        self.register_output(out_name, distance_array)

if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    mesh_in_name_test = 'solid_mesh'
    in_shape_test = (5, 3)
    mesh_out_name_test = 'fluid_mesh'
    out_shape_test = (8, 3)
    out_name_test = 'distance_array'

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # create test model that wraps the NodalMap model
    test_model = Model()
    test_model.add(DistanceCalculation(mesh_in_name=mesh_in_name_test, in_shape=in_shape_test, 
                            mesh_out_name=mesh_out_name_test, out_shape=out_shape_test, 
                            out_name=out_name_test), name='distancecalculation_model')

    test_model.create_input('solid_mesh_input', val=rng_solid_mesh)
    test_model.create_input('fluid_mesh_input', val=rng_fluid_mesh)
    test_model.connect('solid_mesh_input', 'distancecalculation_model.{}'.format(mesh_in_name_test))
    test_model.connect('fluid_mesh_input', 'distancecalculation_model.{}'.format(mesh_out_name_test))
    distance_calc_test_sim = Simulator(test_model)

    distance_calc_test_sim.run()

    print("Projection map:")
    print(distance_calc_test_sim[out_name_test])
