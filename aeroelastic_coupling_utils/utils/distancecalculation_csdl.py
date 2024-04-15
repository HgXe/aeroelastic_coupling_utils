# from csdl import Model
# from python_csdl_backend import Simulator
# import csdl

import csdl_alpha as csdl

def distance_calculation(mesh_input: csdl.Variable, mesh_output: csdl.Variable):

    mesh_in_shape = mesh_input.shape
    mesh_out_shape = mesh_output.shape

    # Compute distances between in_mesh and out_mesh
    # first we expand both meshes to fit the same shape
    in_mesh_exp = csdl.expand(mesh_input, tuple([mesh_in_shape[0], mesh_out_shape[0], mesh_in_shape[1]]), 'ik->ijk')
    out_mesh_exp = csdl.expand(mesh_output, tuple([mesh_in_shape[0], mesh_out_shape[0], mesh_out_shape[1]]), 'jk->ijk')
    # then we subtract their coordinates
    mesh_dist_exp = in_mesh_exp - out_mesh_exp
    # lastly we compute the 2-norm along the last axis of mesh_dist_exp
    distance_array = csdl.norm(mesh_dist_exp, axes=tuple([2]), ord=2)

    return distance_array

if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    in_shape_test = (5, 3)
    out_shape_test = (8, 3)

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # create test model
    recorder = csdl.Recorder()
    recorder.start()

    # construct mesh csdl variables 
    solid_mesh_csdl = csdl.Variable(value=rng_solid_mesh)
    fluid_mesh_csdl = csdl.Variable(value=rng_fluid_mesh)

    distance_array = distance_calculation(solid_mesh_csdl, fluid_mesh_csdl)
    recorder.stop()

    recorder.execute()

    print("Distance array:")
    print(distance_array.value)
