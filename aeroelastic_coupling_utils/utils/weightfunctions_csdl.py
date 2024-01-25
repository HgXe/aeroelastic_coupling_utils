from csdl import Model
from python_csdl_backend import Simulator
import csdl

class WeightFunctions(Model):
    def initialize(self):
        self.parameters.declare('Weight_func_name', types=str, default='Gaussian')
        # self.parameters.declare('Weight_eps', types=float, default=1.)
        self.parameters.declare('Weight_eps_name', types=str, default='weight_eps')
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('in_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
    
    def define(self):
        weight_func_name = self.parameters['Weight_func_name']
        weight_eps_name = self.parameters['Weight_eps_name']
        # weight_eps = self.parameters['Weight_eps']
        in_name = self.parameters['in_name']
        in_shape = self.parameters['in_shape']
        out_name = self.parameters['out_name']

        weight_eps = self.declare_variable(weight_eps_name, shape=(1,))
        distance_array = self.declare_variable(in_name, shape=in_shape)

        # TODO: Add a csdl.expand operation to expand the weight_eps parameter into a vector (for elementwise multiplication)

        if weight_func_name == 'Gaussian':
            # we first create a csdl object of the right shape that contains weight_eps 
            # weight_eps_arr = self.create_input('weight_eps_arr', val=weight_eps*np.ones(in_shape))
            weight_eps_arr = csdl.expand(weight_eps, in_shape)
            r_vec = (weight_eps_arr*distance_array)**2
            weight_weight_vec = csdl.exp(-r_vec)
        elif weight_func_name == 'ThinPlateSpline':
            dist_log = csdl.log(distance_array)
            dist_squared = distance_array**2
            weight_weight_vec = dist_squared*dist_log

        self.register_output(out_name, weight_weight_vec)



if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)
    # Test basic functionality of CSDL model
    # weight_eps = 0.1
    # input_dist_arr = (np.array([[1, 2, 3], [4, 5, 6]]))
    input_dist_arr_shape = (3, 2)
    in_name = 'test_array'
    out_name = 'test_weights'
    rng_dist_arr = np.random.random(input_dist_arr_shape)

    # test Gaussian function
    Weight_test_Gaussian = WeightFunctions(Weight_func_name='Gaussian', in_name=in_name, in_shape=input_dist_arr_shape, out_name=out_name)
    Weight_test_Gaussian_sim = Simulator(Weight_test_Gaussian)
    Weight_test_Gaussian_sim[in_name] = rng_dist_arr  # set random array as test input
    Weight_test_Gaussian_sim.run()

    # test ThinPlateSpline function
    Weight_test_ThinPlateSpline = WeightFunctions(Weight_func_name='ThinPlateSpline', in_name=in_name, in_shape=input_dist_arr_shape, out_name=out_name)
    Weight_test_ThinPlateSpline_sim = Simulator(Weight_test_ThinPlateSpline)
    Weight_test_ThinPlateSpline_sim[in_name] = rng_dist_arr  # set random array as test input
    Weight_test_ThinPlateSpline_sim.run()

    print(Weight_test_Gaussian_sim[out_name])
    print(Weight_test_ThinPlateSpline_sim[out_name])
    print(Weight_test_Gaussian_sim['weight_eps'])
    # print(Weight_test_ThinPlateSpline_sim['weight_eps'])