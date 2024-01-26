from csdl import Model
from python_csdl_backend import Simulator
import csdl

class WeightNormalization(Model):
    def initialize(self):
        self.parameters.declare('weight_array_in_name', types=str)
        self.parameters.declare('weight_array_in_shape', types=tuple)
        self.parameters.declare('column_scaling_vec_name', types=str)
        self.parameters.declare('out_name', types=str)
        self.parameters.declare('out_shape', types=tuple)

    def define(self):
        weight_array_in_name = self.parameters['weight_array_in_name']
        weight_array_in_shape = self.parameters['weight_array_in_shape']
        column_scaling_vec_name = self.parameters['column_scaling_vec_name']
        out_name = self.parameters['out_name']
        out_shape = self.parameters['out_shape']

        weight_array = self.declare_variable(weight_array_in_name, shape=weight_array_in_shape)
        column_scaling_vector = self.declare_variable(column_scaling_vec_name, shape=(weight_array_in_shape[0],))

        dist_weights_scaled = csdl.einsum(weight_array, column_scaling_vector, subscripts='ij,i->ji')
        # and compute the columnwise sum, before expanding the result
        dist_weights_scaled_colsums = csdl.sum(dist_weights_scaled, axes=(1,))
        dist_weights_scaled_colsums_exp = csdl.expand(dist_weights_scaled_colsums, tuple([out_shape[0], weight_array_in_shape[0]]), 'i->ij')
        # lastly we divide the scaled distance weights by the column sums
        dist_weights = dist_weights_scaled / dist_weights_scaled_colsums_exp

        self.register_output(out_name, dist_weights)