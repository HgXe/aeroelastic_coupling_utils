import csdl_alpha as csdl

class WeightFunctions:
    def __init__(self, weight_func_name: str='Gaussian', weight_eps: float=1.):
        """_summary_

        Args:
            weight_func_name (str, optional): _description_. Defaults to Gaussian.
            weight_eps (float, optional): _description_. Defaults to 1..
        """
        self.weight_func_name = weight_func_name
        self.weight_eps = weight_eps

    def evaluate(self, dist_array: csdl.Variable):
        """_summary_

        Args:
            dist_array (csdl.Variable): _description_

        Returns:
            _type_: _description_
        """
        if self.weight_func_name == 'Gaussian':
            weights_array = compute_gaussian(dist_array, self.weight_eps)
        elif self.weight_func_name == 'ThinPlateSpline':
            weights_array = compute_thinplatespline(dist_array)

        return weights_array

def compute_gaussian(dist_array: csdl.Variable, weight_eps: float=1.):
    weight_eps_times_dist_array = weight_eps * dist_array
    r_vec = csdl.square(weight_eps_times_dist_array)
    weights_array = csdl.exp(-r_vec)
    return weights_array

def compute_thinplatespline(dist_array: csdl.Variable):
    dist_log = csdl.log(dist_array, base=10)
    dist_squared = csdl.square(dist_array)
    weights_array = dist_squared*dist_log
    return weights_array

if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)
    # Test basic functionality of CSDL model
    input_dist_arr_shape = (3, 2)
    rng_dist_arr = np.random.random(input_dist_arr_shape)

    # Construct Gaussian test model
    recorder_gaussian = csdl.Recorder()
    recorder_gaussian.start()
    test_model_Gaussian = WeightFunctions(weight_func_name='Gaussian', weight_eps=0.1)
    dist_array_gaussian = csdl.Variable(value=rng_dist_arr)
    # run Gaussian function test model
    f_gaussian = test_model_Gaussian.evaluate(dist_array_gaussian)
    recorder_gaussian.stop()

    # Construct ThinPlateSpline test model
    recorder_thinplate = csdl.Recorder()
    recorder_thinplate.start()
    test_model_thinplate = WeightFunctions(weight_func_name='ThinPlateSpline', weight_eps=0.1)
    dist_array_thinplate = csdl.Variable(value=rng_dist_arr)
    # run Gaussian function test model
    f_thinplate = test_model_thinplate.evaluate(dist_array_thinplate)
    recorder_thinplate.stop()

    # execute both test models
    recorder_gaussian.execute()
    recorder_thinplate.execute()

    print(f_gaussian.value)
    print(f_thinplate.value)
    # print(Weight_test_Gaussian_sim['weightfunction_model.weight_eps'])
    # print(Weight_test_ThinPlateSpline_sim['weight_eps'])