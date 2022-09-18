import warnings
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from classes.Gaussian2D import Gaussian2D

def f(x):
    F_tmp1 = 1.5 - 1.0 * 1.6 * np.exp((-0.05 * ((x[:,0]*5 + 3)** 2 + (x[:,1]*5 + 2)** 2)))
    F_tmp2 = 1 + 0.5 * 1.6 * np.exp(-0.05 * (2 * (x[:,0]*5 - 3)** 2 + (x[:,1]*5 + 3)** 2))
    F_tmp3 = -2 + 1.6 * np.exp(-0.08 * (3 * (x[:,0]*5 + 3)** 2 + (x[:,1]*5 - 3)** 2))
    F_tmp4 = 0.5 - np.exp(-0.05 * ((x[:,0]*5 + 3)**2 + 4 * (x[:,1]*5 - 3)** 2))
    return F_tmp1+F_tmp2+F_tmp3+F_tmp4

class GaussianProcessForIPP():
    def __init__(self):#, node_coords):
        self.kernel = Matern(length_scale=0.45)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, optimizer=None, n_restarts_optimizer=0)
        self.observed_points = []
        self.observed_value = []
#        self.node_coords = node_coords

    def remove(self, sampled_coords, sampled_values):
        for i in range(len(sampled_values)):
            self.observed_points.remove(sampled_coords[i])
            self.observed_value.remove(sampled_values[i])


    def get_ground_truth(self):
        x1 = np.linspace(-1, 1)
        x2 = np.linspace(-1, 1)
        x1x2 = np.array(list(product(x1, x2)))
        y_true = f(x1x2)
        return y_true

    def add_observed_point(self, point_pos, value):
        self.observed_points.append(point_pos)
        self.observed_value.append(value)

    def update_gp(self):
        if self.observed_points:
            X = np.array(self.observed_points, dtype=float).reshape(-1,2)
            y = np.array(self.observed_value, dtype=float).reshape(-1,1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)

#    def update_node(self):
#        y_pred, std = self.gp.predict(self.node_coords, return_std=True)
#        return y_pred, std

    def flexi_updates(self, node_coord):
        y_pred, std = self.gp.predict(node_coord, return_std=True)
        return y_pred, std

    def evaluate_RMSE(self, y_true):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        RMSE = np.sqrt(mean_squared_error(y_pred, y_true))
        return RMSE

    def evaluate_F1score(self, y_true):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)
        x1x2 = np.array(list(product(x1, x2)))
        score = self.gp.score(x1x2,y_true)
        return score

    def evaluate_cov_trace(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1, 30)
            x2 = np.linspace(0, 1, 30)
            X = np.array(list(product(x1, x2)))
        _, std = self.gp.predict(X, return_std=True)
        trace = np.sum(std**2) 
        return trace

    def evaluate_mutual_info(self, X=None):
        if X is None:
            x1 = np.linspace(0, 1)
            x2 = np.linspace(0, 1)
            X = np.array(list(product(x1, x2)))
        n_sample = X.shape[0]
        _, cov = self.gp.predict(X, return_cov=True)

        mi = (1 / 2) * np.log(np.linalg.det(0.01*cov.reshape(n_sample, n_sample) + np.identity(n_sample)))
        return mi

    def get_high_info_area(self, t=0.4, beta=1):
        x1 = np.linspace(0, 1, 30)
        x2 = np.linspace(0, 1, 30)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)
        high_measurement_area = []
        for i in range(900):
            if y_pred[i] + beta * std[i] >= t:
                high_measurement_area.append(x1x2[i])
        high_measurement_area = np.array(high_measurement_area)
        return high_measurement_area




    def plot(self, y_true):
        x1 = np.linspace(0, 1)
        x2 = np.linspace(0, 1)

        x1x2 = np.array(list(product(x1, x2)))
        y_pred, std = self.gp.predict(x1x2, return_std=True)

        X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)
        y_pred = np.reshape(y_pred, (50, 50))
        std = std.reshape(50,50)

        X = np.array(self.observed_points)

        fig = plt.figure(figsize=(15,5))
        #if self.observed_points:
        #    plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=10, c='r')
        plt.subplot(1, 3, 2) # ground truth
        plt.pcolormesh(X0p, X1p, y_true.reshape(50, 50), shading='auto', vmin=0, vmax=1)
        plt.subplot(1, 3, 3) # stddev
        plt.pcolormesh(X0p, X1p, std, shading='auto', vmin=0, vmax=1)
        plt.subplot(1, 3, 1) # mean
        plt.pcolormesh(X0p, X1p, y_pred, shading='auto', vmin=0, vmax=1)
        # if self.observed_points:
        #     plt.scatter(X[:, 0].reshape(1, -1), X[:, 1].reshape(1, -1), s=2, c='r')
        # plt.show()

if __name__ == '__main__':
    example = Gaussian2D()
    x1 = np.linspace(0, 1)
    x2 = np.linspace(0, 1)
    x1x2 = np.array(list(product(x1, x2)))
    y_true = example.distribution_function(X=x1x2)
    # print(y_true.shape)
    node_coords = np.random.uniform(0,1,(100,2))
    gp_ipp = GaussianProcessForIPP(node_coords)
    gp_ipp.plot(y_true.reshape(50,50))
    for i in range(node_coords.shape[0]):
        y_observe = example.distribution_function(node_coords[i].reshape(-1,2))
        # print(node_coords[i], y_observe)
        gp_ipp.add_observed_point(node_coords[i], y_observe)
        gp_ipp.update_gp()
        y_pre, std = gp_ipp.update_node()
        print(gp_ipp.evaluate_cov_trace())
    gp_ipp.plot(y_true)
    print(gp_ipp.evaluate_F1score(y_true))
    print(gp_ipp.gp.kernel_)