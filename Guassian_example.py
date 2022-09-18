
import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from itertools import product
a = [np.array([0, 0.25]), np.array([0.3, 0.46]), np.array([0.36, 0.95]), np.array([0.6, 0.57])]
# a = [np.array([0, 0.25]), np.array([0, 0.25])]
estimator = KMeans(n_clusters=1)
data = a
estimator.fit(data)
centroids = estimator.cluster_centers_
# mean = centroids[0]
mean = np.array([0.5, 0.5])
from matplotlib import cm
cov = np.array([[1/28**2, 0], [0, 1/28**2]])
# cov = np.cov(data, rowvar=False)
# print(f"centroids is {centroids}")
print(f"cov is {cov}")


def maxminnorm(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t


def Gaussian_Distribution(cov, mean, N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = mean
    # mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    print(f"mean is {mean}")
    # cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma
    cov = cov
    # print(f"mean is {mean}", "\n", f"cov is {cov}")
    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


M = 1000
data, Gaussian = Gaussian_Distribution(cov, mean, N=2, M=M, sigma=0.1)
# 生成二维网格平面
X, Y = np.meshgrid(np.linspace(0,1,M), np.linspace(0,1,M))
# 二维坐标数据
# X = 0.665155
# Y = -1.36803

d = np.dstack([X,Y])
# d = np.array([X, Y])
# 计算二维联合高斯概率
t = np.zeros(M)
# Z = Gaussian.pdf(d).reshape(M, M) / Gaussian.pdf(mean)
Z = Gaussian.pdf(d).reshape(M, M)
t = t + Z
print(Gaussian.pdf(mean))
# Z = maxminnorm(Z)
# Z = Gaussian.pdf(d)
# print(Z)

# '''二元高斯概率分布图'''
plt.figure()
plt.axis('off')
plt.xlabel("X")
plt.ylabel("Y")
x, y = data.T
# plt.plot(x, y, 'ko', alpha=0.3)
levels = [0.1*i for i in range(1250)]
# print(levels)
cset1 = plt.contourf(X, Y, t, levels,cmap=cm.jet)
#
plt.colorbar()
plt.subplots_adjust(top=1, bottom=0.02, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('minimum_gaussian_distribution.eps', dpi=300)
plt.show()
a = np.zeros((5,5))

print(a)
