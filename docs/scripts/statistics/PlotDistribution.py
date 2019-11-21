#!/usr/bin/python
#coding:utf-8
# ***************************************************************
# 绘制正态分布曲线
# author:   pruce
# email:    1756983926@qq.com
# date:     20180919
# ***************************************************************

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def normDistribution():
    mu, sigma , num_bins = 0, 1, 50
    x = mu + sigma * np.random.randn(1000000)
    # 正态分布的数据
    n, bins, patches = plt.hist(x, num_bins, normed=True, facecolor = 'black', alpha = 0.5)
    # 拟合曲线
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Expectation')
    plt.ylabel('Probability')
    plt.title('$N(0,1)$')

    plt.subplots_adjust(left = 0.15)
    plt.show()

normDistribution()