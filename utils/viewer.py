import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable


def display_coverage(data, value_range, bins=10, name=''):
    """

    :param data:
    :param value_range (tuple): 要么是（min,max）表示所有维度的上下界，要么是（[mins],[maxs]）的形式表示每个维度的上下界
    :param bins: 直方图bin个数
    :param name:
    :return:
    """
    if isinstance(data, list):
        data = np.array(data)

    if not isinstance(value_range[0], Iterable):
        value_range = ([value_range[0]]*data.shape[-1], [value_range[1]]*data.shape[-1])

    for i, values in enumerate(np.split(data, data.shape[-1], axis=-1)):
        plt.figure()
        plt.suptitle(name + f'_{i + 1}')
        plt.subplot(121)
        plt.hist(values, bins=bins, range=(value_range[0][i], value_range[1][i]))

        plt.subplot(122)
        uniques = np.unique(values)
        plt.scatter(np.arange(len(uniques)), sorted(uniques), vmin=value_range[0], vmax=value_range[1], marker='X',
                    linewidths=1)
        plt.show()
