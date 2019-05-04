import numpy as np
import random

class Kernalpegasos():
    """ this is implementation of the kernal peggasos """

    def __init__(self, lamb):
        self.lamb = lamb

    def fit(self, X, y, number_of_iter, change_classes, ker):
        y = list(y)
        if change_classes:
            y = self.change_classes(y)
        m = X.shape[0] # number of samples
        n = X.shape[1] # number of features
        self.alpha = np.zeros(m)
        for i in range(number_of_iter):
            random_num = random.randint(0, m - 1)
            sample_x = X[random_num]
            sample_y = y[random_num]
            eta = 1 / (self.lamb * (i+1))
            if sample_y * eta * self.caluclate_sum(y, sample_x, X, ker) < 1:
                self.alpha[random_num] += 1

    def caluclate_sum(self, y, sample_x, X, kernal):
        sum = 0
        for i in range(len(X)):
            x = X[i]
            val = self.find_kernal(x, sample_x, kernal)
            sum += y[i] * val * self.alpha[i]
        return sum

    def find_kernal(self, x, sample_x, kernal):
        if kernal == 'gauss':
            sig = 5
            dx = sample_x - x
            dx_square = np.dot(dx, dx)
            val = np.exp(-dx_square/(2*sig*sig))
            return val
        elif kernal == 'rbf':
            sig = 1
            dx = sample_x - x
            dx_square = np.dot(dx, dx)
            val = np.exp(-dx_square/sig)
            return val
        elif kernal == 'pkh':
            d = 2
            dx_square = np.dot(x, sample_x)
            val = (dx_square)**d
            return val
        elif kernal == 'pknh':
            d = 2
            dx_square = np.dot(x, sample_x)
            val = (dx_square + 1) ** d
            return val

    def change_classes(self, y):
        new_list = []
        for i in y:
            if i == 0:
                new_list.append(-1)
            else:
                new_list.append(1)
        return new_list

    def check_test_points(self, train_x, train_y, test_x, ker, change_classes):
        if change_classes:
            train_y = self.change_classes(train_y)
        test_y = []
        for i in range(len(test_x)):
            x = test_x[i]
            if self.caluclate_sum(train_y, x, train_x, ker) < 0:
                test_y.append(0)
            else:
                test_y.append(1)
        return np.array(test_y)
