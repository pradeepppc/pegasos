import numpy as np
import random


class linear_svm_pegasos():
    """ class for implementing linear svm pegasos"""

    def __init__(self):
        self.lamb = 0.1

    def fit(self, X, y, number_of_iter, change_classes):
        y = list(y)
        # change the classes to either +1 or -1
        if change_classes:
            y = self.change_classes(y)
        m = X.shape[0] # number of samples
        n = X.shape[1] # number of features
        self.w = np.zeros(n) # w is the decision boundary
        for i in range(number_of_iter):
            random_num = random.randint(0, m-1)
            sample_x = X[random_num]
            sample_y = y[random_num]
            eta = 1/(self.lamb*(i+1))
            check = self.w.dot(sample_x)
            if sample_y*check < 1:
                self.w = (1 - eta*self.lamb)*self.w + (eta*sample_y)*sample_x
            else:
                self.w = (1 - eta*self.lamb)*self.w

    def change_classes(self, y):
        new_list = []
        for i in y:
            if i == 0:
                new_list.append(-1)
            else:
                new_list.append(1)
        return new_list

    def check_single_point(self, x_new):
        return np.dot(x_new, self.w)

    def check_test_points(self, X):
        label_list = []
        for row in X:
            val = np.dot(row, self.w)
            if val >= 0:
                label_list.append(1)
            else:
                label_list.append(0)
        label_array = np.array(label_list)
        return label_array