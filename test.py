import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets.samples_generator import make_blobs, make_moons
from pagasos import linear_svm_pegasos
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import warnings
from kernalpagasos import Kernalpegasos

# a general dataset for testing binary classification with linear svm
def run_svm_non_kernal_test():
    bankdata = pd.read_csv("bill_authentication.csv")
    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    X = X.values
    y = y.values
    number_of_samples = X_train.shape[0]
    new_list = []
    for i in range(number_of_samples):
        new_list.append([1])
    new_array = np.array(new_list)
    new_linear_X = np.append(X_train, new_array, axis=1)
    number_of_test_samples = X_test.shape[0]
    new_list = []
    for i in range(number_of_test_samples):
        new_list.append([1])
    new_test_array = np.array(new_list)
    new_test_X = np.append(X_test, new_test_array, axis=1)
    pegasos = linear_svm_pegasos()
    kernel_pegasos = Kernalpegasos(0.1)
    kernel_pegasos.fit(X_train, y_train, 10000, True, 'rbf')
    print(kernel_pegasos.alpha)
    pegasos.fit(new_linear_X, y_train, 10000, True)
    y_pegasos_pred = pegasos.check_test_points(new_test_X)
    y_kernal_pred = kernel_pegasos.check_test_points(X_train, y_train, X_test, 'rbf', True)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    print('metrices evaluated for pegasos') # this is implemented linear pegasos
    print(confusion_matrix(y_test, y_pegasos_pred))
    print(classification_report(y_test, y_pegasos_pred))
    print('metrices evaluated for linear svm') # python svm class one
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('metrices caluclated for the kernal svm')
    print(classification_report(y_test, y_kernal_pred))

#run_svm_non_kernal_test()

# creating a test 2d daataset for visualization purposes

# linear_X, linear_y = make_blobs(n_samples=100, centers=2, n_features=2)
# number_of_samples = linear_X.shape[0]
# new_list = []
# for i in range(number_of_samples):
#     new_list.append([1])
# new_array = np.array(new_list)
# new_linear_X = np.append(linear_X, new_array, axis=1)

def svm_non_kernal_visualization(new_linear_X, linear_y, n):
    # print(new_linear_X)
    pegasos = linear_svm_pegasos()
    pegasos.fit(new_linear_X, linear_y, n, True)
    plot_x = np.linspace(-12, 12, 10000)
    plot_y = -(pegasos.w[0]/pegasos.w[1]) * plot_x - pegasos.w[2]/pegasos.w[1]
    df = pd.DataFrame(dict(x=linear_X[:, 0], y=linear_X[:, 1], label=linear_y))
    colors = {0: 'red', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    plt.plot(plot_x, plot_y, '-g')
    plt.show()


# svm_non_kernal_visualization(new_linear_X, linear_y, 1)
# svm_non_kernal_visualization(new_linear_X, linear_y, 10)
# svm_non_kernal_visualization(new_linear_X, linear_y, 20)
# svm_non_kernal_visualization(new_linear_X, linear_y, 30)
# svm_non_kernal_visualization(new_linear_X, linear_y, 40)
# svm_non_kernal_visualization(new_linear_X, linear_y, 50)
# svm_non_kernal_visualization(new_linear_X, linear_y, 60)
# svm_non_kernal_visualization(new_linear_X, linear_y, 70)
# svm_non_kernal_visualization(new_linear_X, linear_y, 80)
# svm_non_kernal_visualization(new_linear_X, linear_y, 90)
# svm_non_kernal_visualization(new_linear_X, linear_y, 100)






def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


def svm_kernal():
    # creating dataset for checking kernal svm for binary classification
    X, y = make_moons(n_samples=100, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    kernal_pegasos = Kernalpegasos(0.5)
    kernal_pegasos.fit(X_train, y_train, 100000, True, 'rbf')
    classifier = SVC(kernel='rbf', random_state=0, gamma=1, C=1)
    classifier.fit(X_train, y_train)
    # predicting using inbuilt kernal svm
    y_pred = classifier.predict(X_test)
    y_kernal_pred = kernal_pegasos.check_test_points(X_train, y_train, X_test, 'rbf')
    # print(y_pred)
    # print(y_kernal_pred)
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # print(classification_report(y_test, y_pred))
    # print(classification_report(y_test, y_kernal_pred))
    # plt.scatter(X[y == 1, 0],
    #             X[y == 1, 1],
    #             c='b', marker='x',
    #             label='1')
    # plt.scatter(X[y == 0, 0],
    #             X[y == 0, 1],
    #             c='r',
    #             marker='s',
    #             label='0')

    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    # plt.legend(loc='best')
    # plt.tight_layout()

    plot_decision_regions(X, y, classifier=classifier)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
