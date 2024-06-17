import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm


if __name__ == "__main__":

    X, y = datasets.load_iris(return_X_y=True)
    print(X.shape, y.shape)