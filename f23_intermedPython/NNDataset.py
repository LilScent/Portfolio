"""
Neural Networks: Datasets

Author: Drew Xavier
Date: 9/27/2023

Description: Methods to manage training and testing datasets.
"""
from enum import Enum
import numpy as np


@staticmethod
def percentage_limiter(percentage):
    """Static Method to limit the percentage allowed so that it falls between 0 and 1"""
    if percentage < 0:
        return 0
    elif percentage > 1:
        return 1
    else:
        return percentage


class NNData:
    class Order(Enum):
        SHUFFLE = 1
        STATIC = 2

    class Set(Enum):
        TRAIN = 1
        TEST = 2

    def __init__(self, features=None, labels=None, train_factor=.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._labels = None
        self._features = None
        self._train_factor = percentage_limiter(train_factor)
        self.load_data(features, labels)

    def load_data(self, features=None, labels=None):
        if features is None:
            self._features = None
            self._labels = None
        if labels is None:
            self._features = None
            self._labels = None
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError

        try:
            self._features = np.array(features, dtype=np.float64)
        except:
            self._features = None
            self._labels = None
            raise ValueError
        try:
            self._labels = np.array(labels, dtype=np.float64)
        except:
            self._features = None
            self._labels = None
            raise ValueError


def load_XOR(features=None, labels=None):
    features = [[1, 0], [1, 1], [0, 1], [0, 0]]
    labels = [[1], [0], [1], [0]]
    NNData(features, labels, train_factor=1)
    if features == [0, 1] or [1, 0]:
        labels = 1
    else:
        labels = 0


class DataMismatchError(Exception):
    pass


def unit_test():
    test_features_valid = [[0, 0], [1, 1]]
    test_labels_valid = [[0], [1]]
    features_array = np.array(test_features_valid)
    labels_array = np.array(test_labels_valid)
    empty_array = np.array([])
    test_labels_short = [[0]]
    test_labels_bad_data = [["a"], ["b"]]
    # Test constructor defaults
    print("Calling constructor with default arguments:", end="")
    my_data = NNData()
    if np.array_equal(my_data._features, empty_array) \
            and np.array_equal(my_data._labels, empty_array):
        print("PASS")
    else:
        print("FAIL")
    print("Calling constructor with valid arguments:", end="")
    my_data = NNData(test_features_valid, test_labels_valid)
    if np.array_equal(my_data._features, features_array) \
            and np.array_equal(my_data._labels, labels_array):
        print("PASS")
    else:
        print("FAIL")
    # Test mismatched constructor arguments
    print("Calling constructor with mismatched arguments:", end="")
    try:
        my_data = NNData(test_features_valid, test_labels_short)
        print("FAIL")
    except DataMismatchError:
        print("PASS")
    except:
        print("FAIL")
    # Test load_data with valid arguments
    print("Calling load_data() with valid arguments:", end="")
    my_data = NNData()
    my_data.load_data(test_features_valid, test_labels_valid)
    if np.array_equal(my_data._features, features_array) \
            and np.array_equal(my_data._labels, labels_array):
        print("PASS")
    else:
        print("FAIL")
    # Test load_data with mismatched arguments (start with a populated
    # dataset
    print("Calling load_data() with mismatched arguments:", end="")
    my_data = NNData(test_features_valid, test_labels_valid)
    try:
        my_data.load_data(test_features_valid, test_labels_short)
        print("FAIL")  # Execution should not reach this point
    except DataMismatchError:
        if my_data._features is None and my_data._labels is None:
            print("PASS")
        else:
            print("FAIL")
    # Test load_data with bad arguments (start with a populated
    # dataset
    print("Calling load_data() with bad arguments:", end="")
    my_data = NNData(test_features_valid, test_labels_valid)
    try:
        my_data.load_data(test_features_valid, test_labels_bad_data)
        print("FAIL")  # Execution should not reach this point
    except ValueError:
        if my_data._features is None and my_data._labels is None:
            print("PASS")
        else:
            print("FAIL")
    # Test percent limiter
    print("Testing train factor within bounds:", end="")
    my_data = NNData(test_features_valid, test_labels_valid, .5)
    if my_data._train_factor == .5:
        print("PASS")
    else:
        print("FAIL")
    print("Testing train factor too low:", end="")
    my_data = NNData(test_features_valid, test_labels_valid, -.1)
    if my_data._train_factor == 0:
        print("PASS")
    else:
        print("FAIL")
    print("Testing train factor within bounds:", end="")
    my_data = NNData(test_features_valid, test_labels_valid, 1.5)
    if my_data._train_factor == 1:
        print("PASS")
    else:
        print("FAIL")


unit_test()
load_XOR()

"""
/usr/bin/python3.10 /home/lilscent/PycharmProjects/f23_intermedPython/NNDataset.py 
Calling constructor with default arguments:PASS
Calling constructor with valid arguments:PASS
Calling constructor with mismatched arguments:PASS
Calling load_data() with valid arguments:PASS
Calling load_data() with mismatched arguments:PASS
Calling load_data() with bad arguments:PASS
Testing train factor within bounds:PASS
Testing train factor too low:PASS
Testing train factor within bounds:PASS

Process finished with exit code 0
"""