"""
Neural Networks: Datasets

Author: Drew Xavier
Date: 9/27/2023

Description: Methods to manage training and testing datasets.
"""

from enum import Enum
import numpy as np
from collections import deque
import random


class NNData:
    class Order(Enum):
        SHUFFLE = 1
        STATIC = 2

    class Set(Enum):
        TRAIN = 1
        TEST = 2

    @staticmethod
    def percentage_limiter(percentage):
        """Static Method to limit the percentage allowed so that it falls between 0 and 1"""
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        else:
            return percentage

    def __init__(self, features=None, labels=None, train_factor=.9):
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._labels = None
        self._features = None
        self._train_factor = self.percentage_limiter(train_factor)

        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque(self._train_indices)
        self._test_pool = deque(self._test_indices)

        try:
            self.load_data(features, labels)
        except (ValueError, DataMismatchError):
            pass

    def data_clear(self):
        self._features = None
        self._labels = None
        self._test_indices = []
        self._train_indices = []
        self._train_pool = deque()
        self._test_pool = deque()

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
            self.data_clear()
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
        self.split_set()

    def split_set(self, new_train_factor=None):
        if new_train_factor is not None:
            self._train_factor = self.percentage_limiter(new_train_factor)

        try:
            dataset_size = len(self._labels)
            dataset_indices = list(range(dataset_size))
            train_set_size = int(dataset_size * self._train_factor)
            random.shuffle(dataset_indices)
        except:
            self.data_clear()
            raise DataMismatchError

        self._train_indices = dataset_indices[:train_set_size]
        self._test_indices = dataset_indices[train_set_size:]

        if len(self._train_indices) + len(self._test_indices) != dataset_size:
            raise DataMismatchError
        self.prime_data()

    def prime_data(self, target_set=None, order=None):
        if target_set is None:
            self._train_pool = deque(self._train_indices)
            self._test_pool = deque(self._test_indices)
        elif target_set is NNData.Set.TRAIN:
            self._train_pool = self._train_indices
        elif target_set is NNData.Set.TEST:
            self._test_pool = self._test_indices
        else:
            raise Exception("target_set must be TRAIN, TEST, or None")

        if order is NNData.Order.SHUFFLE:
            random.shuffle(self._train_pool)
            random.shuffle(self._test_pool)
        elif order is NNData.Order.STATIC or order is None:
            pass
        else:
            raise Exception("order must be STATIC, SHUFFLE, or None")

    def get_one_item(self, target_set=None):
        if target_set is NNData.Set.TRAIN or target_set is None:
            try:
                index = self._train_pool.popleft()
                return self._features[index], self._labels[index]
            except self._train_pool.popleft() is None:
                return None

        elif target_set is NNData.Set.TEST:
            try:
                index = self._test_pool.popleft()
                return self._features[index], self._labels[index]
            except self._test_pool.popleft() is None:
                return None

    def number_of_samples(self, target_set=None):
        if target_set is None:
            return len(self._train_pool) + len(self._test_pool)
        elif target_set is NNData.Set.TRAIN:
            return len(self._train_pool)
        elif target_set is NNData.Set.TEST:
            return len(self._test_pool)
        else:
            raise Exception("target_set must be TRAIN, TEST, or None")

    def pool_is_empty(self, target_set=None):
        if target_set is NNData.Set.TRAIN or target_set is None:
            if not self._train_pool:
                return True
            else:
                return False
        elif target_set is NNData.Set.TEST:
            if not self._test_pool:
                return True
            else:
                return False
        else:
            raise Exception("target_set must be TRAIN, TEST, or None")


def load_XOR(features=None, labels=None):
    features = [[1, 0], [1, 1], [0, 1], [0, 0]]
    labels = [[1], [0], [1], [0]]
    NNData(features, labels, train_factor=1)
    if features == [0, 1] or [1, 0]:
        labels = 1
    else:
        labels = 0


class DataMismatchError(Exception):
    """Label and example lists have different lengths"""


def unit_test_1():
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


def unit_test_2():
    errors = False
    try:
        # Create a valid small and large dataset to be used later
        x = [[i] for i in range(10)]
        y = x
        our_data_0 = NNData(x, y)
        x = [[i] for i in range(100)]
        y = x
        our_big_data = NNData(x, y, .5)

        # Try loading lists of different sizes
        y = [[1]]
        try:
            our_bad_data = NNData()
            our_bad_data.load_data(x, y)
            raise Exception
        except DataMismatchError:
            pass
        except:
            raise Exception

        # Create a dataset that can be used to make sure the
        # features and labels are not confused
        x = [[1.0], [2.0], [3.0], [4.0]]
        y = [[.1], [.2], [.3], [.4]]
        our_data_1 = NNData(x, y, .5)

    except:
        print("There are errors that likely come from __init__ or a "
              "method called by __init__")
        errors = True

    # Test split_set to make sure the correct number of examples are in
    # each set, and that the indices do not overlap.
    try:
        our_data_0.split_set(.3)
        print(f"Train Indices:{our_data_0._train_indices}")
        print(f"Test Indices:{our_data_0._test_indices}")

        assert len(our_data_0._train_indices) == 3
        assert len(our_data_0._test_indices) == 7
        assert (list(set(our_data_0._train_indices +
                         our_data_0._test_indices))) == list(range(10))
    except:
        print("There are errors that likely come from split_set")
        errors = True

    # Make sure prime_data sets up the deques correctly, whether
    # static or shuffle.
    try:
        assert our_data_0.number_of_samples(NNData.Set.TEST) == 7
        assert our_data_0.number_of_samples(NNData.Set.TRAIN) == 3
        assert our_data_0.number_of_samples() == 10
    except:
        print("Return value of number_of_samples does not match the "
              "expected value.")
    try:
        our_data_0.prime_data(order=NNData.Order.STATIC)

        assert len(our_data_0._train_pool) == 3
        assert len(our_data_0._test_pool) == 7
        assert our_data_0._train_indices == list(our_data_0._train_pool)
        assert our_data_0._test_indices == list(our_data_0._test_pool)
        our_big_data.prime_data(order=NNData.Order.SHUFFLE)
        assert our_big_data._train_indices != list(our_big_data._train_pool)
        assert our_big_data._test_indices != list(our_big_data._test_pool)
    except:
        print("There are errors that likely come from prime_data")
        errors = True

    # Make sure get_one_item is returning the correct values, and
    # that pool_is_empty functions correctly.
    try:
        our_data_1.prime_data(order=NNData.Order.STATIC)
        my_x_list = []
        my_y_list = []
        while not our_data_1.pool_is_empty():
            example = our_data_1.get_one_item()
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert len(my_x_list) == 2
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        while not our_data_1.pool_is_empty(our_data_1.Set.TEST):
            example = our_data_1.get_one_item(our_data_1.Set.TEST)
            my_x_list.append(list(example[0]))
            my_y_list.append(list(example[1]))
        assert my_x_list != my_y_list
        my_matched_x_list = [i[0] * 10 for i in my_y_list]
        assert my_matched_x_list == my_x_list
        assert set(i[0] for i in my_x_list) == set(i[0] for i in x)
        assert set(i[0] for i in my_y_list) == set(i[0] for i in y)
    except:
        print("There are errors that may come from prime_data, but could "
              "be from another method")
        errors = True

    # Summary
    if errors:
        print("You have one or more errors.  Please fix them before "
              "submitting")
    else:
        print("No errors were identified by the unit test.")
        print("You should still double check that your code meets spec.")
        print("You should also check that PyCharm does not identify any "
              "PEP-8 issues.")


if __name__ == "__main__":
    # unit_test_1()
    unit_test_2()
    # load_XOR()


"""
UNIT TEST 1

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

"""
UNIT TEST 2

/usr/bin/python3.10 /home/lilscent/PycharmProjects/f23_intermedPython/NNDataset.py 
Train Indices:[4, 9, 3]
Test Indices:[8, 7, 1, 6, 0, 2, 5]
No errors were identified by the unit test.
You should still double check that your code meets spec.
You should also check that PyCharm does not identify any PEP-8 issues.

Process finished with exit code 0

"""
