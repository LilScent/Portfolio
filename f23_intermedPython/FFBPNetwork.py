"""
Assignment 5: FFBPNetwork

Author: Drew Xavier
Date: 11-27-2023

Description: Combines all previous assignments including, the NNDataset, FFBPNeurode, and DoublyLinkedList files to make
    a working network.
"""

import copy
import random
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum

import numpy as np


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
            self._train_pool = deque(self._train_indices)
        elif target_set is NNData.Set.TEST:
            self._test_pool = deque(self._test_indices)
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


class LayerType(Enum):
    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2


class MultiLinkNode(ABC):
    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def __init__(self):
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0, MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [], MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self):
        # Include the ability to check the reference value and the reporting node's binary value
        return ("This object's ID is", id(self), "Its upstream neighbor's ID is", id(self.Side.UPSTREAM),
                "and its downstream neighbor's ID is", id(self.Side.DOWNSTREAM))

    @abstractmethod
    def _process_new_neighbor(self, node: list, side):
        pass

    def reset_neighbors(self, nodes: list, side: Side):
        self._neighbors[side] = copy.copy(nodes)
        # if side is MultiLinkNode.Side.UPSTREAM:
            # self._weights
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = 2 ** len(nodes) - 1
        self._reporting_nodes[side] = 0


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}
        super().__init__()

    def _process_new_neighbor(self, node, side):
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side):
        node_index = self._neighbors[side].index(node)
        bitvalue = 1 << node_index
        self._reporting_nodes[side] = self._reporting_nodes[side] | bitvalue

        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    def get_weight(self, node):
        return self._weights[node]

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate


class FFNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def _calculate_value(self):
        weighted_value = 0
        for neighbor in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            weighted_value += (neighbor.value * self._weights[neighbor])
        self._value = self._sigmoid(weighted_value)

    def fire_downstream(self):
        for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            node.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        try:
            if self._check_in(node, MultiLinkNode.Side.UPSTREAM) is True:
                self._calculate_value()
                self.fire_downstream()
        except:
            pass

    def set_input(self, input_value):
        self._value = input_value
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            neighbor.data_ready_upstream(self)


class BPNeurode(Neurode):
    def __init__(self, my_type):
        super().__init__(my_type)
        self._delta = 0

    @staticmethod
    def _sigmoid_derivative(value):
        return value*(1-value)

    def _calculate_delta(self, expected_value: int = None):
        if expected_value is None:
            if self._node_type is LayerType.INPUT:
                pass
            elif self._node_type is LayerType.HIDDEN:
                sum_of_weighted_deltas = 0
                for node in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
                    sum_of_weighted_deltas += node.get_weight(self) * node._delta
                self._delta = sum_of_weighted_deltas * self._sigmoid_derivative(self._value)

        else:
            self._delta = (expected_value - self._value) * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node):
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM) is True:
            self._calculate_delta()
            self._fire_upstream()

            self._update_weights()

    def set_expected(self, expected_value):
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node, adjustment):
        self._weights[node] += adjustment

    def _update_weights(self):
        for neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = neighbor.delta * self.learning_rate * self._value
            neighbor.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        for neighbor in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            neighbor.data_ready_downstream(self)

    @property
    def delta(self):
        return self._delta


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class Node:
    def __init__(self, data=None):
        self._data = data
        self._next = None
        self._prev = None


class DoublyLinkedList:
    def __init__(self):
        self._head = None
        self._tail = None
        self._curr = self._head

    def move_forward(self):
        self._curr = self._curr._next
        if self._curr is None:
            self.reset_to_tail()
            raise IndexError

    def move_back(self):
        self._curr = self._curr._prev
        if self._curr is None:
            self.reset_to_head()
            raise IndexError

    def reset_to_head(self):
        self._curr = self._head

    def reset_to_tail(self):
        self._curr = self._tail

    def add_to_head(self, data):
        new_node = Node(data)
        if self._head is None:
            self._head = new_node
            self._tail = new_node
        else:
            self._head._prev = new_node
            new_node._next = self._head
            self._head = new_node
        self.reset_to_head()

    def add_after_cur(self, data):
        new_node = Node(data)
        new_node._next = self._curr._next
        if self._curr._next is not None:
            self._curr._next._prev = new_node
        self._curr._next = new_node
        self._curr._next._prev = self._curr
        if self._curr._next._next is None:
            self._tail = new_node

    def remove_from_head(self):
        if self._head is None:
            raise self.EmptyListError
        self._head = self._head._next
        if self._head is None:
            self._tail = None
        else:
            self._head._prev = None

    def remove_after_cur(self):
        if self._curr is not self._tail:
            self._curr._next = self._curr._next._next
            if self._curr._next:
                self._curr._next._prev = self
            else:
                self._tail = self
        else:
            raise IndexError

    def get_current_data(self):
        if self._curr is None:
            raise self.EmptyListError
        else:
            return self._curr._data

    class EmptyListError(Exception):
        pass


class LayerList(DoublyLinkedList):
    def __init__(self, inputs: int, outputs: int, neurode_type: type(Neurode)):
        super().__init__()
        self._type = neurode_type

        input_nodes = [neurode_type(LayerType.INPUT) for a in range(inputs)]
        output_nodes = [neurode_type(LayerType.OUTPUT) for a in range(outputs)]

        DoublyLinkedList.add_to_head(self, input_nodes)
        DoublyLinkedList.add_after_cur(self, output_nodes)

        self.connect_curr_to_next()

    def connect_curr_to_next(self):
        for node in self._curr._data:
            node.reset_neighbors(self._curr._next._data, MultiLinkNode.Side.DOWNSTREAM)
        for node in self._curr._next._data:
            node.reset_neighbors(self._curr._data, MultiLinkNode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        if self is LayerType.OUTPUT:
            raise IndexError
        else:
            hidden_nodes = [self._type(LayerType.HIDDEN) for a in range(num_nodes)]
            DoublyLinkedList.add_after_cur(self, hidden_nodes)
            self.connect_curr_to_next()

    def remove_layer(self):
        if self._curr._next is self._tail:
            raise IndexError('Cannot remove tail list')
        else:
            self._curr._next = self._curr._next._next
            self._curr._next._prev = self._curr

    @property
    def input_nodes(self):
        return self._head._data

    @property
    def output_nodes(self):
        return self._tail._data


class FFBPNetwork(LayerList):
    class EmptySetException(Exception):
        "Target set is empty."

    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__(num_inputs, num_outputs, neurode_type=FFBPNeurode)
        self.inputs = num_inputs
        self.outputs = num_outputs

    def add_hidden_layer(self, num_nodes: int, position=0):
        self.reset_to_head()
        for x in range(position):
            self.move_forward()
        LayerList.add_layer(self, num_nodes)

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.SHUFFLE):
        if data_set is None:
            raise FFBPNetwork.EmptySetException
        for epoch in range(epochs):
            data_set.prime_data(NNData.Set.TRAIN, order)
            while data_set.pool_is_empty(NNData.Set.TRAIN) is False:
                data_tuple = data_set.get_one_item()
                set_features = data_tuple[0]
                input_data = zip(FFBPNetwork.input_nodes, set_features)
                for node, feature in input_data:
                    node.set_input(feature)
                self.reset_to_tail()
                output_val = self.get_current_data()
                FFBPNeurode.set_expected(self, data_tuple[1])
            if verbosity > 0:
                print(f"The RMSE for the last epoch was: {output_val-data_tuple[1]}")
        print("Training Complete")
        print(f"Final RMSE is {output_val-data_tuple[1]}")


def run_iris():
    network = FFBPNetwork(4, 3)
    network.add_hidden_layer(3)
    Iris_X = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
              [5, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5, 3.4, 1.5, 0.2],
              [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
              [4.8, 3, 1.4, 0.1], [4.3, 3, 1.1, 0.1], [5.8, 4, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
              [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
              [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1, 0.2], [5.1, 3.3, 1.7, 0.5],
              [4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 0.2], [5, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
              [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
              [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5, 3.2, 1.2, 0.2],
              [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
              [5, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5, 3.5, 1.6, 0.6],
              [5.1, 3.8, 1.9, 0.4], [4.8, 3, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
              [5.3, 3.7, 1.5, 0.2], [5, 3.3, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
              [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
              [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1], [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5, 2, 3.5, 1],
              [5.9, 3, 4.2, 1.5], [6, 2.2, 4, 1], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
              [5.6, 3, 4.5, 1.5], [5.8, 2.7, 4.1, 1], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
              [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
              [6.4, 2.9, 4.3, 1.3], [6.6, 3, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3, 5, 1.7], [6, 2.9, 4.5, 1.5],
              [5.7, 2.6, 3.5, 1], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1], [5.8, 2.7, 3.9, 1.2],
              [6, 2.7, 5.1, 1.6], [5.4, 3, 4.5, 1.5], [6, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5],
              [6.3, 2.3, 4.4, 1.3], [5.6, 3, 4.1, 1.3], [5.5, 2.5, 4, 1.3], [5.5, 2.6, 4.4, 1.2],
              [6.1, 3, 4.6, 1.4], [5.8, 2.6, 4, 1.2], [5, 2.3, 3.3, 1], [5.6, 2.7, 4.2, 1.3], [5.7, 3, 4.2, 1.2],
              [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3, 1.1], [5.7, 2.8, 4.1, 1.3],
              [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8],
              [6.5, 3, 5.8, 2.2], [7.6, 3, 6.6, 2.1], [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8],
              [6.7, 2.5, 5.8, 1.8], [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2], [6.4, 2.7, 5.3, 1.9],
              [6.8, 3, 5.5, 2.1], [5.7, 2.5, 5, 2], [5.8, 2.8, 5.1, 2.4], [6.4, 3.2, 5.3, 2.3], [6.5, 3, 5.5, 1.8],
              [7.7, 3.8, 6.7, 2.2], [7.7, 2.6, 6.9, 2.3], [6, 2.2, 5, 1.5], [6.9, 3.2, 5.7, 2.3],
              [5.6, 2.8, 4.9, 2], [7.7, 2.8, 6.7, 2], [6.3, 2.7, 4.9, 1.8], [6.7, 3.3, 5.7, 2.1],
              [7.2, 3.2, 6, 1.8], [6.2, 2.8, 4.8, 1.8], [6.1, 3, 4.9, 1.8], [6.4, 2.8, 5.6, 2.1],
              [7.2, 3, 5.8, 1.6], [7.4, 2.8, 6.1, 1.9], [7.9, 3.8, 6.4, 2], [6.4, 2.8, 5.6, 2.2],
              [6.3, 2.8, 5.1, 1.5], [6.1, 2.6, 5.6, 1.4], [7.7, 3, 6.1, 2.3], [6.3, 3.4, 5.6, 2.4],
              [6.4, 3.1, 5.5, 1.8], [6, 3, 4.8, 1.8], [6.9, 3.1, 5.4, 2.1], [6.7, 3.1, 5.6, 2.4],
              [6.9, 3.1, 5.1, 2.3], [5.8, 2.7, 5.1, 1.9], [6.8, 3.2, 5.9, 2.3], [6.7, 3.3, 5.7, 2.5],
              [6.7, 3, 5.2, 2.3], [6.3, 2.5, 5, 1.9], [6.5, 3, 5.2, 2], [6.2, 3.4, 5.4, 2.3], [5.9, 3, 5.1, 1.8]]
    Iris_Y = [[1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ], [1, 0, 0, ],
              [1, 0, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ], [0, 1, 0, ],
              [0, 1, 0, ], [0, 1, 0, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ],
              [0, 0, 1, ], [0, 0, 1, ], [0, 0, 1, ]]
    data = NNData(Iris_X, Iris_Y, .7)
    network.train(data, 10001, order=NNData.Order.RANDOM)
    network.test(data)


def run_sin():
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    sin_X = [[0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12],
             [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25],
             [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38],
             [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51],
             [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64],
             [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77],
             [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9],
             [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1], [1.01], [1.02], [1.03],
             [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16],
             [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29],
             [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42],
             [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55],
             [1.56], [1.57]]
    sin_Y = [[0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342],
             [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727],
             [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919],
             [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246],
             [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461],
             [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523],
             [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134],
             [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814],
             [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161],
             [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066],
             [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158],
             [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847],
             [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969],
             [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481],
             [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434],
             [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537],
             [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334],
             [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523],
             [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859],
             [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969],
             [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777],
             [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197],
             [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363],
             [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986],
             [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435],
             [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883],
             [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167],
             [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697],
             [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692],
             [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734],
             [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659],
             [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846],
             [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588],
             [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054],
             [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479],
             [0.999783764189357], [0.999941720229966], [0.999999682931835]]
    data = NNData(sin_X, sin_Y, .1)
    network.train(data, 10001, order=NNData.Order.SHUFFLE)
    # network.test(data)


def run_XOR():
        # Student should replace both lines of code below
        print("Student Code is missing")
        assert False


if __name__ == "__main__":
    run_sin()




