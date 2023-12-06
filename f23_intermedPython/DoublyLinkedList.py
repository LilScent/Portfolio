"""
Assignment 4: Doubly Linked List

Author: Drew Xavier
Date: 10-16-2023

Description: Doubly Linked List class for Layer Lists of Neurodes
"""
import FFBPNeurode as FFBPN


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
    def __init__(self, inputs: int, outputs: int, neurode_type: type(FFBPN.Neurode)):
        super().__init__()
        self._type = neurode_type

        input_nodes = [neurode_type(FFBPN.LayerType.INPUT) for a in range(inputs)]
        output_nodes = [neurode_type(FFBPN.LayerType.OUTPUT) for a in range(outputs)]

        DoublyLinkedList.add_to_head(self, input_nodes)
        DoublyLinkedList.add_after_cur(self, output_nodes)

        self.connect_curr_to_next()

    def connect_curr_to_next(self):
        for node in self._curr._data:
            node.reset_neighbors(self._curr._next._data, FFBPN.MultiLinkNode.Side.DOWNSTREAM)
        for node in self._curr._next._data:
            node.reset_neighbors(self._curr._data, FFBPN.MultiLinkNode.Side.UPSTREAM)

    def add_layer(self, num_nodes: int):
        if self is FFBPN.LayerType.OUTPUT:
            raise IndexError
        else:
            hidden_nodes = [self._type(FFBPN.LayerType.HIDDEN) for a in range(num_nodes)]
            DoublyLinkedList.add_after_cur(self, hidden_nodes)
            self.connect_curr_to_next()
            self.move_forward()
            self.connect_curr_to_next()
            self.move_back()

    def remove_layer(self):
        if self._curr._next is self._tail:
            raise IndexError('Cannot remove tail list')
        else:
            self._curr._next = self._curr._next._next
            self._curr._next._prev = self._curr
            self.connect_curr_to_next()

    @property
    def input_nodes(self):
        return self._head._data

    @property
    def output_nodes(self):
        return self._tail._data


def dll_test():
    my_list = DoublyLinkedList()
    try:
        my_list.get_current_data()
    except DoublyLinkedList.EmptyListError:
        print("Pass")
    else:
        print("Fail")
    for a in range(3):
        my_list.add_to_head(a)
    if my_list.get_current_data() != 2:
        print("Error")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_forward()
    try:
        my_list.move_forward()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    if my_list.get_current_data() != 0:
        print("Fail")
    my_list.move_back()
    my_list.remove_after_cur()
    if my_list.get_current_data() != 1:
        print("Fail")
    my_list.move_back()
    if my_list.get_current_data() != 2:
        print("Fail")
    try:
        my_list.move_back()
    except IndexError:
        print("Pass")
    else:
        print("Fail")
    my_list.move_forward()
    if my_list.get_current_data() != 1:
        print("Fail")


def layer_list_test():
    # create a LayerList with two inputs and four outputs
    my_list = LayerList(2, 4, FFBPN.FFBPNeurode)
    # get a list of the input and output nodes, and make sure we have the right number
    inputs = my_list.input_nodes
    outputs = my_list.output_nodes
    assert len(inputs) == 2
    assert len(outputs) == 4
    # check that each has the right number of connections
    for node in inputs:
        assert len(node._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM]) == 4
    for node in outputs:
        assert len(node._neighbors[FFBPN.FFBPNeurode.Side.UPSTREAM]) == 2
    # check that the connections go to the right place
    for node in inputs:
        out_set = set(node._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM])
        check_set = set(outputs)
        assert out_set == check_set
    for node in outputs:
        in_set = set(node._neighbors[FFBPN.FFBPNeurode.Side.UPSTREAM])
        check_set = set(inputs)
        assert in_set == check_set
    # add a couple layers and check that they arrived in the right order, and that iterate and rev_iterate work
    my_list.reset_to_head()
    my_list.add_layer(3)
    my_list.add_layer(6)
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # save this layer to make sure it gets properly removed later
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 3
    # check that information flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    # try to remove an output layer
    try:
        my_list.remove_layer()
        assert False
    except IndexError:
        pass
    except:
        assert False
    # move and remove a hidden layer
    save_list = my_list.get_current_data()
    my_list.move_back()
    my_list.remove_layer()
    # check the order of layers again
    my_list.reset_to_head()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_forward()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.OUTPUT
    assert len(my_list.get_current_data()) == 4
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.HIDDEN
    assert len(my_list.get_current_data()) == 6
    my_list.move_back()
    assert my_list.get_current_data()[0].node_type == FFBPN.LayerType.INPUT
    assert len(my_list.get_current_data()) == 2
    # save a value from the removed layer to make sure it doesn't get changed
    saved_val = save_list[0].value
    # check that information still flows through all layers
    save_vals = []
    for node in outputs:
        save_vals.append(node.value)
    for node in inputs:
        node.set_input(1)
    for i, node in enumerate(outputs):
        assert save_vals[i] != node.value
    # check that information still flows back as well
    save_vals = []
    for node in inputs[1]._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM]:
        save_vals.append(node.delta)
    for node in outputs:
        node.set_expected(1)
    for i, node in enumerate(inputs[1]._neighbors[FFBPN.FFBPNeurode.Side.DOWNSTREAM]):
        assert save_vals[i] != node.delta
    assert saved_val == save_list[0].value


dll_test()
layer_list_test()

"""
Work in Process, I am unsure why the classes in the other module cannot be called.

/usr/bin/python3.10 /home/lilscent/PycharmProjects/f23_intermedPython/DoublyLinkedList.py 
Traceback (most recent call last):
  File "/home/lilscent/PycharmProjects/f23_intermedPython/DoublyLinkedList.py", line 252, in <module>
    layer_list_test()
  File "/home/lilscent/PycharmProjects/f23_intermedPython/DoublyLinkedList.py", line 149, in layer_list_test
    my_list = LayerList(2, 4, FFBPN)
  File "/home/lilscent/PycharmProjects/f23_intermedPython/DoublyLinkedList.py", line 90, in __init__
    input_nodes = [neurode_type(FFBPN.LayerType.INPUT) for a in range(inputs)]
  File "/home/lilscent/PycharmProjects/f23_intermedPython/DoublyLinkedList.py", line 90, in <listcomp>
    input_nodes = [neurode_type(FFBPN.LayerType.INPUT) for a in range(inputs)]
TypeError: 'module' object is not callable
Pass
Pass
Pass

Process finished with exit code 1
"""
