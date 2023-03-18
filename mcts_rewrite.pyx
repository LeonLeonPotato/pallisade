# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True

import os
import platform

if "mac" in platform.platform():
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

np.import_array()

cdef float cpuct = 3.0
cdef float dirichlet_val = 0.03

cdef diri = torch.distributions.dirichlet.Dirichlet(torch.tensor([dirichlet_val] * 7 * 7))

# @cython.auto_pickle(True)

cdef class Node:
    cdef: 
        unsigned int W
        unsigned int visits
        float Q, P
        list children
        Node parent
        char turn
        np.ndarray state
        bint leaf
        unsigned char move_x, move_y
        
    cdef float uct(self):
        return self.Q / (1 + self.W) + cpuct * self.P / (1 + self.visits)

    cdef void backprop(self, float value, int w):
        cdef Node current = self
        cdef int turn = -1
        while True:
            current.W += w
            current.Q += value * turn
            current = current.parent
            turn *= -1
            if not current:
                break

    cdef Node expand(self):
        if not self.leaf:
            return None

        cdef:
            np.ndarray copied
            Node child
            Node best
            float tmp_uct = -999
            float best_uct = -999
            int i, t
            signed char[:, :] moves
            unsigned char x, y
            float total_value = 0

        self.leaf = False
        moves = np.array(np.where(self.state == 0), dtype=np.int8).T
        i = 0
        t = moves.shape[0]
        while i < t:
            x = moves[i][0]
            y = moves[i][1]
            copied = self.state.copy()
            copied[x, y] = -self.turn
            child = Node(self, -self.turn, copied, x, y)
            child.Q = 0.4
            child.P = 0.4
            tmp_uct = child.uct()
            if tmp_uct > best_uct:
                best_uct = tmp_uct
                best = child
            self.children.append(child)
            total_value += child.Q
            i += 1
        
        self.backprop(total_value, i)
        return best


    # # Auto generated code from ChatGPT
    # def __reduce__(self):
    #     # Convert the NumPy array to a tuple of data and dtype
    #     data = self.state.data, self.state.dtype
    #     # Return the Node constructor and initialization arguments as well as the data tuple
    #     return Node, (self.parent, self.turn, np.zeros((7, 7), dtype=np.int32)), data

cdef void search(int num, Node root):
    pass

cdef void play_game(model):
    cdef:
        list states
        list policy
        list value
        Node root = Node(None, -1, np.zeros((7, 7), dtype=np.int8), 0, 0)
        char depth = 0
        int searches = 0

    while True:
        while searches < 256:
            searches += 1

from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cdef double get_time() nogil:
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current

cdef int i = 0
cdef double last = get_time()
cdef Node current

cdef int [:] data_x = np.zeros((1000,), dtype=np.int32)
cdef double [:] data_y = np.zeros((1000,), dtype=np.double)
while i < 1000:
    current = Node(None, -1, np.zeros((7, 7), dtype=np.int8), 0, 0)
    current.visits += 1
    while True:
        tmp = current.expand()
        current.visits += 1
        if len(current.children) == 1:
            break
        current = tmp
    i += 1
    data_x[i] = i
    data_y[i] = get_time() - last
    last = get_time()

balls = data_x
ballsy = data_y
