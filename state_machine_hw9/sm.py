from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        pass


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = None # Change

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass


class Reverser(SM):
    start_state = None

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        # Your code here
        pass

    def transition_fn(self, s, i):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass
