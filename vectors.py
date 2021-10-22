import numpy as np
#In Machine Learning the features are represented as column vectors
def tp(A):
    """returns transpose of a np vector"""
    return np.transpose(A)

def rv(value_list):
    """returns a np rew vector from a list eg. [1, 2, 3]"""
    return np.array([value_list])


def cv(value_list):
    """returns a np column vector from a python list eg from [1, 2, 3...., n]"""
    return np.transpose(rv(value_list))


def length(col_v):
    """returns the lenth of a np column vecctor"""
    return np.sum(col_v * col_v)**0.5

def normalize(col_v):
    """returns normalized length from a np col vector"""
    return col_v/length(col_v)


def index_final_col(A):
    """Takes a 2D matrix;  returns last column as 2D matrix"""
    return A[:,-1:]