import vectors as vc
import numpy as np

#The data feed into the functions should havw dimention d by n
#where n is the number of datapoints

#th and th0 is the linear separator- a hyperplane. the th and th0
#can be generated in various way. a randomly generated th and th0 
#is supposed to be feed into the functions in this file

# x is dimension d by 1
# th is dimension d by 1
# th0 is a scalar
# return 1 by 1 matrix of signed distance
def signed_dist(x, th, th0):
    """th and th0 represent a hiperplane and signed_dist() returns 
        1 by 1 matrix of signed distance of x from the hiperplane"""
    return ((th.T@x) + th0) / vc.length(th)


# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
    return np.sign(np.dot(np.transpose(th), x) + th0)



# data is dimension d by n
# labels is dimension 1 by n
# ths is dimension d by 1
# th0s is dimension 1 by 1
# return 1 by 1 matrix of integer indicating number of data points correct for
# each separator.
def score(data, labels, th, th0):
    return np.sum(positive(data, th, th0) == labels)


# data is dimension d by n
# labels is dimension 1 by n
# ths is dimension d by m
# th0s is dimension 1 by m
# return matrix of integers indicating number of data points correct for
# each separator:  dimension m x 1
def score_mat(data, labels, ths, th0s):
    pos = np.sign(np.dot(np.transpose(ths), data) + np.transpose(th0s))
    return np.sum(pos == labels, axis = 1, keepdims = True)

def best_separator(data, labels, ths, th0s):
    best_index = np.argmax(score_mat(data, labels, ths, th0s))
    return vc.cv(ths[:,best_index]), th0s[:,best_index:best_index+1]




