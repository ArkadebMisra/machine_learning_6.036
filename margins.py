import numpy as np

def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])

def y(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is a scalar
    return a 1 by 1 matrix
    '''
    return np.dot(np.transpose(th), x) + th0

def positive(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is dimension 1 by 1
    return 1 by 1 matrix of +1, 0, -1
    '''
    return np.sign(y(x, th, th0))

def score(data, labels, th, th0):
    '''
    data is dimension d by n
    labels is dimension 1 by n
    ths is dimension d by 1
    th0s is dimension 1 by 1
    return 1 by 1 matrix of integer indicating number of data points correct for
    each separator.
    '''
    return np.sum(positive(data, th, th0) == labels)

def length(col_v):
    """returns the lenth of a np column vecctor"""
    return np.sum(col_v * col_v)**0.5

def signed_dist(x, th, th0):
    """th and th0 represent a hiperplane and signed_dist() returns 
        1 by 1 matrix of signed distance of x from the hiperplane"""
    return ((th.T@x) + th0) / length(th)





def margin(data, labels, th, th0):
    sd = signed_dist(data, th, th0)
    sd =  sd * labels
    return sd
    
def s_sum(data, labels, th, th0):
    return margin(data, labels, th, th0).sum(axis = 1) 

def s_min(data, labels, th, th0):
    return np.amin(margin(data, labels, th, th0)) 

def s_max(data, labels, th, th0):
    return np.amax(margin(data, labels, th, th0)) 



def gama(data, labels, th, th0):
    return margin(data, labels, th, th0)

def loss_individual(data, labels, th, th0, gama_ref):
    g = gama(data, labels, th, th0)
    gama_by_gama_ref = g/gama_ref
    g = g<gama_ref
    g = g*1
    g = np.where(g==1, 1-gama_by_gama_ref, 0)
    return g




# data = np.array([[1, 2, 1, 2, 10, 10.3, 10.5, 10.7],
#                  [1, 1, 2, 2,  2,  2,  2, 2]])
# labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1]])
# blue_th = np.array([[0, 1]]).T
# blue_th0 = -1.5
# red_th = np.array([[1, 0]]).T
# red_th0 = -2.5

# print(margin(data, labels, blue_th, blue_th0))

# print(s_sum(data, labels, blue_th, blue_th0))
# print(s_min(data, labels, blue_th, blue_th0))
# print(s_max(data, labels, blue_th, blue_th0))

# print(s_sum(data, labels, red_th, red_th0))
# print(s_min(data, labels, red_th, red_th0))
# print(s_max(data, labels, red_th, red_th0))


data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
gma_ref = (2**(1/2))/2

print(loss_individual(data, labels, th, th0, gma_ref))