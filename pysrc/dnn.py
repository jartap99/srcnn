#
# @rajeevp
#

import numpy as np

class DNN(object):
    def __init__(self):
        np.random.seed(9076301)
        self.description = "DNN functions"

    def __str__(self):
        return self.description

    def conv2D(ifm, weights, biases, padding, activation):
        print("*** DNN.conv2D ***")
        print("ifm     : ", ifm.shape)
        print("weights : ", weights.shape)
        print("biases  : ", biases.shape)
        m, n, c = ifm.shape
        k, l, c1, numFilters = weights.shape
        if padding == "valid":
            I = m-k+1
            J = n-l+1
            ifmPadded = ifm
        else:
            I = m
            J = n
            ifmPadded = np.pad(ifm, (  ( (k-1)//2, (k-1)//2 ), ( (l-1)//2, (l-1)//2 ), (0,0) ), 
                                mode="constant")
        ofm = np.zeros((I, J, numFilters))
        print("ofm     : ", ofm.shape)

        for nc in range(numFilters):
            for i in range(I):
                for j in range(J):
                    patch = ifmPadded[i:i+k, j:j+l, :]
                    filt = weights[:,:,:,nc]
                    ofm[i,j,nc] = np.sum(np.multiply(patch, filt)) + biases[nc]
                    if (activation=='relu'):
                        if (ofm[i,j,nc] < 0):
                            ofm[i,j,nc] = 0
        return ofm




