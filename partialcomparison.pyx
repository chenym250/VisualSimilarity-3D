# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:05:50 2016

@author: chenym
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs
cimport cython

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compare_partial(np.ndarray[DTYPE_t, ndim=2] subset_query, 
                    np.ndarray[DTYPE_t, ndim=2] subset_other, 
                    int query_lf_size, int other_set_size, int other_lf_size, 
                    int image_size, np.ndarray[DTYPE_t, ndim=1] scores):

    cdef int number_of_coeff = subset_query.shape[1]
    cdef int i,j,k,i2,j2,t1,t2
    
    cdef double z0,z1

    assert other_set_size == scores.shape[0]
    assert query_lf_size*image_size == subset_query.shape[0]
    assert other_lf_size*image_size*other_set_size == subset_other.shape[0]
    
    cdef double low_score, curr_score
        
    for i in range(other_set_size):
        low_score = -1
        
        for j in range(query_lf_size):
            for k in range(other_lf_size):
                curr_score = 0
                for i2 in range(image_size):
                    t1 = j*image_size
                    t2 = k*image_size+i*other_lf_size*image_size
                    for j2 in range(number_of_coeff):
                        z0 = subset_query[i2+t1,j2]
                        z1 = subset_other[i2+t2,j2]
                        curr_score += fabs(z0-z1)
                if low_score < 0 or curr_score < low_score:
                    low_score = curr_score

        scores[i] = low_score