# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:19:11 2016

@author: chenym
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs
cimport cython

ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_INT_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compare_full(np.ndarray[DTYPE_t, ndim=2] subset_query, 
                 np.ndarray[DTYPE_t, ndim=2] subset_other, 
                 int query_lf_size, int other_set_size, int other_lf_size, 
                 int image_size, np.ndarray[DTYPE_INT_t, ndim=2] rotation, 
                 np.ndarray[DTYPE_t, ndim=1] scores):
    
    cdef int number_of_coeff = subset_query.shape[1]
    cdef int rotation_size = rotation.shape[0]
    cdef int i,j,k,i2,j2,k2,i3,t1,t2
    cdef int count = 0
    
    cdef double z0,z1
 
    assert other_set_size == scores.shape[0]
    assert query_lf_size*image_size == subset_query.shape[0]
    assert other_lf_size*image_size*other_set_size == subset_other.shape[0]
    
    cdef double low_score, curr_score
    
    for i in range(other_set_size):
        low_score = -1
        
        for j in range(query_lf_size):
            for k in range(other_lf_size):
                for k3 in range(rotation_size):
                    curr_score = 0
                    for i2 in range(image_size):
                        i3 = rotation[k3][i2]
                        t1 = j*image_size
                        t2 = k*image_size+i*other_lf_size*image_size
                        for j2 in range(number_of_coeff):
                            z0 = subset_query[i2+t1,j2]
                            z1 = subset_other[i3+t2,j2]
                            count += 1
                            curr_score += fabs(z0-z1)
                    if low_score < 0 or curr_score < low_score:
                        low_score = curr_score

        scores[i] = low_score