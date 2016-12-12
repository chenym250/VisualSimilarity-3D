# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:05:50 2016

@author: chenym
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs

def compare_and_reject(int query_id, np.ndarray features, np.ndarray ids_of_set, 
                       int number_of_coeff, np.ndarray image_selection, 
                       np.ndarray lfs_of_queries, np.ndarray scores):
#    cdef double [:, :] features_view = features

    cdef int set_size = ids_of_set.shape[0]
    cdef int image_size = image_selection.shape[0]
    cdef int lfs_size_q = lfs_of_queries.shape[0]
    cdef int i,j,k,i2,j2
    cdef int shape_id,image_id,lf_id_q
    cdef double low_score, new_score, curr_score, summ
    cdef double z0,z1
    
    for i in range(set_size):
        shape_id = ids_of_set[i]
        if shape_id == query_id:
            scores[i] = -1
            continue
        low_score = -1
        for j in range(lfs_size_q):
            lf_id_q = lfs_of_queries[j]
            for k in range(10):
                curr_score = 0
                for i2 in range(image_size):
                    image_id = image_selection[i2]
                    for j2 in range(number_of_coeff): # hideous...
                        z0 = features[image_id+query_id*100+lf_id_q*10,j2]
                        z1 = features[image_id+shape_id*100+k*10,j2]
                        curr_score += fabs(z0-z1)
                if low_score < 0 or curr_score < low_score:
                    low_score = curr_score
        scores[i] = low_score