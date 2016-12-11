# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 01:20:01 2016

@author: chenym
"""

##############################################################################
# imports
##############################################################################
import time
from visual_similarity import constants
import numpy as np
import scipy.spatial.distance as distance
import pickle

##############################################################################
# constants
# make this configurable in the future
##############################################################################
INPUTDIR = 'db\\'

def unfold_map_to_tables(whole_data, table_height, table_width, 
                         skip_mesh_id=[], skip_lf_id=[], skip_angle_id=[]):
    table = np.ones((table_height,table_width))*np.inf
    index_lookup = np.ones((table_height,3))*np.inf
    mesh_id = 0
    row_pointer = 0
    for mesh_data in whole_data:
        if mesh_id in skip_mesh_id:
            continue
        for lf_id in mesh_data.keys():
            if lf_id in skip_lf_id:
                continue
            angle_id = 0
            for image_feature in mesh_data[lf_id]:
                if angle_id in skip_angle_id:
                    continue
                table[row_pointer,:] = image_feature
                index_lookup[row_pointer,:] = \
                np.array([mesh_id,lf_id,angle_id])
                angle_id += 1
                row_pointer += 1
        mesh_id += 1
        
    return index_lookup, table

def dissimilarity_between_descriptors(z1,z2):
    """
    Calculate the L1 distance between two sets of shape descriptors. 
    
    If the length of the sets is equal to 10, the calculation will be repeated
    for 60 times, each time corresponds to a rotation of a dodecahedron; if the 
    length of the sets is not 10, a simple one-to-one comparison will be made.
    
    Parameters
    ----------
    z1 : iterable
        The first set to compare. z1[0] is a n-array.
    param2 : iterable
        The second set to compare, same size as z1.

    Returns
    -------
    float
        The final score
    """
    # DA
    
    
    if len(z1) is not 10:
        min_score = 0
        for i in xrange(len(z1)):
            min_score += distance.cityblock(z1[i], z2[i])
        return min_score

           
    permutation = constants.PERMUTATION_DODECAHEDRON_HALF # shape (60,10)
    min_score = np.inf
    for p0 in permutation:
        curr_score = 0
        for i in xrange(10):
            curr_score += np.random.rand()#distance.cityblock(z1[i], z2[p0[i]]) # <-- p0[i]
        if curr_score < min_score:
            min_score = curr_score

    return min_score

def computefullscore(query_index,_z_all):
    # compute scores (full)
    
    z_all = _z_all[:]    
    
    t8 = time.time()
    scores = []
    z0 = z_all.pop(query_index)
    for z1 in z_all:
        min_score = np.inf
        for zm0 in z0.values():
            # dissimilarity between meshes from different orientations
            for zm1 in z1.values():
                curr_score = dissimilarity_between_descriptors(zm0,zm1)
                if curr_score < min_score:
                    min_score = curr_score
        scores.append(min_score)
    t9 = time.time()
    print 'computing the scores takes: %fs' % (t9-t8)


def load_obj(name ):
    """
    Open a file and load an object using `pickle`. 
    
    Parameters
    ----------
    name: str
        The name of the file without the extension (assume .pkl)
    
    Returns
    -------
    object
        The loaded object
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def compare_and_reject(query_id, features, ids_of_set, number_of_coeff, 
                       image_selection, lfs_of_queries, lfs_of_rest):
    """
    Compare the coefficients (or part of the coefficients) between a selected
    query shape and the other shapes. 
    
    Select a subset of the whole features as the features of the querying 
    shape; select other subset as the features of other shapes. The comparison
    can be partial, meaning some of the shapes, lightfields, images and/or 
    coefficients will not be used. 
    
    Note that if query_id is presented in ids_of_rest, then one of the scores
    will be set as infinite and may affect the mean calculations. 
    
    Parameters
    ----------
    query_id: int
        The id of the query
    features: ndarray
        The whole feature set of all shapes
    ids_of_set: array-like
        A full list of shape id(s)
    number_of_coeff: int
        number of coefficients used for each feature
    image_selection: array-like
        The id(s) of the images used in each LightField
    lfs_of_queries: iterable of int
        The id(s) of the LightField used by the query shape
    lfs_of_rest: iterable of int
        The id(s) of the LightField used by other shape
        
    Returns
    -------
    array-like
        A shrunkened/halved list of id(s)
    """
    
    scores = np.ones((len(ids_of_set),)) *np.inf
    
    count = 0
    summ = 0
    
    for i in ids_of_set:
        if i == query_id:
            continue
        low_score = np.inf
        for lf_id_q in lfs_of_queries:
            for lf_id_rest in lfs_of_rest:
                z0 = features[np.array(image_selection)+query_id*100+lf_id_q*10,\
                0:number_of_coeff]
                z1 = features[np.array(image_selection)+i*100+lf_id_rest*10,\
                0:number_of_coeff]
                new_score = dissimilarity_between_descriptors(z0,z1)
                if new_score < low_score:
                    low_score = new_score
        scores[count] = low_score
        count += 1
        summ += low_score

#    return ids_of_set[scores < summ/float(count)]
    return ids_of_set[scores < np.median(scores)], scores

def _combine_returns(return_old,return_new,size):
    l1 = return_old[0]
    l2 = return_new[0]
    r1 = return_old[1]
    r2 = return_new[1]
    if len(l2) >= size:
        return l2[0:size]
    l1 = l1[np.argsort(r1)]
    l2 = l2[np.argsort(r2)]
    result = np.empty((len(l1),),dtype='int16')
    result[0:len(l2)] = l2
    result[len(l2):len(l1)] = np.array([i for i in l1 if i not in l2])
    return result[0:size]
    
def retrieval_process(query_id, features, number_of_shape, steps=[0,1,2,3], 
                      min_return_size=5):
    """
    The retrieval process described in Chen .etc to find meshes that look 
    similar to the query. Everything about bit quantization is currently
    ignored. 
    
    There are 6 steps with increasing complexity and decreasing sample size. 
    That is, this is an early rejection algorithm that filters out most of the 
    samples using the fastest computation before dedicating all its power in 
    finding the best solutions. 
        
    Parameters
    ----------
    query_id: int
        The id of the query
    features: ndarray
        The whole feature set of all shapes
    number_of_shape: int
        The total number of shapes
    steps: list of int
        Select which of the 6 steps will be executed
    return_size: int
        Minimum size of the return. 
    
    Returns
    -------
    array-like
        A list of mesh indices with size larger or equal to `min_return_size`
    """
    image_select = np.arange(10)
    lf_select = np.arange(10)
    np.random.shuffle(image_select)
    np.random.shuffle(lf_select)
    shape_set = np.arange(number_of_shape)
    
    ## step 1
    # "
    # In the initial stage, all 3D models in the database are
    # compared with the queried one. Two LightField Descriptors
    # of the queried model are compared with ten of those in the
    # database. Three images of each light field are compared, and
    # each image is compared using 8 coefficients of Zernike moment. 
    # "
    t1 = time.time()
    step1result = compare_and_reject(query_id,features,shape_set,8,\
    image_select[0:3],lf_select[0:2],range(10))
    t2 = time.time()
    print 'step 1 done in %f s. remaining shapes: %d' % (t2-t1,len(step1result[0]))
    # remove half of the data
    shape_set = step1result[0]    
    if len(shape_set) <= min_return_size:
        return shape_set

    ## step 2
    # "
    # In the second stage, five LightField Descriptors of
    # the queried model are compared to ten of the others in the 
    # database. Five images of each light field are compared, and
    # each image is compared using 16 coefficients of Zernike moment.
    # "
    t3 = time.time()
    step2result = compare_and_reject(query_id,features,shape_set,16,\
    image_select[0:5],lf_select[0:5],range(10))
    t4 = time.time()
    print 'step 2 done in %f s. remaining shapes: %d' % (t4-t3,len(step2result[0]))
    shape_set = step2result[0]    
    if len(shape_set) <= min_return_size:
        return _combine_returns(step1result,step2result,min_return_size)
    
    ## step 3
    # "
    # Thirdly, seven LightField Descriptors of queried
    # model are compared with ten of the others in the database.
    # The other is the same as the second stage, while another five
    # images of each light field are compared.
    # "
    t5 = time.time()
    step3result = compare_and_reject(query_id,features,shape_set,16,\
    image_select,lf_select[0:7],range(10))
    shape_set = step3result[0]
    t6 = time.time()
    print 'step 3 done in %f s. remaining shapes: %d' % (t6-t5,len(shape_set))
    if len(shape_set) <= min_return_size:
        return _combine_returns(step2result,step3result,min_return_size)
    
    ## step 4
    # "
    # The fourth stage is the same as full comparison, but
    # only the Zernike moment coefficients, quantized to 4 bits,
    # are used. In addition, the top 16 of the 5,460 rotations are
    # recorded between the queried one and others.
    # "
    t7 = time.time()
    step4result = compare_and_reject(query_id,features,shape_set,35,\
    image_select,lf_select,range(10))
    shape_set = step4result[0]    
    t8 = time.time()
    print 'step 4 done in %f s. remaining shapes: %d' % (t8-t7,len(shape_set))
    if len(shape_set) <= min_return_size:
        return _combine_returns(step3result,step4result,min_return_size)
    
    return shape_set

def loaddata():
    full_data = load_obj(INPUTDIR + 'zernike3_200_shapes_01_10_12_11')
    number_of_shape = 200
    number_of_lf = 10
    number_of_angle = 10
    table_height = number_of_shape*number_of_lf*number_of_angle
    two_table = unfold_map_to_tables(full_data, table_height, 35)
#    reference_table = two_table[0]
    feature_table = two_table[1]
    assert np.max(feature_table) < np.inf
    
    return feature_table

##############################################################################
# main method
##############################################################################
def main():
    feature_table = loaddata()
    t1 = time.time()
    a = retrieval_process(21, feature_table, 200, steps=[0])
    print a
    t2 = time.time()
    print 'stage 1 takes %f s' % (t2-t1)   
    
if __name__ == '__main__':
    main()
