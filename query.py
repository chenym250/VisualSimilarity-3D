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
from psb_meshes import readfrompsb
from mayavi import mlab
from partialcomparison import compare_and_reject as compare_and_reject2

##############################################################################
# constants
# make this configurable in the future
##############################################################################
INPUTDIR = 'db\\'
lookup = np.random.rand(16,16)

##############################################################################
# methods
##############################################################################
def unfold_map_to_tables(whole_data, table_height, table_width, 
                         skip_mesh_id=[], skip_lf_id=[], skip_angle_id=[]):
    table = np.ones((table_height,table_width))*np.inf
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
                angle_id += 1
                row_pointer += 1
        mesh_id += 1
        
    return table

def _find_digits(zs):
    return np.ceil(np.log10(zs))

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
            curr_score += distance.cityblock(z1[i], z2[p0[i]]) # <-- p0[i]
        if curr_score < min_score:
            min_score = curr_score

    return min_score

def dissimilarity_between_descriptors_2(z1,z2):
    """
    Calculate the L1 distance between two sets of shape descriptors. 
    
    If the length of the sets is equal to 10, the calculation will be repeated
    for 60 times, each time corresponds to a rotation of a dodecahedron; if the 
    length of the sets is not 10, a simple one-to-one comparison will be made.
    
    Parameters
    ----------
    z1 : ndarray
        The first set to compare. 
    z2 : ndarray
        The second set to compare, same size as z1.

    Returns
    -------
    float
        The final score
    """
    # DA
    
    
    if z1.shape[0] != 10:
        distances = np.abs(z1-z2)
        return np.sum(distances)

           
    permutation = constants.PERMUTATION_DODECAHEDRON_HALF # shape (60,10)
    min_score = np.inf
    for p0 in permutation:        
        z2_new = z2[p0,:]
        distances = np.abs(z1-z2_new)
        curr_score = np.sum(distances)
        if curr_score < min_score:
            min_score = curr_score

    return min_score

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
            count += 1
            continue
        low_score = np.inf
        for lf_id_q in lfs_of_queries:
            for lf_id_rest in lfs_of_rest:
                z0 = features[np.array(image_selection)+query_id*100+lf_id_q*10,\
                0:number_of_coeff]
                z1 = features[np.array(image_selection)+i*100+lf_id_rest*10,\
                0:number_of_coeff]
                new_score = dissimilarity_between_descriptors_2(z0,z1)
                if new_score < low_score:
                    low_score = new_score
        scores[count] = low_score
        count += 1
        summ += low_score

#    return ids_of_set[scores < summ/float(count)
    print scores
    return ids_of_set[scores < np.median(scores)], scores[scores < np.median(scores)]

def _combine_returns(return_old,return_new,size):
    if not return_old:
        return _single_returns(return_new,size)
    l1 = return_old[0]
    l2 = return_new[0]
    r1 = return_old[1]
    r2 = return_new[1]
    l1 = l1[np.argsort(r1)]
    l2 = l2[np.argsort(r2)]
    if len(l2) >= size:
        return l2[0:size]
    result = np.empty((len(l1),),dtype='int16')
    result[0:len(l2)] = l2
    result[len(l2):len(l1)] = np.array([i for i in l1 if i not in l2])
    return result[0:size]

def _single_returns(return_new,size):
    l2 = return_new[0]
    r2 = return_new[1]
    l2 = l2[np.argsort(r2)]
    size = min(size,len(l2))
    return l2[0:size]
    
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
    result = None
    args = []
    
    ## step 1
    # "
    # In the initial stage, all 3D models in the database are
    # compared with the queried one. Two LightField Descriptors
    # of the queried model are compared with ten of those in the
    # database. Three images of each light field are compared, and
    # each image is compared using 8 coefficients of Zernike moment. 
    # "
    args.append([8,image_select[0:3],lf_select[0:2],range(10)])

    ## step 2
    # "
    # In the second stage, five LightField Descriptors of
    # the queried model are compared to ten of the others in the 
    # database. Five images of each light field are compared, and
    # each image is compared using 16 coefficients of Zernike moment.
    # "
    args.append([16,image_select[0:5],lf_select[0:5],range(10)])
    
    ## step 3
    # "
    # Thirdly, seven LightField Descriptors of queried
    # model are compared with ten of the others in the database.
    # The other is the same as the second stage, while another five
    # images of each light field are compared.
    # "
    args.append([16,image_select,lf_select[0:7],range(10)])
    
    ## step 4
    # "
    # The fourth stage is the same as full comparison, but
    # only the Zernike moment coefficients, quantized to 4 bits,
    # are used. In addition, the top 16 of the 5,460 rotations are
    # recorded between the queried one and others.
    # "
    args.append([35,image_select,lf_select,range(10)])
        
    for step in steps:
        t0 = time.time()
        prev_result = result
        if result is None: # initial state
            result = [np.arange(number_of_shape),]
        result = compare_and_reject(query_id,features,result[0],*args[step])
        t1 = time.time()
        print 'step %d done in %f s. remaining shapes: %d' % (step+1,t1-t0,len(result[0]))
        if len(result[0]) <= min_return_size:
            return _combine_returns(prev_result,result,min_return_size)
    
    return _single_returns(result,min_return_size)

def loaddata(full_data):
        
    number_of_shape = len(full_data['mesh_list'])
    number_of_lf = len(full_data['lightfield_id'])
    number_of_angle = 10
    table_height = number_of_shape*number_of_lf*number_of_angle
    feature_table = unfold_map_to_tables(full_data['features'], table_height, 35)
    assert np.max(feature_table) < np.inf
    
    return feature_table

##############################################################################
# main method
##############################################################################
def main():
    full_data = load_obj(INPUTDIR + 'descriptor_zernike_shapecount_907_timedate__1241_1211')
    feature_table = loaddata(full_data)
    query_id = 6
#    print 'started'
#    t1 = time.time()
#    a = retrieval_process(query_id, feature_table, 907, steps=[0,1,2,3])
#    t2 = time.time()
#    print 'comparison takes %f s to run' % (t2-t1)
    
#    compare_and_reject2(query_id, feature_table, [],5,[],[],[])
#    indices = np.loadtxt('indexlist1.txt',dtype='int16',delimiter=',').tolist()
    
    t1 = time.time()
    compare_and_reject(query_id,feature_table,np.arange(907),8,
                           np.array([0,1,2]),np.array([3,4]),range(10))  
    t2 = time.time()
    b = np.zeros(907)
    print '--------------'
    
    t3 = time.time()
    compare_and_reject2(query_id,feature_table,np.arange(907),8,
                            np.array([0,1,2]),np.array([3,4]),b)  
    t4 = time.time()
    print b
    
    print (t2-t1),(t4-t3)
    
    
#    indices = full_data['mesh_list']
#    meshes = [readfrompsb(indices[query_id])]
#    for i in a:
#        meshes.append(readfrompsb(indices[i]))
#        
#    for m in meshes:
#        mlab.figure()
#        mlab.triangular_mesh(m.verts[:,0],m.verts[:,1],m.verts[:,2],m.faces)
#    
if __name__ == '__main__':
    main()
    pass
