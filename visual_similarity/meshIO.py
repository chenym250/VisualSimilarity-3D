# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 15:47:08 2016

@author: chenym
"""
from mesh import Mesh
import numpy as np

##########
# import #
##########

def read_off_and_store_as_vf(reader):
    """
    read (line-by-line) an opened OFF file. Store results in numpy arrays
    
    Param: 
        reader: an open('file_name') reader
    
    Return: 
        verts: x,y,z coords of all vertices
        faces: vertex-index of all faces
    """
    first_line = reader.readline()
    if first_line.split()[0] != 'OFF':
        raise ValueError('the first line of the file is not "OFF"; is instead "' +
        first_line + '"')

    second_line_split = reader.readline().split()       
    num_v = int(second_line_split[0])
    num_f = int(second_line_split[1])

    verts_list = []

    # read vertices    
    for l in range(num_v):
        x,y,z = [float(num) for num in reader.readline().split()]
        verts_list.append([x,y,z])
        
    faces_list = []
    
    # read triangles
    for f in range(num_f):
        count, v1, v2, v3 = [long(num) for num in reader.readline().split()]
        if count > 3:
            raise ValueError("face has more than 3 vertices; not triangular mesh.")
        faces_list.append([v1,v2,v3])
        
    verts = np.array(verts_list)
    faces = np.array(faces_list)
    return Mesh(verts, faces)
    
##########
# export #
##########
    
def customizePatchColor(mesh_scene, cdata):
    """
    a wrapper for mayavi functionalities that allow you to color the mesh by
    1. a rank
    2. a set of rgb or rgba values, hence an image
    
    mesh_scene is the return of a mayavi plot, like this:
        mesh_scene = mlab.triangular_mesh(x,y,z,f)
    cdata is the vector/image you want to plot on the mesh
    
    this can be applied to either the vertices or the faces;therefore, cdata 
    can have the following shapes:
    (V,1) or (F,1) <-- these two for ranks
    
    (V,2 or 3 or 4) or (F,2 or 3 or 4)<-- hese two for images ***
    
    *** this method will automatically fill cdata if it has less than 4 columns
    """
    #####################
    # cdata is a vector #
    #####################
    h2,w2 = cdata.shape
    if w2 == 1:
        mesh_scene.mlab_source.scalars = cdata # plot the color coded rank of cdata
        return
        
    
    #####################
    # cdata is an image #
    #####################
    # make sure cdata is in the range of [0,255]
    if np.min(cdata) < 0:
        cdata -= np.min(cdata)
    if np.max(cdata) > 255:
        cdata = cdata/np.max(cdata)*255
    # stuff cdata if it does not have enough columns (no alpha value)
    if w2 < 4:
        cdata = np.concatenate((cdata,255*np.ones((h2,4-w2))),axis=1)
    mesh_scene.mlab_source.scalars = np.arange(h2)
    mesh_scene.module_manager.scalar_lut_manager.lut.number_of_colors = h2
    mesh_scene.module_manager.scalar_lut_manager.lut.table = cdata[:,0:4]