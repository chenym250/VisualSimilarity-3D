# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 18:58:24 2016

@author: chenym
"""
from mesh import Mesh
import numpy as np

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