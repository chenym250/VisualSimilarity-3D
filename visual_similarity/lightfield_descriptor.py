# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:42:15 2016

@author: chenym
"""

import numpy as np

class LightField(object):
    """
    LightField class can be seen as a container of different camera angles. 
    It stores the camera angles as a NumPy array of shape (N,2), where N is the
    total number of angles, and 2 is the inclination and azimuth angles of the
    spherical coordinates (radius does not need to be defined since the 
    projections are always assumed to be from the unit sphere). This set of 
    camera angles can be seen as "fields" of light rays running through a 3D 
    model from angles. Hence the name light field. 
    
    LightField will be used as a reference when creating LightFieldDescriptor
    (image renderer will use the angles stored in a LightField to create
    2D projection images (silhouettes) of shapes, and the set of images of the 
    same LightField and the same mesh is stored in a LightFieldDescriptor)
    """
    
    # a class variable, used to keep track of LightField obj created
    _class_id = 0
    
    # camera_anles = (azimuthal, polar), following this convention
    # http://mathworld.wolfram.com/SphericalCoordinates.html
    def __init__(self, camera_angles):
        self.camera_angles = camera_angles
        self.id = LightField._class_id # get current id
        LightField._class_id += 1 # increment class id
        # each instance of LightField will have different id

class LightFieldDescriptor(object):
    """
    LightFieldDescriptor is the LightField Descriptor (See reference paper) of 
    a mesh. It can be seen as a collection of 2D projection images (of a
    3D object) taken from different angles. The images are stored in memory 
    as NumPy ndarrays. The class is iterable so it can be easily used
    in loops.
    
    There is method that compares two LightFieldDescriptors using some image
    metric function (provided by the caller) and output their L1 distance. 
    """
    
    _class_id = 0
    
    def __init__(self, lf, images = []):
        self.lf = lf
        self.images = images
        self.id = LightFieldDescriptor._class_id # get current id
        LightFieldDescriptor._class_id += 1 # increment class id
    
    def __getitem__(self,key):
        return(self.images[key])