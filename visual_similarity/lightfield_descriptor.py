# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 21:42:15 2016

@author: chenym
"""

import numpy as np
import constants

class LightField(object):
    """
    LightField class can be seen as a container of different camera angles. 
    It stores the camera angles as a NumPy array of shape (N,2), where N is the
    total number of angles, and 2 is the inclination and azimuth angles of the
    spherical coordinates (radius does not need to be defined since the 
    projections are always assumed to be from the unit sphere). This set of 
    camera angles can be seen as a "field" of rays ready to run through a 3D 
    model from different angles. Hence the name light field.  The result is a 
    set of shadow-like 2D projection images called silhouettes. 
    
    LightField could be used by other modules for multiple purposes. For
    instance, it provides the angles that will be used by an image renderer to 
    create LightFieldDescriptors; it may be preprocessed to save computation
    time, and thus needs save/load functions; although we do not need to know 
    the light field when comparing two meshes (and theirLightFieldDescriptors),
    we may need that information later to evaluate and analyze the performance. 
    
    There are class methods that will compute the distance between two or more 
    LightField objects. It is essential to create a set of light fields that 
    uniformly covers a whole sphere without any one of them overlapping with 
    the others. 
    """
    
    # a class variable, used to keep track of LightField obj created
    _class_id = 0
    
    # camera_anles = (azimuthal, polar), following this convention
    # http://mathworld.wolfram.com/SphericalCoordinates.html
    def __init__(self, camera_angles = None):
        self.camera_angles = camera_angles
        self.id = LightField._class_id # get current id
        LightField._class_id += 1 # increment class id
        # each instance of LightField will have different id
    
    def set_as_dodecahedron(self):
        """
        choose the 20 vertices of a regular Dodecahedron as camera angles
        """
        self.camera_angles = constants.AZIMUTHAL_POLAR_DODECAHEDRON

class LightFieldDescriptor(object):
    """
    LightFieldDescriptor is the LightField Descriptor (See reference paper) of a
    mesh object. It can be seen as a collection of 2D projection images (of a
    3D object) taken from different angles. The images are stored in memory 
    runtime as NumPy ndarrays. The class is iterable so it can be easily used
    in for loops.
    
    There is method that compares two LightFieldDescriptors using some image
    metric function (provided by the caller) and output their L1 distance. 
    """
    
    def __init__(self, images = []):
        self.images = images
        
    def shuffle(self):
        pass
    
    def __getitem__(self,key):
        return(self.images[key])