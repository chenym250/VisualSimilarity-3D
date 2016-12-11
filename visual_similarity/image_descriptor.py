# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:54:27 2016

@author: chenym
"""

import mahotas


class ZernikeMoments:
    """
    Source: http://www.pyimagesearch.com/2014/04/07/
    building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/
    """
    def __init__(self, radius=128, degree = 10, com = [127.5,127.5]):
        """
        
        """
        self.radius = radius
        self.degree = degree
        self.com = com
 
    def describe(self, image):
        return mahotas.features.zernike_moments(\
        image, self.radius, degree=self.degree, cm=self.com)[1:]
        # ignore the first moment
        # first moment is the center of the mass
        
    def copy(self):
        return ZernikeMoments(self.radius,self.degree,self.com)
        