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
    def __init__(self, radius=256, degree = 10, com = [127.5,127.5]):
        """
        
        """
        self.radius = radius
        self.degree = degree
 
    def describe(self, image):
        return mahotas.features.zernike_moments(image, self.radius, degree=self.degree)