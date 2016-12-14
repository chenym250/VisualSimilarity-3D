# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 22:54:27 2016

@author: chenym
"""

import mahotas
from skimage import measure
import numpy as np

class FourierMoments:
    
    def __init__(self,order = 15,fftlength = 16):
        self.order = order
        self.fftlength = fftlength
        self.name = 'fourier'
        assert fftlength > order
    
    def describe(self, image):
        contours = measure.find_contours(image, 0.8)
        final_contour = np.zeros(0)
        for contour in contours:
            if contour.shape[0] > final_contour.shape[0]:
                final_contour = contour 
        
        if final_contour.shape[0] < self.fftlength: 
            return np.ones(self.order)*np.inf # an image with very small or no contour!
        
        final_contour = final_contour[0:self.fftlength,0] + \
        final_contour[0:self.fftlength,1]*1j
        coef = np.fft.fft(final_contour[0:self.fftlength])
        return coef[1:self.order+1]
    
    def attributes(self):
        return {'length':self.order}

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
        self.name = 'zernike'
 
    def describe(self, image):
        return mahotas.features.zernike_moments(\
        image, self.radius, degree=self.degree, cm=self.com)[1:]
        # ignore the first moment
        # first moment is the center of the mass
        
    def copy(self):
        return ZernikeMoments(self.radius,self.degree,self.com)
    
    def attributes(self):
        return {'radius':self.radius,'degree':self.degree,'com':self.com}
