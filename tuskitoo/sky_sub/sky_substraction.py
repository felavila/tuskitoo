from astropy.io import fits
import numpy as np 
from .tools import extract_wavelength_array

class skysubstraction:
    def __init__(self, path_images,path_response=None,band="NIR"):
        self.path_images = path_images
        self.path_response = path_response
        self.read_files_to_array()
        
    #def __call__(self, img):
     #   return img - self.path
    
    def read_files_to_array(self):
        self.images = np.array([fits.getdata(name) for name in self.path_images ])
        self.wavelenghts = np.array([extract_wavelength_array(name) for name in self.path_images ])
        self.headers =  [fits.getheader(name) for name in self.path_images ]
    
    