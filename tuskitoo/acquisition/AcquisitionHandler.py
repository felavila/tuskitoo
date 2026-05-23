from copy import deepcopy
import numpy as np 
import matplotlib.pyplot as plt 

from astroquery.gaia import Gaia

from astropy.wcs import WCS,FITSFixedWarning
from astropy.coordinates import SkyCoord,Angle
from astropy import units as u
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import Angle
from astropy.nddata import CCDData
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs import WCS

from photutils.detection import DAOStarFinder, find_peaks
from regions import RectangleSkyRegion
from reproject import reproject_interp

import warnings

from .ploting import arrow_plot,plot_image_cut
from tuskitoo.acquisition.utils import get_image_inclination,get_objects_in_image,get_gaia_cone

warnings.simplefilter("ignore", category=FITSFixedWarning)


class AcquisitionHandler:#ACQUISITION
    def __init__(self,image,header,cut_size = 40,fwhm=5,threshold=5,plot=False,gaia_coords=None,match_threshold=1,coordinates_images_sky=None):
        
        self.header = header
        wcs= WCS(header)
        data = image
        self.ccd = CCDData(data,unit="adu",wcs = wcs)
        
        self.sky_pointing = SkyCoord(self.header["RA"],self.header["DEC"], unit='deg')
        self.cut_size=cut_size
        ra,dec = str(self.header["HIERARCH ESO TEL TARG ALPHA"]),str(self.header["HIERARCH ESO TEL TARG DELTA"])
        if len(ra.split(".")[0])<6:
            ra = "0" + ra
        if len(dec.split(".")[0])<6:    
            dec = "0" + dec
        #print(self.sky_pointing )
        #self.sky_pointing = SkyCoord(f"{ra[0:2]} {ra[2:4]} {ra[4:]}",f"{dec[0:3]} {dec[3:5]} {dec[5:]}", unit=(u.hourangle, u.deg),frame="fk5")
        self.angle_region = get_image_inclination(self.sky_pointing,self.ccd.wcs)
        self.pixel_scale =  proj_plane_pixel_scales(self.ccd.wcs) * 3600  # arcsec/pixel
        #print(self.pixel_scale * min(self.ccd.data.shape))
        self.gaia_coords = get_gaia_cone(self.sky_pointing,radius = min(self.pixel_scale * self.ccd.data.shape)//2)
        
    
    def plot_image(self,add_images=True,fwhm=5,threshold=5,add_gaia_points=True):
        #this can be just a part of a major function
        fig = plt.figure(figsize=(25, 10))
        ax1 = fig.add_subplot(1, 1, 1, projection=self.ccd.wcs)
        ax1.imshow(np.log10(self.ccd.data),origin="lower", cmap=plt.cm.viridis)
        if add_images:
            self.coords_objs = get_objects_in_image(self.ccd,fwhm=fwhm,threshold=threshold)
            ax1.scatter(*self.coords_objs["coords_pix"],c="k")
        if add_gaia_points: #(only matched ones..)
            self.gaia_coords_pixel = np.array(self.ccd.wcs.world_to_pixel(self.gaia_coords))
            ax1.scatter(*self.gaia_coords_pixel,c="r",label="gaia")
        ax1.coords['ra'].set_axislabel('Right Ascension')
        ax1.coords['dec'].set_axislabel('Declination')
        ax1.coords['ra'].set_axislabel('Right Ascension')
        ax1.coords['dec'].set_axislabel('Declination')
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel(ax1.get_ylabel(),     fontsize=20)
        ax1.set_xlabel(ax1.get_xlabel(), fontsize=20)
        ax1.set_ylabel(ax1.get_xlabel(),     fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        plt.legend()
        plt.show()
    
        
    def calculate_objecs(self,fwhm=5,threshold=5):
        self.coords_objs = get_objects_in_image(self.ccd,fwhm=fwhm,threshold=threshold)
        

            
