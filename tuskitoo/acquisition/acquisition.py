from copy import deepcopy
from photutils.detection import DAOStarFinder, find_peaks
from regions import RectangleSkyRegion
from astropy.wcs import WCS,FITSFixedWarning
import numpy as np 
from astropy.coordinates import SkyCoord,Angle
from astropy import units as u
from astropy.nddata.utils import Cutout2D
import matplotlib.pyplot as plt 
from .ploting import arrow_plot,plot_image_cut
from astroquery.gaia import Gaia
import warnings
from reproject import reproject_interp
from astropy.coordinates import Angle

warnings.simplefilter("ignore", category=FITSFixedWarning)

def get_objects_in_image(cutout2D,fwhm=5,threshold=5.):
    #
    data_cutout,wcs_cutout = cutout2D.data,cutout2D.wcs
    mean, median, std = np.mean(data_cutout), np.median(data_cutout), np.std(data_cutout)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind(data_cutout-median)
    coords_sky = wcs_cutout.pixel_to_world(sources['xcentroid'],sources['ycentroid'])
    #coords_pixel = wcs_cutout.world_to_pixel(coords_sky)
    return coords_sky#,np.array(coords_pixel)

def get_gaia_cone(coord_center):
    radius = 200* u.arcsec
    Gaia.ROW_LIMIT = -1
    j = Gaia.cone_search_async(coord_center, radius=radius)
    gaia_table = j.get_results()
    gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.deg, u.deg))
    return gaia_coords
def get_image_inclination(sky_pointing,wcs):
    #
    inclination = Angle(0, 'deg')
    sky_reg = RectangleSkyRegion(center=sky_pointing,
                                    width=1 * u.arcsec, height=1 * u.arcsec,
                                    angle=inclination)
    pix_reg = sky_reg.to_pixel(wcs)
    inclination = Angle(-pix_reg.angle, 'deg')
    sky_reg = RectangleSkyRegion(center=sky_pointing,
                                    width=1 * u.arcsec, height=1 * u.arcsec,
                                    angle=inclination)
    angle_region = pix_reg.angle.to_value("rad")
    return angle_region

def mach_photo_gaia(coords_sky,gaia_coords,match_threshold=0):
    #print(coords_sky)
    idx, d2d, d3d = coords_sky.match_to_catalog_sky(gaia_coords)
    match_threshold = match_threshold * u.arcsec  # Adjust as needed
    print(d2d)
    good_matches = d2d > 0#match_threshold
    gaia_positions = np.column_stack((gaia_coords.ra[idx][good_matches].value, gaia_coords.dec[idx][good_matches].value))
    coords_sky = np.column_stack((coords_sky.ra[good_matches].value, coords_sky.dec[good_matches].value))
    ra_offset = np.nanmedian(coords_sky[:,0]-gaia_positions[:,0])
    dec_offset = np.nanmedian(coords_sky[:,1]-gaia_positions[:,1])
    if np.isnan(ra_offset):
        print("ra_offset is nan ",ra_offset,dec_offset)
        ra_offset = 0
        dec_offset = 0 
    return ra_offset,dec_offset,idx,good_matches

def cut_astro_image(image,wcs,header,cut_size = 40,fwhm=5,threshold=5,plot=False,gaia_coords=None,match_threshold=1,coordinates_images_sky=None):
    sky_pointing = SkyCoord(header["RA"],header["DEC"], unit='deg')
    height_cut,width_cut = cut_size * u.arcsec, cut_size* u.arcsec
    cutout_2d = Cutout2D(image,sky_pointing,size=(height_cut,width_cut),wcs=wcs)
    coords_sky,coords_pixel = get_objects_in_image(cutout_2d,fwhm=fwhm)
    angle_region = get_image_inclination(sky_pointing,wcs)
    if gaia_coords:
        ra_offset,dec_offset,idx,good_matches=mach_photo_gaia(coords_sky,gaia_coords,match_threshold=match_threshold)
        header_copy = deepcopy(header)
        header_copy['CRVAL1'] -= ra_offset
        header_copy['CRVAL2'] -= dec_offset
        wcs_copy = WCS(header_copy)
        #coords_pixel_new_wcs = np.array(coords_sky.to_pixel(wcs_copy))
        cutout_2d_deshift = Cutout2D(image,sky_pointing, (height_cut,width_cut), wcs=wcs_copy)
        gaia_pixel_positions_deshift = np.column_stack(cutout_2d_deshift.wcs.world_to_pixel(gaia_coords[idx][good_matches]))
        gaia_pixel_positions_c = np.column_stack(cutout_2d.wcs.world_to_pixel(gaia_coords[idx][good_matches]))
        #print(ra_offset,dec_offset)
        if not plot:
            if coordinates_images_sky:
                dic_images = {}
                for key,values in coordinates_images_sky.items():
                    coord = cutout_2d_deshift.wcs.world_to_pixel(values)
                    dic_images[key] = coord
            return coords_sky,cutout_2d,cutout_2d_deshift,angle_region,dic_images
    if plot:
        if gaia_coords:
            fig = plt.figure(figsize=(20, 10))
            axis1 = fig.add_subplot(1, 2, 1, projection=cutout_2d.wcs)
            axis2 = fig.add_subplot(1, 2, 2, projection=cutout_2d_deshift.wcs)
            axis1.imshow(np.log10(cutout_2d.data),origin="lower", cmap=plt.cm.viridis)
            axis2.imshow(np.log10(cutout_2d_deshift.data),origin="lower", cmap=plt.cm.viridis)    
            arrow_plot(axis1,cutout_2d.data,angulo_radianes=angle_region)
            arrow_plot(axis2,cutout_2d_deshift.data,angulo_radianes=angle_region)
            for i in gaia_pixel_positions_c:
                axis1.scatter(*i,color="k")
            for i in gaia_pixel_positions_deshift:
                axis2.scatter(*i,color="k")
            if coordinates_images_sky:
                dic_images = {}
                for key,values in coordinates_images_sky.items():
                    coord = cutout_2d_deshift.wcs.world_to_pixel(values)
                    axis2.scatter(*coord,color="w")
                    axis2.text(*coord,key)
                    dic_images[key] = coord
                #plt.legend()
            plt.show()
            return coords_sky,cutout_2d,cutout_2d_deshift,angle_region,dic_images
    return coords_sky,cutout_2d

class AcquisitionClass:#ACQUISITION
    def __init__(self,image,header,cut_size = 40,fwhm=5,threshold=5,plot=False,gaia_coords=None,match_threshold=1,coordinates_images_sky=None):
        self.data=image
        self.header=header
        self.wcs= WCS(header)
        self.cut_size=cut_size
        self.sky_pointing = SkyCoord(self.header["RA"],self.header["DEC"], unit='deg')
        ra,dec = str(self.header["HIERARCH ESO TEL TARG ALPHA"]),str(self.header["HIERARCH ESO TEL TARG DELTA"])
        if len(ra.split(".")[0])<6:
            ra = "0" + ra
        if len(dec.split(".")[0])<6:    
            dec = "0" + dec
        #print(self.sky_pointing )
        #self.sky_pointing = SkyCoord(f"{ra[0:2]} {ra[2:4]} {ra[4:]}",f"{dec[0:3]} {dec[3:5]} {dec[5:]}", unit=(u.hourangle, u.deg),frame="fk5")
        self.angle_region = get_image_inclination(self.sky_pointing,self.wcs)

    
    def cut2d(self,center_correction=False,cut_size=None,sky_pointing=None,plot=False,wcs=None,fwhm=5,threshold=5,match_threshold=1):
        wcs = wcs or self.wcs
        cut_size= cut_size or self.cut_size
        print(cut_size)
        center_cut = sky_pointing or self.sky_pointing
        cutout2d = Cutout2D(self.data,center_cut, (cut_size * u.arcsec, cut_size* u.arcsec),wcs=wcs)
        sky_coords = None
        gaia_coords = None
        cutout2d_previous = None
        sky_coords = get_objects_in_image(cutout2d,fwhm=fwhm,threshold=threshold)
        if center_correction:
            gaia_coords = get_gaia_cone(center_cut)
            ra_offset,dec_offset,idx,good_matches = mach_photo_gaia(sky_coords,gaia_coords,match_threshold=match_threshold)
            #print(ra_offset,dec_offset)
            header_copy = deepcopy(self.header)
            header_copy['CRVAL1'] -= ra_offset
            header_copy['CRVAL2'] -= dec_offset
            wcs_copy = WCS(header_copy)
            #coords_pixel_new_wcs = np.array(coords_sky.to_pixel(wcs_copy))
            cutout2d_previous = cutout2d
            cutout2d = Cutout2D(self.data,center_cut, (cut_size * u.arcsec, cut_size* u.arcsec),wcs=wcs_copy)
        
        #print(gaia_coords)
        
        if plot:
            title = "Original image"
            if cutout2d_previous:
                title = "Corrected"
                plot_image_cut(self,cutout2d_previous,sky_coords=sky_coords,gaia_coords=gaia_coords,title="Not Corrected")
            plot_image_cut(self,cutout2d,sky_coords=None,gaia_coords=gaia_coords,title=title)
        self.cutout2d = cutout2d
        x0, y0 = self.cutout2d.data.shape[0]//2,self.cutout2d.data.shape[1]//2
        print(x0, y0 )
        self.center_image_world = cutout2d.wcs.pixel_to_world(x0, y0 )
        self.sky_coords = sky_coords
    
    
    
    def slit_plot(self,cut_size=None,dic_images=None,height_slit=11,width_slit=1.2,inclination_slit=None
                  ,target_center=None,title=None,zoom=40,slit_dic=None,slit_to_plot=None,fwhm=5,threshold=5):
        #target_center skycoord
        #inclination_slit in angle  Angle(inclination_slit, 'deg')
        cutout2d = self.cutout2d#always will require call this first.
        dict_plot = {}
        height_slit,width_slit = height_slit * u.arcsec,width_slit* u.arcsec
        #cut_size= cut_size or self.cut_size
        fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': cutout2d.wcs})
        dict_plot["fig"] = (fig, ax)
        sky_coords = get_objects_in_image(cutout2d,fwhm=fwhm,threshold=threshold)
        ax.imshow(np.log10(cutout2d.data),origin="lower", cmap="grey")
        if slit_to_plot:
            edge_colors_list = ["r","g"]
            for n,(key,val) in enumerate(slit_to_plot.items()):
                target_center = val["target_center_header"]
                #print(target_center)
                inclination_slit = val["inclination_slit_header"]
                sky_slit = RectangleSkyRegion(center=target_center,width=width_slit, height=height_slit,angle=inclination_slit,)
                pix_reg = sky_slit.to_pixel(cutout2d.wcs)
                pix_reg.plot(ax=ax, facecolor="none", edgecolor=edge_colors_list[n],lw=2, label=f"position {n+1}",)     
        if dic_images:
            for key,values in dic_images.items():
                values = cutout2d.wcs.world_to_pixel(values)
                #ax.scatter(*values,color="k")
                if "G" in key:
                    ax.text(*values,key,fontsize=20,c="tab:orange", fontweight="bold")
                else:
                    ax.text(*values,key,fontsize=20,c="tab:purple", fontweight="bold")
            x0, y0 = cutout2d.wcs.world_to_pixel(sky_coords)
            ax.scatter(x0, y0)
        if len(ax.get_legend_handles_labels()[0])>0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
        if target_center is not None:
            x0, y0 = cutout2d.wcs.world_to_pixel(target_center)
            zoom_x = zoom
            zoom_y = zoom
            ax.set_xlim(x0 - zoom_x, x0 + zoom_x)
            ax.set_ylim(y0 - zoom_y, y0 + zoom_y)
        ax.set_title(title)
        arrow_plot(ax,angulo_radianes=self.angle_region,color="r")
        ax.set_xlabel("RA", fontsize=16)
        ax.set_ylabel("Dec", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid()
        plt.show()
        
        return ax,dict_plot
        
        
    def slit_plot_old(self,header,cut_size=None,dic_images=None):
        #TODO add condition of only accept slit images to do this plot 
        #TODO the cutout never can be done here? always you have to choose it before 
        cut_size= cut_size or self.cut_size
        ra,dec = str(header["HIERARCH ESO TEL TARG ALPHA"]),str(header["HIERARCH ESO TEL TARG DELTA"])
        if len(ra.split(".")[0])<6:
            ra = "0" + ra
        if len(dec.split(".")[0])<6:    
            dec = "0" + dec
        target_ = SkyCoord(f"{ra[0:2]} {ra[2:4]} {ra[4:]}",f"{dec[0:3]} {dec[3:5]} {dec[5:]}", unit=(u.hourangle, u.deg),frame="fk5")
        
        #Cutout2D(self.data,self.sky_pointing, (cut_size * u.arcsec, cut_size* u.arcsec), wcs=wcs_copy)
        fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': cutout2d.wcs})
        ax.imshow(np.log10(cutout2d.data),origin="lower", cmap=plt.cm.viridis)
        arrow_plot(ax,cutout2d.data,angulo_radianes=self.angle_region)
        header_slit_angle = header["HIERARCH ESO ADA POSANG"]
        title = header['HIERARCH ESO OBS NAME']
        if dic_images:
            for key,values in dic_images.items():
                values = cutout2d.wcs.world_to_pixel(values)
                ax.scatter(*values,color="w")
                ax.text(*values,key)
        #the header_slit_angle is - the sky angle
        inclination_slit = Angle(-header_slit_angle, 'deg')
        height_slit,width_slit = 11 * u.arcsec,1.2* u.arcsec
        sky_slit = RectangleSkyRegion(center=target_,width=width_slit, height=height_slit,angle=inclination_slit)
        pix_reg = sky_slit.to_pixel(cutout2d.wcs)
        pix_reg.plot(ax=ax, facecolor='none', edgecolor='r', lw=2,
                    label='slit')
        ax.set_title(title)
        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()
    
    def plot_n_e(self,dic_images=None):
        cutout2d = self.cutout2d
        header_2=deepcopy(self.cutout2d.wcs.to_header())
        #header_2['PC1_1'] = -4.6811e-05
        header_2['PC1_2'] = 0.0
        header_2['PC2_1'] = 0.0
        #header_2['PC2_2'] = 4.6811e-05
        new_wcs = WCS(header_2)
        
        array, _ = reproject_interp((cutout2d.data,cutout2d.wcs),header_2,shape_out = cutout2d.data.shape)
        self.array = array
        fig, ax1 = plt.subplots(figsize=(10,10),subplot_kw={'projection': new_wcs})
        ax1.imshow(np.log10(array))
        ax1.coords['ra'].set_axislabel('Right Ascension')
        ax1.coords['dec'].set_axislabel('Declination')
        if dic_images:
            for key,values in dic_images.items():
                values = new_wcs.world_to_pixel(values)
                ax1.scatter(*values,color="w")
                ax1.text(*values,key)
        arrow_plot(ax1,array,angulo_radianes=0)
        #ax1.set_title('Reprojected MSX band E image')
        ax1.grid()
        plt.show()
        
    
    