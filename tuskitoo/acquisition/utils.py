from astropy.io import fits
import os 
import pandas as pd 
from astropy.wcs import WCS,FITSFixedWarning
from glob import glob 
from astropy.coordinates import Angle
from regions import RectangleSkyRegion
from astropy import units as u
import numpy as np 
from photutils.detection import DAOStarFinder, find_peaks
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord,Angle

# def make_a_fits_list_csv(path,save=False,add_eso_keys=False,add_4most_keys =False):
#     """script that make a pandas csv  with all the fits files in a directory. That means it look for all the fits inside all the inside directories, 
#     and keywords based on the necessity of the user if the keyword is not in the header fill the space with a None
#     path: directory were to search
#     save=path+name of where wants to save the files"""
#     data=[] 
#     keys = ["OBJECT","RA","DEC","TELESCOP","INSTRUME","DATE"]
#     if add_eso_keys:
#         keys +=  ["HIERARCH ESO DPR CATG","HIERARCH ESO ADA POSANG","HIERARCH ESO ADA ABSROT END"
#         ,"HIERARCH ESO ADA ABSROT START","RA","DEC","TELESCOP","INSTRUME","OBJECT","DATE","HIERARCH ESO TEL TARG ALPHA"
#         ,"HIERARCH ESO TEL TARG DELTA","HIERARCH ESO OBS NAME","HIERARCH ESO OBS PROG ID"]
#     # if add_4most_keys:
        
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if file.endswith(".fits") and "c1" not in file and "c2" not in file:
#                 full_path = os.path.join(root, file)                
#                 hdulist = fits.open(os.path.join(full_path))
#                 header = hdulist[0].header  # Access the header of the first HDU
#                 #print(header["SRVID1"])
#                 values = [None if key.replace("HIERARCH ","") not in list(header.keys()) else header[key] for key in keys]
#                 if add_4most_keys:
#                     _,header = fits.getdata(full_path,header=True)
#                     keys_4most = ["SRVID1","SRVID2","SRVID3"]
#                     values += [None if key.replace("HIERARCH ","") not in list(header.keys()) else header[key] for key in keys_4most]
#                 data.append([full_path,file,*values])
#     if add_4most_keys:
#         keys += keys_4most
#     data_pandas = pd.DataFrame(data,columns=["path","file_name",*keys])
#     if save:
#         data_pandas.to_csv(f"{save}",index=False)
#     return data_pandas.sort_values("DATE")

# import os
# import pandas as pd
# from astropy.io import fits


def make_a_fits_list_csv(path, save=False, add_eso_keys=False, add_4most_keys=False):
    """
    Build a pandas DataFrame with FITS files found recursively inside a directory.

    Parameters
    ----------
    path : str
        Directory where the FITS files will be searched recursively.
    save : str or bool, optional
        Output CSV path. If False, the table is not saved.
    add_eso_keys : bool, optional
        If True, add a set of ESO header keywords.
    add_4most_keys : bool, optional
        If True, read additional 4MOST keywords from the extension header.

    Returns
    -------
    pandas.DataFrame
        Table with file paths, names, and selected header keywords.
    """

    base_keys = ["OBJECT", "RA", "DEC", "TELESCOP", "INSTRUME", "DATE"]

    eso_keys = [
        "HIERARCH ESO DPR CATG",
        "HIERARCH ESO ADA POSANG",
        "HIERARCH ESO ADA ABSROT END",
        "HIERARCH ESO ADA ABSROT START",
        "HIERARCH ESO TEL TARG ALPHA",
        "HIERARCH ESO TEL TARG DELTA",
        "HIERARCH ESO OBS NAME",
        "HIERARCH ESO OBS PROG ID",
    ]

    keys_4most = ["SRVID1", "SRVID2", "SRVID3"]

    # Avoid duplicates while preserving order
    keys = list(dict.fromkeys(base_keys + (eso_keys if add_eso_keys else [])))

    data = []

    for root, _, files in os.walk(path):
        for file in files:
            if not file.endswith(".fits"):
                continue
            if "c1" in file or "c2" in file:
                continue

            full_path = os.path.join(root, file)

            try:
                # Read headers only, not data
                primary_header = fits.getheader(full_path, ext=0)

                row = [full_path, file]
                row.extend(primary_header.get(key, None) for key in keys)

                if add_4most_keys:
                    try:
                        ext1_header = fits.getheader(full_path, ext=1)
                        row.extend(ext1_header.get(key, None) for key in keys_4most)
                    except Exception:
                        row.extend([None] * len(keys_4most))

                data.append(row)

            except Exception as e:
                print(f"Skipping {full_path}: {e}")

    final_keys = keys + (keys_4most if add_4most_keys else [])
    data_pandas = pd.DataFrame(data, columns=["path", "file_name", *final_keys])

    if "DATE" in data_pandas.columns:
        data_pandas = data_pandas.sort_values("DATE", kind="stable")

    if save:
        data_pandas.to_csv(save, index=False)

    return data_pandas

def get_fits_header_wcs(path):
    fits_file = fits.open(path)#os.path.join('2038_adq','XSHOO.2022-10-01T01_11_07.327.fits')
    header = fits_file[0].header
    wcs = WCS(fits_file[0].header)
    data = fits_file[0].data
    fits_file.close()
    return data,wcs,header#,slit_angle,ra,dec

def get_image_inclination(sky_pointing,wcs):
    "retrive angle in radians"
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


def get_objects_in_image(ccd,fwhm=5,threshold=5.):
    "in an image it get the objects that can eb observe in it"
    data_cutout,wcs_cutout = ccd.data,ccd.wcs
    mean, median, std = np.mean(data_cutout), np.median(data_cutout), np.std(data_cutout)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    sources = daofind(data_cutout-median)
    coords_sky = wcs_cutout.pixel_to_world(sources['xcentroid'],sources['ycentroid'])
    coords_pixel = wcs_cutout.world_to_pixel(coords_sky)
    
    return {"coords_sky":coords_sky,"coords_pix":np.array(coords_pixel)}#,np.array(coords_pixel)

def get_gaia_cone(coord_center,radius=200):
    radius = radius* u.arcsec
    Gaia.ROW_LIMIT = -1
    j = Gaia.cone_search_async(coord_center, radius=radius)
    gaia_table = j.get_results()
    gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.deg, u.deg))
    return gaia_coords


class AstroImage:
    def __init__(self,path):
        fits_file = fits.open(path)
        self.header = fits_file[0].header
        self.wcs = WCS(fits_file[0].header)
        self.data = fits_file[0].data
        self.category = self.header["HIERARCH ESO DPR CATG"]
    