from astropy.io import fits
import os 
import pandas as pd 
from astropy.wcs import WCS,FITSFixedWarning
from glob import glob 

def make_a_fits_list_csv(path,save=False):
    """script that make a pandas csv  with all the fits files in a directory. That means it look for all the fits inside all the inside directories, 
    and keywords based on the necessity of the user if the keyword is not in the header fill the space with a None
    path: directory were to search
    save=path+name of where wants to save the files"""
    data=[] 
    keys = ["HIERARCH ESO DPR CATG","HIERARCH ESO ADA POSANG","HIERARCH ESO ADA ABSROT END"
        ,"HIERARCH ESO ADA ABSROT START","RA","DEC","TELESCOP","INSTRUME","OBJECT","DATE","HIERARCH ESO TEL TARG ALPHA"
        ,"HIERARCH ESO TEL TARG DELTA","HIERARCH ESO OBS NAME","HIERARCH ESO OBS PROG ID"]                    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".fits") and "c1" not in file and "c2" not in file:
                full_path = os.path.join(root, file)
                hdulist = fits.open(os.path.join(full_path))
                header = hdulist[0].header  # Access the header of the first HDU
                values = [None if key.replace("HIERARCH ","") not in list(header.keys()) else header[key] for key in keys]
                data.append([full_path,file,*values])
    data_pandas = pd.DataFrame(data,columns=["path","file_name",*keys])
    if save:
        data_pandas.to_csv(f"{save}",index=False)
    return data_pandas.sort_values("DATE")

def get_fits_header_wcs(path):
    fits_file = fits.open(path)#os.path.join('2038_adq','XSHOO.2022-10-01T01_11_07.327.fits')
    header = fits_file[0].header
    wcs = WCS(fits_file[0].header)
    data = fits_file[0].data
    fits_file.close()
    return data,wcs,header#,slit_angle,ra,dec

class AstroImage:
    def __init__(self,path):
        fits_file = fits.open(path)
        self.header = fits_file[0].header
        self.wcs = WCS(fits_file[0].header)
        self.data = fits_file[0].data
        self.category = self.header["HIERARCH ESO DPR CATG"]
    