from astropy.io import fits
import copy 
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import numpy as np
import warnings
from .utils import find_signal,guess_picks_image,gaussian_with_error,integrated_gaussian,integrated_moffat,moffat_with_error
from .fitting import parallel_fit
from tuskitoo.utils.utils import sigma_clip_1d
from scipy.signal import savgol_filter
import pandas as pd 
import pickle



class Expectra2D:
    "Main class to handle 2D spectra and extract the spectra"
    
    def __init__(self,object,center_cut = None,size_cut=None,distances=None,verbose=False,header=None,apply_cut=True,**kwargs):
        """
        Initialize the Expectra_2D class.
        
        Parameters:
        -----------
        object : str or array-like
            The input data for the spectra. If a string ending with 'fits', it is treated as a filepath 
            to a FITS file. Otherwise, if a 2D numpy array, it is used directly.
        center_cut : int or None, optional, default=None
            The center position (row index) for cutting the 2D image. If None, the center is estimated.
        size_cut : int or None, optional, default=None
            The size of the cut-out region. If None, a default value (40) is used.
        distances : optional
            Not used in the current version but may be intended for future use (e.g., spatial calibration).
        verbose : bool, optional, default=False
            If True, prints additional debugging information.
        header : dict or None, optional, default=None
            Header information, typically from a FITS file.
        kwargs : dict
            Additional keyword arguments. In particular, can include:
                - band: instrument band information (e.g., "NIR", "VIS", "UVB")
                - name: object name
        """
        self.band = kwargs.get("band",None) #none or ""?
        self.name = kwargs.get("name",None) 
        self.header = header 
        if isinstance(object,str) and  object.endswith("fits"):
            print(object)
            self.object = object
            self.fits_image = fits.open(object,center_cut=None,size_cut=None)
            if len(self.fits_image)>=3:
                print("Fits image has a len bigger than 1 be aware of in what layer is the image")
                self.original_data,self.header = self.fits_image[0].data,self.fits_image[0].header
                self.original_error = self.fits_image[1].data
                self.original_quality = self.fits_image[2].data
            elif len(self.fits_image)==1:
                self.original_data,self.header = self.fits_image[0].data,self.fits_image[0].header
                #self.original_data = self.fits_image[0].data
        elif isinstance(object,np.ndarray) and len(object.shape)==2:
            self.object = 'no_name'
            print("Object is a numpy array you can also add the Header later")
            self.original_data = np.nan_to_num(self.object,0)
        else:
            raise Exception("Check if is a fits file or numpy array-len(shape) = 2")
        self.get_header_keys()
        if not hasattr(self, 'original_error'):
            self.original_error = np.ones_like(self.original_data)
        if not hasattr(self, 'original_quality'):
            self.original_quality = np.zeros_like(self.original_data)
        if self.original_data.shape[1] < self.original_data.shape[0]:
            self.original_data = self.original_data.T
            self.original_quality = np.zeros_like(self.original_data).T
            self.original_error = np.ones_like(self.original_data).T
        #self.center_cut = center_cut or self.original_data.shape[0]//2 
        #self.size_cut = size_cut or 40 
        self.cut_data = self.original_data #Expectra2D.cut_2d_image(self.original_data,center=self.center_cut,size=size_cut,verbose=True)
        self.cut_error = self.original_error #Expectra2D.cut_2d_image(self.original_error,center=self.center_cut,size=size_cut,verbose=False)
        self.cut_quality = self.original_quality #Expectra2D.cut_2d_image(self.original_quality,center=self.center_cut,size=size_cut,verbose=False)
        
        if apply_cut:
            self.center_cut = center_cut or self.original_data.shape[0]//2 
            self.size_cut = size_cut or 40 
            self.cut_data = Expectra2D.cut_2d_image(self.original_data,center=self.center_cut,size=size_cut,verbose=True)
            self.cut_error = Expectra2D.cut_2d_image(self.original_error,center=self.center_cut,size=size_cut,verbose=False)
            self.cut_quality = Expectra2D.cut_2d_image(self.original_quality,center=self.center_cut,size=size_cut,verbose=False)
        
        self.stacked_median = np.nanmedian(self.cut_data,axis=1)
        
    def get_header_keys(self,distances=None):
        """
        Retrieve and store a subset of header keys relevant for further processing.
        
        Parameters:
        -----------
        distances : optional
            If provided as a dictionary, it may be used for additional processing related to distances.
        
        Notes:
        ------
        If no header is available, a warning is issued.
        """
        if not self.header:   
            warnings.warn(
                "Warning: 'self.header' is not defined. "
                "Please add a header to the class to take extra advantage of the code.",
                UserWarning
            )
            return
        self.relevant_keywords_header = {i:self.header[i] for i in ["ORIGIN","INSTRUME","OBJECT","NAXIS1","CRVAL1","CD1_1","CUNIT1","BUNIT","CD2_2","OBJECT","ESO SEQ ARM"] if i in list(self.header.keys()) }
        self.name = self.relevant_keywords_header["OBJECT"]
        self.band = self.relevant_keywords_header.get("ESO SEQ ARM","No info")
        #if self.relevant_keywords_header["CUNIT1"]=="nm": to_angs=10
        #self.original_wavelength =  np.array([(self.relevant_keywords_header["CRVAL1"]+i*self.relevant_keywords_header["CD1_1"])*10 for i in self.original_data.shape[1]])
        #calculate wavelenght here for example  
        # if "BUNIT" in self.relevant_keywords_header and self.relevant_keywords_header['INSTRUME'] == 'EFOSC':
        #     factor = convert_to_float(self.relevant_keywords_header["BUNIT"])
        #     print(f"Corrected by factor={factor} BUNIT")
        #     self.data2d = factor * self.data2d
        # if isinstance(distances,dict):
        #     self.distances_arc = distances
        #     if "CD2_2"  in self.relevant_keywords_header.keys():
        #    
    # def arc_to_pix(self,value):
    #     distances_pix = value/self.relevant_keywords_header["CD2_2"]
    #     return distances_pix
    #     #{key:value/self.relevant_keywords_header["CD2_2"] for key,value in distances.items()}
    
    
    

    @staticmethod
    def cut_2d_image(image,center=None,size=None,verbose=False):
        """
        Cut a 2D image to the specified region.

        Parameters:
        ----------
        image : array-like
            The input 2D image to be cut.
            
        center : int or None, optional, default=None
            The center position for cutting the 2D image. If None, the center will be estimated.
            
        size : int or None, optional, default=None
            The size of the cut-out region. If None, a default size of 70 will be used.
            
        verbose : bool, optional, default=False
            If True, print additional information during processing.

        Returns:
        -------
        array-like
            The cut-out 2D image.
        """
        if image.shape[0]//2 != 0:
            nan_row = np.full((1, image.shape[1]), np.nan)
            # Append the row to the bottom of the image
            image = np.vstack([image, nan_row])
        if not center:
            center = int(np.nanmedian(np.array([find_signal(i) for i in image.T])))
        if not size:
            size = 70 # should be fine as initial value
        if verbose:
            print(f"cut center {center} and cut size {size}")
        return image[int(center-size//2):int(center+size//2),:]
    
    
    def plot_cut_out(self,trace=None,sep=None):
        """
        Plot the 2D cut-out image and the stacked median profile.
        """
        norm_image = self.cut_data/self.cut_data.max(axis=0)
        fig,axs = plt.subplots(1, 2, figsize=(18, 5))
        # Plot data on the first subplot
        im = axs[0].imshow(norm_image,aspect="auto",vmin=0,vmax=1)
        if trace is not None:
            axs[0].plot(trace,lw=2)
            if sep is not None:
                if not isinstance(sep,list):
                    sep = [sep]
                for s in sep:
                    axs[0].plot(trace+s,lw=2)
        axs[0].set_title('2d cut')
        axs[0].set_xlabel('X-pixel')
        axs[0].set_ylabel('Y-pixel')
        
        plt.colorbar(im, ax=axs[0], label="normalized intensity")

        axs[1].plot(np.nanmedian(norm_image,axis=1), color='orange')
        axs[1].set_xlim(np.arange(len(np.nanmedian(norm_image,axis=1)))[[0,-1]])
        axs[1].axhline(0, ls= '--')
        axs[1].set_title('stacked median')
        axs[1].set_xlabel('y-pixels')
        axs[1].set_ylabel('intensity')
        plt.tight_layout()
        plt.show()
    
    ######################################
    def save_fit_keywords_as_pickle(self,filename):
        """
        Save the dictionary of fitting keywords (parameters used in the fit) to a pickle file.
        
        Parameters:
        -----------
        filename : str
            The base filename for saving (without extension).
        """
        try:
            filename = f'{filename}.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(self.keywords_fit, f)
            print(f"Dictionary successfully saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving the dictionary: {e}")
    
    def save_spectra_as_pickle(self,save=None,band=None):
        """_summary_

        Args:
            save (_type_, optional): _description_. Defaults to None.
            band (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        result = self.results['result_panda']
        band = band or self.band
        if band is None:
            band = "?"
            print("Warning band not found")
        dic_result = {}
        for i in self.images:
            band = band.lower()
            dic_result[f"{i}_{band}"] = {"wavelength":result["wavelength"].values,"flux":result[f"flux_{i}"].values,"std":result[f"std_flux_{i}"].values,"band":band}
        if save:
            if len(list(dic_result.keys()))>0:
                with open(f"{save}_{band}.pickle", "wb") as file:
                    print("Save as",f"{save}_{band}.pickle")
                    pickle.dump(dic_result, file)
            else:
                print("Empty dictionary ")
        else:
            return dic_result
    
    def save_to_fits(self,filename,person="F. Avila-Vera"):
        """
        Save the extracted spectra (results) to a FITS file.
        
        Parameters:
        -----------
        filename : str
            The base filename for the FITS file.
        person : str, optional, default="F. Avila-Vera"
            Name of the person responsible for the extraction; stored in the FITS header.
        
        Raises:
        -------
        AttributeError:
            If the results have not been computed (i.e., 'array_to_pandas' has not been run).
        """
        if not hasattr(self, 'results'):
            raise AttributeError(
                "Error: 'self.results' is not defined. \n"
                "Could be an Error in runing 'array_to_pandas'")
        df = self.results['result_panda']
        flux_columns = [i for i in df.columns.values if 'flux' in i.split('_')[0]]
        flux_columns_std = ["std_"+i for i in flux_columns]
        columns_to_save = ["wavelength"] + flux_columns+flux_columns_std
        n_rows = len(df)
        dtype = [(col, '>f4') for col in columns_to_save]
        data = np.empty(n_rows, dtype=dtype)
        for col in columns_to_save:
            data[col] = df[col].values.astype('>f4')
        primary_hdu = fits.PrimaryHDU()
        for key, value in self.header.items():    
            if 'ESO' in key:
                continue
            primary_hdu.header[key] = value
        if isinstance(self.object,str):
            primary_hdu.header["2DFILE"] = self.object
        table_hdu = fits.BinTableHDU(data)
        table_hdu.header["PERSON"] = (person, "who extract")
        # Combine into an HDUList and write to file
        hdul = fits.HDUList([primary_hdu, table_hdu])
        filename = f"{filename}_extracted_spectra.fits"
        hdul.writeto(filename, overwrite=True)
        print(f"FITS file '{filename}' created successfully.")
    #TODO maybe save the keys from the fiting process 
    