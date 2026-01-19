#import astropy
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

def df_get(df, key, default=None):
    return df[key] if key in df.columns else default

#from .spectra_extraction_results import spectral_extraction_results_handler

#Change all to object dosent sound to bad (?)
class Expectra2D:
    "Main class to handle 2D spectra and extract the spectra"
    
    def __init__(self,object,center_cut = None,size_cut=None,distances=None,verbose=False,header=None,**kwargs):
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
            self.object = 'mmm'
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
    def arc_to_pix(self,value):
        distances_pix = value/self.relevant_keywords_header["CD2_2"]
        return distances_pix
        #{key:value/self.relevant_keywords_header["CD2_2"] for key,value in distances.items()}
    

    def run_parallel_fit(self,n_picks=2,pixel_limit=[],bound_sigma=[2],distribution="gaussian",
                        param_value=None,param_limit=None,param_fix=None,no_use_real_error=False,initial_separation=[],initial_center=None,**kwargs):
        """
        Run the parallel fitting process on the instance's image data.

        This function prepares the fitting parameters based on the instance attributes,
        defines masks based on the instrument band, and calls `parallel_fit` to perform
        the actual parallel fitting. It also stores the local parameters used for fitting
        in the attribute `keywords_fit` and the final results in `fit_result`.
        TODO add the init_trace condition.
        
        Parameters:
        -----------
        n_picks : int, optional, default=2
            Number of sources (or picks) to consider in the fitting. This should match the number
            of distinct peaks expected in the data.
        pixel_limit : list or tuple, optional, default=[]
            Pixel (column) limits to process. Example: [start_column, end_column]. If empty, all columns are processed.
        bound_sigma : list, optional, default=[2]
            List of component indices for which the sigma value should be bounded to that of the first component.
        distribution : str, optional, default="gaussian"
            Type of distribution to use for fitting. Options include "gaussian" and "moffat".
        param_value : dict or None, optional, default=None
            Dictionary of initial parameter values. For example:
                {
                    "height_1": 10.0,
                    "sigma_1": 2.0,
                    "center_1": 150.0
                }
            These values provide starting points for the fitting algorithm.
        param_limit : dict or None, optional, default=None
            Dictionary of limits (min, max) for parameters. For example:
                {
                    "sigma_1": (0.1, 5.0),
                    "center_1": (100, 200)
                }
            This restricts the range over which parameters can be optimized.
        param_fix : list or None, optional, default=None
            List of parameter names to be fixed (kept constant) during fitting. For example:
                ["height_1", "center_1"]
            Parameters listed here will not be varied during the optimization process.
        no_use_real_error : bool, optional, default=False
            If True, the function uses a constant error (i.e., ones) instead of the real error provided.
        initial_separation : list, optional, default=[]
            Initial guess for the separation between the sources/components. Must have a length corresponding to n_picks - 1.
        initial_center : float or None, optional, default=None
            Initial guess for the center position. If None, the function estimates it from the data.

        Examples:
        ---------
        >>> # Example: Fix the height of the first source and set an initial value for sigma.
        >>> param_fix_example = ["height_1"]
        >>> param_value_example = {"sigma_1": 2.0, "center_1": 150.0}
        >>> self.run_paralel_fit(n_picks=2, pixel_limit=[0, 1024], bound_sigma=[2],
        ...                      distribution="gaussian", param_fix=param_fix_example,
        ...                      param_value=param_value_example, init_separation=[20], init_center=150)
        
        Returns:
        --------
        None
            The results are stored in the instance attributes `keywords_fit` and `fit_result`.
        """
        init_trace = kwargs.get("init_trace")
        if n_picks>1:
            picks=np.array([guess_picks_image(i,n_picks) for i in self.cut_data.T])
            if not initial_center:
                print('Given a init_center was not added we will guess one')
                initial_center = np.nanmedian(picks[:,0])
            if len(initial_separation) != n_picks-1:
                print('Given a init_separation  was not added we will guess it')
                initial_separation = np.nanmedian(picks,axis=0)[1:] - initial_center
        if n_picks ==1 and not initial_separation:
            initial_center = np.argmax(np.nanmedian(self.cut_data,axis=1))
            initial_separation = []
        print("initial_center:",initial_center,"initial_separation:",initial_separation)
        if isinstance(initial_separation,(float,int)):
            initial_separation = [initial_separation]
        band = kwargs.get("band",self.band)
        if band == "NIR":
            mask_list=[[5800,7005],[13500,15900]] #teluric
        elif band =="VIS":
            mask_list = [[0,1000],[int(self.cut_data.shape[1]-50),int(self.cut_data.shape[1]-1)]]
        elif band =="UVB":
            mask_list = [[0,500]]
        else:
            mask_list = []
        #guess_separation how to work with something like this?
        # guess_separation
        #print(self.cut_data.shape,self.cut_error.shape)
        #self.wavelength =  np.array([(self.relevant_keywords_header["CRVAL1"]+i*self.relevant_keywords_header["CD1_1"])*unit_factor for i in self.cleaned_panda["n_pixel"].values])
        error = self.cut_error
        data = self.cut_data
        
        if no_use_real_error:
            error = np.ones_like(self.cut_data)
        self.keywords_fit = locals() #maybe add some "remove keys"
        self.keywords_fit.pop("self")
        if 'picks' in self.keywords_fit.keys():
            self.keywords_fit.pop('picks')
        self.fit_result = parallel_fit(data,error,n_picks,initial_center=initial_center,initial_separation=initial_separation,pixel_limit=pixel_limit,bound_sigma=bound_sigma,distribution=distribution,mask_list=mask_list,\
                        param_value=param_value,param_limit=param_limit,param_fix=param_fix,init_trace =  init_trace)
        
        
        #TODO will be necesary add the self.header but with a non usefull variable?
        #self.name,self.band,self.header?
        #self.serh_1_nir=spectral_extraction_results_handler(full_result_step_1_nir,conditions={"rsquared":0.8},header=self.header,band=self.band,name=self.name,names,wavelength)
    
    
    def array_to_pandas(self,max_iter=5,sigma=2,region_size=20,over_write = False,images=[] ):
        """
        Convert the fitting results into a pandas DataFrame.
        
        Processes the output of the parallel fit, applies sigma clipping,
        and organizes the results into a DataFrame for further analysis or plotting.
        
        Parameters:
        -----------
        max_iter : int, optional, default=5
            Maximum number of iterations for sigma clipping.
        sigma : float, optional, default=2
            Sigma threshold for sigma clipping.
        region_size : int, optional, default=20
            Size of the region to be considered in the sigma clipping routine.
        over_write : bool, optional, default=False
            If True, overwrites any existing results in the instance attribute.
        images : list, optional, default=[]
            Optional list of image names corresponding to the different sources.
        
        Returns:
        --------
        DataFrame or dictionary
            Returns the DataFrame if not overwriting; otherwise, the results are stored in the instance.
        """
        results = self.fit_result
        name_params = results.get("name_params")
        num_source  = results.get("num_source")
        distribution = results.get("distribution")
        image_shape = results.get("normalized_image").shape
        num_parameter = results.get("parameter_number")
        normalize_matrix = results.get("normalize_matrix")
        values = results.get('value').copy()
        std = results.get('std').copy()
        dist_func =  gaussian_with_error if distribution=="gaussian" else moffat_with_error
        int_func = integrated_gaussian if distribution=="gaussian" else integrated_moffat
        flux_columns =[f"flux_{n}" for n in range(1,num_source+1)]
        extra_columns = ["chisqr","redchi","aic","bic","rsquared","n_pixel","x_num"]
        result_panda = pd.DataFrame()
        result_panda[["value_"+i if not "height" in i else "value_norm_"+i for i in name_params]] = values
        result_panda[["std_"+i if not "height" in i else "std_norm_"+i for i in name_params]] = std
        result_panda[extra_columns] = results.get("extra_params")
        values[:,["height" in i for i in name_params]] = values[:,["height" in i for i in name_params]] * normalize_matrix
        std[:,["height" in i for i in name_params]] = std[:,["height" in i for i in name_params]] * normalize_matrix
        if any("separation" in i for i in result_panda.columns):
            sep_to_cen = result_panda["value_center_1"].values[:,None] + result_panda[[i for i in result_panda.columns if "value_separation" in i ]].values
            std_sep_to_cen = np.sqrt(result_panda["std_center_1"].values[:,None]**2 + result_panda[[i for i in result_panda.columns if "std_separation" in i ]].values**2)
            result_panda[[f"value_center_{i}" for i in range(1,num_source+1) if i!=1]] = sep_to_cen#result_panda["value_center_1"].values[:,None] - result_panda[[i for i in result_panda.columns if "value_separation" in i ]].values
            result_panda[[f"std_center_{i}" for i in range(1,num_source+1) if i!=1]] = std_sep_to_cen
            values[:,["separation" in i for i in name_params]] = sep_to_cen
            std[:,["separation" in i for i in name_params]] = std_sep_to_cen
        re_shape_results_m = np.concatenate((values.reshape(-1, num_source, num_parameter),std.reshape(-1, num_source, num_parameter)),axis=2)
        multiple_dist,error_dist = dist_func(np.arange(0,image_shape[0])[:, np.newaxis, np.newaxis],*re_shape_results_m.T)
        multiple_dist = np.nan_to_num(np.moveaxis(multiple_dist,0,1),0)  #* normalize_matrix.T
        error_dist = np.nan_to_num(np.moveaxis(error_dist,0,1),0) #* normalize_matrix.T
        image_2d_model = multiple_dist.sum(axis=0) #* normalize_matrix.T
        fluxes,errors =  int_func(*re_shape_results_m.T) #* normalize_matrix.T
        result_panda[['raw_'+i for i in flux_columns]] = fluxes.T
        result_panda[['std_'+i for i in flux_columns]] = errors.T
        result_panda[ flux_columns] = np.array([sigma_clip_1d(result_panda['raw_'+i].values,max_iter=max_iter,sigma=sigma,region_size=region_size,error=result_panda['std_'+i].values) for i in [i for i in flux_columns]]).T
        result_panda['units_flux'] = len(result_panda) * ["flux"]
        errors[errors>fluxes] = 0 
        #result_panda[['std_'+i for i in flux_columns]] = errors.T
        #TODO what happend if it is not difine? i should ask for it?
        result_panda['wavelength'] =  np.array([(self.relevant_keywords_header["CRVAL1"]+i*self.relevant_keywords_header["CD1_1"])*10 for i in result_panda['n_pixel'].values])
        result_panda['units_flux'] = len(result_panda) * [self.relevant_keywords_header.get("BUNIT","No info")]
        if len(images) > 0:
            if len(images) == num_source:
                print(f'setting names of images {np.arange(1, num_source+1).astype(str).tolist()} to {images}')
                result_panda = result_panda.rename(columns={i:i.replace(i.split("_")[-1],images[int(i.split("_")[-1])-1]) for i in result_panda.columns.values if i.split("_")[-1] in np.arange(1, num_source+1).astype(str).tolist()})#{'A': 'Alpha', 'B': 'Beta'}) 
                self.images = images
            else:
                print(f'The number of image ({images}) is different of the number of source ({num_source}) check it')

        if over_write or not hasattr(self, 'results'):
            print("saving")
            self.results = {'result_panda':result_panda,"multiple_dist":multiple_dist,'image_2d_model':image_2d_model}
            return 
        
        return results
    
    
    
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
    
    
    def plot_column(self,):
        return 
    
    def plot_data_model(self,n):
        """
        Plot the data, individual model components, and the residual for the nth column.
        
        Parameters:
        -----------
        n : int
            Index of the column (pixel) to be plotted.
        
        Raises:
        -------
        AttributeError:
            If the results have not been computed (i.e., 'array_to_pandas' has not been run).
        """
        if not hasattr(self, 'results'):
            raise AttributeError(
                "Error: 'self.results' is not defined. \n"
                "try runing 'array_to_pandas' first")
        df = self.results['multiple_dist'].T
        x_axis = np.arange(self.cut_data.shape[0])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 15), gridspec_kw={'height_ratios': [2, 1]})#, gridspec_kw={'height_ratios': [2, 1]})
        sumx = df[n].T.sum(axis=0)
        for dis in df[n].T:
            #x_axis = np.linspace(0,self.cut_data.shape[0]-1,len(dis))
            ax1.plot(x_axis,dis)
        ax1.plot(x_axis,self.cut_data.T[n])
        ax1.plot(x_axis,sumx) 
        ax2.scatter(x_axis,self.cut_data.T[n]-df[n].T.sum(axis=0)) 
        ax2.axhline(0,ls='--')
        ax1.set_xlim(0,x_axis[-1])  # Set x-axis label font size
        ax2.set_xlim(0,x_axis[-1])  # Set x-axis label font size
        ax1.xaxis.label.set_size(40)  # Set x-axis label font size
        ax1.yaxis.label.set_size(40)  # Set y-axis label font size
        ax1.tick_params(which="both",bottom=False,top=False,left=True,right=False,length=10,width=2,labelsize=20,labelbottom=False)
        ax2.tick_params(which="both",bottom=True,top=False,left=True,right=False,length=10,width=2,labelsize=20,labelbottom=True )
        plt.legend(loc='best', prop={'size': 24}, frameon=False)
        plt.show()
    
    
    def plot_spectra(self,add_error=False,add_raw=False,save='',force_pix=False,z_s=None,add_lines=False,rest_frame=False,flux_columns=None,smooth=False,**kwargs):
        """
        Plot the extracted spectra with optional error bars, raw spectra, and emission/absorption lines.
        Parameters:
        -----------
        add_error : bool, optional, default=False
            If True, adds error bars to the plot.
        add_raw : bool, optional, default=False
            If True, plots the raw flux values.
        save : str, optional, default=''
            If provided, the plot is saved to the specified filename.
        force_pix : bool, optional, default=False
            If True, the x-axis will be in pixel units instead of wavelength.
        z_s : float or None, optional, default=None
            Redshift value for converting wavelengths to the rest frame.
        add_lines : bool, optional, default=False
            If True, vertical lines for known spectral features will be added.
        rest_frame : bool, optional, default=False
            If True and z_s is provided, the wavelengths are converted to the rest frame.
        kwargs : dict
            Additional keyword arguments for customizing the plot (e.g., xlim, ylim, text_fontsize).
        
        Raises:
        -------
        AttributeError:
            If the results have not been computed (i.e., 'array_to_pandas' has not been run).
        """
        if not hasattr(self, 'results'):
            raise AttributeError(
                "Error: 'self.results' is not defined. \n"
                "try runing 'array_to_pandas' first")
        df = self.results['result_panda']
        wavelength = np.arange(len(df))
        xlabel = "pixel"
        ylabel = f"Flux [{df['units_flux'].values[0]}]"
        if "wavelength" in df.columns and not force_pix:
            #rest frame?
            wavelength = df["wavelength"].values
            xlabel = r"Wavelength [Å]"
            if rest_frame and z_s:
                wavelength = df["wavelength"].values/(1+z_s)
                xlabel = "rest frame wavelength (A)"
        fig, ax = plt.subplots(1, 1, figsize=(30, 8))#, gridspec_kw={'height_ratios': [2, 1]})
        if not flux_columns:
            flux_columns = [i for i in df.columns.values if 'flux' in i.split('_')[0]]
        alpha = 0.75
        if len(flux_columns)>2:
            alpha  = 0.6
        colors = ['b','r','g']
        colors = ['dodgerblue','crimson','forestgreen']
        #colors = ['navy','firebrick','limegreen']
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        colors = ['#4c72b0', '#dd8452', '#55a868']
        # Alternative 3: ColorBrewer Set1 (vibrant and high-contrast colors)
        colors = ['#377eb8', '#e41a1c', '#4daf4a']
        ecolors = ['lightskyblue','LightCoral',"LightGreen"]
        all_flux = []
        for i,flux in enumerate(flux_columns):
            flux_=df[flux].values
            if smooth:
                dlam = np.median(np.diff(wavelength))
                win = max(15, int(round(8.0/dlam)) | 1)  # ~8 Å window, odd
                flux_ = savgol_filter(flux_, win, 2, mode="mirror")
            error_ = None
            if add_raw:
                ax.plot(wavelength,flux_raw,label='raw_'+flux)
            if add_error:
                error_ = df['std_'+flux].values
                error_[error_>flux_] = 0
                print("For plotting convenience the errors>flux will be set to 0")
            if "G" not in flux:
                flux = "Image "+flux.replace("flux_","")
            else:
                flux = "Lens "+flux.replace("flux_","")
            ax.errorbar(wavelength,flux_,yerr=error_,color=colors[i], ecolor=ecolors[i],label=flux,alpha=0.9)
            all_flux.append(flux_)
        all_flux = np.concatenate(all_flux)
        ylim_lower, ylim_upper = np.percentile(all_flux, [1, 99.99])
        ax.tick_params(which="both", bottom=True, top=False, left=True, right=False,
            length=10, width=2, labelsize=35)  # Increase tick length and width
        xlim = kwargs.get('xlim',wavelength[[0,-1]])
        ylim =kwargs.get('ylim',[-0.01e-16, ylim_upper*1.05])
        text_fontsize = kwargs.get("text_fontsize",20)
        text_rotation = kwargs.get("text_rotation",0)
        if z_s and add_lines:
            agn_lines = {
            "Lya": 1216,         # Lyman-alpha
            "CIV": 1549,         # Carbon IV
            "CIII_1909": 1909,   # Carbon III]
            "MgII": 2800,        # Magnesium II
            "HeII_4686": 4686,   # Helium II
            "Hβ": 4861,          # Hydrogen Balmer beta
            "OIII_4959": 4959,   # [O III] 4959
            "OIII_5007": 5007,   # [O III] 5007
            "OI_6300": 6300,     # [O I] 6300
            "NII_6548": 6548,    # [N II] 6548
            "Hα": 6563,          # Hydrogen Balmer alpha
            "NII_6583": 6583,    # [N II] 6583
            "SII_6716": 6716,    # [S II] 6716
            "SII_6731": 6731     # [S II] 6731
            }

            for line_name,central_wavelength in agn_lines.items():
                if rest_frame:
                    central_wavelength = central_wavelength
                else:
                    central_wavelength = central_wavelength*(1+z_s)
                if max(xlim)>central_wavelength and min(xlim)<central_wavelength:
                    ax.axvline(central_wavelength, linestyle="--", color="k", linewidth=2,alpha=0.5)
                    ax.text(central_wavelength, ylim[1], f" {line_name}", fontsize=text_fontsize, rotation=text_rotation,
                            verticalalignment="top", color="k",zorder=10,horizontalalignment="left")
        # offset_text = ax.yaxis.get_offset_text()
        # offset_text.set_fontsize(30)
        zl = 0.228
        # ax.axvline(3933.66*(1+ zl),lw=3,color = "k", ls = "--")
        # ax.text(3933.66*(1+ zl), 0.8*1e-16, "Lens Ca II K", fontsize=25, rotation=90,
        #                     verticalalignment="top", color="k",zorder=10,horizontalalignment="right")
        # ax.axvline(3968.47*(1+ zl),lw=3,color = "k", ls = "--")
        # ax.text(3968.47*(1+ zl), 0.8*1e-16, "Lens Ca II H", fontsize=25, rotation=90,
        #                     verticalalignment="top", color="k",zorder=10,horizontalalignment="right")
        
        # ax.axvline(2800*(1+ 0.77),lw=3,color = "purple", ls = "--")
        # ax.text(2800*(1+ 0.77), 1.05*1e-16, "QSO Mg II", fontsize=25, rotation=0,
        #                      verticalalignment="top", color="purple",zorder=10,horizontalalignment="left")
        ax.set_xlabel(xlabel, fontsize=30)
        ax.set_ylabel(r"Flux [$\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]", fontsize=30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        offset = ax.yaxis.get_offset_text()
        offset.set_fontsize(30)  # or whatever size you like
        ax.xaxis.label.set_size(30)  # Set x-axis label font size
        ax.yaxis.label.set_size(30)  # Set y-axis label font size 
        plt.legend(loc='best', prop={'size': 30}, frameon=False)
        if save:
            plt.savefig(f"images/{save}.pdf", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_cut_out(self):
        """
        Plot the 2D cut-out image and the stacked median profile.
        """
        norm_image = self.cut_data/self.cut_data.max(axis=0)
        fig,axs = plt.subplots(1, 2, figsize=(18, 5))
        # Plot data on the first subplot
        im = axs[0].imshow(norm_image,aspect="auto",vmin=0,vmax=1)
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
    def run_cut_2d(self,center,size,verbose=False):
        return 
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
    
    
    def plot_slices_overlay(
        self,
        x_step=25,
        x_min=0,
        x_max=None,
        normalize=True,
        subtract_bg=True,
        bg_percentile=10,
        alpha=0.35,
        show_trace_centroid=True,
        threshold_frac=0.25,
        vmin=None,
        vmax=None,
        show_x_lines=True,
        x_line_alpha=0.35,
        x_line_lw=1.5,
        cmap_name="viridis",
        trace_array = None):
        """
        Plot the 2D cut-out and overlay many spatial profiles (y-slices) for different x columns.
        Optionally draw vertical lines on the 2D image at each sampled x, with matching colors.
        """
        data = self.cut_data
        ny, nx = data.shape
        if x_max is None:
            x_max = nx

        fig, (ax_img, ax_prof) = plt.subplots(
            2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.0, 1.0]}, sharex=False
        )

        if vmin is None or vmax is None:
            vmin_, vmax_ = np.nanpercentile(data, [5, 95])
            if vmin is None:
                vmin = vmin_
            if vmax is None:
                vmax = vmax_

        im = ax_img.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
        ax_img.set_title("2D cut")
        ax_img.set_xlabel("X (dispersion pixel)")
        ax_img.set_ylabel("Y (spatial pixel)")
        fig.colorbar(im, ax=ax_img, label="intensity")

        ys = np.arange(ny)

        x_list = list(range(int(x_min), int(x_max), int(x_step)))
        n_lines = len(x_list)

        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / max(n_lines - 1, 1)) for i in range(n_lines)]

        xs_cent, y0s = [], []

        for idx, x in enumerate(x_list):
            prof = data[:, x].astype(float)

            if subtract_bg:
                prof = prof - np.nanpercentile(prof, bg_percentile)

            m = np.nanmax(prof)
            if not np.isfinite(m) or m <= 0:
                continue

            prof_plot = (prof / m) if normalize else prof
            ax_prof.plot(ys, prof_plot, alpha=alpha, color=colors[idx])

            if show_x_lines:
                ax_img.axvline(x, color=colors[idx], alpha=x_line_alpha, lw=x_line_lw)

            if show_trace_centroid:
                mask = prof > (threshold_frac * m)
                if mask.sum() >= 3:
                    w = prof[mask]
                    y = ys[mask]
                    y0 = np.sum(y * w) / np.sum(w)
                    xs_cent.append(x)
                    y0s.append(y0)

        ax_prof.set_title(f"Spatial profiles (step={x_step})")
        ax_prof.set_xlabel("Y (spatial pixel)")
        ax_prof.set_ylabel("Normalized flux" if normalize else "Flux")
        ax_prof.grid(alpha=0.2)
        
        if isinstance(trace_array,np.ndarray):
            ax_img.plot(np.arange(len(trace_array)), trace_array, lw=2, label="centroid trace",c="red")
            ax_img.legend(loc="best")
            return 
        if show_trace_centroid and len(xs_cent) > 0:
            ax_img.plot(xs_cent, y0s, lw=2, label="centroid trace")
            ax_img.legend(loc="best")

        plt.tight_layout()
        plt.show()

        if show_trace_centroid:
            return np.array(xs_cent), np.array(y0s)
        return None
    

    def get_trace(
        self,
        pixel_limit=None,
        no_use_real_error=False,

        # ---- trace centroiding ----
        trace_bg_percentile=10,
        trace_threshold_frac=0.25,
        trace_smooth=31,
        trace_min_valid_frac=0.2,

        # ---- robust / windowed trace ----
        trace_half_window=15,
        trace_use_prev=True,
        trace_clip_percentile=99.5,

        # ---- NEW: polynomial trace ----
        trace_poly_order=None,          # e.g. 2 or 3, None = disable
        trace_poly_robust=True,
        trace_poly_clip_sigma=4.0,
        trace_poly_max_iter=5,

        # ---- optional separation guessing ----
        guess_separation_from_peaks=True,
        peak_window=10,
        initial_separation=None,
        initial_center=None,

        # ---- sky subtraction ----
        do_sky_subtract=False,
        sky_inner=12,
        sky_outer=30,
        sky_stat="median",
        sky_poly_order=1,

        # ---- CR clipping ----
        cr_clip_sigma=6.0,
        cr_clip_maxiters=1,

        **kwargs,):
        """TODO add that for the values outside the limit use the same function as the other ones"""
        # ------------------------------------------------------------
        # Setup
        # ------------------------------------------------------------
        data0 = self.cut_data
        err0  = self.cut_error if not no_use_real_error else np.ones_like(data0)

        ny, nx = data0.shape

        if pixel_limit is None or pixel_limit == []:
            x_min, x_max = 0, nx
        else:
            x_min, x_max = int(pixel_limit[0]), int(pixel_limit[1])
            x_min = max(0, x_min)
            x_max = min(nx, x_max)

        xs = np.arange(nx, dtype=float)
        ys = np.arange(ny, dtype=float)

        # ------------------------------------------------------------
        # 1) Build centroid trace
        # ------------------------------------------------------------
        if initial_center is not None:
            y0 = np.full(nx, float(initial_center), dtype=float)

        else:
            y0 = np.full(nx, np.nan, dtype=float)
            y_prev = None

            for x in range(x_min, x_max):
                prof = data0[:, x].astype(float)

                bg = np.nanpercentile(prof, trace_bg_percentile)
                prof = prof - bg
                prof[~np.isfinite(prof)] = 0.0
                prof[prof < 0] = 0.0
                if np.nanmax(prof) <= 0:
                    continue

                yc = y_prev if (trace_use_prev and y_prev is not None) else np.nanargmax(prof)

                ylo = int(max(0, yc - trace_half_window))
                yhi = int(min(ny, yc + trace_half_window + 1))

                prof_w = prof[ylo:yhi]
                ys_w   = ys[ylo:yhi]

                m = np.nanmax(prof_w)
                if not np.isfinite(m) or m <= 0:
                    continue

                mask = prof_w > (trace_threshold_frac * m)
                if mask.sum() < 3:
                    continue

                w = prof_w[mask]
                if trace_clip_percentile is not None:
                    cap = np.nanpercentile(w, trace_clip_percentile)
                    w = np.clip(w, 0, cap)

                y0[x] = np.sum(ys_w[mask] * w) / np.sum(w)
                y_prev = y0[x]

            good = np.isfinite(y0) & (xs >= x_min) & (xs < x_max)
            if good.sum() < trace_min_valid_frac * (x_max - x_min):
                raise RuntimeError("Trace centroiding failed: too few valid columns")

            y0[x_min:x_max] = np.interp(
                xs[x_min:x_max],
                xs[good],
                y0[good],
            )

            if trace_smooth and trace_smooth > 1:
                win = trace_smooth + (trace_smooth % 2 == 0)
                half = win // 2
                y0_sm = y0.copy()
                for x in range(x_min, x_max):
                    lo = max(x_min, x - half)
                    hi = min(x_max, x + half + 1)
                    y0_sm[x] = np.nanmedian(y0[lo:hi])
                y0 = y0_sm

        # ------------------------------------------------------------
        # 2) Polynomial trace fit (NEW)
        # ------------------------------------------------------------
        if trace_poly_order is not None and trace_poly_order >= 1:
            fit_mask = np.isfinite(y0) & (xs >= x_min) & (xs < x_max)
            xfit = xs[fit_mask]
            yfit = y0[fit_mask]

            mask = np.ones_like(yfit, dtype=bool)

            for _ in range(trace_poly_max_iter if trace_poly_robust else 1):
                coef = np.polyfit(xfit[mask], yfit[mask], trace_poly_order)
                ymod = np.polyval(coef, xfit)
                resid = yfit - ymod

                med = np.nanmedian(resid[mask])
                mad = np.nanmedian(np.abs(resid[mask] - med))
                sig = 1.4826 * mad if mad > 0 else np.nanstd(resid[mask])
                if not np.isfinite(sig) or sig <= 0:
                    break

                new_mask = np.abs(resid - med) < trace_poly_clip_sigma * sig
                if new_mask.sum() == mask.sum():
                    break
                mask = new_mask

            coef = np.polyfit(xfit[mask], yfit[mask], trace_poly_order)
            y0[x_min:x_max] = np.polyval(coef, xs[x_min:x_max])

            self.trace_poly_coef = coef  # store for diagnostics

        init_center_arr = y0

        # ------------------------------------------------------------
        # 3) Sky subtraction (optional)
        # ------------------------------------------------------------
        data = data0.copy()
        err  = err0.copy()

        if do_sky_subtract:
            for x in range(x_min, x_max):
                yc = init_center_arr[x]
                if not np.isfinite(yc):
                    continue

                y1 = int(max(0, yc - sky_outer))
                y2 = int(max(0, yc - sky_inner))
                y3 = int(min(ny, yc + sky_inner))
                y4 = int(min(ny, yc + sky_outer))

                sky_idx = np.r_[y1:y2, y3:y4]
                if sky_idx.size < 5:
                    continue

                col = data[:, x]
                sky_v = col[sky_idx]
                good = np.isfinite(sky_v)

                if good.sum() < 5:
                    continue

                if sky_stat == "poly1":
                    p = np.polyfit(sky_idx[good], sky_v[good], sky_poly_order)
                    bg = np.polyval(p, ys)
                else:
                    bg = np.nanmedian(sky_v[good])

                data[:, x] -= bg

        # ------------------------------------------------------------
        # 4) Cosmic ray clipping
        # ------------------------------------------------------------
        if cr_clip_sigma is not None and cr_clip_sigma > 0:
            for _ in range(int(cr_clip_maxiters)):
                for x in range(x_min, x_max):
                    yc = init_center_arr[x]
                    if not np.isfinite(yc):
                        continue

                    ylo = int(max(0, yc - trace_half_window))
                    yhi = int(min(ny, yc + trace_half_window + 1))

                    seg = data[ylo:yhi, x]
                    med = np.nanmedian(seg)
                    mad = np.nanmedian(np.abs(seg - med))
                    sig = 1.4826 * mad if mad > 0 else np.nanstd(seg)

                    if np.isfinite(sig) and sig > 0:
                        cap = med + cr_clip_sigma * sig
                        data[ylo:yhi, x] = np.minimum(seg, cap)

        return init_center_arr, initial_separation




    # -------------------------------------------------------------------------
    # NEW: Optimal extraction helpers (IRAF-like; instrument-agnostic)
    # -------------------------------------------------------------------------


    def _sky_subtract_2d_from_trace(
        self, data, y0_arr, x_min, x_max,
        sky_inner=10, sky_outer=28,
        sky_stat="median", sky_poly_order=1,
        sky_side="both",   # "both" | "up" | "down"
        ):
        ny, nx = data.shape
        ys = np.arange(ny, dtype=float)

        data_sky = data.astype(float).copy()
        sky_model = np.zeros_like(data_sky, dtype=float)

        for x in range(int(x_min), int(x_max)):
            yc = y0_arr[x]
            if not np.isfinite(yc):
                continue

            y1 = int(max(0, np.floor(yc - sky_outer)))
            y2 = int(max(0, np.floor(yc - sky_inner)))
            y3 = int(min(ny, np.ceil(yc + sky_inner)))
            y4 = int(min(ny, np.ceil(yc + sky_outer)))

            sky_idx = []

            # "down" = lower-y side (y0 - outer : y0 - inner)
            if sky_side in ("both", "down"):
                if y2 > y1:
                    sky_idx.append(np.arange(y1, y2))

            # "up" = higher-y side (y0 + inner : y0 + outer)
            if sky_side in ("both", "up"):
                if y4 > y3:
                    sky_idx.append(np.arange(y3, y4))

            if len(sky_idx) == 0:
                continue

            sky_idx = np.concatenate(sky_idx)
            col = data_sky[:, x]

            sky_y = ys[sky_idx]
            sky_v = col[sky_idx]
            good = np.isfinite(sky_v)

            if np.count_nonzero(good) < 10:
                continue

            sky_y = sky_y[good]
            sky_v = sky_v[good]

            if sky_stat == "poly1":
                try:
                    p = np.polyfit(sky_y, sky_v, deg=int(sky_poly_order))
                    bg = np.polyval(p, ys)
                except Exception:
                    bg = np.nanmedian(sky_v) * np.ones_like(ys)
            else:
                bg = np.nanmedian(sky_v) * np.ones_like(ys)

            sky_model[:, x] = bg
            data_sky[:, x] = col - bg

        return data_sky, sky_model


    # -------------------------------------------------------------------------
    # 2) EMPIRICAL PROFILE BUILDER (as you had it)
    # -------------------------------------------------------------------------
    def _build_empirical_profile_from_trace(
        self, data_sky, y0_arr, x_min, x_max,
        half_window=6, step=5, clip_percentile=99.5
    ):
        ny, nx = data_sky.shape
        w = 2 * int(half_window) + 1
        profs = []

        for x in range(int(x_min), int(x_max), int(step)):
            yc = y0_arr[x]
            if not np.isfinite(yc):
                continue

            ylo = int(max(0, np.floor(yc - half_window)))
            yhi = int(min(ny, np.floor(yc + half_window + 1)))
            if (yhi - ylo) != w:
                continue

            seg = data_sky[ylo:yhi, x].astype(float)
            seg[~np.isfinite(seg)] = 0.0
            seg[seg < 0] = 0.0

            s = np.nansum(seg)
            if not np.isfinite(s) or s <= 0:
                continue

            if clip_percentile is not None:
                cap = np.nanpercentile(seg, clip_percentile)
                if np.isfinite(cap) and cap > 0:
                    seg = np.clip(seg, 0, cap)

            seg = seg / np.nansum(seg)
            profs.append(seg)

        if len(profs) < 10:
            raise RuntimeError(
                "Not enough valid columns to build an empirical profile. "
                "Try increasing pixel_limit range, adjusting half_window, or lowering clip_percentile."
            )

        P = np.nanmedian(np.array(profs), axis=0)
        P[P < 0] = 0.0
        P /= np.nansum(P)
        return P


    # -------------------------------------------------------------------------
    # 3) OPTIMAL EXTRACTION (passes sky_side + stores diagnostics)
    # -------------------------------------------------------------------------
    def optimal_extract_from_trace(
        self, y0_arr, pixel_limit=None, half_window=6,
        do_sky_subtract=True,
        sky_inner=10, sky_outer=28, sky_stat="median", sky_poly_order=1,
        sky_side="both",
        profile_step=5, profile_clip_percentile=99.5,
        cr_clip_sigma=6.0
    ):
        data0 = self.cut_data.astype(float)
        err0  = self.cut_error.astype(float) if hasattr(self, "cut_error") else np.ones_like(data0)
        
        ny, nx = data0.shape
        if pixel_limit is None or pixel_limit == []:
            x_min, x_max = 0, nx
        else:
            x_min, x_max = int(pixel_limit[0]), int(pixel_limit[1])
            x_min = max(0, x_min); x_max = min(nx, x_max)

        if do_sky_subtract:
            data_sky, sky_model = self._sky_subtract_2d_from_trace(
                data0, y0_arr, x_min, x_max,
                sky_inner=sky_inner, sky_outer=sky_outer,
                sky_stat=sky_stat, sky_poly_order=sky_poly_order,
                sky_side=sky_side
            )
        else:
            data_sky = data0.copy()
            sky_model = np.zeros_like(data0)

        P = self._build_empirical_profile_from_trace(
            data_sky, y0_arr, x_min, x_max,
            half_window=half_window, step=profile_step,
            clip_percentile=profile_clip_percentile
        )

        w = 2 * int(half_window) + 1
        flux = np.full(nx, np.nan, dtype=float)
        ferr = np.full(nx, np.nan, dtype=float)

        for x in range(x_min, x_max):
            yc = y0_arr[x]
            if not np.isfinite(yc):
                continue

            ylo = int(max(0, np.floor(yc - half_window)))
            yhi = int(min(ny, np.floor(yc + half_window + 1)))
            if (yhi - ylo) != w:
                continue

            D = data_sky[ylo:yhi, x].astype(float)
            V = (err0[ylo:yhi, x].astype(float) ** 2)

            good = np.isfinite(D) & np.isfinite(V) & (V > 0)
            if np.count_nonzero(good) < 3:
                continue

            if cr_clip_sigma is not None and cr_clip_sigma > 0:
                med = np.nanmedian(D[good])
                mad = np.nanmedian(np.abs(D[good] - med))
                sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(D[good])
                if np.isfinite(sig) and sig > 0:
                    cap = med + cr_clip_sigma * sig
                    D = np.minimum(D, cap)

            num = np.sum(P[good] * D[good] / V[good])
            den = np.sum((P[good] ** 2) / V[good])

            if np.isfinite(den) and den > 0:
                flux[x] = num / den
                ferr[x] = np.sqrt(1.0 / den)

        # store last-run diagnostics
        self.optimal_results = dict(
            x_min=x_min, x_max=x_max,
            y0_arr=y0_arr,
            half_window=half_window,
            data_sky=data_sky,
            sky_model=sky_model,
            profile=P,
            flux=flux,
            ferr=ferr,
            settings=dict(
                do_sky_subtract=do_sky_subtract,
                sky_inner=sky_inner, sky_outer=sky_outer,
                sky_stat=sky_stat, sky_poly_order=sky_poly_order,
                sky_side=sky_side,
                profile_step=profile_step,
                profile_clip_percentile=profile_clip_percentile,
                cr_clip_sigma=cr_clip_sigma,
            )
        )

        #return flux, ferr
