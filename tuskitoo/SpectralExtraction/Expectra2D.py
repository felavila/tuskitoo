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
        self.band = self.relevant_keywords_header["ESO SEQ ARM"]
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
                        param_value=param_value,param_limit=param_limit,param_fix=param_fix)
        
        
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
        result_panda['units_flux'] = len(result_panda) * [self.relevant_keywords_header["BUNIT"]]
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
    
    
    def plot_spectra(self,add_error=False,add_raw=False,save='',force_pix=False,z_s=None,add_lines=False,rest_frame=False,flux_columns=None,**kwargs):
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
        ylabel = df['units_flux'].values[0]
        if "wavelength" in df.columns and not force_pix:
            #rest frame?
            wavelength = df["wavelength"].values
            xlabel = "wavelength (A)"
            if rest_frame and z_s:
                wavelength = df["wavelength"].values/(1+z_s)
                xlabel = "rest frame wavelength (A)"
        fig, ax = plt.subplots(1, 1, figsize=(35, 15))#, gridspec_kw={'height_ratios': [2, 1]})
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
            flux_raw=df['raw_'+flux].values
            error_ = None
            if add_raw:
                ax.plot(wavelength,flux_raw,label='raw_'+flux)
            if add_error:
                error_ = df['std_'+flux].values
                error_[error_>flux_] = 0
                print("For plotting convenience the errors>flux will be set to 0")
            ax.errorbar(wavelength,flux_,yerr=error_,color=colors[i], ecolor=ecolors[i],label=flux,alpha=0.9)
            all_flux.append(flux_)
        all_flux = np.concatenate(all_flux)
        ylim_lower, ylim_upper = np.percentile(all_flux, [1, 99.99])
        ax.tick_params(which="both", bottom=True, top=False, left=True, right=False,
            length=10, width=2, labelsize=35)  # Increase tick length and width
        xlim = kwargs.get('xlim',wavelength[[0,-1]])
        ylim =kwargs.get('ylim',[0, ylim_upper*1.05])
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
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.xaxis.label.set_size(40)  # Set x-axis label font size
        ax.yaxis.label.set_size(40)  # Set y-axis label font size 
        plt.legend(loc='best', prop={'size': 24}, frameon=False)
        if save:
            plt.savefig(f"images/{save}.jpg", dpi=300, bbox_inches='tight')
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