from .function_maker import create_multigaussian_model, create_multimoffat_model
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import cpu_count
from parallelbar import progress_imap, progress_map, progress_imapu
from parallelbar.tools import cpu_bench, fibonacci
import numpy as np
from copy import deepcopy

def make_fit(ydata:np.array,num_source=2,initial_center=None,initial_separation=None
                 ,bound_sigma=None,fix_sep=None,fix_height=None,custom_expr=None
                 ,weights=None,param_limit=None,param_fix=None,param_value=None,verbose=False,\
                distribution="gaussian"):
    """
    Fits a model to the provided data using either a Gaussian or Moffat profile.

    Parameters:
    -----------
    ydata : array-like
        The dependent data (observations or measurements) to be fitted by the model.
    num_source : int, optional, default=2
        Number of sources (or components) to include in the fitting model.
    initial_center : float, int, or None, optional, default=None
        Initial estimate for the center (mean or peak position) of each source/component.
    initial_separation : list or None, optional, default=None
        Initial guess for the separation between the sources/components.
    bound_sigma : array-like or None, optional, default=None
        Bounds or constraints on the sigma (standard deviation) for components beyond the first.
    fix_sep : list, bool, or None, optional, default=None
        List of source indices for which the separation is fixed (i.e., not varied during fitting).
    fix_height : bool or None, optional, default=None
        Flag to fix the height (amplitude) of the sources/components.
    custom_expr : dict or None, optional, default=None
        Dictionary mapping parameter names to custom expressions for the fitting function.
        Refer to https://lmfit.github.io/lmfit-py/index.html for details.
    weights : array-like or None, optional, default=None
        Weights for each data point in ydata; useful for weighted fitting.
    param_limit : dict or None, optional, default=None
        Dictionary specifying limits (min and max) for parameters. Example:
        {'sigma_1': (0.1, 5.0), 'center_1': (100, 200)}
    param_fix : list or None, optional, default=None
        List of parameter names to be fixed at a specified value during fitting.
    param_value : dict or None, optional, default=None
        Dictionary of initial parameter values. Example:
        {'height_1': 10.0, 'sigma_1': 2.0}
    verbose : bool, optional, default=False
        If True, prints detailed information during the fitting process.
    distribution : str, optional, default="gaussian"
        Type of distribution to use for fitting. Supported options are "gaussian" and "moffat".

    Returns:
    --------
    result : lmfit.model.ModelResult
        The result of the fitting process, including fitted parameters and fit statistics.

    Raises:
    -------
    ValueError:
        If the provided distribution is not among the supported types.
    """
    if distribution=="moffat":
        model, params,xdata = create_multimoffat_model(num_source,ydata, initial_separation=initial_separation,initial_center=initial_center)
    elif distribution=="gaussian":
        model, params,xdata = create_multigaussian_model(num_source,ydata, initial_separation=initial_separation,initial_center=initial_center)
    else:
        raise ValueError(f"the distribution {distribution} not available, only can be use [gaussian, moffat]")
    if bound_sigma:
        #se podria definir a cual sigma atar
        for i in bound_sigma:
            if i>num_source:
                continue
            params["sigma_"+str(i)].expr=f'sigma_{1}'
    if fix_sep and initial_separation:
        for i in fix_sep:
            if i>num_source:
                continue
            params["separation_"+str(i)].vary = False
    if custom_expr:
        for param, expr in custom_expr.items():
            params[param].expr = expr
    if param_limit:
        for param, limit in param_limit.items():
            params[param].min, params[param].max = limit
    if param_value:
        for param, value in param_value.items():
            params[param].value = value
    if param_fix:
        for param in param_fix:
            params[param].vary = False
    result = model.fit(ydata, params, x=xdata, weights=weights)#,max_nfev=200 
    if verbose:
        print(f"Model parameters {params}")
    return result




def parallel_fit(image,error,num_source,pixel_limit=None,n_cpu=None,mask_list=[],**kwargs):
    ##Weights = 1 / (error^2)
    """
    Perform parallel fitting on a 2D image using multiple sources.

    Parameters:
    -----------
    image : 2D array-like
        The image data to be fitted.
    error : 2D array-like
        The error (sigma) associated with the image data.
    num_source : int
        Number of sources (or components) to include in the fitting model.
    pixel_limit : list or tuple of two ints, optional, default=None
        Column indices [start, end] that limit the processing region. If None, all columns are processed.
    n_cpu : int or None, optional, default=None
        Number of CPU cores to use for parallel processing. Defaults to the total available cores.
    wavelenght : list, optional, default=[]
        (Not used yet) Intended to hold wavelength information for each pixel.
    mask_list : list, optional, default=[]
        List of column ranges to mask (exclude) from fitting. Each element should be a two-element iterable (start, stop).
    **kwargs : dict
        Additional keyword arguments to be passed to the make_fit function. These can include:
            - initial_center
            - initial_separation
            - bound_sigma
            - fix_sep
            - fix_height
            - custom_expr
            - weights
            - param_limit
            - param_fix
            - param_value
            - verbose
            - distribution

    Returns:
    --------
    spectral_extraction_results : dict
        A dictionary containing:
            - "value": Fitted parameter values.
            - "std": Parameter uncertainties.
            - "name_params": Names of the fitted parameters.
            - "extra_params": Extra fit statistics (e.g., chisqr, redchi, AIC, BIC, R-squared).
            - "normalized_image": The normalized image used for fitting.
            - "normalize_matrix": Normalization factors for each pixel column.
            - "num_source": Number of sources used.
            - "distribution": Distribution type used.
            - "mask": The mask applied to the image.
            - "parameter_number": Number of parameters per source.
    """
    
    if not n_cpu:
        n_cpu = cpu_count()
    distribution = kwargs.get("distribution", "gaussian")
    #if "distribution" not in kwargs.keys():
     #   kwargs["distribution"]= "gaussian"
    proc_image = np.copy(image)
    proc_error = np.copy(error)
     # Determine pixel limits along the column axis.
    if isinstance(pixel_limit, (list, tuple)) and len(pixel_limit) == 2:
        col_start, col_end = pixel_limit
    else:
        col_start, col_end = 0, image.shape[1]
        pixel_limit = [0, image.shape[1]]
    if isinstance(mask_list, list):
        # Create a boolean mask with the same shape as the image.
        mask = np.ones_like(proc_image, dtype=bool)
        for mask_range in mask_list:
            # Use slice to define the range of columns to mask out.
            # Assumes mask_range is a two-element iterable: (start, stop)
            mask[:, slice(*mask_range)] = False
        proc_image = proc_image * mask
        # Optionally, you could also mask the error if needed:
        proc_error = proc_error * mask
    parameter_number = 3 if distribution == "gaussian" else 4
    proc_image = proc_image[:, col_start:col_end]
    proc_error = proc_error[:, col_start:col_end]
    x_num = len(proc_image.T)
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = 1.0 / np.square(proc_error)
    norm_factor = np.abs(proc_image).max(axis=0)
    norm_factor[norm_factor == 0] = 1
    normalized_image = np.nan_to_num(proc_image / norm_factor)
    normalized_weight = np.nan_to_num(weight / norm_factor)
    global process_pixel
    def process_pixel(args):
        n_pixel, pixel,pixel_weight = args
        #if i want to add a parameter for the stats part of the matrix always should be and the left of it
        if np.all(pixel== 0) or  np.all(pixel== np.nan):
            return list([0]*parameter_number*num_source)+[1e15]*parameter_number*num_source+ list([np.nan]*parameter_number*num_source)+[1e15,1e15,1e15,1e15,1e15,n_pixel,x_num] 
        fiting =  make_fit(pixel, num_source=num_source,weights=pixel_weight,**kwargs)
        if not np.all(fiting.covar):
            return list([0]*parameter_number*num_source)+[1e15]*parameter_number*num_source+ list(fiting.values.keys())+[1e15,1e15,1e15,1e15,1e15,n_pixel,x_num] 
        return list(np.array([[value.value,value.stderr] for key,value in fiting.params.items()]).T.reshape(parameter_number*num_source*2,))+ list(fiting.values.keys()) + [fiting.chisqr,fiting.redchi,fiting.aic,fiting.bic,fiting.rsquared,n_pixel,x_num] 
    print(f"The code will be executed in {n_cpu} core using {num_source} sources an a {distribution} distribution")
    args = [(n_pixel, pixel,pixel_weight) for n_pixel, pixel,pixel_weight in zip(np.arange(*pixel_limit) ,normalized_image.T,normalized_weight.T)]
    normalize_matrix = norm_factor[:,np.newaxis]
    full_fit = np.array(progress_map(process_pixel, args, process_timeout=20, n_cpu=n_cpu,need_serialize=False))
    results = full_fit[:,0:parameter_number*num_source].astype(float)
    errors = full_fit[:,parameter_number*num_source:2*parameter_number*num_source].astype(float)
    name_params = full_fit[:,2*parameter_number*num_source:-7]
    mask_names=~np.all(name_params == 'nan', axis=1)
    name_params = name_params[mask_names][0]
    extra_params = full_fit[:,-7:].astype(float)
    #print(normalized_image *norm_factor == proc_image)
    spectral_extraction_results = {"value":results,"std":errors,"name_params":name_params,"extra_params":extra_params,"normalized_image":normalized_image,"normalize_matrix":normalize_matrix,"num_source":num_source,"distribution":distribution,"mask":mask,"parameter_number":parameter_number} #,"original_image":image
    return spectral_extraction_results

