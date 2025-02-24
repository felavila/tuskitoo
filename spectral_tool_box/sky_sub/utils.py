import numpy as np 
from scipy import signal
from copy import deepcopy
import glob


def sigma_clip_1d(spectra,region_size=10,sigma=2,max_iter=5,replace_with='median',**kwargs):
    error = kwargs.get("error",spectra)
    return np.hstack([sigma_clip_replace(spectra[i:i+region_size], sigma=sigma, max_iter=max_iter, replace_with="median",error=error[i:i+region_size])[0] for i in range(0,spectra.shape[0],region_size)])

def small_fun(p,rang,recon_s,Targ):
    return np.sum((p*recon_s[rang[0]:rang[1]]-Targ[rang[0]:rang[1]])**2)*1e30

def mad(data,axis=0):
    median = np.median(data,axis=axis)
    absolute_deviations = np.abs(data - median[...,np.newaxis])
    return  np.median(absolute_deviations,axis=axis)

def get_images_paths(system_name,band,OB,data_path=""):
    if not data_path:
        data_path = f"../../spectra_lens_paper/spectroscopic_data/{system_name}/{OB}/{band}/"
    images_paths = sorted(glob.glob(f'{data_path}/*SCI_SLIT_MERGE2D*'))
    if len(images_paths)==0:
        print("CHECK THE PATH")
    if isinstance(images_paths,str):
        images_paths = [images_paths]
    #name_images.sort()
    response_paths = sorted(glob.glob(f'{data_path}/RESPONSE*'))
    if isinstance(response_paths,str):
        response_paths = [response_paths]
    if len(response_paths)==1:
        response_paths = response_paths * len(images_paths)
    #images_paths.sort(),response_paths.sort()
    telluric_path = sorted(glob.glob(f'{data_path}/*/*_TELLURIC*'))
    if len(telluric_path) != len(images_paths):
        if len(telluric_path) == 1:
            print('We will use the same telluric for all the images ')
            telluric_path = telluric_path * len(images_paths)
        else :
            telluric_path = sorted(glob.glob(f'{data_path}/*/TELLURIC*'))
            #len(telluric_path) != len(images_paths):
            if len(telluric_path) == 1:
                print('We will use the same telluric for all the images ')
                telluric_path = telluric_path * len(images_paths)
            else:
                print("CHECK THE telluric_path")
                telluric_path = [None]*len(images_paths)
    stare_spectrums = sorted(glob.glob(f'{data_path}/*/*MERGE1D*'))
    if len(stare_spectrums) != len(images_paths):
        if len(stare_spectrums) == 1:
            print('We will use the same stare_spectrums for all the images ')
            stare_spectrums = stare_spectrums * len(images_paths)
        else:
            stare_spectrums = [None]*len(images_paths)
            print("CHECK THE stare_spectrums")
    return images_paths,response_paths,telluric_path,stare_spectrums



def list_builder(list_,value):
    try:
        if len(list_)==len(value) and any(isinstance(value[i],list) for i in range(len(value))):
            return value
        elif len(value) != len(list_) and any(isinstance(value[i],list) for i in range(len(value))):
            raise ValueError(
                f"IT CANT BE DONE CHECK THE INPUT length value: {value} "
                f"is different from number of images: {list_}"
            )
        elif isinstance(value,list):
            return [value] *len(list_)
    except:
        if value is None:
            return [None]*len(list_)
        elif isinstance(value,bool) or isinstance(value,str) or isinstance(value,int):
            return [value] *len(list_)
        
def inpaint_nans(im, kernel_size=5):
    # Taken from http://stackoverflow.com/a/21859317/6519723
    ipn_kernel = np.ones((kernel_size, kernel_size)) # kernel for inpaint_nans
    ipn_kernel[int(kernel_size/2), int(kernel_size/2)] = 0

    nans = np.isnan(im)
    while np.sum(nans)>0:
        im[nans] = 0
        vNeighbors = signal.convolve2d((nans == False), ipn_kernel, mode='same', boundary='symm')
        im2 = signal.convolve2d(im, ipn_kernel, mode='same', boundary='symm')
        im2[vNeighbors > 0] = im2[vNeighbors > 0]/vNeighbors[vNeighbors > 0]
        im2[vNeighbors == 0] = np.nan
        im2[(nans == False)] = im[(nans == False)]
        im = im2
        nans = np.isnan(im)
    return im


def interpolate_array_with_nans_np(array,mask=None):
    if mask is None:
        mask = np.isnan(array)
    x = np.where(~mask)[0]
    y = array[~mask]
    if len(x)==0:
        return np.zeros(array.shape) 
    y_new = np.interp(np.where(mask), x, y,right=np.nan,left=np.nan)
    array[np.where(mask)] = y_new
    return array

def remove_nan(array,mask,verbose=False):
    #we are agreement at 1e-7 for 0.9999 of the values with the method of the guys 
    if mask.shape != mask.shape:
        return print("The mask and the array should have the same shape")
    if len(array.shape) == 2:
        array = array[np.newaxis]
        mask = mask[np.newaxis]
    A = deepcopy(array)
    mask = mask.astype(bool)
    for i in range(array.shape[0]):
        a = A[i].copy()
        #R code not implemented yet
        # if(OLD=="TRUE"){
        # Mc[1:2,]<-0
        # Mc[,1:2]<-0
        # Mc[dim(Mc)[1]-(0:1),]<-0
        # Mc[,dim(Mc)[2]-(0:1)]<-0
        # }
        a[~mask[i]] = np.nan
        UT = 1
        UT2 = []
        while UT != 0:
            Na1 = np.sum(~mask[i])
            #Mc = interpolate_array_3d_reshape(A[i],mask[i])
            Mc = np.apply_along_axis(interpolate_array_with_nans_np, axis=1, arr=a)
            Na2 = np.isnan(Mc).sum()
            if verbose:
                print("na",Na1)
                print("na2",Na2)
            if Na1 == Na2:
                Mc = np.apply_along_axis(interpolate_array_with_nans_np, axis=1, arr=Mc.T).T
            UT = np.isnan(Mc).sum()
            UT2.append(UT)
            if len(UT2)>6:
                lu = len(UT2)
                #this is a condition to avoid infinite loops
                if (UT2[lu-1] == UT2[lu-2]) & (UT2[lu-2] == UT2[lu-3]) & (UT2[lu-3] == UT2[lu-4]): #for R the last element is len(l) for python is len(l)-1
                    UT = 0
        A[i] = Mc
        local_nan = np.isnan(Mc)
        if local_nan.sum() != 0:
            A[i][local_nan] = array[i][local_nan]
    return np.squeeze(A)


def sigma_clip_replace(data, sigma=3.0, max_iter=5, replace_with='mean',**kwargs):
    """
    Perform iterative sigma-clipping on 'data' and replace outliers with
    either the mean or median of the *valid* data in each iteration.
    
    Parameters
    ----------
    data : array_like
        1D or 2D numpy array of your data (e.g. pixel values).
    sigma : float, optional
        Sigma-clipping limit (number of standard deviations).
    max_iter : int, optional
        Maximum number of iterations.
    replace_with : {'mean', 'median'}, optional
        Replacement strategy for outliers.

    Returns
    -------
    clipped_data : numpy.ndarray
        Copy of 'data' with outliers replaced.
    mask : numpy.ndarray (bool)
        Boolean mask array of the same shape as data. `True` indicates
        pixels considered outliers in the final iteration.
    """

    # Make a copy so we don't overwrite original data
    clipped_data = np.array(data, copy=True, dtype=float)
    clipped_error = kwargs.get("error",clipped_data)
    for i in range(max_iter):
        # Compute statistics on the current valid data
        valid_mask = ~np.isnan(clipped_data)
        current_valid_data = clipped_data[valid_mask]

        if current_valid_data.size < 2:
            # Not enough data to compute meaningful stats
            break

        mean_val = np.mean(current_valid_data)
        median_val = np.median(current_valid_data)
        std_val = np.std(current_valid_data)

        # Define "low" and "high" cutoff
        lower_bound = mean_val - sigma * std_val
        upper_bound = mean_val + sigma * std_val

        # Determine which pixels are outliers
        old_outlier_mask = (clipped_data < lower_bound) | (clipped_data > upper_bound) | (clipped_error>clipped_data)

        # Stop if there are no more outliers
        if not np.any(old_outlier_mask):
            break

        # Decide on the replacement value: mean or median
        if replace_with.lower() == 'mean':
            replacement_value = mean_val
        else:
            replacement_value = median_val

        # Replace outliers
        clipped_data[old_outlier_mask] = replacement_value

        # After replacing, check if the outlier mask remains the same
        # in the next iteration, or continue until max_iter is reached
        new_valid_mask = ~np.isnan(clipped_data)
        new_valid_data = clipped_data[new_valid_mask]
        new_mean_val = np.mean(new_valid_data)
        new_std_val = np.std(new_valid_data)

        # If the change in mean or std is negligible, we can consider stopping early
        if np.isclose(mean_val, new_mean_val, rtol=1e-5) and np.isclose(std_val, new_std_val, rtol=1e-5):
            break

    # One final mask of outliers, using the final statistics
    final_valid_data = clipped_data[~np.isnan(clipped_data)]
    if final_valid_data.size < 2:
        final_mask = np.zeros_like(data, dtype=bool)
    else:
        final_mean = np.mean(final_valid_data)
        final_std = np.std(final_valid_data)
        final_lower = final_mean - sigma * final_std
        final_upper = final_mean + sigma * final_std
        final_mask = (clipped_data < final_lower) | (clipped_data > final_upper)
    
    return clipped_data, final_mask

def clipping_region(image,pieces,slice_, sigma=2.0, max_iter=5, replace_with='mean',where_is_the_signal=[]):
    image_clipped = np.zeros_like(image)
    for j in range(image.shape[0]):
        sigma_i = sigma 
        if j in where_is_the_signal:
            sigma_i = 3
        for i in range(0,pieces):
            image_clipped[j,slice_*i:slice_*(i+1)] = sigma_clip_replace(image[j,slice_*i:slice_*(i+1)], sigma=sigma_i, max_iter=max_iter, replace_with=replace_with)[0]
    return image_clipped