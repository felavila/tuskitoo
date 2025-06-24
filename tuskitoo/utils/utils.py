import numpy as np 


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

def sigma_clip_1d(spectra,region_size=10,sigma=2,max_iter=5,replace_with='median',**kwargs):
    error = kwargs.get("error",spectra)
    return np.hstack([sigma_clip_replace(spectra[i:i+region_size], sigma=sigma, max_iter=max_iter, replace_with="median",error=error[i:i+region_size])[0] for i in range(0,spectra.shape[0],region_size)])
