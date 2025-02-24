import numpy as np 
from .utils import remove_nan,small_fun,mad
from .dummy_pca import dummies_pca
from multiprocessing import cpu_count
from parallelbar import progress_map
from scipy.optimize import minimize

def sky_subtraction(data,band,mask_image_x=None,mask_image_y=None,not_considering_pixels=[],by_eye_signal_position=None,force_median=False):   
    keywords_function = locals()
    mask =(data > 0.0) #& (quality_pixel == 0)#mask for the data, in their code they make this as nan values maybe is best save it as a mask     
    sky_sub_work = False
    if mask_image_y:
        if len(mask_image_y) == 2:
            mask[int(max(mask_image_y)):,:] = False #value: desde
            mask[:int(min(mask_image_y)),:] = False #:value means hasta
        elif len(mask_image_y) != 2:
            for ii in mask_image_y:
                mask[ii,:] = False
        #mask[[27,28,29],:] = False
    if mask_image_x:
        mask[:,int(max(mask_image_x)):] = False #value: desde
        mask[:,:int(min(mask_image_x))] = False #:value means hasta
    ######################
    data_clean =remove_nan(data,mask,verbose=False) #remove nan values
    Sky0 = np.nanmedian(data_clean,axis=0)
    Im = np.nanmedian(data_clean-Sky0,axis=1)
    Q1 = np.percentile(Im, 5)
    Q3 = np.percentile(Im, 95)
    IQR = Q3 - Q1
    # Define the outlier cutoff based on the IQR
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR
    Im[~((Im >= lower_bound) & (Im <= upper_bound))] = 0
    Im[np.isnan(Im)] = 0
    where_is_the_signal = np.zeros_like(Im)
    ####################################
    COF = np.zeros_like(Im)
    (recon_sky,
    image_recon_sky_subtracted,
    image_recon_cof_sky_subtracted,
    image_pca_sky_subtracted,
    pca_sky,
    image_median_sky_subtracted,image_median_sky_subtracted_local) = [np.zeros_like(data_clean) for _ in range(7)]
    if by_eye_signal_position:
        print("pre-defined signal position")
        valuu = np.arange(min(by_eye_signal_position),max(by_eye_signal_position)+1)
    else:
        max_val = np.argmax(Im)
        #########################
        tr1 = Im[:max_val]
        tr2 = Im[max_val:]
        xr1 = np.arange(max_val+1)
        xr2 = np.arange(max_val,Im.shape[0])
        v1 = np.where(np.abs(tr1)<3)[0]
        v2 = np.where(np.abs(tr2)<3)[0]
        try:
            q1 = np.where(np.abs(xr1[v1]-max_val) == np.min(np.abs(xr1[v1]-max_val)))[0][0]
            q2 = np.where(np.abs(xr2[v2]-max_val) == np.min(np.abs(xr2[v2]-max_val)))[0][0]
        except:
            print("Signal detection dont work")
            return {"recon_sky":recon_sky,"COF":COF,"image_recon_sky_subtracted":image_recon_sky_subtracted,"image_recon_cof_sky_subtracted":image_recon_cof_sky_subtracted\
                ,"pca_sky":pca_sky, "image_pca_sky_subtracted" : image_pca_sky_subtracted,"image_median_sky_subtracted":image_median_sky_subtracted,\
                    "where_is_the_signal":where_is_the_signal,"keywords_function":keywords_function,"sky_sub_work":sky_sub_work,"data_clean":data_clean,"image_median_sky_subtracted_local":image_median_sky_subtracted_local}
        cut = np.sort([xr1[v1[q1]] , xr2[v2[q2]]]) + np.array([-3,3])
        valuu = np.arange(np.maximum(cut[0], 0),np.minimum(cut[1]+1, np.shape(data_clean)[0]))
        ##########################
    sky_sub_work = True
    #y_axis_length = data_clean.shape[0]
    where_is_the_signal[valuu] = 1 #where is the signal
    #print(len(valuu),min(valuu),max(valuu),data_clean.shape[0]/2)
    local_sky_region = np.arange(min(valuu)-len(valuu),min(valuu)) if (min(valuu)>=data_clean.shape[0]/2 or (max(valuu)+len(valuu)+1)> data_clean.shape[0]) else np.arange(max(valuu)+1,max(valuu)+len(valuu)+1)
    #TODO add constraint on negative values 
    
    #print("valuu",valuu)
    #print(len(local_sky_region),len(valuu),local_sky_region)
    va0 = np.where(where_is_the_signal==0)[0].tolist()
    va0 = list(set(va0) - set([0,1,data_clean.shape[0]-1,data_clean.shape[0]]+not_considering_pixels))
    data_clean[np.isnan(data_clean)] = 0 
    median_sky = np.median(data_clean[va0],axis=0)
    #print(local_sky_region)
    median_sky_local = np.median(data_clean[local_sky_region],axis=0)
    if band=="UVB":
        median_sky[:800] = 0
    image_median_sky_subtracted = data_clean - median_sky
    image_median_sky_subtracted_local = data_clean - median_sky_local
    
    if mask_image_y:
        if len(mask_image_y) == 2:
            image_median_sky_subtracted_local[int(max(mask_image_y)):,:] = 0 #value: desde
            image_median_sky_subtracted_local[:int(min(mask_image_y)),:] = 0 #:value means hasta
            image_median_sky_subtracted[int(max(mask_image_y)):,:] = 0 #value: desde
            image_median_sky_subtracted[:int(min(mask_image_y)),:] = 0 #:value means hasta
        elif len(mask_image_y) != 2:
            for ii in mask_image_y:
                image_median_sky_subtracted_local[ii,:] = 0
                image_median_sky_subtracted[ii,:] = 0
    #image_median_sky_subtracted_local[[mask_image_y]] = 0
    if (band=="NIR" or band=="VIS") and not force_median:
        if band=="NIR":
            rang = [15000,23030] # this is a variable that depends on the system
        else:
            rang =[1700,22100] #visible
        global run_pca_pixel
        def run_pca_pixel(args_tuple):
            """
            Unpacks the single tuple of arguments and calls sky_subtraction.
            """
            uj,va0,band,data_clean,rang = args_tuple
            return pca_pixel(uj,va0,band,data_clean,rang)
        args = [(uj,va0,band,data_clean,rang) for uj in range(data_clean.shape[0])]
        parallel_fit = progress_map(run_pca_pixel, args,process_timeout=40, n_cpu=cpu_count(),need_serialize=False)
        recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,image_pca_sky_subtracted,pca_sky = [np.stack(arrays) for arrays in zip(*parallel_fit)]
    else:
        return {"recon_sky":recon_sky,"COF":COF,"image_recon_sky_subtracted":image_recon_sky_subtracted,"image_recon_cof_sky_subtracted":image_recon_cof_sky_subtracted\
                ,"pca_sky":pca_sky, "image_pca_sky_subtracted" : image_pca_sky_subtracted,"image_median_sky_subtracted":image_median_sky_subtracted,\
                    "where_is_the_signal":where_is_the_signal,"keywords_function":keywords_function,"sky_sub_work":sky_sub_work,"data_clean":data_clean,"image_median_sky_subtracted_local":image_median_sky_subtracted_local}    
    
    return {"recon_sky":recon_sky,"COF":COF,"image_recon_sky_subtracted":image_recon_sky_subtracted,"image_recon_cof_sky_subtracted":image_recon_cof_sky_subtracted\
                ,"pca_sky":pca_sky, "image_pca_sky_subtracted" : image_pca_sky_subtracted,"image_median_sky_subtracted":image_median_sky_subtracted,\
                    "where_is_the_signal":where_is_the_signal,"keywords_function":keywords_function,"sky_sub_work":sky_sub_work,"data_clean":data_clean,"image_median_sky_subtracted_local":image_median_sky_subtracted_local}



# def run_pca_pixel(args_tuple):
#     """
#     Unpacks the single tuple of arguments and calls sky_subtraction.
#     """
#     data_clean,uj,va0,band = args_tuple
#     return pca_pixel(uj,va0,band,data_clean)

def vectorized_pca_pixel(uj, va0, band, data_clean, rang):
    """
    A vectorized version of the PCA pixel reconstruction using only NumPy.
    """
    # Exclude the target pixel (uj) from the reference sky pixels (va0)
    va0 = np.array(va0)
    va = va0[va0 != uj] if uj in va0 else va0

    # Build the Sky matrix and the target vector
    Sky = data_clean[va, :]   # shape: (n_sky, n_pix)
    Targ = data_clean[uj, :].copy()  # shape: (n_pix,)
    Targ[Targ < 0] = 0  # enforce non-negative

    # Compute standard deviation difference for ordering
    Sd0 = np.std(Sky - Targ, axis=1)
    sort_idx = np.argsort(Sd0)
    Sky0 = Sky[sort_idx, :].astype(np.float64)

    # Normalize each row of Sky0
    row_norms = np.sqrt(np.sum(Sky0**2, axis=1))
    row_norms[row_norms == 0] = 1.0
    Sky0_norm = Sky0 / row_norms[:, None]
    
    # (Optional) Normalize the entire matrix globally.
    global_norm = np.sqrt(np.sum(Sky0_norm**2))
    if global_norm == 0:
        global_norm = 1.0
    Sky0_norm /= global_norm

    # Compute projection coefficients of Targ on each normalized sky pixel
    s_pred = Targ.dot(Sky0_norm.T)  # shape: (n_sky,)

    # --- Vectorized cumulative reconstruction ---
    # Compute contributions: each sky component multiplied by its coefficient.
    prod = s_pred[:, None] * Sky0_norm  # shape: (n_sky, n_pix)
    
    # Compute cumulative sum (reconstruction for each number of components)
    cum_recon = np.cumsum(prod, axis=0)  # shape: (n_sky, n_pix)

    # Compute residuals between Targ and each cumulative reconstruction
    residuals = cum_recon - Targ  # Broadcast Targ to each row
    stds = np.std(residuals, axis=1)  # Standard deviation per reconstruction

    # Choose the reconstruction with minimal std deviation.
    vsky = np.argmin(stds) + 1  # +1 because we start with 1 component
    recon_s = cum_recon[vsky - 1]

    # --- Coefficient estimation for VIS or NIR bands ---
    if band in ["VIS", "NIR"]:
        def wrapper(p):
            return small_fun(p, rang, recon_s, Targ)
        p0 = 1.4
        coef = minimize(wrapper, p0, method='BFGS').x[0]
    else:
        coef = 1.0

    # --- PCA-based reconstruction using dummy PCA ---
    res0 = dummies_pca(Sky0_norm)
    s_pr = -Targ.dot(res0["components_"].T)  # shape: (n_components,)
    pca_prod = s_pr[:, None] * (-res0["components_"])  # shape: (n_components, n_pix)
    cum_pca = np.cumsum(pca_prod, axis=0)
    pca_res = Targ - cum_pca
    # Compute MAD for each PCA reconstruction
    def mad(arr, axis=0):
        med = np.median(arr, axis=axis, keepdims=True)
        return np.median(np.abs(arr - med), axis=axis)
    mad_vals = 1.4826 * np.apply_along_axis(mad, 1, pca_res)
    vsky2 = np.argmin(mad_vals) + 1
    re_s2 = cum_pca[vsky2 - 1]

    # Return all computed results.
    return recon_s, coef, Targ - recon_s, Targ - recon_s * coef, Targ - re_s2, re_s2

def pca_pixel(uj,va0,band,data_clean,rang):
    #TODO think about error propagation produced by pca
    if any(np.array(va0)==uj):
        va = list(set(va0) - set([uj])) # set 
    
    else:
        va = va0 #set
    
    Sky = data_clean[va,:]
    Sky[np.isnan(Sky)] = 0
    Targ = data_clean[uj,:]#[np.newaxis]
    Targ[Targ<0] = 0
    Sd0 = np.std(Sky-Targ,axis=1)
    Sky0 = Sky[np.argsort(Sd0)]
    S_0 = np.sum(Sky0,axis=1)
    Sky0[S_0 != 0] = Sky0[S_0 != 0]/np.sqrt(np.sum(Sky0[S_0 != 0]**2,axis=1))[...,np.newaxis]
    #star to appear differences
    Sky0 = Sky0/np.sqrt(np.sum(Sky0**2))
    s_pred = Targ @ Sky0.T
    recon_s_all = np.squeeze(np.array([s_pred[:i] @ Sky0[:i, :] for i in range(1, Sky0.shape[0] + 1)]))
    SX = np.std(Targ[np.newaxis] - recon_s_all,axis=1)
    vsky = np.argmin(SX) +1 #avoid exces of list with this 
    #print(Sky0.shape,Targ.shape)
    #const = Sky0 @ Targ
    recon_s = s_pred[:vsky] @ Sky0[:vsky]
    if band in ["VIS","NIR"]:
        def wrapper(p):
            return small_fun(p,rang,recon_s,Targ)
        p0  = 1.4 # this also could be a dependence on the the system
        coef = minimize(wrapper, p0, method='BFGS').x
    else:
        coef = [1]
        #return print(f"the Band:{band} not found")
    res0 = dummies_pca(Sky0)
    #recon_m = res0["transform"] @ res0["components_"]
    s_pr = Targ @ - res0["components_"].T
    re_s = np.squeeze(np.array([s_pr[:i] @ -res0["components_"][:i, :] for i in range(1, Sky0.shape[0] + 1)]))
    SX2 = 1.4826*mad(Targ-re_s,axis=1)
    vsky = np.argmin(SX2) + 1
    re_s2 = s_pr[:vsky] @ - res0["components_"][:vsky]
    return recon_s,coef[0],Targ-recon_s,Targ-recon_s*coef,Targ-re_s2,re_s2