from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import pandas as pd
from copy import deepcopy
from csaps import csaps
import os 
import datetime
from pathlib import Path 
#Here try to lef codes that are use for astronomical stuff 
module_dir = os.path.dirname(os.path.abspath(__file__))
#print(module_dir)


def combine_2D_spectra_sky_sub(
    paths,
    method="mean",
    verbose=True,
    pre_combine=False
    ,save = "",person="F. Avila-Vera",telluric_corrected=True):
    """
    Combine multiple 2D spectra (and their errors/quality arrays) into a single
    2D output using weighted combination. The function can also return the
    pre-combined arrays for inspection.
    This is results that came for our sky sub
    Parameters
    ----------
    paths : list of str
        List of file paths to FITS files. Each file should have:
          - EXT 0: 2D flux image
          - EXT 1: 2D error map
          - EXT 2: 2D quality map (integer or float)
    method : str, optional
        Combination method. Currently only 'mean' (weighted) is implemented.
        Could be extended to 'median' or other methods in the future.
    verbose : bool, optional
        If True, print diagnostic information.
    pre_combine : bool, optional
        If True, the function returns the stacked array (without weighting)
        before doing the final combination.

    Returns
    -------
    combined_flux : 2D np.ndarray
        Weighted-combined flux image (shape: [Y, X]).
    combined_error : 2D np.ndarray
        Propagated error image for the combined flux (shape: [Y, X]).
    combined_quality : 2D np.ndarray
        Average (or some combined) quality map (shape: [Y, X]).

    Notes
    -----
    - This function attempts to correct image offsets in the Y dimension by
      using the `correct(headers)` function (not shown here) to determine shifts.
    - If `pre_combine` is True, it returns the large stacked arrays without
      applying weighted combination. This can be used for debugging.
    """
    
    # -------------------------------------------------------------------------
    # 1) Read data from each file
    # -------------------------------------------------------------------------
    # images[i], errors[i], quality[i] each has shape = (Y_i, X_i)
    images  = [fits.getdata(p, ext=0, header=False) for p in paths]
    errors  = [fits.getdata(p, ext=1, header=False) for p in paths]
    quality = [fits.getdata(p, ext=2, header=False) for p in paths]  # check actual values
    
    # -------------------------------------------------------------------------
    # 2) Check the difference in Y-axis size between consecutive images
    #    (i.e., shape[0] differences). We'll try to correct for this.
    # -------------------------------------------------------------------------
    y_sizes = [img.shape[0] for img in images]  # list of Y-dim sizes
    # np.diff(...) will give consecutive differences: y_sizes[1]-y_sizes[0], etc.
    # multiply by -1 so that we can interpret positive/negative as needed
    y_diffs = -1 * np.diff(y_sizes)
    if verbose and len(y_diffs) > 0:
        print(
            "Differences in Y-axis pixel counts between images:", y_diffs,
            "\nThe code will attempt to correct alignment. If you want to inspect "
            "the intermediate stacked array, set pre_combine=True."
        )

    # -------------------------------------------------------------------------
    # 4) Allocate large arrays to hold all images with potential Y-offset
    #    We create final arrays with shape:
    #       (N_images, max_possible_Y, X)
    #    so that we can shift each image up/down as needed.
    # -------------------------------------------------------------------------
    n_images    = len(images)
    ref_img     = images[0]            # reference image
    ref_y, ref_x = ref_img.shape       # e.g. (Y0, X0)
    # Our final "stack" in Y dimension: we do 2*ref_y + 1 to accommodate shifts
    big_y_size  = ref_y * 2 + 1
    if np.all(y_diffs == 0):
        big_y_size = ref_y
        ref_x = ref_x
    # Initialize with np.nan for flux/error, and large dummy values for quality
    # (so later we can compute an average or do a mask if needed)
    final_image = np.full((n_images, big_y_size, ref_x), np.nan)
    final_error = np.full((n_images, big_y_size, ref_x), np.nan)
    final_quality = np.full((n_images, big_y_size, ref_x), 1000, dtype=float)

    # -------------------------------------------------------------------------
    # 5) Populate the big arrays with each image, offset as needed
    # -------------------------------------------------------------------------
    for i, (img, err, qual) in enumerate(zip(images, errors, quality)):
        
        # Y size of current image
        cur_y_size = img.shape[0]
        shift = 0
        if i > 0:
            shift = y_diffs[i-1]  # might be negative or positive
            if verbose:
                print(f"Applying an additional shift of {shift} in Y dimension.")
        # Range in final stack
        start_y = 0 #+ shift
        end_y   = cur_y_size #+ shift
        
        # Insert data
        final_image[i, start_y:end_y, :]   = img
        final_error[i, start_y:end_y, :]   = err
        final_quality[i, start_y:end_y, :] = qual

    # -------------------------------------------------------------------------
    # 6) If 'pre_combine' is True, return the stacked arrays for debugging
    # -------------------------------------------------------------------------
    if pre_combine:
        print("pre_combine set to true save will not do nothing")
        return final_image, final_error, final_quality

    # -------------------------------------------------------------------------
    # 7) Now actually combine them (Weighted Mean)
    #    If method="mean", we do a weighted combination with 1/error^2
    #    (assuming typical inverse-variance weighting).
    # -------------------------------------------------------------------------
    # Avoid divide-by-zero => treat error=0 as np.nan
    final_error[final_error == 0] = np.nan
    final_image[final_image == 0] = np.nan

    # Weights = 1 / (error^2)
    weights = 1.0 / (final_error**2)

    # Sum of weights along the "image" axis (axis=0).
    sum_weights = np.nansum(weights, axis=0)  # shape = (big_y_size, X)
    # If sum_weights=0 => set it to nan to avoid /0
    sum_weights[sum_weights == 0] = np.nan

    # Weighted flux
    flux_comb  = np.nansum(final_image * weights, axis=0) / sum_weights

    # Propagated error = 1 / sqrt(sum_of_weights)
    error_comb = 1.0 / np.sqrt(sum_weights)

    # Combine quality by a simple average (or you could do min, max, etc.)
    quality_comb = np.nanmean(final_quality, axis=0)
    # Convert to int if that makes sense for your context
    quality_comb = quality_comb.astype(int)

    # Clean up any remaining NaNs
    flux_comb  = np.nan_to_num(flux_comb, nan=0.0)
    # For error, you might choose a large sentinel value or 0. 
    # The '10e+10' in your code is 1e+11, which is quite large. 
    # We'll keep that logic:
    error_comb = np.nan_to_num(error_comb, nan=1e11)
    
    
    if save:
        header = [fits.getheader(paths[0], ext=p) for p in range(3)]
        files_used = ', '.join([os.path.basename(path) for path in paths]) 
        comb_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Current date in YYYY-MM-DD format
        comb_method = method        # For example, the combination method used 
        for h in header:
            h['FILES'] = (files_used, "files comb")
            h['COMBDATE'] = (comb_date, "Date comb")
            h['COMB_MTH'] = (comb_method, "Meth to comb")
            h['PERSON'] = (person, "who COMB")
            h['TELCOR'] = (str(telluric_corrected), "Tell corrected")    
        primary_hdu = fits.PrimaryHDU(data=flux_comb)
        primary_hdu.header = header[0]        # Example header keyword

        # Create an ImageHDU for the second data array and add a header keyword
        hdu2 = fits.ImageHDU(data=error_comb)
        hdu2.header = header[1]                # Example header keyword
        # Create an ImageHDU for the third data array and add a header keyword
        hdu3 = fits.ImageHDU(data=quality_comb)
        hdu3.header = header[2]                 # Example header keyword
        # Combine the HDUs into an HDUList
        hdulist = fits.HDUList([primary_hdu, hdu2, hdu3])
        if isinstance(save,bool):
            print("check")
            dir_path = str(Path(paths[0]).parent.parent.parent.parent) + "/Science"
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            path_ = os.path.basename(paths[0]).split("_")
            path_0 = "_".join([path_[0]]+path_[2:])
            save_path = os.path.join(dir_path,path_0).replace("_OB1","").replace("_OB2","").replace("_OB3","").replace(".fits","_COMB.fits")
            hdulist.writeto(save_path, overwrite=True)
            print("Will be saved in",save_path)
    # -------------------------------------------------------------------------
    # 8) Return final combined data
    # -------------------------------------------------------------------------
    return flux_comb, error_comb, quality_comb

def off(hdr):
    Y_1 = hdr['ESO SEQ CUMOFF Y']
    return  -Y_1

def Ref(header_1,header_2):
    # Extract and convert header values to float
    RA_1 = float(header_1['RA'])
    RA_2 = float(header_2['RA'])
    DEC_1 = float(header_1['DEC'])
    DEC_2 = float(header_2['DEC'])
    dif =  np.sqrt( (RA_1-RA_2)**2 + (DEC_1-DEC_2)**2  )
    return dif * 3600

def ToAng(hdr):
    """
    Converts header information to an array of arcsecond values.

    Parameters:
    hdr (astropy.io.fits.Header): FITS header containing necessary keywords.

    Returns:
    numpy.ndarray: Array of arcsecond values.
    """
    # Extract and convert header values to float
    Len = float(hdr['NAXIS2'])
    Pix = float(hdr['CRPIX2'])
    Val = float(hdr['CRVAL2'])
    Del = float(hdr['CDELT2'])
    
    # Generate pixel indices (1-based to match FITS standard)
    pix = np.arange(1, Len + 1)
    
    # Calculate arcsecond values
    arcsec = pix * Del + (Val - Del * Pix)
    
    return arcsec

def correct(hdrs):
    """
    header list
    TODO change this name i dont really like this 
    """
    Ang = [ToAng(hdr) for hdr in hdrs]
    r = [off(hdr) for hdr in hdrs] #off
    return Ang,r

def extract_wavelength_array(fits_file):
    # Open the FITS file
    with fits.open(fits_file) as hdul:
        header = hdul[0].header  # Assuming the data is in the primary HDU
        data = hdul[0].data

    # Check if WCS information is available in the header
    if 'CTYPE1' in header and 'CDELT1' in header and 'CRVAL1' in header:
        # Get WCS parameters
        crpix1 = header.get('CRPIX1', 1)  # Reference pixel (default to 1 if missing)
        crval1 = header['CRVAL1']         # Reference value
        cdelt1 = header['CDELT1']         # Wavelength increment
        
        # Compute the wavelength array
        n_pix = data.shape[0] if data.ndim == 1 else data.shape[1]  # Length of the spectral axis
        pixel_indices = np.arange(1, n_pix + 1)  # FITS is 1-indexed
        wavelength = crval1 + (pixel_indices - crpix1) * cdelt1

        return wavelength * 10 #in Angstroms
    else:
        raise ValueError("The FITS file does not contain WCS information for the spectral axis.")

def flux_correction(image_in_counts,Re,exptime,airmass,gain,bin,extinction):
    return image_in_counts * (Re/exptime)*(10**(0.4*airmass*extinction))/(bin/gain) #why here is a 0 ? xd theory is because is NIR band

# pieces = 22
#     slice_ = 1125
#     n = 6000
#     n_total = image_in_counts.shape[1]
#     if band=="VIS" or band=="UVB":
#         pieces = 21
#         slice_=n_total//21
#         n = 12159

def flux_correctionv2(image_in_counts,Re,exptime,airmass,band,separations=None,gain=2.12,bin=1,extinction=0,mask_image_x=None,do_mask=False): #the seted values are the values from Infrared
    print("runing flux correction \n")
    pieces = 22
    slice_ = 1125
    n = 6000
    n_total = image_in_counts.shape[1]
    if band=="VIS" or band=="UVB":
        pieces = 21
        slice_=n_total//21
        n = 12159
    keywords_function = locals()
    #if they reduce NIR in 22 pieces
    #vis was in 21
    #PCA2<-PCA.L0[[uqi]]*(Re/exptime)*(10**(0.4*airmass*0))/(1/2.12)
    #PCA2<-PCA.L0[[uqi]]*(Re/exptime)*(10**(0.4*airmass*extintion))/(binn/gain)
    image_flux = image_in_counts * (Re/exptime)*(10**(0.4*airmass*0))/(bin/gain) #why here is a 0 ? xd theory is because is NIR band
    PCAMask = np.zeros_like(image_flux)
    if do_mask:
        for j in range(image_in_counts.shape[0]):
            C0 = image_in_counts[j]*0
            for i in range(0,pieces):#why  (in our case is 23 for the array differences)? something with the band maybe? 
                C01 = image_flux[j,slice_*i:slice_*(i+1)]
                #spline
                C0[slice_*i:slice_*(i+1)] = csaps(np.arange(slice_), C01, np.arange(slice_), smooth=0.25)
            nom = np.where(np.abs(image_flux[j]-C0)>5*1.4826*mad(image_flux[j]-C0))[0] #why 5 times ? xd
            C1 = deepcopy(image_flux[j])
            C1a = deepcopy(C1)
            Xpos = np.arange(len(C1),dtype=float) 
            C1a[nom] = np.nan
            for i in range(0,pieces):
                c1 = C1[slice_*i:slice_*(i+1)]
                c1a = C1a[slice_*i:slice_*(i+1)]
                Xpos1 = Xpos[slice_*i:slice_*(i+1)]
                weights = np.ones_like(Xpos1)
                no_fit = np.where(np.isnan(c1a))[0]
                if len(no_fit) != 0:
                    weights[no_fit] = 1e-12
                C1[slice_*i:slice_*(i+1)] = csaps(Xpos1, c1, Xpos1, smooth=0.6,weights=weights)
            #i guess this has 2 options
            nom0 = np.where(np.abs(image_flux[j]-C1)>5*1.4826*mad(image_flux[j]-C1))[0]
            #in vis they put 12159 
            if band in ["VIS","NIR"]:
                nom0a = np.where(np.logical_or(np.logical_and((image_flux[j,0:n]-C1[0:n])>0, np.abs(image_flux[j,0:n]-C1[0:n])),\
                    np.logical_and((image_flux[j,0:n]-C1[0:n])<0,np.abs(image_flux[j,0:n]-C1[0:n])>5*1.4826*mad(image_flux[j]-C1))))[0]
                nom0b = np.where(np.logical_or(np.logical_and((image_flux[j,n:n_total]-C1[n:n_total])>0, np.abs(image_flux[j,n:n_total]-C1[n:n_total])),\
                    np.logical_and((image_flux[j,n:n_total]-C1[n:n_total])<0,np.abs(image_flux[j,n:n_total]-C1[n:n_total])>5*1.4826*mad(image_flux[j]-C1))))[0]
                nom0 = np.concatenate((nom0a,nom0b+n))
            
            C3 = np.ones_like(image_in_counts[j],dtype=float)
            C3[nom0] = 0
            PCAMask[j] = C3
    #print(PCAMask)
    #if False:
    #   PCA2[PCAMask.astype(bool)] = np.nan
    #  return inpaint_nans(PCA2)
    if mask_image_x:
        image_flux[:,int(max(mask_image_x)):] = 0 #value: desde
        image_flux[:,:int(min(mask_image_x))] = 0 #:value means hasta
    
    return image_flux#,keywords_function#remove_nan(PCA2,PCAMask)
    

def create_fits_with_table(data_array, header_dict=None, extra_table=None, table_columns=None, output_filename="output.fits"):
    """
    Create a FITS file from a 1D array with an optional header and extra table.

    Parameters:
    - data_array: 1D array of data to save in the primary HDU.
    - header_dict: Dictionary containing header key-value pairs to add to the primary header.
    - extra_table: 2D array or structured data for the additional table (optional).
    - table_columns: List of tuples (name, format, data) for creating table columns if extra_table is None.
    - output_filename: The output FITS filename.
    """
    # Create the primary HDU
    primary_hdu = fits.PrimaryHDU(data=data_array)

    # Add custom header entries if provided
    if header_dict:
        header = primary_hdu.header
        for key, value in header_dict.items():
            header[key] = value

    # Create a list to hold HDUs
    hdulist = [primary_hdu]

    # Add an additional table HDU if needed
    if extra_table is not None:
        # Automatically determine column format
        col_list = []
        for i, col_data in enumerate(extra_table.T):
            col_name = f'COL{i+1}'
            col_list.append(fits.Column(name=col_name, format='E', array=col_data))
        table_hdu = fits.BinTableHDU.from_columns(col_list)
        hdulist.append(table_hdu)

    elif table_columns:
        # Create a table from provided columns
        cols = [fits.Column(name=name, format=fmt, array=data) for name, fmt, data in table_columns]
        table_hdu = fits.BinTableHDU.from_columns(cols)
        hdulist.append(table_hdu)

    # Write to file
    hdul = fits.HDUList(hdulist)
    hdul.writeto(output_filename, overwrite=True)
    #print(f"FITS file '{output_filename}' created successfully.")


def image_response_reader(image_path,response_path,**kwargs):
    #TODO just calculate the flux for one method remove the other is not way to use the pca in UVB meanwhile in vis and nir the "median" method dosent make any sense
    print(f"read {image_path} and {response_path}")
    ##############
    fits_image = fits.open(image_path)
    #data,header = fits_image[0].data,fits_image[0].header
    #data_error = fits_image[2].data#,fits_image[0].header
    #quality_pixel = fits_image[2].data #https://ftp.eso.org/pub/dfs/pipelines/xshooter/xsh-manual-12.0.pdf page 112
    data,header  = fits.getdata(image_path, ext=0, header=True)
    errors,header_errors  = fits.getdata(image_path, ext=1, header=True)
    quality,quality_header = fits.getdata(image_path, ext=2, header=True)
    
    #data_clean[:,int(max(mask_image_x)):] = 0
    #data_clean[:,int(max(mask_image_x)):] = 0
    startH = header['ESO TEL AIRM START']
    endH = header['ESO TEL AIRM END']
    airmass = (startH + endH)/2
    exptime = header["EXPTIME"]
    band = header["HIERARCH ESO SEQ ARM"]
    wavelength = ((header["CRVAL1"] + header["CDELT1"] * np.arange(data.shape[1]))).astype(np.float32) #this is nm
    if isinstance(response_path,str):
        response_fits = fits.open(response_path)
        header_response = response_fits[0].header 
        if header_response["HIERARCH ESO SEQ ARM"] == band:
            w_Re = response_fits[1].data["LAMBDA"].astype(np.float32)
            argmin_w = np.where(wavelength[0] == w_Re)[0][0]
            argmax_w = np.where(wavelength[-1] == w_Re)[0][0]
            Re = response_fits[1].data["RESPONSE"][argmin_w:argmax_w+1]
            #plt.plot(response_fits[1].data["LAMBDA"],response_fits[1].data["RESPONSE"])
            #plt.plot(response_fits[1].data["LAMBDA"][argmin_w:argmax_w+1],Re)
            #plt.show()
        else:
            return print(f"the Band of image and respondes are not the same")
    extinction_model = fits.getdata(f"{module_dir}/extinction/xsh_paranal_extinct_model_{band.lower()}.fits") #this 
    extinction = np.interp(wavelength,extinction_model["lambda"],extinction_model["extinction"])#, smooth=1)#.values
    #plt.plot(extinction_model["lambda"],extinction_model["extinction"])
    #plt.plot(wavelength,extinction)
    #plt.show()
    try:
        gain = header["ESO DET OUT1 CONAD"] 
        bin = header['ESO DET WIN1 BINX'] 
    except KeyError as e:
        print(f"We dont found the keys {e}")
        gain = 2.12
        bin = 1
    return {'data':data,'header':header,'errors':errors,"header_errors":header_errors,'quality':quality,"quality_header":quality_header,'airmass':airmass,'exptime':exptime,"band":band,'extinction':extinction,'bin':bin,'gain':gain,'response':Re,"image_path":image_path,"response_path":response_path}


