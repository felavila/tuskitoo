from astropy.io import fits
from typing import Union, List
import numpy as np 
from tuskitoo.sky_sub.tools import correct #? very unclear name for a function think more about it
from pathlib import Path 
import os 
import datetime
#flux_clipped_corrected_pre_combine[i,Lx- au0:Lx+au1,:] = self.flux_clipped_corrected[n]/telluric

def apply_telluric_correction(path_images: Union[str, List[str]], 
                            path_tellurics: Union[str, List[str]], 
                            save: bool = False) -> List:
    """
    Apply telluric correction to one or more image FITS files using corresponding telluric files.

    Parameters
    ----------
    path_images : Union[str, List[str]]
        A single file path or a list of file paths to the image FITS file(s).
    path_tellurics : Union[str, List[str]]
        A single file path or a list of file paths to the telluric FITS file(s).
    save : bool, optional
        If True, saves the corrected images to new FITS files.
    
    Returns
    -------
    List
        A list containing the telluric-corrected image data arrays.
    
        
    Raises
    ------
    ValueError
        If the number of image paths and telluric paths differ, or if a telluric file does not contain 2 HDUs.
    """
    
    corrected_images = []

    # Ensure the inputs are lists
    if isinstance(path_images, str):
        path_images = [path_images]
    if isinstance(path_tellurics, str):
        path_tellurics = [path_tellurics]

    # Validate that the number of image and telluric files match
    if len(path_images) != len(path_tellurics):
        raise ValueError("Error: The number of image paths and telluric paths are different.")

    # Process each pair of image and telluric file
    for idx, (img_path, tel_path) in enumerate(zip(path_images, path_tellurics)):
        # Open the image and telluric FITS files
        image_hdulist = fits.open(img_path)
        tel_hdulist = fits.open(tel_path)
        
        # Check if the telluric file contains exactly 2 HDUs
        if len(tel_hdulist) == 2:
            #raise ValueError("Telluric file must have exactly 2 HDUs. Found: {}".format(len(tel_hdulist)))
            n = 1
        # Get the telluric data from the second HDU (index 1)
            telluric_data = tel_hdulist[1].data
        elif len(tel_hdulist) == 5:
            n=4
            telluric_data = tel_hdulist[4].data['mtrans']
        # Apply the telluric correction (element-wise division)
        image_data = image_hdulist[0].data
        corrected_data = image_data / telluric_data
        
        # Optionally save the corrected file
        if save:
            # Update the header to indicate telluric correction
            image_hdulist[0].header['TELCOR'] = (True, "Telluric corrected")
            
            # Create a new filename by appending '_TELL_COR.fits'
            output_filename = img_path.replace('.fits', '_TELL_COR.fits')
            
            # Replace the image data with the corrected data
            image_hdulist[0].data = corrected_data
            
            # Append the telluric HDU to the HDUList (optional; may be useful for reference)
            new_tel_hdu = fits.ImageHDU(data=telluric_data, header=tel_hdulist[n].header)
            image_hdulist.append(new_tel_hdu)
            
            # Write the updated HDUList to a new file, overwriting if necessary
            image_hdulist.writeto(output_filename, overwrite=True)
            print(f"Saved corrected file: {output_filename}")
            
            # Close the file to free resources
            image_hdulist.close()
        else:
            # If not saving, close the opened image file
            image_hdulist.close()

        # Close the telluric file
        tel_hdulist.close()

        # Append the corrected image data to our list
        corrected_images.append(corrected_data)
        
    return corrected_images

#TODO add it to the sky sub pipeline
def combine_2D_spectra(
    paths,
    method="mean",
    verbose=True,
    pre_combine=False
    ,save = "",person="F. Avila-Vera",telluric_corrected=True,sub_name='',extra_shif=[]):
    """
    Combine multiple 2D spectra (and their errors/quality arrays) into a single
    2D output using weighted combination. The function can also return the
    pre-combined arrays for inspection.

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
    # 3) Read FITS headers & apply 'correct' function for Y-offset alignment
    # -------------------------------------------------------------------------
    headers = [fits.getheader(p, ext=0) for p in paths]
    # This function presumably returns something like correct_[0][i], correct_[1][i].
    # Not shown in your snippet, but we'll trust it.
    correction_values = correct(headers)  # shape or structure depends on your code
    
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

    # Initialize with np.nan for flux/error, and large dummy values for quality
    # (so later we can compute an average or do a mask if needed)
    final_image = np.full((n_images, big_y_size, ref_x), np.nan)
    final_error = np.full((n_images, big_y_size, ref_x), np.nan)
    final_quality = np.full((n_images, big_y_size, ref_x), 1000, dtype=float)

    # -------------------------------------------------------------------------
    # 5) Populate the big arrays with each image, offset as needed
    # -------------------------------------------------------------------------
    for i, (img, err, qual) in enumerate(zip(images, errors, quality)):
        # 'correction_values' might be a 2D structure. In original code:
        #   an0 = correct_[0][i] - correct_[1][i]
        #   ind = np.argmin(abs(an0))
        # We replicate that logic here:
        an0 = correction_values[0][i] - correction_values[1][i]
        # Possibly 'an0' is an array; we pick the index that gives minimal shift
        idx = np.argmin(np.abs(an0))

        # Y size of current image
        cur_y_size = img.shape[0]

        # Example logic from your code:
        #   Lx = image.shape[0]
        #   au1 = Lx - ind
        #   au0 = Lx - au1
        # We'll rename them for clarity:
        top_part  = cur_y_size - idx    # au1
        bottom_part = cur_y_size - top_part  # au0

        if verbose:
            print("Image", i, ": shape =", img.shape)
            print("  Y offset info => top_part =", top_part, ", bottom_part =", bottom_part)

        # 'move' is how many pixels we shift *this* image compared to the reference
        # Your code does: if i>0, move = y_diffs[i-1]
        # Possibly we do that, or we do an additive shift.
        # We'll replicate your logic:
        shift = 0
        extra_shif_ = 0
        if i > 0:
            shift = y_diffs[i-1]  # might be negative or positive
            if verbose:
                print(f"Applying an additional shift of {shift} in Y dimension.")
        if i>0 and extra_shif:
            extra_shif_ = extra_shif[i-1]
        # Range in final stack
        start_y = cur_y_size - bottom_part + shift+extra_shif_
        end_y   = cur_y_size + top_part + shift+extra_shif_
        
        # Insert data
        final_image[i, start_y:end_y, :]   = img
        final_error[i, start_y:end_y, :]   = err
        final_quality[i, start_y:end_y, :] = qual

    # -------------------------------------------------------------------------
    # 6) If 'pre_combine' is True, return the stacked arrays for debugging
    # -------------------------------------------------------------------------
    if pre_combine:
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
            dir_path = str(Path(paths[0]).parent.parent.parent) + "/Science"
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            path_ = os.path.basename(paths[0]).split("_")
            path_0 = "_".join([path_[0]]+path_[2:])
            save_path = os.path.join(dir_path,path_0).replace(".fits",f"_COMB{sub_name}.fits")
            hdulist.writeto(save_path, overwrite=True)
            print("Will be saved in",save_path)
    # -------------------------------------------------------------------------
    # 8) Return final combined data
    # -------------------------------------------------------------------------
    return flux_comb, error_comb, quality_comb
