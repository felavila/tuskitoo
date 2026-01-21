# Expectra2D.py
# -------------
# Main class to handle 2D spectra and extract the spectra.
#
# This file supports:
# - FITS or numpy-array 2D inputs
# - Generic FITS headers (not only X-Shooter): wavelength computed from CRVAL1/CDELT1(or CD1_1)/CRPIX1/CUNIT1
# - Two extraction approaches:
#     (A) PSF-model fitting per column via parallel_fit (gaussian/moffat)
#     (B) Trace + IRAF-like sky subtraction + (optional) optimal extraction (recommended for asymmetric PSFs)
#
# NOTE:
# - The "parallel_fit" path is model-based and can struggle when the spatial PSF is non-Gaussian/asymmetric.
# - The "optimal_extract_from_trace" path is robust and closer in spirit to IRAF/apall.
#

from astropy.io import fits
import copy
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import numpy as np
import warnings

from .utils import (
    find_signal,
    guess_picks_image,
    gaussian_with_error,
    integrated_gaussian,
    integrated_moffat,
    moffat_with_error,
)
from .fitting import parallel_fit
from tuskitoo.utils.utils import sigma_clip_1d
import pandas as pd
import pickle


def df_get(df, key, default=None):
    return df[key] if key in df.columns else default


class Expectra2D:
    "Main class to handle 2D spectra and extract the spectra"

    def __init__(
        self,
        object,
        center_cut=None,
        size_cut=None,
        distances=None,
        verbose=False,
        header=None,
        **kwargs,
    ):
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

        Notes:
        ------
        For FITS inputs, this code tries to interpret common conventions:
        - If len(hdul) >= 3: [0]=data, [1]=error, [2]=quality
        - If len(hdul) == 1: [0]=data, error and quality default to ones/zeros.

        Orientation handling:
        ---------------------
        If the data comes with a transposed orientation (dispersion axis shorter than spatial),
        this class transposes it so that:
            shape = (ny_spatial, nx_dispersion)
        """
        self.band = kwargs.get("band", None)
        self.name = kwargs.get("name", None)
        self.header = header

        if isinstance(object, str) and object.endswith("fits"):
            print(object)
            self.object = object
            self.fits_image = fits.open(object)

            if len(self.fits_image) >= 3:
                print("Fits image has a len bigger than 1 be aware of in what layer is the image")
                self.original_data, self.header = self.fits_image[0].data, self.fits_image[0].header
                self.original_error = self.fits_image[1].data
                self.original_quality = self.fits_image[2].data
            elif len(self.fits_image) == 1:
                self.original_data, self.header = self.fits_image[0].data, self.fits_image[0].header
        elif isinstance(object, np.ndarray) and len(object.shape) == 2:
            self.object = object
            print("Object is a numpy array you can also add the Header later")
            self.original_data = np.nan_to_num(self.object, 0)
        else:
            raise Exception("Check if is a fits file or numpy array-len(shape) = 2")

        self.get_header_keys()

        if not hasattr(self, "original_error"):
            self.original_error = np.ones_like(self.original_data)
        if not hasattr(self, "original_quality"):
            self.original_quality = np.zeros_like(self.original_data)

        # Ensure orientation: (ny, nx) = (spatial, dispersion)
        if self.original_data.shape[1] < self.original_data.shape[0]:
            self.original_data = self.original_data.T
            self.original_quality = self.original_quality.T
            self.original_error = self.original_error.T

        self.center_cut = center_cut or self.original_data.shape[0] // 2
        self.size_cut = size_cut or 40

        self.cut_data = Expectra2D.cut_2d_image(
            self.original_data, center=self.center_cut, size=size_cut, verbose=True
        )
        self.cut_error = Expectra2D.cut_2d_image(
            self.original_error, center=self.center_cut, size=size_cut, verbose=False
        )
        self.cut_quality = Expectra2D.cut_2d_image(
            self.original_quality, center=self.center_cut, size=size_cut, verbose=False
        )

        self.stacked_median = np.nanmedian(self.cut_data, axis=1)

    def get_header_keys(self, distances=None):
        """
        Retrieve and store a subset of header keys relevant for further processing.

        Parameters:
        -----------
        distances : optional
            If provided as a dictionary, it may be used for additional processing related to distances.

        Notes:
        ------
        If no header is available, a warning is issued.

        Instrument-agnostic behavior:
        -----------------------------
        - The code tries to read common WCS/spectral keywords: CRVAL1, CDELT1 or CD1_1, CRPIX1, CUNIT1
        - If ESO-only keys exist (like "ESO SEQ ARM"), it uses them to set 'band'
        - If not, it tries fallbacks for other instruments (e.g., GTC) using ARM/GRISM/GRATING/DISPERSR/FILTER/INSTRUME
        """
        if not self.header:
            warnings.warn(
                "Warning: 'self.header' is not defined. "
                "Please add a header to the class to take extra advantage of the code.",
                UserWarning,
            )
            return

        self.relevant_keywords_header = {
            i: self.header[i]
            for i in [
                "ORIGIN",
                "INSTRUME",
                "OBJECT",
                "NAXIS1",
                "CRVAL1",
                "CD1_1",
                "CDELT1",
                "CRPIX1",
                "CUNIT1",
                "BUNIT",
                "CD2_2",
                "ESO SEQ ARM",
            ]
            if i in list(self.header.keys())
        }

        # SAFE: OBJECT might also be missing in some products
        self.name = self.relevant_keywords_header.get("OBJECT", self.name)

        # SAFE: ESO-only keyword
        if "ESO SEQ ARM" in self.relevant_keywords_header:
            self.band = self.relevant_keywords_header["ESO SEQ ARM"]
        else:
            # Non-ESO fallback (e.g. GTC)
            self.band = (
                self.band
                or self.header.get("ARM")
                or self.header.get("GRISM")
                or self.header.get("GRATING")
                or self.header.get("DISPERSR")
                or self.header.get("FILTER")
                or self.header.get("INSTRUME")
                or "UNKNOWN"
            )

    def arc_to_pix(self, value):
        distances_pix = value / self.relevant_keywords_header["CD2_2"]
        return distances_pix

    def run_parallel_fit(
        self,
        n_picks=2,
        pixel_limit=[],
        bound_sigma=[2],
        distribution="gaussian",
        param_value=None,
        param_limit=None,
        param_fix=None,
        no_use_real_error=False,
        initial_separation=[],
        initial_center=None,
        **kwargs,
    ):
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
        if n_picks > 1:
            picks = np.array([guess_picks_image(i, n_picks) for i in self.cut_data.T])
            if not initial_center:
                print("Given a init_center was not added we will guess one")
                initial_center = np.nanmedian(picks[:, 0])
            if len(initial_separation) != n_picks - 1:
                print("Given a init_separation  was not added we will guess it")
                initial_separation = np.nanmedian(picks, axis=0)[1:] - initial_center
        if n_picks == 1 and not initial_separation:
            initial_center = np.argmax(np.nanmedian(self.cut_data, axis=1))
            initial_separation = []

        print("initial_center:", initial_center, "initial_separation:", initial_separation)
        if isinstance(initial_separation, (float, int)):
            initial_separation = [initial_separation]

        mask_list = []

        band = kwargs.get("band", self.band)
        if band == "NIR":
            mask_list = [[5800, 7005], [13500, 15900]]  # teluric
        elif band == "VIS":
            mask_list = [[0, 1000], [int(self.cut_data.shape[1] - 50), int(self.cut_data.shape[1] - 1)]]
        elif band == "UVB":
            mask_list = [[0, 500]]

        error = self.cut_error
        data = self.cut_data

        if no_use_real_error:
            error = np.ones_like(self.cut_data)

        self.keywords_fit = locals()
        self.keywords_fit.pop("self")
        if "picks" in self.keywords_fit.keys():
            self.keywords_fit.pop("picks")

        self.fit_result = parallel_fit(
            data,
            error,
            n_picks,
            initial_center=initial_center,
            initial_separation=initial_separation,
            pixel_limit=pixel_limit,
            bound_sigma=bound_sigma,
            distribution=distribution,
            mask_list=mask_list,
            param_value=param_value,
            param_limit=param_limit,
            param_fix=param_fix,
        )

    def array_to_pandas(self, max_iter=5, sigma=2, region_size=20, over_write=False, images=[]):
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
        num_source = results.get("num_source")
        distribution = results.get("distribution")
        image_shape = results.get("normalized_image").shape
        num_parameter = results.get("parameter_number")
        normalize_matrix = results.get("normalize_matrix")
        values = results.get("value").copy()
        std = results.get("std").copy()

        dist_func = gaussian_with_error if distribution == "gaussian" else moffat_with_error
        int_func = integrated_gaussian if distribution == "gaussian" else integrated_moffat

        flux_columns = [f"flux_{n}" for n in range(1, num_source + 1)]
        extra_columns = ["chisqr", "redchi", "aic", "bic", "rsquared", "n_pixel", "x_num"]

        result_panda = pd.DataFrame()
        result_panda[[("value_" + i) if ("height" not in i) else ("value_norm_" + i) for i in name_params]] = values
        result_panda[[("std_" + i) if ("height" not in i) else ("std_norm_" + i) for i in name_params]] = std
        result_panda[extra_columns] = results.get("extra_params")

        # denormalize heights
        values[:, ["height" in i for i in name_params]] = values[:, ["height" in i for i in name_params]] * normalize_matrix
        std[:, ["height" in i for i in name_params]] = std[:, ["height" in i for i in name_params]] * normalize_matrix

        # convert separations -> centers
        if any("separation" in i for i in result_panda.columns):
            sep_to_cen = result_panda["value_center_1"].values[:, None] + result_panda[
                [i for i in result_panda.columns if "value_separation" in i]
            ].values
            std_sep_to_cen = np.sqrt(
                result_panda["std_center_1"].values[:, None] ** 2
                + result_panda[[i for i in result_panda.columns if "std_separation" in i]].values ** 2
            )
            result_panda[[f"value_center_{i}" for i in range(1, num_source + 1) if i != 1]] = sep_to_cen
            result_panda[[f"std_center_{i}" for i in range(1, num_source + 1) if i != 1]] = std_sep_to_cen

            values[:, ["separation" in i for i in name_params]] = sep_to_cen
            std[:, ["separation" in i for i in name_params]] = std_sep_to_cen

        re_shape_results_m = np.concatenate(
            (values.reshape(-1, num_source, num_parameter), std.reshape(-1, num_source, num_parameter)), axis=2
        )
        multiple_dist, error_dist = dist_func(np.arange(0, image_shape[0])[:, np.newaxis, np.newaxis], *re_shape_results_m.T)
        multiple_dist = np.nan_to_num(np.moveaxis(multiple_dist, 0, 1), 0)
        error_dist = np.nan_to_num(np.moveaxis(error_dist, 0, 1), 0)
        image_2d_model = multiple_dist.sum(axis=0)

        fluxes, errors = int_func(*re_shape_results_m.T)
        result_panda[["raw_" + i for i in flux_columns]] = fluxes.T
        result_panda[["std_" + i for i in flux_columns]] = errors.T

        result_panda[flux_columns] = np.array(
            [
                sigma_clip_1d(
                    result_panda["raw_" + i].values,
                    max_iter=max_iter,
                    sigma=sigma,
                    region_size=region_size,
                    error=result_panda["std_" + i].values,
                )
                for i in flux_columns
            ]
        ).T

        # wavelength solution (instrument-agnostic)
        hdr = self.header if self.header is not None else {}

        crval1 = hdr.get("CRVAL1", self.relevant_keywords_header.get("CRVAL1", 0.0))
        cdelt1 = hdr.get("CDELT1", hdr.get("CD1_1", self.relevant_keywords_header.get("CD1_1", 1.0)))
        crpix1 = hdr.get("CRPIX1", self.relevant_keywords_header.get("CRPIX1", 1.0))  # FITS is 1-indexed
        cunit1 = (hdr.get("CUNIT1", self.relevant_keywords_header.get("CUNIT1", "")) or "").strip().lower()

        unit_factor = 1.0
        if cunit1 in ["nm", "nanometer", "nanometers"]:
            unit_factor = 10.0
        elif cunit1 in ["um", "micron", "microns", "µm"]:
            unit_factor = 1e4
        elif cunit1 in ["m", "meter", "meters"]:
            unit_factor = 1e10
        elif cunit1 in ["a", "aa", "angstrom", "angstroms", "å"]:
            unit_factor = 1.0
        # if unit is missing/unknown, assume Angstrom

        pix = result_panda["n_pixel"].values.astype(float)
        lam = (crval1 + (pix + 1.0 - crpix1) * cdelt1) * unit_factor
        result_panda["wavelength"] = lam

        result_panda["units_flux"] = len(result_panda) * [self.relevant_keywords_header.get("BUNIT", "flux")]

        if len(images) > 0:
            if len(images) == num_source:
                print(f"setting names of images {np.arange(1, num_source+1).astype(str).tolist()} to {images}")
                result_panda = result_panda.rename(
                    columns={
                        i: i.replace(i.split("_")[-1], images[int(i.split("_")[-1]) - 1])
                        for i in result_panda.columns.values
                        if i.split("_")[-1] in np.arange(1, num_source + 1).astype(str).tolist()
                    }
                )
                self.images = images
            else:
                print(f"The number of image ({images}) is different of the number of source ({num_source}) check it")

        if over_write or not hasattr(self, "results"):
            print("saving")
            self.results = {"result_panda": result_panda, "multiple_dist": multiple_dist, "image_2d_model": image_2d_model}
            return

        return results


    # -------------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------------
    def save_fit_keywords_as_pickle(self, filename):
        """
        Save the dictionary of fitting keywords (parameters used in the fit) to a pickle file.

        Parameters:
        -----------
        filename : str
            The base filename for saving (without extension).
        """
        try:
            filename = f"{filename}.pickle"
            with open(filename, "wb") as f:
                pickle.dump(self.keywords_fit, f)
            print(f"Dictionary successfully saved to {filename}")
        except Exception as e:
            print(f"An error occurred while saving the dictionary: {e}")

    def save_spectra_as_pickle(self, save=None, band=None):
        """_summary_

        Args:
            save (_type_, optional): _description_. Defaults to None.
            band (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        result = self.results["result_panda"]
        band = band or self.band
        if band is None:
            band = "?"
            print("Warning band not found")
        dic_result = {}
        for i in self.images:
            band = band.lower()
            dic_result[f"{i}_{band}"] = {
                "wavelength": result["wavelength"].values,
                "flux": result[f"flux_{i}"].values,
                "std": result[f"std_flux_{i}"].values,
                "band": band,
            }
        if save:
            if len(list(dic_result.keys())) > 0:
                with open(f"{save}_{band}.pickle", "wb") as file:
                    print("Save as", f"{save}_{band}.pickle")
                    pickle.dump(dic_result, file)
            else:
                print("Empty dictionary ")
        else:
            return dic_result

    def save_to_fits(self, filename, person="F. Avila-Vera"):
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
        if not hasattr(self, "results"):
            raise AttributeError(
                "Error: 'self.results' is not defined. \n"
                "Could be an Error in runing 'array_to_pandas'"
            )
        df = self.results["result_panda"]
        flux_columns = [i for i in df.columns.values if "flux" in i.split("_")[0]]
        flux_columns_std = ["std_" + i for i in flux_columns]
        columns_to_save = ["wavelength"] + flux_columns + flux_columns_std

        n_rows = len(df)
        dtype = [(col, ">f4") for col in columns_to_save]
        data = np.empty(n_rows, dtype=dtype)
        for col in columns_to_save:
            data[col] = df[col].values.astype(">f4")

        primary_hdu = fits.PrimaryHDU()
        if self.header is not None:
            for key, value in self.header.items():
                if "ESO" in key:
                    continue
                primary_hdu.header[key] = value
        if isinstance(self.object, str):
            primary_hdu.header["2DFILE"] = self.object

        table_hdu = fits.BinTableHDU(data)
        table_hdu.header["PERSON"] = (person, "who extract")

        hdul = fits.HDUList([primary_hdu, table_hdu])
        filename = f"{filename}_extracted_spectra.fits"
        hdul.writeto(filename, overwrite=True)
        print(f"FITS file '{filename}' created successfully.")

    # -------------------------------------------------------------------------
    # Plotting / diagnostics
    # -------------------------------------------------------------------------
    def plot_column(self):
        return

    def plot_data_model(self, n):
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
        if not hasattr(self, "results"):
            raise AttributeError(
                "Error: 'self.results' is not defined. \n" "try runing 'array_to_pandas' first"
            )
        df = self.results["multiple_dist"].T
        x_axis = np.arange(self.cut_data.shape[0])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 15), gridspec_kw={"height_ratios": [2, 1]})
        sumx = df[n].T.sum(axis=0)
        for dis in df[n].T:
            ax1.plot(x_axis, dis)
        ax1.plot(x_axis, self.cut_data.T[n])
        ax1.plot(x_axis, sumx)
        ax2.scatter(x_axis, self.cut_data.T[n] - df[n].T.sum(axis=0))
        ax2.axhline(0, ls="--")
        ax1.set_xlim(0, x_axis[-1])
        ax2.set_xlim(0, x_axis[-1])
        ax1.xaxis.label.set_size(40)
        ax1.yaxis.label.set_size(40)
        ax1.tick_params(
            which="both",
            bottom=False,
            top=False,
            left=True,
            right=False,
            length=10,
            width=2,
            labelsize=20,
            labelbottom=False,
        )
        ax2.tick_params(
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            length=10,
            width=2,
            labelsize=20,
            labelbottom=True,
        )
        plt.legend(loc="best", prop={"size": 24}, frameon=False)
        plt.show()

    def plot_spectra(
        self,
        add_error=False,
        add_raw=False,
        save="",
        force_pix=False,
        z_s=None,
        add_lines=False,
        rest_frame=False,
        flux_columns=None,
        cap_errors=True,
        cap_percentile=99.5,
        **kwargs,
    ):
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
        if not hasattr(self, "results"):
            raise AttributeError("Error: 'self.results' is not defined. \ntry runing 'array_to_pandas' first")

        df = self.results["result_panda"]

        wavelength = np.arange(len(df))
        xlabel = "pixel"
        ylabel = df["units_flux"].values[0] if "units_flux" in df.columns else "flux"

        if "wavelength" in df.columns and not force_pix:
            wavelength = df["wavelength"].values
            xlabel = "wavelength (A)"
            if rest_frame and z_s:
                wavelength = df["wavelength"].values / (1 + z_s)
                xlabel = "rest frame wavelength (A)"

        fig, ax = plt.subplots(1, 1, figsize=(35, 15))

        if not flux_columns:
            flux_columns = [i for i in df.columns.values if "flux" in i.split("_")[0]]

        colors = ["#377eb8", "#e41a1c", "#4daf4a"]
        ecolors = ["lightskyblue", "LightCoral", "LightGreen"]

        if add_error and cap_errors:
            print(f"Plotting note: yerr is capped at the {cap_percentile}th percentile for readability (plot only).")

        all_flux = []

        for i, flux in enumerate(flux_columns):
            flux_ = df[flux].values

            if add_raw and ("raw_" + flux) in df.columns:
                ax.plot(wavelength, df["raw_" + flux].values, label="raw_" + flux, alpha=0.7)

            error_ = None
            if add_error:
                std_col = "std_" + flux
                if std_col in df.columns:
                    error_ = df[std_col].values.astype(float).copy()
                    error_[~np.isfinite(error_)] = np.nan
                    if cap_errors:
                        cap = np.nanpercentile(error_, cap_percentile)
                        if np.isfinite(cap) and cap > 0:
                            error_ = np.clip(error_, 0, cap)

            ax.errorbar(
                wavelength,
                flux_,
                yerr=error_,
                color=colors[i % len(colors)],
                ecolor=ecolors[i % len(ecolors)],
                label=flux,
                alpha=0.9,
            )

            all_flux.append(flux_)

        all_flux = np.concatenate(all_flux)
        ylim_lower, ylim_upper = np.nanpercentile(all_flux, [1, 99.99])

        ax.tick_params(which="both", bottom=True, top=False, left=True, right=False, length=10, width=2, labelsize=35)

        xlim = kwargs.get("xlim", wavelength[[0, -1]])
        ylim = kwargs.get("ylim", [0, ylim_upper * 1.05])

        text_fontsize = kwargs.get("text_fontsize", 20)
        text_rotation = kwargs.get("text_rotation", 0)

        if z_s and add_lines:
            agn_lines = {
                "Lya": 1216,
                "CIV": 1549,
                "CIII_1909": 1909,
                "MgII": 2800,
                "HeII_4686": 4686,
                "Hβ": 4861,
                "OIII_4959": 4959,
                "OIII_5007": 5007,
                "OI_6300": 6300,
                "NII_6548": 6548,
                "Hα": 6563,
                "NII_6583": 6583,
                "SII_6716": 6716,
                "SII_6731": 6731,
            }

            for line_name, central_wavelength in agn_lines.items():
                if not rest_frame:
                    central_wavelength = central_wavelength * (1 + z_s)

                if max(xlim) > central_wavelength and min(xlim) < central_wavelength:
                    ax.axvline(central_wavelength, linestyle="--", color="k", linewidth=2, alpha=0.5)
                    ax.text(
                        central_wavelength,
                        ylim[1],
                        f" {line_name}",
                        fontsize=text_fontsize,
                        rotation=text_rotation,
                        verticalalignment="top",
                        color="k",
                        zorder=10,
                        horizontalalignment="left",
                    )

        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontsize(20)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # if wavelength decreases with pixel, flip the x-axis for display
        if (not force_pix) and ("wavelength" in df.columns):
            try:
                if np.isfinite(wavelength[0]) and np.isfinite(wavelength[-1]) and (wavelength[0] > wavelength[-1]):
                    ax.set_xlim(xlim)
                    ax.invert_xaxis()
                else:
                    ax.set_xlim(xlim)
            except Exception:
                ax.set_xlim(xlim)
        else:
            ax.set_xlim(xlim)

        ax.set_ylim(ylim)

        ax.xaxis.label.set_size(40)
        ax.yaxis.label.set_size(40)

        plt.legend(loc="best", prop={"size": 24}, frameon=False)

        if save:
            plt.savefig(f"images/{save}.jpg", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_cut_out(self):
        """
        Plot the 2D cut-out image and the stacked median profile.
        """
        norm_image = self.cut_data / self.cut_data.max(axis=0)
        vmin, vmax = np.nanpercentile(self.cut_data, [5, 95])
        fig, axs = plt.subplots(1, 2, figsize=(18, 5))
        im = axs[0].imshow(self.cut_data, aspect="auto", vmin=vmin, vmax=vmax)
        axs[0].set_title("2d cut")
        axs[0].set_xlabel("X-pixel")
        axs[0].set_ylabel("Y-pixel")

        plt.colorbar(im, ax=axs[0], label="normalized intensity")

        axs[1].plot(np.nanmedian(norm_image, axis=1), color="orange")
        axs[1].set_xlim(np.arange(len(np.nanmedian(norm_image, axis=1)))[[0, -1]])
        axs[1].axhline(0, ls="--")
        axs[1].set_title("stacked median")
        axs[1].set_xlabel("y-pixels")
        axs[1].set_ylabel("intensity")
        plt.tight_layout()
        plt.show()

    def run_cut_2d(self, center, size, verbose=False):
        return

    @staticmethod
    def cut_2d_image(image, center=None, size=None, verbose=False):
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
        if (image.shape[0] % 2) != 0:
            nan_row = np.full((1, image.shape[1]), np.nan)
            image = np.vstack([image, nan_row])
        if center is None:
            center = int(np.nanmedian(np.array([find_signal(i) for i in image.T])))
        if size is None:
            size = 70
        if verbose:
            print(f"cut center {center} and cut size {size}")
        return image[int(center - size // 2) : int(center + size // 2), :]

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
    ):
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

        if show_trace_centroid and len(xs_cent) > 0:
            ax_img.plot(xs_cent, y0s, lw=2, label="centroid trace")
            ax_img.legend(loc="best")

        plt.tight_layout()
        plt.show()

        if show_trace_centroid:
            return np.array(xs_cent), np.array(y0s)
        return None


    def run_parallel_fit_trace(
        self,
        n_picks=1,
        pixel_limit=None,
        bound_sigma=[2],
        distribution="gaussian",
        param_value=None,
        param_limit=None,
        param_fix=None,
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

        # ---- polynomial trace ----
        trace_poly_order=None,          # e.g. 2 or 3, None = disable
        trace_poly_robust=True,
        trace_poly_clip_sigma=4.0,
        trace_poly_max_iter=5,

        # ---- NEW: left-edge anchoring (fixes "left trace too low") ----
        trace_left_use_peak_until=200,   # use peak instead of centroid for x <= this
        trace_left_peak_half_window=6,   # +/- pixels around yc to search peak

        # ---- optional: left-weighted polynomial fit (optional) ----
        trace_poly_left_xmax=200.0,      # upweight points with x <= this in polyfit
        trace_poly_left_weight=6.0,      # 3-10 typical

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

        **kwargs,
    ):

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
        # 1) Build trace (centroid, with left-edge peak anchoring)
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

                # start search around previous center (continuity) or around global peak
                yc = y_prev if (trace_use_prev and y_prev is not None) else np.nanargmax(prof)

                # window around yc for trace computation
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

                # --- NEW: left-edge anchoring uses PEAK (argmax) instead of centroid ---
                if trace_left_use_peak_until is not None and x <= int(trace_left_use_peak_until):
                    ylo2 = int(max(0, yc - trace_left_peak_half_window))
                    yhi2 = int(min(ny, yc + trace_left_peak_half_window + 1))
                    seg = prof[ylo2:yhi2]
                    if seg.size < 3 or not np.isfinite(np.nanmax(seg)):
                        continue
                    y0[x] = ys[ylo2:yhi2][np.nanargmax(seg)]
                else:
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
        # 2) Polynomial trace fit (with optional left-weighting)
        # ------------------------------------------------------------
        if trace_poly_order is not None and trace_poly_order >= 1:
            fit_mask = np.isfinite(y0) & (xs >= x_min) & (xs < x_max)
            xfit = xs[fit_mask]
            yfit = y0[fit_mask]

            # optional left weighting for better boundary behavior
            wfit = np.ones_like(xfit, dtype=float)
            if trace_poly_left_xmax is not None and trace_poly_left_weight is not None:
                left = xfit <= float(trace_poly_left_xmax)
                wfit[left] *= float(trace_poly_left_weight)

            mask = np.ones_like(yfit, dtype=bool)

            for _ in range(trace_poly_max_iter if trace_poly_robust else 1):
                coef = np.polyfit(xfit[mask], yfit[mask], trace_poly_order, w=wfit[mask])
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

            coef = np.polyfit(xfit[mask], yfit[mask], trace_poly_order, w=wfit[mask])
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

        # ------------------------------------------------------------
        # 5) Call parallel_fit
        # ------------------------------------------------------------
        self.fit_result = parallel_fit(
            data,
            err,
            n_picks,
            initial_center=init_center_arr,
            initial_separation=initial_separation,
            pixel_limit=[x_min, x_max],
            bound_sigma=bound_sigma,
            distribution=distribution,
            param_value=param_value,
            param_limit=param_limit,
            param_fix=param_fix,
            **kwargs,
        )

        if n_picks > 1 and initial_separation is not None and len(initial_separation) > 0:
            centers_all = [init_center_arr]
            for j, sep in enumerate(initial_separation, start=2):
                centers_all.append(init_center_arr + float(sep))
            return init_center_arr, np.array(centers_all)  # shape (n_picks, nx)
        else:
            return init_center_arr, None





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

        return flux, ferr


    # -------------------------------------------------------------------------
    # 4) TWO-TRACE convenience helper (requires separation_2 in result_panda)
    # -------------------------------------------------------------------------
    def optimal_extract_two_traces_from_fit(
        self, centers_1, pixel_limit=None,
        sep_col="value_separation_2",
        half_window=6, **kwargs
    ):
        if not hasattr(self, "results") or "result_panda" not in self.results:
            raise AttributeError("Run array_to_pandas() first so separation_2 exists in result_panda.")

        df = self.results["result_panda"]
        nx = self.cut_data.shape[1]

        if pixel_limit is None or pixel_limit == []:
            x_min, x_max = 0, nx
        else:
            x_min, x_max = int(pixel_limit[0]), int(pixel_limit[1])
            x_min = max(0, x_min); x_max = min(nx, x_max)

        if sep_col not in df.columns:
            raise KeyError(f"Column '{sep_col}' not found in result_panda. Available: {list(df.columns)}")

        sep2_arr = np.full(nx, np.nan, dtype=float)
        sep2_arr[x_min:x_max] = df[sep_col].values.astype(float)

        centers_2 = centers_1 + sep2_arr

        flux1, err1 = self.optimal_extract_from_trace(
            centers_1, pixel_limit=[x_min, x_max], half_window=half_window, **kwargs
        )
        flux2, err2 = self.optimal_extract_from_trace(
            centers_2, pixel_limit=[x_min, x_max], half_window=half_window, **kwargs
        )

        return (flux1, err1), (flux2, err2)



    # -------------------------------------------------------------------------
    # 6) QUICK SKY DIAGNOSTIC PLOT (to compare to IRAF intuition)
    # -------------------------------------------------------------------------
    def plot_optimal_sky_diagnostics(self, xlim=None):
        if not hasattr(self, "optimal_results"):
            raise AttributeError("Run optimal_extract_from_trace() first.")

        x_min = self.optimal_results["x_min"]
        x_max = self.optimal_results["x_max"]
        sky_model = self.optimal_results["sky_model"]

        # show median sky level per column (over full y; this is just a diagnostic)
        sky_col = np.nanmedian(sky_model[:, x_min:x_max], axis=0)
        xs = np.arange(x_min, x_max)

        if xlim is not None:
            m = (xs >= xlim[0]) & (xs <= xlim[1])
            xs = xs[m]
            sky_col = sky_col[m]

        plt.figure(figsize=(12, 3))
        plt.plot(xs, sky_col)
        plt.xlabel("x (cut coords)")
        plt.ylabel("median sky model")
        plt.title("Sky model diagnostic (median over y per column)")
        plt.tight_layout()
        plt.show()

    def plot_real_geometry(
        self, opt, title="",
        show_signal=True,
        signal_mode="snr",        # "snr" or "flux"
        snr_thresh=3.0,           # if signal_mode="snr"
        flux_thresh=0.0,          # if signal_mode="flux"
        max_points=60000,         # downsample points if too many
    ):
        data = self.cut_data.astype(float)
        err  = self.cut_error.astype(float) if hasattr(self, "cut_error") else np.ones_like(data)

        ny, nx = data.shape
        x0, x1 = int(opt["x_min"]), int(opt["x_max"])
        y0 = opt["y0_arr"]
        hw = int(opt["half_window"])
        sky_inner = int(opt["settings"]["sky_inner"])
        sky_outer = int(opt["settings"]["sky_outer"])
        sky_side  = opt["settings"]["sky_side"]

        # Prefer the actual sky-subtracted data used by the extractor (if present)
        data_sky = opt.get("data_sky", None)
        if data_sky is None and hasattr(self, "optimal_results"):
            data_sky = self.optimal_results.get("data_sky", None)
        if data_sky is None:
            data_sky = data.copy()

        vmin, vmax = np.nanpercentile(data, [5, 95])

        plt.figure(figsize=(16, 6))
        plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(label="counts")

        xs_all = np.arange(x0, x1)
        ys_all = y0[x0:x1]
        m = np.isfinite(ys_all)
        xs = xs_all[m]
        ys = ys_all[m]

        # ---- trace center
        plt.plot(xs, ys, lw=2.5, label="trace center")

        # ---- aperture region + edges
        y_ap_lo = np.clip(ys - hw, 0, ny - 1)
        y_ap_hi = np.clip(ys + hw, 0, ny - 1)
        plt.fill_between(xs, y_ap_lo, y_ap_hi, alpha=0.20, label="aperture (extraction)")
        plt.plot(xs, y_ap_lo, lw=1.8, alpha=0.9)
        plt.plot(xs, y_ap_hi, lw=1.8, alpha=0.9)

        # ---- USED sky region(s)
        def draw_band(dy1, dy2, label):
            y1 = np.clip(ys + dy1, 0, ny - 1)
            y2 = np.clip(ys + dy2, 0, ny - 1)
            plt.fill_between(xs, y1, y2, alpha=0.14, label=label)
            plt.plot(xs, y1, ls="--", lw=2, alpha=0.9)
            plt.plot(xs, y2, ls="--", lw=2, alpha=0.9)

        if sky_side in ("down", "both"):
            draw_band(-sky_outer, -sky_inner, "USED sky (y smaller)")
        if sky_side in ("up", "both"):
            draw_band(+sky_inner, +sky_outer, "USED sky (y larger)")

        # ---- SIGNAL overlay (inside aperture)
        if show_signal:
            # Build point list inside the aperture where sky-subtracted data is "signal"
            yy = np.arange(ny)

            pts_x = []
            pts_y = []

            for x, yc in zip(xs.astype(int), ys):
                ylo = int(max(0, np.floor(yc - hw)))
                yhi = int(min(ny, np.floor(yc + hw + 1)))
                if yhi <= ylo:
                    continue

                D = data_sky[ylo:yhi, x]
                E = err[ylo:yhi, x]
                good = np.isfinite(D) & np.isfinite(E) & (E > 0)

                if not np.any(good):
                    continue

                if signal_mode == "snr":
                    mask_sig = np.zeros_like(D, dtype=bool)
                    mask_sig[good] = (D[good] / E[good]) >= snr_thresh
                else:  # "flux"
                    mask_sig = np.zeros_like(D, dtype=bool)
                    mask_sig[good] = D[good] >= flux_thresh

                yseg = yy[ylo:yhi]
                ysig = yseg[mask_sig]

                if ysig.size:
                    pts_x.append(np.full(ysig.size, x))
                    pts_y.append(ysig)

            if len(pts_x):
                pts_x = np.concatenate(pts_x)
                pts_y = np.concatenate(pts_y)

                # Downsample if too many points
                if pts_x.size > max_points:
                    idx = np.random.choice(pts_x.size, size=max_points, replace=False)
                    pts_x = pts_x[idx]
                    pts_y = pts_y[idx]

                plt.scatter(pts_x, pts_y, s=2, alpha=0.35, label=f"signal ({signal_mode})")

        plt.xlim(x0, x1)
        plt.ylim(0, ny - 1)
        plt.xlabel("x (dispersion pixel in cut)")
        plt.ylabel("y (spatial pixel in cut)")
        plt.title(title or f"Real extraction geometry (sky_side={sky_side})")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_extraction_and_sky(
        self,
        y0_arr,
        pixel_limit=None,
        half_window=6,
        sky_inner=10,
        sky_outer=25,
        sky_side="both",      # "up" | "down" | "both"
        title="",
        show_center=True,
        vmin=None,
        vmax=None,
        origin="lower",
        alpha_signal=0.28,
        alpha_sky=0.18,
    ):
        """
        Clean geometry plot:
        - shaded SIGNAL band (extraction aperture)
        - shaded SKY band(s) actually used
        - optional center trace line

        No extra edge lines, no sigma curves. Intended to be visually "realistic".
        """
        import numpy as np
        import matplotlib.pyplot as plt

        data = self.cut_data.astype(float)
        ny, nx = data.shape

        # x-range
        if pixel_limit is None or pixel_limit == []:
            x0, x1 = 0, nx
        else:
            x0, x1 = int(pixel_limit[0]), int(pixel_limit[1])
            x0 = max(0, x0)
            x1 = min(nx, x1)

        # image scaling
        if vmin is None or vmax is None:
            vmin_, vmax_ = np.nanpercentile(data, [5, 95])
            vmin = vmin_ if vmin is None else vmin
            vmax = vmax_ if vmax is None else vmax

        xs = np.arange(x0, x1)
        ys = y0_arr[x0:x1].astype(float)
        m = np.isfinite(ys)

        xs = xs[m]
        ys = ys[m]
        if len(xs) == 0:
            raise RuntimeError("No finite trace points to plot in this pixel_limit range.")

        # signal band (aperture)
        sig_lo = np.clip(ys - half_window, 0, ny - 1)
        sig_hi = np.clip(ys + half_window, 0, ny - 1)

        plt.figure(figsize=(16, 6))
        plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, origin=origin)
        plt.colorbar(label="counts")

        # --- SIGNAL shaded band (one color) ---
        plt.fill_between(xs, sig_lo, sig_hi, alpha=alpha_signal, label="signal (extraction)")

        # --- SKY shaded band(s) (one color) ---
        def shade_sky(dy1, dy2, label):
            y1 = np.clip(ys + dy1, 0, ny - 1)
            y2 = np.clip(ys + dy2, 0, ny - 1)
            plt.fill_between(xs, y1, y2, alpha=alpha_sky, label=label)

        if sky_side in ("up", "both"):
            shade_sky(+sky_inner, +sky_outer, "sky used (up)")
        if sky_side in ("down", "both"):
            shade_sky(-sky_outer, -sky_inner, "sky used (down)")

        # --- center line (optional) ---
        if show_center:
            plt.plot(xs, ys, lw=2.0, label="trace center")

        plt.xlim(x0, x1)
        plt.ylim(0, ny - 1)
        plt.xlabel("x (dispersion pixel in cut)")
        plt.ylabel("y (spatial pixel in cut)")
        plt.title(title if title else f"Signal + sky geometry (sky_side={sky_side})")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_AB_on_one_image(
        self,
        centA, centB,
        pixel_limit,
        half_window=6,
        sky_inner=10,
        sky_outer=25,
        sky_side_A="up",
        sky_side_B="down",
        title="A+B: signal (aperture) + sky (used)",
        vmin=None,
        vmax=None,
        origin="lower",
        alpha_signal=0.28,
        alpha_sky=0.18,
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        data = self.cut_data.astype(float)
        ny, nx = data.shape
        x0, x1 = pixel_limit

        # shared scaling
        if vmin is None or vmax is None:
            vmin_, vmax_ = np.nanpercentile(data, [5, 95])
            vmin = vmin_ if vmin is None else vmin
            vmax = vmax_ if vmax is None else vmax

        fig, ax = plt.subplots(1, 1, figsize=(18, 7))
        im = ax.imshow(data, aspect="auto", vmin=vmin, vmax=vmax, origin=origin)

        def overlay_one(y0_arr, sky_side, label_prefix, color_signal, color_sky, color_center):
            xs = np.arange(x0, x1)
            ys = y0_arr[x0:x1]
            m = np.isfinite(ys)
            xs, ys = xs[m], ys[m]

            # signal aperture band
            ysig_lo = np.clip(ys - half_window, 0, ny - 1)
            ysig_hi = np.clip(ys + half_window, 0, ny - 1)
            ax.fill_between(
                xs, ysig_lo, ysig_hi,
                color=color_signal, alpha=alpha_signal,
                label=f"{label_prefix} signal (extraction)"
            )

            # sky band(s)
            def shade_sky(dy1, dy2, label):
                y1 = np.clip(ys + dy1, 0, ny - 1)
                y2 = np.clip(ys + dy2, 0, ny - 1)
                ax.fill_between(
                    xs, y1, y2,
                    color=color_sky, alpha=alpha_sky,
                    label=label
                )

            if sky_side in ("up", "both"):
                shade_sky(+sky_inner, +sky_outer, f"{label_prefix} sky used (up)")
            if sky_side in ("down", "both"):
                shade_sky(-sky_outer, -sky_inner, f"{label_prefix} sky used (down)")

            # center trace
            ax.plot(xs, ys, lw=2.2, color=color_center, label=f"{label_prefix} trace center")

        # Two overlays, same image
        overlay_one(centA, sky_side_A, "A:", color_signal="deepskyblue", color_sky="salmon", color_center="dodgerblue")
        overlay_one(centB, sky_side_B, "B:", color_signal="limegreen",   color_sky="goldenrod", color_center="orange")

        ax.set_xlim(x0, x1)
        ax.set_ylim(0, ny - 1)
        ax.set_xlabel("x (dispersion pixel in cut)")
        ax.set_ylabel("y (spatial pixel in cut)")
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("counts")

        ax.legend(loc="upper left", frameon=True)
        plt.tight_layout()
        plt.show()



    def _fit_trace_polynomial(
        self,
        x,
        y,
        order=3,
        clip_sigma=4.0,
        max_iter=5,
        use_robust=True,
    ):
        """
        Fit y(x) with a polynomial, optionally sigma-clipping outliers.
        Returns polynomial coefficients (highest power first) and the mask used.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        good = np.isfinite(x) & np.isfinite(y)
        xg, yg = x[good], y[good]

        if len(xg) < (order + 2):
            raise RuntimeError(f"Not enough points ({len(xg)}) to fit poly order {order}.")

        mask = np.ones_like(yg, dtype=bool)

        for _ in range(max_iter if use_robust else 1):
            coef = np.polyfit(xg[mask], yg[mask], deg=order)
            yfit = np.polyval(coef, xg)

            resid = yg - yfit
            med = np.nanmedian(resid[mask])
            mad = np.nanmedian(np.abs(resid[mask] - med))
            sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid[mask])

            if not np.isfinite(sig) or sig <= 0:
                break

            new_mask = np.abs(resid - med) < clip_sigma * sig
            if new_mask.sum() == mask.sum():
                mask = new_mask
                break
            mask = new_mask

        # final fit with final mask
        coef = np.polyfit(xg[mask], yg[mask], deg=order)
        return coef, good, mask


    def boxcar_extract_from_trace(
        self, y0_arr, pixel_limit=None, half_window=6,
        do_sky_subtract=True,
        sky_inner=10, sky_outer=28, sky_stat="median", sky_poly_order=1,
        sky_side="both",
        cr_clip_sigma=None
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

            # optional CR cap inside aperture
            if cr_clip_sigma is not None and cr_clip_sigma > 0:
                med = np.nanmedian(D[good])
                mad = np.nanmedian(np.abs(D[good] - med))
                sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(D[good])
                if np.isfinite(sig) and sig > 0:
                    cap = med + cr_clip_sigma * sig
                    D = np.minimum(D, cap)

            flux[x] = np.nansum(D[good])
            ferr[x] = np.sqrt(np.nansum(V[good]))

        # store diagnostics like the optimal path
        self.boxcar_results = dict(
            x_min=x_min, x_max=x_max,
            y0_arr=y0_arr,
            half_window=half_window,
            data_sky=data_sky,
            sky_model=sky_model,
            flux=flux,
            ferr=ferr,
            settings=dict(
                do_sky_subtract=do_sky_subtract,
                sky_inner=sky_inner, sky_outer=sky_outer,
                sky_stat=sky_stat, sky_poly_order=sky_poly_order,
                sky_side=sky_side,
                cr_clip_sigma=cr_clip_sigma,
            )
        )

        return flux, ferr
