import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.ndimage import uniform_filter
from scipy.special import gamma, psi  # psi is the digamma function

#Here try to lef codes that are use for list,arrays,paths stuff like that 

def convert_to_float(notation):
        # Extract the exponent part after '10^(' and before ')'
        exponent = notation.split('^(')[1].split(')')[0]
        
        # Convert to float using scientific notation
        float_value = float(f"1e{exponent}")
        
        return float_value
    
def find_signal(data):
    """
    Find the most prominent peak in the given data.

    Parameters:
    ----------
    data : array-like
        Input data array in which to find the signal.

    Returns:
    -------
    int
        The index of the most prominent peak in the data.
    """
    data_copy = deepcopy(data)
    data_copy[data_copy<0] = 0
    #data_copy[data_copy>5*np.nanstd(data_copy)]= 0
    size = data_copy.shape
    if data_copy.shape[0]>800:
        mask = np.ones(size[0], dtype=bool)
        I = np.r_[0:200, 800:1000]
        mask[I]=False
        data_copy = data_copy * mask
    if np.all(data_copy==0):
        return np.nan
    #why this works with 10?
    window =signal.windows.general_gaussian(10, p=1, sig=5)
    filtered = signal.convolve(window, data_copy)
    filtered = (np.average(data_copy) / np.average(filtered)) * filtered
    filtered = np.roll(filtered,0)
    #just to avoid negatives that are not ussefull to do this
    filtered[filtered<0] = 0
    peaks, _ = signal.find_peaks(filtered)
    prominences = signal.peak_prominences(filtered, peaks)[0]
    return  peaks[np.argmax(prominences)] - (filtered.shape[0]-data.shape[0])


def guess_picks_image(image,objects_guess=2,plot=False):
    """
    Guess the locations of peaks in a 2D image.

    Parameters:
    ----------
    image : array-like
        2D input image.
    
    objects_guess : int, optional, default=2
        Number of peaks to guess.
    
    plot : bool, optional, default=False
        Whether to plot the intermediate results.

    Returns:
    -------
    array-like
        Indices of the guessed peaks.
    """
    
    data = deepcopy(image)
    data[data<0] = 0
    if np.all(data==0):
        return np.nan*np.ones(objects_guess)
    #why this works with 2?
    window =signal.windows.general_gaussian(2, p=1, sig=5)
    filtered = signal.convolve(window, data)
    filtered = (np.average(data) / np.average(filtered)) * filtered
    filtered = np.roll(filtered,-(filtered.shape[0]-data.shape[0]))
    #just to avoid negatives that are not ussefull to do this
    filtered[filtered<0] = 0
    peaks, _ = signal.find_peaks(filtered)
    #print(peaks)
    prominences = signal.peak_prominences(filtered, peaks)[0]
    sorted_indices = np.argsort(prominences)[::-1][:objects_guess]
    if plot:
        plt.plot(window)
        plt.show()
        plt.plot(filtered)
        plt.plot(data)
        plt.show()
    picks =  np.array([peaks[i] for i in sorted_indices])  + (filtered.shape[0]-data.shape[0])
    if len(picks)<objects_guess:
        return np.array(list(picks)+[np.nan]*(objects_guess-len(picks)))
    return picks

def gaussian(x, center, height, sigma):
    """
    Gaussian distribution function.

    Parameters:
    ----------
    x : array-like
        Input array.
    
    center : float
        Center of the Gaussian peak.
    
    height : float
        Height of the Gaussian peak.
    
    sigma : float
        Standard deviation of the Gaussian peak.

    Returns:
    -------
    array-like
        Gaussian distribution evaluated at x.
    """
    return height * np.exp(-(x - center)**2 / (2 * sigma**2))


def gaussian_with_error(x, center, height, sigma, err_center, err_height, err_sigma):
    """
    Compute the Gaussian and propagate the errors from the parameters.
    
    Parameters:
    -----------
    x : array-like
        The independent variable.
    center : float
        Center of the Gaussian.
    height : float
        Height (amplitude) of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.
    err_center : float
        Uncertainty in the center.
    err_height : float
        Uncertainty in the height.
    err_sigma : float
        Uncertainty in the sigma.
    
    Returns:
    --------
    f : array-like
        The Gaussian function evaluated at x.
    err_f : array-like
        The propagated uncertainty in f.
    """
    # Calculate the Gaussian function
    f = gaussian(x, center, height, sigma)
    
    # Pre-calculate common factor: the exponential term
    exp_term = np.exp(- (x - center)**2 / (2 * sigma**2))
    
    # Partial derivatives:
    df_dh = exp_term
    df_dc = height * exp_term * (x - center) / sigma**2
    df_dsigma = height * exp_term * (x - center)**2 / sigma**3
    
    # Propagate errors (assuming parameters are independent)
    err_f = np.sqrt((df_dh * err_height)**2 +
                    (df_dc * err_center)**2 +
                    (df_dsigma * err_sigma)**2)
    # A = height * sigma * np.sqrt(2 * np.pi)
    
    # # Partial derivatives:
    # dA_dh = sigma * np.sqrt(2 * np.pi)
    # dA_dsigma = height * np.sqrt(2 * np.pi)
    
    # # Error propagation:
    # err_A = np.sqrt((dA_dh * err_height)**2 + (dA_dsigma * err_sigma)**2)
    return f, err_f


def integrated_gaussian(center,height, sigma,err_center, err_height, err_sigma):
    """
    Compute the integrated area under the Gaussian and propagate errors.
    
    For a Gaussian: A = height * sigma * sqrt(2*pi)
    
    Parameters:
    -----------
    height : float
        Height (amplitude) of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian.
    err_height : float
        Uncertainty in the height.
    err_sigma : float
        Uncertainty in sigma.
    
    Returns:
    --------
    A : float
        Integrated area under the Gaussian.
    err_A : float
        Propagated uncertainty in the area.
    """
    A = height * sigma * np.sqrt(2 * np.pi)
    
    # Partial derivatives:
    dA_dh = sigma * np.sqrt(2 * np.pi)
    dA_dsigma = height * np.sqrt(2 * np.pi)
    
    # Error propagation:
    err_A = np.sqrt((dA_dh * err_height)**2 + (dA_dsigma * err_sigma)**2)
    
    return A, err_A

def gaussian_with_error(x, center, height, sigma, err_center, err_height, err_sigma):
    """
    Compute the Gaussian and propagate the errors from the parameters.
    
    Parameters:
    -----------
    x : array-like
        The independent variable.
    center : float
        Center of the Gaussian.
    height : float
        Height (amplitude) of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.
    err_center : float
        Uncertainty in the center.
    err_height : float
        Uncertainty in the height.
    err_sigma : float
        Uncertainty in the sigma.
    
    Returns:
    --------
    f : array-like
        The Gaussian function evaluated at x.
    err_f : array-like
        The propagated uncertainty in f.
    """
    # Calculate the Gaussian function
    f = gaussian(x, center, height, sigma)
    
    # Pre-calculate common factor: the exponential term
    exp_term = np.exp(- (x - center)**2 / (2 * sigma**2))
    
    # Partial derivatives:
    df_dh = exp_term
    df_dc = height * exp_term * (x - center) / sigma**2
    df_dsigma = height * exp_term * (x - center)**2 / sigma**3
    
    # Propagate errors (assuming parameters are independent)
    err_f = np.sqrt((df_dh * err_height)**2 +
                    (df_dc * err_center)**2 +
                    (df_dsigma * err_sigma)**2)
    
    return f, err_f
def moffat_with_error(x, center, height, alpha, sigma,
                      err_center, err_height, err_alpha, err_sigma):
    """
    Compute the Moffat function and propagate the errors from the parameters.
    
    Using the Moffat function:
    
        f(x) = h * [1 + ((x-c)^2/sigma^2)]^(-alpha)
    
    the error on f(x) is computed by standard error propagation:
    
        δf(x) = sqrt( (∂f/∂h * δh)^2 + (∂f/∂c * δc)^2 + 
                     (∂f/∂α * δα)^2 + (∂f/∂σ * δσ)^2 )
    
    Parameters:
    -----------
    x : array-like
        Independent variable.
    center : float
        Center of the Moffat peak.
    height : float
        Height (amplitude) of the Moffat peak.
    alpha : float
        Alpha parameter of the Moffat function.
    sigma : float
        Sigma parameter of the Moffat function.
    err_center : float
        Uncertainty in the center.
    err_height : float
        Uncertainty in the height.
    err_alpha : float
        Uncertainty in alpha.
    err_sigma : float
        Uncertainty in sigma.
    
    Returns:
    --------
    f : array-like
        Moffat function evaluated at x.
    err_f : array-like
        Propagated uncertainty of f.
    """
    # Compute the basic function value
    g = 1 + ((x - center)**2) / sigma**2
    f = height * g**(-alpha)
    
    # Partial derivative with respect to height: ∂f/∂h = g^(-alpha)
    df_dh = g**(-alpha)
    
    # Partial derivative with respect to center:
    # g = 1 + ((x-center)^2)/sigma^2, so d/dcenter g = -2*(x-center)/sigma^2.
    # Then, ∂f/∂center = h * (-alpha)*g^(-alpha-1) * (dg/dcenter)
    #                  = h * (-alpha)*g^(-alpha-1) * (-2*(x-center)/sigma^2)
    #                  = 2*h*alpha*(x-center)/(sigma**2)*g**(-alpha-1)
    df_dc = 2 * height * alpha * (x - center) / sigma**2 * g**(-alpha-1)
    
    # Partial derivative with respect to sigma:
    # d/dsigma g = d/dsigma [1 + ((x-center)^2)/sigma^2] = -2*(x-center)^2/sigma^3.
    # Then, ∂f/∂sigma = h * (-alpha)*g^(-alpha-1) * (dg/dsigma)
    #                  = 2 * h * alpha * (x-center)**2 / sigma**3 * g**(-alpha-1)
    df_dsigma = 2 * height * alpha * (x - center)**2 / sigma**3 * g**(-alpha-1)
    
    # Partial derivative with respect to alpha:
    # ∂f/∂alpha = -h * g^(-alpha) * ln(g)
    df_dalpha = -height * g**(-alpha) * np.log(g)
    
    # Propagate errors (assuming independent errors)
    err_f = np.sqrt((df_dh * err_height)**2 +
                    (df_dc * err_center)**2 +
                    (df_dalpha * err_alpha)**2 +
                    (df_dsigma * err_sigma)**2)
    
    return f, err_f

def integrated_moffat(center,height, alpha, sigma, err_center,err_height, err_alpha, err_sigma):
    """
    Compute the integrated area under the Moffat function and propagate the errors.
    
    For the Moffat function, the integral from -infty to infty is given by:
    
        I = h * sigma * sqrt(pi) * Gamma(alpha - 1/2) / Gamma(alpha)
    
    Note: The center does not affect the area.
    
    Parameters:
    -----------
    height : float
        Height (amplitude) of the Moffat function.
    alpha : float
        Alpha parameter.
    sigma : float
        Sigma parameter.
    err_height : float
        Uncertainty in the height.
    err_alpha : float
        Uncertainty in alpha.
    err_sigma : float
        Uncertainty in sigma.
    
    Returns:
    --------
    I : float
        Integrated area.
    err_I : float
        Propagated uncertainty of the area.
    """
    # Calculate the integrated area I
    I = height * sigma * np.sqrt(np.pi) * gamma(alpha - 0.5) / gamma(alpha)
    
    # Partial derivatives:
    # dI/dh = sigma * sqrt(pi) * Gamma(alpha - 1/2) / Gamma(alpha)
    dI_dh = sigma * np.sqrt(np.pi) * gamma(alpha - 0.5) / gamma(alpha)
    
    # dI/dsigma = height * sqrt(pi) * Gamma(alpha - 1/2) / gamma(alpha)
    dI_dsigma = height * np.sqrt(np.pi) * gamma(alpha - 0.5) / gamma(alpha)
    
    # dI/dalpha: First write I = h * sigma * sqrt(pi) * F(alpha) where F(alpha) = Gamma(alpha-0.5)/Gamma(alpha)
    # Then, dF/dalpha = F(alpha) * [psi(alpha - 0.5) - psi(alpha)].
    F_alpha = gamma(alpha - 0.5) / gamma(alpha)
    dF_dalpha = F_alpha * (psi(alpha - 0.5) - psi(alpha))
    dI_dalpha = height * sigma * np.sqrt(np.pi) * dF_dalpha
    
    # Propagate errors (center is not included since it doesn't affect the integral)
    err_I = np.sqrt((dI_dh * err_height)**2 +
                    (dI_dsigma * err_sigma)**2 +
                    (dI_dalpha * err_alpha)**2)
    
    return I, err_I
def moffat(x,center,height,alpha,sigma):
    """
    Moffat distribution function.

    Parameters:
    ----------
    x : array-like
        Input array.
    
    center : float
        Center of the Moffat peak.
    
    height : float
        Height of the Moffat peak.
    
    alpha : float
        Alpha parameter of the Moffat function.
    
    sigma : float
        Sigma parameter of the Moffat function.

    Returns:
    -------
    array-like
        Moffat distribution evaluated at x.
    """
    
    return height*(1+((x-center)**2/(sigma**2)))**-alpha

def smooth_boxcar(y, filtwidth,var=None, verbose=True):
        """
        Apply a boxcar smoothing to the spectrum.

        Note: This function is not authored by me.

        Parameters:
        ----------
        y : array-like
            Input spectrum to be smoothed.
        
        filtwidth : int
            Width of the smoothing filter.
        
        var : array-like or None, optional
            Variance spectrum for inverse variance weighting. If None, uniform weighting is used.
        
        verbose : bool, optional, default=True
            Whether to print verbose messages.

        Returns:
        -------
        tuple
            Smoothed spectrum and smoothed variance (or None if var is None).
        """
        
        """
        this is not mine 
        Does a boxcar smooth of the spectrum.
        The default is to do inverse variance weighting, using the variance
         spectrum if it exists.
        The other default is not to write out an output file.  This can be
        changed by setting the outfile parameter.
        """

        """ Set the weighting """
        if var is not None:
            if verbose:
                pass
                #print('Weighting by the inverse variance')
            wht = 1.0 / var
        else:
            if filtwidth==0:
                 ysmooth,varsmooth=y,None
                 return ysmooth, varsmooth
            if verbose:
                pass
                #print('Uniform weighting')
            wht = 0.0 * y + 1.0
        """ Smooth the spectrum and variance spectrum """
        yin = wht * y
        smowht = uniform_filter(wht, filtwidth)
        ysmooth = uniform_filter(yin, filtwidth)
        ysmooth /= smowht
        if var is not None:
            varsmooth = 1.0 / (filtwidth * smowht)
        else:
            varsmooth = None

        return ysmooth, varsmooth