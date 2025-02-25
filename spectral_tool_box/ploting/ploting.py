import matplotlib.pyplot as plt
import numpy as np

def plot_2d(
    data, 
    vmax=None, 
    vmin=None, 
    interpolation=None,
    title="image",
    mask_x=None,
    mask_y=None,
    mask_color='gray',
    mask_alpha=0.9,
    origin=None,
    save=""
):
    """
    Plots one or more 2D arrays as images.

    Parameters
    ----------
    data : np.ndarray or list of np.ndarray
        A single 2D array or a list of 2D arrays to be plotted.
    vmax : float, optional
        The upper bound of the colormap. If None, computed from np.nanquantile of each array (0.95).
    vmin : float, optional
        The lower bound of the colormap. If None, computed from np.nanquantile of each array (0.05).
    interpolation : str, optional
        The interpolation method to use (e.g., 'nearest', 'bilinear', etc.).
    title : str, optional
        Base title to apply to each subplot.
    mask_x : tuple, optional
        A tuple (x_start, x_end) specifying a vertical mask region on the plot.
    mask_y : tuple, optional
        A tuple (y_start, y_end) specifying a horizontal mask region on the plot.
    mask_color : str, optional
        Color to use for the masking region.
    mask_alpha : float, optional
        Alpha (transparency) to use for the masking region.
    origin : str, optional
        Place the [0,0] index of the array in the upper-left or lower-left corner of the axes
        (e.g., origin='upper' or origin='lower').
    """

    # If input is a single array, wrap it in a list so logic is the same
    if len(data.shape) == 2:
        arrays = [data]
    else:
        arrays = data
    
    # Number of images
    n_images = len(arrays)
    
    # Decide how many rows/columns of subplots
    # If n_images is multiple of 2, create rows of 2; otherwise, one image per row
    if n_images % 2 == 0:
        n_rows = n_images // 2
        n_cols = 2
        fig_width = 2*10
        fig_height = 5 * n_rows
    else:
        n_rows = n_images
        n_cols = 1
        fig_width = 2*10
        fig_height = 5 * n_images

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    for idx, array in enumerate(arrays):
        # Compute row and column index
        row = idx // n_cols
        col = idx % n_cols
        
        # Compute vmin, vmax if not provided
        _vmax = vmax
        _vmin = vmin
        if _vmax is None:
            _vmax = np.nanquantile(array, 0.95)
        if _vmin is None:
            _vmin = np.nanquantile(array, 0.05)

        ax = axes[row, col]
        im = ax.imshow(
            array,
            vmin=_vmin,
            vmax=_vmax,
            aspect='auto',
            interpolation=interpolation,
            origin=origin
        )
        
        ax.set_title(f"{title} {idx + 1}")
        ax.set_ylabel("Spacial axis")
        ax.set_xlabel("Spectral axis")
        plt.colorbar(im, ax=ax)

        # Mask x-axis region if specified
        if mask_x is not None:
            # mask_x should be a tuple (x_start, x_end)
            ax.axvspan(
                mask_x[0], 
                min(mask_x[1], array.shape[1]-1), 
                facecolor=mask_color, 
                alpha=mask_alpha
            )

        # Mask y-axis region if specified
        if mask_y is not None:
            # mask_y should be a tuple (y_start, y_end)
            ax.axhspan(
                mask_y[0], 
                mask_y[1], 
                facecolor=mask_color, 
                alpha=mask_alpha
            )
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{save}.jpg")
    plt.show()



def plot_cliped_spectra(spectras,lower_percentile=0,upper_percentile=100,ylim=None,add_spectra=None,xlim=None):
    """_summary_

    Args:
        spectras (_type_): _description_
        lower_percentile (int, optional): _description_. Defaults to 0.
        upper_percentile (int, optional): _description_. Defaults to 100.
        ylim (_type_, optional): _description_. Defaults to None.
    """
    #TODO maybe add the windows of telluric could be a great detail to see where are the issues
    #the only problem is it require the addition of "band" keyword
    if isinstance(spectras,np.ndarray):
        if len(spectras.shape)==1:
            spectras = [spectras]
    for i,spectra in enumerate(spectras):
        y = spectra
        x = np.arange(y.shape[0])
        ymin, ymax = np.percentile(y, [lower_percentile, upper_percentile])
        if np.isnan(ymin):
            print(f"cant be done for {i}")
            continue
        plt.figure(figsize=(20,5))
        plt.plot(x, y, label='spectra')
        if isinstance(add_spectra,np.ndarray):
            plt.plot(x, add_spectra, label='add_spectra',alpha=0.7)
        # Clip view to percentile range
        plt.ylim(ymin, ymax)
        if ylim:
            plt.ylim(ylim)
        plt.xlim(x[[0,-1]])
        if xlim:
            plt.xlim(xlim)
        plt.title(f"spectra number {i} Clipped to {lower_percentile}th-{upper_percentile}th Percentiles")
        plt.xlabel("pixel")
        plt.ylabel("Flux")
        plt.legend()
        plt.grid(True)
        plt.show()










def median_image(image,base=[],ylim=None,set_median=False,xlim=None,do_vertical=True,save=""):
    #TODO color change
    #TODO when do_vertical==False for 2 images desapear one 
    if isinstance(image,np.ndarray):
        if len(image.shape)==2:
            image = [image]
    #print(len(image))
    if len(image)%2 == 0 and not do_vertical:
        fig, axes = plt.subplots(len(image)//2,len(image)//2, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(len(image), 1, figsize=(20, 10*len(image)))
    try:
        axis = axes.flat
    except:
        axis = np.atleast_1d(axes)
    for i,ax in enumerate(axis):
        y = np.median(image[i], axis=1)
        x = np.arange(len(y)) 
        ax.plot(x,y) 
        if len(base) == len(image):
            index = np.where(base[i] == 1)[0]
            xmin,xmax = 0,0
            if len(index)>0:
                xmin = np.min(index)
                xmax = np.max(index)
            ax.axvspan(xmin, xmax, facecolor='yellow', alpha=0.3)
            # Assign colors based on region
            colors = np.array(['r'] * len(x))
            colors[(x >= xmin) & (x < xmax)] = 'b'
            ax.axvline(xmin, color='k', linestyle='--')
            ax.axvline(xmax, color='k', linestyle='--')
        if set_median:
            ax.axhline(np.median(y))
        else:
            ax.axhline(0, color='k', linestyle='--')
        if xlim is not None:
           ax.set_xticks(np.arange(*xlim))
           ax.set_xlim(xlim)
        else:
            ax.set_xticks(np.arange(len(y)))
            ax.set_xlim(0, len(y)-1)
        if ylim is not None:
            ax.set_ylim(ylim)
        
        ax.tick_params(axis='y', which='major', labelsize=20)
        ax.set_xlabel("Spacial axis", fontsize=20)
        ax.set_ylabel("counts or flux", fontsize=20)
        # Add a grid along the x-axis
        ax.grid(axis='x')
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{save}.jpg")
    plt.show()