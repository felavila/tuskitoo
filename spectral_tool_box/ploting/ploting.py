import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os 



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
    
    
panda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda")

def plot_spectra(flux_dict,add_error=False,save='',force_pix=False,z_s=None,add_lines=False,rest_frame=False,not_add=[],
                 units_flux=None,units_wavelenght=None,xlim=None,add_SII=False,add_NII=False,factor=None,add_OI=False,add_HeII=False,
                 add_Fe_lines=False,z_l=None,add_galaxy_lines=False,ima_color_dic=None,**kwargs):
    #wavelength = np.arange(len(df))
    xlabel = 'Wavelength (Å)'
    if z_s and rest_frame:
        xlabel = 'Rest frame Wavelength (Å)'
    ylabel =r"Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)"
    fig, ax = plt.subplots(1, 1, figsize=(35, 15))
    alpha = kwargs.get("alpha",0.9)
    import random

    colors = [
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
        '#8da0cb', "green", "brown",
        '#f781bf', '#999999', '#a65628',
        '#ff7f00', '#f781bf', '#999999', '#a65628',
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
        '#8da0cb', "green", "brown",
        '#f781bf', '#999999', '#a65628',
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
        '#ff7f00', '#f781bf', '#999999', '#a65628',
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    ]

    random.shuffle(colors)
    colors_dic = {"red_colors":['#e41a1c',"red","#FF2400","firebrick","crimson","#E0115F","maroon","#800020"]
                ,"green_colors":["teal","#4daf4a","green","forestgreen","olive"],
                "blue_colors":["#8da0cb","#ff7f00","#999999","#984ea3","indigo","lime"]
                ,"others":["skyblue","#377eb8","royalblue","#0047AB","blue","navy"]}

    band_count = {}
    all_flux = []
    all_wavelength = []
    for i,(image,values) in enumerate(flux_dict.items()):
        color=colors[i]
        args = image.split("_")
        Ima,band = args[0],args[1]
        if ima_color_dic:
            color = ima_color_dic[Ima]
        wavelength,flux,=values["wavelength"],values["flux"]
        if z_s and rest_frame:
            wavelength = wavelength/(1+z_s)
        if xlim and (max(wavelength)<min(xlim) or min(wavelength)>max(xlim)):
            continue 
        if image in not_add:
            continue 
        #flux[flux>0.5e-15] = np.nan
        if factor and (image in factor):
            flux = flux * factor[image]
            image = image +" X "+str(factor[image])
        all_flux.append(flux)
        all_wavelength.append(wavelength)
        print(image,colors[i])
        ax.plot(wavelength,flux,color=color,label=image,alpha=alpha)
    #return all_flux,all_wavelength
    all_flux = np.concatenate(all_flux)
    all_wavelength = np.unique(np.concatenate(all_wavelength))
    xlim = xlim or all_wavelength[[0,-1]]
    #kwargs.get('xlim',all_wavelength[[0,-1]])
    ylim_lower, ylim_upper = np.percentile(all_flux, [1, 99.55])
    ylim =kwargs.get('ylim',[0, ylim_upper*1.09])
    text_fontsize = kwargs.get("text_fontsize",20)
    text_rotation = kwargs.get("text_rotation",0)
    if z_s and add_lines:
            agn_lines = {
            "Lya": 1216,         # Lyman-alpha
            "CIV": 1549,         # Carbon IV
            "CIII_1909": 1909,   # Carbon III]
            "MgII": 2800,        # Magnesium II
            
            "Hβ": 4861,          # Hydrogen Balmer beta
            "OIII_4959": 4959,   # [O III] 4959
            "OIII_5007": 5007,   # [O III] 5007
            
            # [N II] 6548
            "Hα": 6563,          # Hydrogen Balmer alpha
                # [N II] 6583
            }
            if add_SII:
                agn_lines.update( {"SII_6716": 6716,    # [S II] 6716
                                    "SII_6731": 6731 })  # [S II] 6731)
            if add_NII:
                agn_lines.update( { "NII_6548": 6548,    # [S II] 6716
                                    "NII_6583": 6583})  # [S II] 6731)
            if add_OI:
                agn_lines.update({"OI_6300": 6300}) #,     # [O I] 6300
            if add_HeII:
                agn_lines.update({"HeII_4686": 4686,"HeII_1640":1640.4}) #,     ,   # Helium II
            for line_name,central_wavelength in agn_lines.items():
                if rest_frame:
                    central_wavelength = central_wavelength
                else:
                    central_wavelength = central_wavelength*(1+z_s)
                if max(xlim)>central_wavelength and min(xlim)<central_wavelength:
                    ax.axvline(central_wavelength, linestyle="--", color="k", linewidth=2,alpha=0.5)
                    label = line_name.replace('_', '$\lambda$')
                    if "_" in line_name:
                        text_rotation = 90
                    ax.text(central_wavelength, ylim[1], label, fontsize=text_fontsize, rotation=text_rotation,
                            verticalalignment="top", color="k",zorder=10,horizontalalignment="right")
                    text_rotation = 0
    if z_s and add_Fe_lines:
        #From fantasy https://github.com/yukawa1/fantasy/tree/main/fantasy_agn/input
        feII_forbidden = pd.read_csv(os.path.join(panda_path,"feII_forbidden.csv"))
        feii_IZw1 = pd.read_csv(os.path.join(panda_path,"feii_IZw1.csv"))
        feii_uv = pd.read_csv(os.path.join(panda_path,"uvfe.csv"))
        feII = pd.concat([feII_forbidden,feii_IZw1,feii_uv])
        for line_name,central_wavelength in feII.values:
                if rest_frame:
                    central_wavelength = central_wavelength
                else:
                    central_wavelength = central_wavelength*(1+z_s)
                if max(xlim)>central_wavelength and min(xlim)<central_wavelength:
                    ax.axvline(central_wavelength, linestyle="--", color="grey", linewidth=1,alpha=0.5)
                    label = line_name.replace('_', '$\lambda$')
                    ax.text(central_wavelength, ylim[1], label, fontsize=10, rotation=90,
                            verticalalignment="top", color="grey",zorder=10,horizontalalignment="right")
    if z_l and add_galaxy_lines:
        galaxy = pd.read_csv(os.path.join(panda_path,"galaxy.csv"))
        for line_name,central_wavelength in galaxy.values:
                if rest_frame:
                    central_wavelength = central_wavelength*(1+z_l)/(1+z_s)
                else:
                    central_wavelength = central_wavelength*(1+z_l)
                if max(xlim)>central_wavelength and min(xlim)<central_wavelength:
                    ax.axvline(central_wavelength, linestyle="--", color="r", linewidth=1,alpha=0.5)
                    label = line_name.replace('_', '$\lambda$')
                    ax.text(central_wavelength, ylim[1], label, fontsize=10, rotation=90,
                            verticalalignment="top", color="r",zorder=10,horizontalalignment="right")
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.label.set_size(40)  # Set x-axis label font size
    ax.yaxis.label.set_size(40)  # Set y-axis label font size
    ax.tick_params(which="both", bottom=True, top=False, left=True, right=False,
            length=10, width=2, labelsize=35)  # Increase tick length and width 
    plt.legend(loc='best', prop={'size': 24}, frameon=False)
    if save:
        plt.savefig(f"images/{save}.jpg", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()