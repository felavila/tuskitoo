import numpy as np
import matplotlib.pyplot as plt
import os 
import matplotlib.colors as mcolors
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


def plot_cut_out(cut_data,save=""):
    norm_image = cut_data/cut_data.max(axis=0)
    fig,axs = plt.subplots(1, 2, figsize=(18, 5))
    # Plot data on the first subplot
    im = axs[0].imshow(norm_image,aspect="auto",vmin=0,vmax=1)
    axs[0].set_title('2d cut')
    axs[0].set_xlabel('X-pixel')
    axs[0].set_ylabel('Y-pixel')
    
    plt.colorbar(im, ax=axs[0], label="normalized intensity")

    axs[1].plot(np.nanmedian(norm_image,axis=1), color='orange')
    axs[1].set_xlim(np.arange(len(np.nanmedian(norm_image,axis=1)))[[0,-1]])
    axs[1].axhline(0, ls= '--')
    axs[1].set_title('stacked median')
    axs[1].set_xlabel('y-pixels')
    axs[1].set_ylabel('intensity')
    plt.tight_layout()
    if save:
        plt.savefig(f"{save}.jpg", bbox_inches='tight')
    plt.show()

def plot_2d_image_residuals(image,image_model,save="",**kwargs):
    #tengo uncertanties puedo agregarla en este "residual calculus"
    residuals = ((image - image_model)/image)**2
    model_result = {"original_image":image/image.max(axis=0),"model_image":image_model/image_model.max(axis=0),"chi":residuals)}
    fig, axes = plt.subplots(1,3, figsize=(50, 10))
    for ax, (key, spectra2d) in zip(axes, model_result.items()):
        vmin,vmax,label=0,1,"normalize"
        ax.set_title(key, fontsize=30)
        if key=="residuals original-model":
            vmin,vmax,label=-1,1,"(image-model)/max"
        im = ax.imshow(spectra2d,aspect="auto",vmin=vmin,vmax=vmax, cmap='coolwarm')
        cbar = plt.colorbar(im, ax=ax, shrink=1)
        ax.set_xlabel("Pixel",fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        #fig.colorbar(im, ax=ax, shrink=1,label=label, fontsize=14)
    if save:
        plt.savefig(f"{save}.jpg", bbox_inches='tight')
    plt.show()
def plot_1d(panda,n_pixel,distribution_function,image,names,columns_distribtuion,source_number,parameter_number,save="",**kwargs):
    "this function is to test how the well has worked the fit just remind if we work only with nomalize or de_normalize"
    parameters=panda[panda['n_pixel'].isin([n_pixel])][[f"value_{c}_{n}"  for n in names for c in columns_distribtuion]]
    pixel_1d =image.T[n_pixel]
    x = np.linspace(0,len(pixel_1d),100)
    plt.plot(image.T[n_pixel],label="raw data")
    separated_sources = np.array([distribution_function(x,*i) for i in parameters.values[0].reshape(source_number,parameter_number)])
    plt.plot(x,np.sum(separated_sources,axis=0),color="k",label="added models")
    [plt.plot(x ,i, linestyle="--", linewidth=1.5,label=f"source {names[n]}") for n,i in  enumerate(separated_sources)]
    plt.plot(np.arange(len(pixel_1d)),pixel_1d-np.sum(np.array([distribution_function(np.arange(len(pixel_1d)),*i) for i in parameters.values[0].reshape(source_number,parameter_number)]),axis=0),label="residuals",alpha=0.5)
    plt.title(f"pixel {n_pixel}")
    plt.legend()
    if save:
        plt.savefig(f"{save}.jpg", bbox_inches='tight')
    plt.show()
def plot_column(panda,column_name="",**kwargs):
        if column_name not in list(panda.columns):
            print(f"{column_name} is not a avalaible column try \n {list(panda.columns)}")
            return 
        fig, ax = plt.subplots(figsize=(20, 6))
        column = panda[column_name].values
        print(f"mean value for {column_name} if {np.nanmedian(column)}")
        mdian = np.nanmedian(column)
        ax.plot(column)
        ax.axhline(mdian,zorder=10,c="k", linewidth=1.5)
        ax.set_title(f"column {column_name.replace('value','')}: {mdian:.3f}")
        if "xlim" in kwargs.keys():
            ax.set_xlim(*kwargs["xlim"])
        ax.set_ylim([mdian*0.2, mdian*1.7])
        if "ylim" in kwargs.keys():
            ax.set_ylim(*kwargs["ylim"])
        ax.tick_params(which="both", bottom=True, top=True, left=True, right=True,
            length=10, width=2, labelsize=35)  # Increase tick length and width
        ax.xaxis.label.set_size(40)  # Set x-axis label font size
        ax.yaxis.label.set_size(40)  # Set y-axis label font size
        plt.show()
        return

def plot_spectra(panda,wavelength,obj=None,xlim=None,ylim=None,save=False,add_lines=False,xlabel=None,**kwargs):
        #it will be interesting can change between clear and not clear in this routine to check the diferences
        #"redo"
        if isinstance(obj,str):
            obj = [obj]
        wavelength = None
        if xlabel=="pixel":
            wavelength = np.arange(len(panda))
        else:
            try:
                wavelength = wavelength
                xlabel="Observe wavelength"
            except:
                print("not wavelength in the class")
                wavelength = np.arange(len(panda))
                xlabel="pixel"
        plt.figure(figsize=(20,10))
        if not obj:
            obj = spectras1d.keys()
        [plt.plot(wavelength,spectras1d[key],label=key,linewidth=0.5) for key in obj]
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel('Flux', fontsize=20)
        #ax1.set_title(title, fontsize=20)
        plt.xlim(np.min(wavelength),np.max(wavelength))
        if xlim:
             plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)
        if add_lines:
            if xlabel=="pixel":
                print("not zs informed")
            else:
                #maybe pre render a kind of plots 
                import os
                #tableau_colors = list(mcolors.TABLEAU_COLORS.values())
                plt.text(0.05, 0.95, r"$z_{source}=$"+f"{self.zs}", transform=plt.gca().transAxes, fontsize=30, 
                         verticalalignment='top', horizontalalignment='left')
                module_dir = os.path.dirname(os.path.abspath(__file__))
                xmin,xmax=plt.gca().get_xlim()
                _,ymax = plt.gca().get_ylim()
                line_name,wv = np.loadtxt(os.path.join(module_dir,"tabuled_values/linelist.txt"),dtype="str").T
                for key,value in zip(line_name,wv):
                    value = float(value)*(1+self.zs)
                    if xmin<value<xmax:
                        #remove lines in masked zone
                        if "Fe" in key or "H1" in key or "H9" in key or "H8" in key:
                            continue#print(key,value)
                        plt.axvline(float(value),c="k",ls="--",alpha=0.2)
                        plt.text(float(value),ymax,key, rotation=90, verticalalignment='bottom', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend(loc='upper right', prop={'size': 24}, frameon=False, ncol=2)
        if save:
            plt.savefig(f"images/{self.name}_{self.band}_spectra.jpg")



# def plot2d_spectra(image,region=None):
#     """
#     Plots a 2D spectra image with optional region highlighting.

#     Parameters:
#     ----------
#     image : array-like
#         The 2D spectra image data to be plotted.
        
#     region : float or None, optional, default=None
#         A float value representing the region to highlight, as a fraction of the image height. 
#         If provided, a horizontal line and shaded area will be added to the plot.

#     Returns:
#     -------
#     None
#     """
#     image_to_plot = np.nan_to_num((image/image.max(axis=0)),0)
#     size = image_to_plot.shape
#     fig, ax = plt.subplots(figsize=(20, 9), dpi=80)
#     img =ax.imshow(image_to_plot,aspect='auto', origin = 'lower', vmin = 0, vmax = 1)
#     if isinstance(region,float):
#         threshold = int(region * size[0])
#         ax.axhline(threshold, color='green', lw=2, alpha=0.7)
#         ax.fill_between(np.arange(size[1]), threshold, size[0],
#                         color='green', alpha=0.5, transform=ax.get_yaxis_transform())
#     colorbar =fig.colorbar(img, ax=ax, label='Norm spectra')
#     colorbar.ax.yaxis.label.set_size(20)
#     ax.set_xlabel('dispersion axis', fontsize=20)  # Add label to x axis
#     ax.set_ylabel('spacial axis', fontsize=20)  # Add label to y axis
#     ax.tick_params(axis=("both"), labelsize=20) 
#     ax.set_title(f'2d spectra {image_to_plot.shape[1]} dispersion x {image_to_plot.shape[0]} spacial', fontsize=20)
#     ax.set_ylim(0,size[0]-1)
#     plt.show()





# #maybe pre render a kind of plots 
# tableau_colors = list(mcolors.TABLEAU_COLORS.values())
# __all__ = ("ploting_result","plot_spectra","plot_three_levels")
# module_dir = os.path.dirname(os.path.abspath(__file__))

# def plot_spectra(wavelenght,flux,title="?",save=None,show=True,xlim=None,ylim=None):
#     plt.figure(figsize=(35, 15))
#     line_name,wv = np.loadtxt(os.path.join(module_dir,"tabuled_values/linelist.txt"),dtype="str").T
#     zs = 0
#     plt.plot(wavelenght,flux,c="k",label=title)
#     _,ymax = plt.gca().get_ylim()
#     #print(plt.gca().get_xlim())
#     plt.xlim(wavelenght[0],wavelenght[-1])
#     if xlim:
#          plt.xlim(*xlim)
#     xmin,xmax=plt.gca().get_xlim()
#     if ylim:
#          plt.ylim(*ylim)
#     ymin,ymax = plt.gca().get_ylim()
#     for key,value in zip(line_name,wv):
#         value = float(value)
#         if xmin<value<xmax:
#             if "Fe" in key or "H1" in key or "H9" in key or "H8" in key:
#                 continue
#             #print(key,value)
#             plt.axvline(float(value)*(1+zs),c="k",ls="--",alpha=0.2)
#             plt.text(float(value)*(1+zs),ymax,key, rotation=90, verticalalignment='bottom', fontsize=20)
#     plt.xlabel('Rest Wavelength', fontsize=20)
#     plt.gca().tick_params(axis='both', which='major', labelsize=20)
#     plt.ylabel('Flux', fontsize=20)
#     plt.legend(loc='upper center',  prop={'size': 24}, frameon=False, ncol=2)
#     if save:
#         plt.savefig(save)
#      # Show the plot if requested
#     if show:
#         plt.show()
#     else:
#         plt.close()


# def plot_multiple_spectra(result,obj=None,bands=None,add_lines=False,xlim=None,ylim=None,save=None,normalize=False):
#     xlabel = "Observe wavelength"
#     zs = result.zs.values[0]
#     plt.figure(figsize=(30,10))
#     m,M  = [],[]
#     n = ""
#     if bands == None:
#         bands = result.band.unique()
#     for band in bands:
#         local_r=result[(result["band"]==band)][[i for i in result.columns if ("flux" in i) or ("wavelength" in i)]]
#         if obj==None:
#             obj = [i.replace("flux_","") for i in local_r.columns if "flux" in i]
#         for i,o in enumerate(obj):
#             try:
#                 flux = local_r[f"flux_{o}"]
#                 if normalize == True:
#                     n = "normalized "
#                     if i==0:
#                         norm = np.sum(flux)
#                     flux = flux/norm

#                 plt.plot(local_r.wavelength,flux,label=f"{o} {band}")
#             except:
#                 print(f' the objs are {[i.replace("flux_","") for i in local_r.columns if "flux" in i]}')
#                 plt.close()
#                 return 
#         m.append(local_r.wavelength.values[0]), M.append(local_r.wavelength.values[-1])
    
#     if m==[]:
#         plt.close()
#         return print(f' the bans are {result.band.unique()}')
    
#     plt.xlim(np.min(m),np.max(M))
#     plt.xlabel(xlabel, fontsize=20)
#     plt.ylabel(n+'Flux', fontsize=20)
#     if xlim:
#         plt.xlim(*xlim)
#     if ylim:
#         plt.ylim(*ylim)
#     if add_lines:
#         #maybe pre render a kind of plots 
#         import os
#         #tableau_colors = list(mcolors.TABLEAU_COLORS.values())
#         __all__ = ("ploting_result","plot_spectra","plot_three_levels")
#         module_dir = os.path.dirname(os.path.abspath(__file__))
#         xmin,xmax=plt.gca().get_xlim()
#         _,ymax = plt.gca().get_ylim()
#         line_name,wv = np.loadtxt(os.path.join(module_dir,"tabuled_values/linelist.txt"),dtype="str").T
#         for key,value in zip(line_name,wv):
#             value = float(value)*(1+zs)
#             if xmin<value<xmax:
#                 #remove lines in masked zone
#                 if "Fe" in key or "H1" in key or "H9" in key or "H8" in key:
#                     continue#print(key,value)
#                 plt.axvline(float(value),c="k",ls="--",alpha=0.2)
#                 plt.text(float(value),ymax,key, rotation=90, verticalalignment='bottom', fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     plt.legend(loc='upper right', prop={'size': 24}, frameon=False, ncol=2)
#     plt.tick_params(axis='both', which='major', labelsize=20)
#     if save:
#         plt.savefig(save)
        