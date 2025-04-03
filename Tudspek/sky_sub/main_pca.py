import numpy as np
from .tools import small_fun, mad, dummies_pca,remove_nan,inpaint_nans,flux_correction,list_builder,correct
from csaps import csaps
from scipy.optimize import minimize
from astropy.io import fits
from parallelbar import progress_imap, progress_map, progress_imapu
from multiprocessing import cpu_count
from .ploting import plot_2d,median_image,plot_cliped_spectra
import os 
import matplotlib.pyplot as plt 

module_dir = os.path.dirname(os.path.abspath(__file__))

#TODO maybe in the end for the x axis the best will be implement the speckit methodology for mask this means in pairs of [] in where the limits are more clear
#TODO https://www.eso.org/sci/software/pipelines/xshooter/xshooter-pipe-recipes.html
#TODO https://ftp.eso.org/pub/dfs/pipelines/instruments/xshooter/xshoo-reflex-tutorial-3.6.8.pdf


class Main_Sky_Sub():
    def __init__(self,images_paths,response_paths,stare_spectrums=None,telluric_path=None,mask_image_x=None,mask_image_y=None,by_eye_signal_position=None,
                not_considering_pixels=None,force_median=False):
        self.images_paths = images_paths
        self.response_paths = response_paths
        self.stare_spectrums =  stare_spectrums
        self.telluric_path= telluric_path
        self.mask_image_x = mask_image_x
        self.mask_image_y = mask_image_y
        self.by_eye_signal_position = by_eye_signal_position
        self.not_considering_pixels = not_considering_pixels
        self.force_median=force_median
        self.results = None
        self.get_header_parameters()
        self.band = [self.images_relevant_info[i]["band"] for i in self.images_relevant_info.keys()][0] # it assumes alot 
    def run_sub(self):
        self.results = multi_image_sky_sub(self.images_paths, self.response_paths,mask_image_x=self.mask_image_x,mask_image_y=self.mask_image_y,by_eye_signal_position=self.by_eye_signal_position,force_median=self.force_median,not_considering_pixels=self.not_considering_pixels)
        for key, value in self.results.items():
            setattr(self, key, value)
        if self.telluric_path or self.band=="UVB":
        #len(self.images_paths) == len(self.telluric_path) 
            self.run_telluric_correction()
    def run_median_spectrum(self,save=False,window_size=5,lower_percentile=1,upper_percentile=99.9):
        medians = []
        if isinstance(upper_percentile,float):
            upper_percentile = [upper_percentile]*len(self.images_paths)
        for i,work in enumerate(self.sky_sub_work):
            if not work:
                print(f"Dosent work for {self.images_paths[i]}")
                continue
            signal_median = np.median(self.image_sky_subtracted_flux_corrected[i][np.where(self.where_is_the_signal[i] == 1)],axis=0)
            post_signal_median = medfilt(signal_median, kernel_size=window_size)
            ymin, ymax = np.percentile(post_signal_median, [lower_percentile, upper_percentile[i]])
            post_signal_median[(post_signal_median < ymin) | (post_signal_median > ymax)] = 0
            if save:
                file_name =  self.stare_spectrums[i]
                data = fits.open(file_name)
                output_filename=f"{file_name.replace('MERGE1D','MEDIAN1D')}"
                data[0].data = post_signal_median.data
                data.writeto(output_filename, overwrite=True)
            medians.append(post_signal_median.data)
        self.medians = np.array(medians)
    def run_telluric_correction(self):
        print("runing telluric correction/combination")
        pca_work = []
        telluric = []
        self.correct_ = correct(self.header)
        for i,work in enumerate(self.sky_sub_work):
            if not work:
                print(f"Dosent work for {self.images_paths[i]}")
                continue
            if self.band != "UVB":
                telluric.append(fits.open(self.telluric_path[i])[4].data['mtrans'])
            pca_work.append(self.image_sky_subtracted_flux_corrected[i])
        self.telluric = np.array(telluric)
        self.pca_work = np.array(pca_work)
        TR = np.zeros((sum(self.sky_sub_work),self.image_sky_subtracted_flux_corrected.shape[1]*2+1,self.image_sky_subtracted_flux_corrected.shape[2]))
        not_TR = np.zeros((sum(self.sky_sub_work),self.image_sky_subtracted_flux_corrected.shape[1]*2+1,self.image_sky_subtracted_flux_corrected.shape[2]))
        for i in range(TR.shape[0]):
            an0 = self.correct_[0][i] - self.correct_[1][i]
            ind = np.argmin(abs(an0))
            Lx = self.pca_work[i].shape[0] #number of pixels in x axi# 
            au1 = Lx-ind
            au0 = Lx-au1
            #print(Lx- au0,Lx+au1-1)
            not_TR[i,Lx- au0:Lx+au1,:] = self.pca_work[i]
            if self.band != "UVB":
                TR[i,Lx- au0:Lx+au1,:] = self.pca_work[i]/ self.telluric[i]
            #au1 = np.arange(ind,Lx)
        self.TR = TR
        self.final_TR = np.median(self.TR,axis=0)
        self.not_TR = np.median(not_TR,axis=0)
    def median_image_plot(self):
        median_image(self.image_sky_subtracted_flux_corrected,self.where_is_the_signal)
    def get_header_parameters(self):
        images_relevant_info = {}
        for i,data_path in enumerate(self.images_paths):
            fits_image = fits.open(data_path)
            response_path = self.response_paths[i]
            data,header = fits_image[0].data,fits_image[0].header
            data_error = fits_image[1].data#,fits_image[0].header
            quality_pixel = fits_image[2].data #https://ftp.eso.org/pub/dfs/pipelines/xshooter/xsh-manual-12.0.pdf page 112
            airmass = (header['ESO TEL AIRM START'] +  header['ESO TEL AIRM END'])/2
            exptime = header["EXPTIME"]
            band = header["HIERARCH ESO SEQ ARM"]
            wavelength = (header["CRVAL1"] + header["CDELT1"] * np.arange(data.shape[1])) #this is nm
            if isinstance(response_path,str):
                response_fits = fits.open(response_path)
                header_response = response_fits[0].header 
                if header_response["HIERARCH ESO SEQ ARM"] == band:
                    w_Re = response_fits[1].data["LAMBDA"]
                    argmin_w = np.where(wavelength[0] == w_Re)[0][0]
                    argmax_w = np.where(wavelength[-1] == w_Re)[0][0]
                    Re = response_fits[1].data["RESPONSE"][argmin_w:argmax_w+1]
                else:
                    return print("the Band of image and respondes are not the same")
            extinction_model = fits.getdata(f"{module_dir}/extinction/xsh_paranal_extinct_model_{band.lower()}.fits") #this 
            extinction = np.interp(wavelength,extinction_model["lambda"],extinction_model["extinction"])#, smooth=1)#.values
            try:
                gain = header["ESO DET OUT1 CONAD"] 
                bin = header['ESO DET WIN1 BINX'] 
            except KeyError as e:
                print(f"We dont found the keys {e}")
                gain = 2.12
                bin = 1
            images_relevant_info[i] = {"band":band,"data":data,"data_error":data_error,"quality_pixel":quality_pixel,"airmass":airmass,"exptime":exptime, \
                                                                                "wavelength":wavelength,"Re":Re,"wavelenght_Re":response_fits[1].data["LAMBDA"][argmin_w:argmax_w+1],"extinction":extinction, \
                                                                                "gain":gain,"bin":bin,"data_path":data_path}
        self.images_relevant_info =  images_relevant_info



def multi_image_sky_sub(images_paths, response_paths,mask_image_x=None,mask_image_y=None,not_considering_pixels=None,by_eye_signal_position=None,force_median=False):
    not_considering_pixels =list_builder(images_paths,not_considering_pixels)
    by_eye_signal_position = list_builder(images_paths,by_eye_signal_position)
    mask_image_y = list_builder(images_paths,mask_image_y)
    mask_image_x = list_builder(images_paths,mask_image_x)
    force_median = list_builder(images_paths,force_median)
    if not all(len(images_paths)==len(i) for i in [response_paths,not_considering_pixels,by_eye_signal_position,mask_image_y,mask_image_x]):
        raise ValueError(
                f"NOT ALL LIST HAVE THE SAME LENGH CHECK "
            )
    list_decomposition = list(map(lambda args: sky_subtraction(args[0], args[1],mask_image_x=args[2],mask_image_y=args[3],not_considering_pixels=args[4],by_eye_signal_position=args[5],force_median=args[6]), zip(images_paths,response_paths,mask_image_x,mask_image_y,not_considering_pixels,by_eye_signal_position,force_median)))
    #recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,pca_sky,image_pca_sky_subtracted,image_median_sky_subtracted,image_sky_subtracted_flux_corrected,where_is_the_signal,data,data_error,data_clean,header = [np.stack(arrays) for arrays in zip(*list_decomposition)]
    list_of_returns =["recon_sky","COF","image_recon_sky_subtracted","image_recon_cof_sky_subtracted","pca_sky","image_pca_sky_subtracted","image_median_sky_subtracted","image_sky_subtracted_flux_corrected","where_is_the_signal",
                      "data","data_error","data_clean","header","keywords_function","sky_sub_work"]
    #return recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,pca_sky,image_pca_sky_subtracted,image_median_sky_subtracted,image_sky_subtracted_flux_corrected,where_is_the_signal,data,data_error,data_clean,header
    print("Process ended \n")
    return {list_of_returns[i]: arrays if list_of_returns[i] in ["header","keywords_function","sky_sub_work"] else  np.stack(arrays) for i,arrays in enumerate(zip(*list_decomposition))}
    #return {i:[np.stack(arrays) for arrays in enumerate(zip(*list_decomposition))] }

def sky_subtraction(data_path,response_path,mask_image_x=None,mask_image_y=None,not_considering_pixels=None,by_eye_signal_position=None,force_median=False):   
    if not not_considering_pixels:
        not_considering_pixels = []
    keywords_function = locals()
    #TODO just calculate the flux for one method remove the other is not way to use the pca in UVB meanwhile in vis and nir the "median" method dosent make any sense
    print(f"Doing {data_path}")
    ##############
    fits_image = fits.open(data_path)
    data,header = fits_image[0].data,fits_image[0].header
    data_error = fits_image[2].data#,fits_image[0].header
    quality_pixel = fits_image[2].data #https://ftp.eso.org/pub/dfs/pipelines/xshooter/xsh-manual-12.0.pdf page 112
    mask =(data > 0.0) #& (quality_pixel == 0)#mask for the data, in their code they make this as nan values maybe is best save it as a mask     
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
    data_clean =remove_nan(data,mask,verbose=False) #remove nan values
    #data_clean[:,int(max(mask_image_x)):] = 0
    #data_clean[:,int(max(mask_image_x)):] = 0
    startH = header['ESO TEL AIRM START']
    endH = header['ESO TEL AIRM END']
    airmass = (startH + endH)/2
    exptime = header["EXPTIME"]
    band = header["HIERARCH ESO SEQ ARM"]
    wavelength = (header["CRVAL1"] + header["CDELT1"] * np.arange(data_clean.shape[1])) #this is nm
    if isinstance(response_path,str):
        response_fits = fits.open(response_path)
        header_response = response_fits[0].header 
        if header_response["HIERARCH ESO SEQ ARM"] == band:
            w_Re = response_fits[1].data["LAMBDA"]
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
        #extinction = np.zeros_like(wavelength)
    ######################
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
    recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,image_pca_sky_subtracted,pca_sky = \
        np.zeros_like(data_clean),np.zeros_like(Im),np.zeros_like(data_clean),np.zeros_like(data_clean),np.zeros_like(data_clean),np.zeros_like(data_clean)
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
            print("method dosent work")
            return recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,pca_sky,image_pca_sky_subtracted,data_clean,\
                data_clean,where_is_the_signal,data,data_error,data_clean,header,keywords_function,False
        cut = np.sort([xr1[v1[q1]] , xr2[v2[q2]]]) + np.array([-3,3])
        valuu = np.arange(np.maximum(cut[0], 0),np.minimum(cut[1]+1, np.shape(data_clean)[0]))
        ##########################
    where_is_the_signal[valuu] = 1 #where is the signal
    #where is no signal
    va0 = np.where(where_is_the_signal==0)[0].tolist()
    #if not_considering_pixels:
    #print(va0)
    va0 = list(set(va0) - set([0,1,data_clean.shape[0]-1,data_clean.shape[0]]+not_considering_pixels))
    print(va0)
    data_clean[np.isnan(data_clean)] = 0 
    median_sky = np.median(data_clean[va0],axis=0)
    if band=="UVB":
        median_sky[:800] = 0
    image_median_sky_subtracted = data_clean - median_sky
    if (band=="NIR" or band=="VIS") and force_median==False:
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
        paralel_fit = progress_map(run_pca_pixel, args,process_timeout=40, n_cpu=cpu_count(),need_serialize=False)
        recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,image_pca_sky_subtracted,pca_sky = [np.stack(arrays) for arrays in zip(*paralel_fit)]
        image_sky_subtracted_flux_corrected,keywords_f = flux_correction(image_pca_sky_subtracted,Re,exptime,airmass,band=band,gain=gain,bin=bin,extinction=extinction,mask_image_x=mask_image_x) #PCA_L
    else:
        print(f"given the band is {band}  or force_median is True we use the median to calculate the sky" )
        image_sky_subtracted_flux_corrected,keywords_f = flux_correction(image_median_sky_subtracted,Re,exptime,airmass,band=band,gain=gain,bin=bin,extinction=extinction,mask_image_x=mask_image_x) #PCA_L
    
    keywords_function.update(keywords_f)
    
    return recon_sky,COF,image_recon_sky_subtracted,image_recon_cof_sky_subtracted,pca_sky,image_pca_sky_subtracted,image_median_sky_subtracted,image_sky_subtracted_flux_corrected,where_is_the_signal,data,data_error,data_clean,header,keywords_function,True



def run_pca_pixel(args_tuple):
    """
    Unpacks the single tuple of arguments and calls sky_subtraction.
    """
    data_clean,uj,va0,band = args_tuple
    return pca_pixel(uj,va0,band,data_clean)


def pca_pixel(uj,va0,band,data_clean,rang):
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