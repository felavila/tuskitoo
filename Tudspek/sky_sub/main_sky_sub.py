from .tools import image_response_reader,flux_correction,correct
from .utils import list_builder,clipping_region
from .sky_sub import sky_subtraction
import warnings
import os 
import numpy as np 
import datetime
from pathlib import Path 
from astropy.io import fits

module_dir = os.path.dirname(os.path.abspath(__file__))


class SkySubtraction:
    def __init__(self,images_paths,response_paths,telluric_path=None,stare_spectrums=None,OB="OB",mask_image_x=None,mask_image_y=None,by_eye_signal_position=None,
                not_considering_pixels=[]):
        self.images_paths = images_paths
        self.response_paths = response_paths
        self.OB = OB
        self.stare_spectrums =   stare_spectrums or [None]*len(images_paths)
        self.telluric_path= telluric_path or [None]*len(images_paths)
        self.dic_paths = {n:{"image_path":self.images_paths[n],"response_path":self.response_paths[n],"telluric_path":self.telluric_path[n],"stare_spectrums":self.stare_spectrums[n]} for n,i in enumerate(self.images_paths )}
        self.run_reader()
        self.mask_image_x = list_builder(self.images_paths,mask_image_x)
        self.mask_image_y = list_builder(self.images_paths,mask_image_y)
        self.by_eye_signal_position = list_builder(self.images_paths,by_eye_signal_position)
        self.not_considering_pixels = list_builder(self.images_paths,not_considering_pixels)

    def run_reader(self):
        self.dic_data = {n:image_response_reader(**d) for n,d in enumerate(self.dic_paths.values())} #a
    
    def multi_image_sky_sub(self,re_do_all = False,force_median=False):
        #self.dic_results = None
        #dictionary of all the things we will use in the multi_image
        dic_sky_sub = {n:{"force_median":force_median,"data":self.dic_data[n]['data'],"band":self.dic_data[n]['band'],"mask_image_x":self.mask_image_x[n],"mask_image_y":self.mask_image_y[n],"by_eye_signal_position":self.by_eye_signal_position[n],"not_considering_pixels":self.not_considering_pixels[n]} for n,_ in enumerate(self.images_paths)}
        #{"data":self.data,"band":self.band,"mask_image_x":self.mask_image_x,"mask_image_y":self.mask_image_y,"by_eye_signal_position":self.by_eye_signal_position,"not_considering_pixels":self.not_considering_pixels}
        if hasattr(self, 'dic_results') and not re_do_all:
            dic_sky_sub_ = {}
            for n in self.dic_results.keys():
                result_ = self.dic_results[n]["keywords_function"]
                local_ = dic_sky_sub[n]
                if not all([np.array(result_[k]==local_[k]).all() for k in result_.keys()]):
                    dic_sky_sub_[n] = dic_sky_sub[n]
                else:
                    print(f"image {n} alrredy done")
            if len(dic_sky_sub_.keys())>0:
                #list_result = list(map(lambda d: sky_subtraction(**d), dic_sky_sub_.values()))
                #dic_results_ = {n:list_result[i] for i,n in enumerate(dic_sky_sub_.keys())}
                self.dic_results.update({n:sky_subtraction(**d) for n,d in dic_sky_sub_.items()})
        else:
            self.dic_results = {n:sky_subtraction(**d) for n,d in dic_sky_sub.items()}
        _ = list(self.dic_results.values())
        a = {key:np.array([i[key] for i in _]) for key in _[0].keys()}
        for key, value in a.items():
            setattr(self, key,value)
            
    def run_flux_correction(self,force_median=False,key_counts = "image_pca_sky_subtracted",sigma=3,max_iter=5,replace_with='mean'):
        
        if not hasattr(self, 'dic_data'):
            raise AttributeError(
                "Error: 'self.dic_results' is not defined. \n"
                "Could be an Error in runing 'run_reader'")
        
        if not hasattr(self, 'dic_results'):
            raise AttributeError(
                "Error: 'self.dic_results' is not defined. \n"
                "Please run 'multi_image_sky_sub' before calling 'run_combine'.")
        
        flux_corrected = {}
        for n,values in self.dic_data.items():
            (Re,exptime,airmass,gain,bin,extinction,band) = [values[key] for key in ["response","exptime","airmass","gain","bin","extinction","band"]]
            self.band = band 
            where_is_the_signal = np.arange(self.dic_results[n]["where_is_the_signal"].shape[0])[self.dic_results[n]["where_is_the_signal"]==1]
            #print(where_is_the_signal)
            if band == "UVB" or force_median:
                key_counts = "image_median_sky_subtracted"
                print('we will use image_median_sky_subtracted to calculate the flux')
                image_counts = self.dic_results[n][key_counts]
            else:
                image_counts = self.dic_results[n][key_counts]
            if np.all(image_counts == np.zeros_like(image_counts)):
                print("The key_counts",key_counts,"is full with zeros for image",n,"we will use instant image_median_sky_subtracted",)
                image_counts = self.dic_results[n]["image_median_sky_subtracted"]
            error_counts = values["errors"] #mmm
            self.key_counts = key_counts
            flux = flux_correction(image_counts,Re,exptime,airmass,gain,bin,extinction)
            error_flux = flux_correction(error_counts,Re,exptime,airmass,gain,bin,extinction)
            #if self.dic_results[n]["keywords_function"]["mask_image_x"]:
                #  flux[:,int(max(self.dic_results[n]["keywords_function"]["mask_image_x"])):] = 0 #value: desde
                # flux[:,:int(min(self.dic_results[n]["keywords_function"]["mask_image_x"]))] = 0 #:value means hasta
            pieces,slice_,_ = 22,1125,6000
            n_total = flux.shape[1] #espectral axis
            if band=="VIS" or band=="UVB":
                pieces,slice_,_ = 21,n_total//21,12159
            flux_clipped = clipping_region(flux,pieces,slice_, sigma=sigma, max_iter=max_iter, replace_with=replace_with,where_is_the_signal=where_is_the_signal)
            flux_ = {"key_counts":key_counts,"flux_corrected":flux_correction(image_counts,Re,exptime,airmass,gain,bin,extinction),\
                "flux_clipped_corrected":flux_clipped,"error_flux":error_flux}
            flux_corrected[n] = flux_
        self.dic_flux_corrected = flux_corrected
        _ = list(self.dic_flux_corrected.values())
        a = {key:np.array([i[key] for i in _]) for key in _[0].keys()}
        for key, value in a.items():
            setattr(self, key,value)
            
    def run_combine(self,images_to_combine=None,do_telluric=True):
        #TODO modify the units in the header check if it this  'BUNIT': 'erg/s/cm2/Angstrom'
        
        if not hasattr(self, 'dic_results'):
            raise AttributeError(
                "Error: 'self.dic_results' is not defined. \n"
                "Please run 'multi_image_sky_sub' before calling 'run_combine'.")
        
        if not hasattr(self, 'flux_corrected'):
            raise AttributeError(
                "Error: 'self.flux_corrected' is not defined. \n"
                "Please run 'run_flux_correction' before calling 'run_combine' The code doesn't allow combine without flux correction (YET)")
        
        self.correct_ = correct([value["header"] for key,value in self.dic_data.items()])
        
        if not images_to_combine:
            images_to_combine = [key for key in self.dic_results.keys() if self.dic_results[key]["sky_sub_work"]]
        print(f"Will be combine images {images_to_combine}")
        flux_clipped_corrected_pre_combine = np.zeros((len(images_to_combine),self.flux_clipped_corrected.shape[1]*2+1,self.flux_clipped_corrected.shape[2]))
        flux_corrected_pre_combine = np.zeros((len(images_to_combine),self.flux_clipped_corrected.shape[1]*2+1,self.flux_clipped_corrected.shape[2]))
        final_error = np.full((len(images_to_combine), self.flux_clipped_corrected.shape[1]*2+1, self.flux_clipped_corrected.shape[2]), np.nan)
        final_quality = np.full((len(images_to_combine), self.flux_clipped_corrected.shape[1]*2+1, self.flux_clipped_corrected.shape[2]), 1000, dtype=float)
        self.comb_info = {}
        for i,n in enumerate(images_to_combine):
            self.comb_info[n] = {}
            an0 = self.correct_[0][n] - self.correct_[1][n]
            ind = np.argmin(abs(an0))
            Lx = self.flux_clipped_corrected.shape[1] #number of pixels in x axi# 
            au1 = Lx-ind
            au0 = Lx-au1
            telluric_data = np.ones(self.flux_clipped_corrected.shape[2])
            if self.dic_data[n]["band"] != "UVB" and do_telluric:
                telluric_data = np.ones(self.flux_clipped_corrected.shape[2])
                telluric_path = self.telluric_path[n] or ""
                if os.path.isfile(telluric_path):
                    fits_telluric = fits.open(self.telluric_path[n])
                    if len(fits_telluric) == 2:
                        telluric_data = fits_telluric[1].data
                        telluric_header = fits_telluric[1].header
                    elif len(fits_telluric) == 5:
                        telluric_data = fits_telluric[4].data['mtrans']
                        telluric_header = fits_telluric[4].header
                    #telluric = fits_telluric[4].data['mtrans']
                    telluric_data[telluric_data == 0] = np.nan
                    self.comb_info[n]['telluric_path'] =  self.telluric_path[n]
                    self.comb_info[n]['telluric_header'] = telluric_header 
                    self.comb_info[n]['telluric_mtrans'] = telluric_data 
                    print(f"doing telluric correction image {n}")
            flux_clipped_corrected_pre_combine[i,Lx- au0:Lx+au1,:] = self.flux_clipped_corrected[n]/telluric_data
            flux_corrected_pre_combine[i,Lx- au0:Lx+au1,:] = self.flux_corrected[n]/telluric_data
            final_error[i,Lx- au0:Lx+au1,:] = self.error_flux[n]/telluric_data
            final_quality[i,Lx- au0:Lx+au1,:] = self.dic_data[n]['quality']
            self.comb_info[n]['header'] =  self.dic_data[n].get('header')
            self.comb_info[n]['header_errors'] =  self.dic_data[n].get('header_errors')
            self.comb_info[n]['quality_header'] =  self.dic_data[n].get('quality_header')
            self.comb_info[n]['image_path'] =  self.dic_data[n].get('image_path')
            self.comb_info[n]['response_path'] =  self.dic_data[n].get('response_path')
        self.flux_clipped_corrected_pre_combine = flux_clipped_corrected_pre_combine
        self.flux_corrected_pre_combine = flux_corrected_pre_combine
        final_error[final_error == 0] = np.nan
        flux_clipped_corrected_pre_combine[flux_clipped_corrected_pre_combine == 0] = np.nan
        flux_corrected_pre_combine[flux_corrected_pre_combine == 0] = np.nan
        # Weights = 1 / (error^2)
        weights = 1.0 / (final_error**2)
        sum_weights = np.nansum(weights, axis=0)
        sum_weights[sum_weights == 0] = np.nan
        # Weighted flux
        flux_clipped_corrected_combine = np.nansum(flux_clipped_corrected_pre_combine * weights, axis=0) / sum_weights
        flux_corrected_combined = np.nansum(flux_corrected_pre_combine * weights, axis=0) / sum_weights
        self.flux_clipped_corrected_combine  = np.nan_to_num(flux_clipped_corrected_combine, nan=0.0)
        self.flux_corrected_combined  = np.nan_to_num(flux_corrected_combined, nan=0.0)
        # # Propagated error = 1 / sqrt(sum_of_weights)
        self.error_comb = np.nan_to_num(1.0 / np.sqrt(sum_weights), nan=1e11)
        #Combine quality by a simple average (or you could do min, max, etc.)
        self.quality_comb = np.nanmean(final_quality, axis=0).astype(int)
        
    def run_median_spectrum(self,save=False,window_size=5,lower_percentile=1,upper_percentile=99.9,where_median=[],**kwargs):
        images_to_combine = [key for key in self.dic_results.keys() if self.dic_results[key]["sky_sub_work"]]
        list_medians = []
        for i,n in enumerate(images_to_combine): 
            print(f"run_median_spectrum for image {n}" )
            if len(where_median) == len(images_to_combine) and len(where_median[n])==len(self.where_is_the_signal[n]):
                print("edited where is the signal")
                where_is_the_signal = where_median[n]
            else:
                where_is_the_signal= self.where_is_the_signal[n]
            signal_median = np.median(self.flux_clipped_corrected[n][np.where(where_is_the_signal== 1)],axis=0)
            list_medians.append(signal_median)
            #post_signal_median = medfilt(signal_median, kernel_size=window_size)
            ymin, ymax = np.percentile(signal_median, [lower_percentile, upper_percentile])
            signal_median[(signal_median < ymin) | (signal_median > ymax)] = 0
            if save:
                file_name =  self.stare_spectrums[n]
                data = fits.open(file_name)
                output_filename=f"{file_name.replace('MERGE1D','MEDIAN1D')}"
                data[0].data = signal_median.data
                data.writeto(output_filename, overwrite=True) 
                print(f"saved file in {output_filename}")
        return list_medians
    def save_sci(self,path="",person="F. Avila-Vera",use_clipped=True):
        #TODO what will happen when we have multiple OB1 to combine them, 
        # I am guessing two scenarios the first one is all the systems are alight at the same pixel so no worries but exist a case in where they are not and T.T
        if not hasattr(self, 'comb_info'):
            raise AttributeError(
                "Error: 'self.comb_info' is not defined. \n"
                "Please run run_combine before calling 'save_sci'.")
        header = [[self.comb_info[key].get("header"),self.comb_info[key].get("header_errors"),self.comb_info[key].get("quality_header")] for key in self.comb_info.keys()][0]
        files_used = ', '.join([os.path.basename(self.comb_info[key].get("image_path")) for key in self.comb_info.keys()]) 
        comb_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Current date in YYYY-MM-DD format
        # comb_method = method        # For example, the combination method used 
        for h in header:
            #'BUNIT': 'erg/s/cm2/Angstrom'
            if "BUNIT" in list(h.keys()):
                h['BUNIT'] = 'erg/s/cm2/Angstrom'
            h['FILES'] = (files_used, "files comb")
            h['COMBDATE'] = (comb_date, "Date comb")
            h['COMB_MTH'] = ("mean", "Meth to comb")
            h['PERSON'] = (person, "who COMB")
            h['TELCOR'] = (str(True), "Tell corrected")
            h["SKYSUB"] = (",".join(self.key_counts),"Meth sky sub")
            h["CLIP"] = (str(use_clipped),"was apply clip")
        primary_hdu = fits.PrimaryHDU(data=self.flux_corrected_combined)
        if use_clipped:
            primary_hdu = fits.PrimaryHDU(data=self.flux_clipped_corrected_combine)
        primary_hdu.header = header[0]        # Example header keyword
        hdu2 = fits.ImageHDU(data=self.error_comb)
        hdu2.header = header[1] 
        hdu3 = fits.ImageHDU(data=self.quality_comb)
        hdu3.header = header[2] 
        hdulist = fits.HDUList([primary_hdu, hdu2, hdu3])
        if not path:
            obj_path = str(Path(self.images_paths[0]).parent)
            dir_path = obj_path + "/Our_red_Comb"
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            path_ = os.path.basename(Path(self.images_paths[0]).parent.parent.parent) + f"_SCI_SLIT_FLUX_MERGE2D_{self.band}_TELL_COR_{self.OB}_OUR.fits" #maybe add something else here?
            save_path = os.path.join(dir_path,path_)
            hdulist.writeto(save_path, overwrite=True)
            if path:
                save_path = path
                hdulist.writeto(save_path, overwrite=True) 
            print("Will be saved in",save_path)    
