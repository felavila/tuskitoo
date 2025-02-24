import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button,RangeSlider
from lmfit import Model
from scipy.integrate import quad
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import multiprocessing
from functools import partial
from .utils import format_scientific
from screeninfo import get_monitors

module_dir = os.path.dirname(os.path.abspath(__file__))

# Get the screen resolution
monitor = get_monitors()[0]  # Assuming you want the primary monitor
screen_width = monitor.width
screen_height = monitor.height

# Calculate the aspect ratio
aspect_ratio = screen_width / screen_height


def linear_fit(x, slope, intercept):
        return slope * x + intercept

def bootstrap_single_iteration(x, y, _):
    pto_medio=np.min(x)+(np.max(x)-np.min(x))/2
    if type(x)==np.ndarray:
        x_L,y_L=x[x<pto_medio],y[x<pto_medio]
        x_R,y_R=x[x>pto_medio],y[x>pto_medio]
    else:
        x_L,y_L=x.values[x<pto_medio],y.values[x<pto_medio]
        x_R,y_R=x.values[x>pto_medio],y.values[x>pto_medio]
    # Resample the dataset with replacement
    resampled_indices_R = np.random.choice(len(x_R), len(x_R), replace=True)
    resampled_x_R = x_R[resampled_indices_R]
    resampled_y_R = y_R[resampled_indices_R]
   
    resampled_indices_L = np.random.choice(len(x_L), len(x_L), replace=True)

    resampled_x_L = x_L[resampled_indices_L]
    resampled_y_L = y_L[resampled_indices_L]
    
    model = Model(linear_fit)
    params = model.make_params(slope=1, intercept=0)
    resampled_x=np.concatenate((resampled_x_L, resampled_x_R))
    
    resampled_y=np.concatenate((resampled_y_L, resampled_y_R))
    result = model.fit(resampled_y, params, x=resampled_x)

    fitted_m = result.params["slope"].value
    fitted_b = result.params["intercept"].value

    
    x_min = min(resampled_x)
    x_max = max(resampled_x)

    area = (fitted_m * (x_max**2 / 2) + fitted_b * x_max) - (fitted_m * (x_min**2 / 2) + fitted_b * x_min)
    return fitted_m, fitted_b, area

def bootstrap_Area(x, y, n_bootstrap=5000):
    ncpu = multiprocessing.cpu_count()
    bootstrap_func = partial(bootstrap_single_iteration, x, y)

    with multiprocessing.Pool(ncpu) as pool:
        results = pool.map(bootstrap_func, range(n_bootstrap))

    slope_fit_estimates, intercept_estimates, area_estimates = zip(*results)
    mean_slope_fit_estimates,std_slope_fit_estimates = np.median(slope_fit_estimates),np.std(slope_fit_estimates)
    mean_intercept_estimates,std_intercept_estimates = np.median(intercept_estimates),np.std(intercept_estimates)
    mean_area,std_area = np.median(area_estimates),np.std(area_estimates)
    
    
    return mean_slope_fit_estimates,std_slope_fit_estimates,mean_intercept_estimates,std_intercept_estimates,mean_area,std_area, slope_fit_estimates, intercept_estimates, area_estimates
    # mean_area, std_area, area_estimates,Slope_fit,Intercept_fit
def update(val, y_noisy_list):
    
    global slider_barrier12,barrier_values,y_curve,y_noisy,x_between_barriers,y_noisy_between_barriers,range_values  #barrier_values, y_curve, y_noisy_between_barriers
    # Extract barrier positions from sliders
    x_barrier1, x_barrier2 = slider_barrier12.val[0], slider_barrier12.val[1]
    x_barrier3, x_barrier4 = slider_barrier34.val[0], slider_barrier34.val[1]
    x_barrier5, x_barrier6 = slider_barrier56.val[0], slider_barrier56.val[1]

    # Additional barrier positions
    x_slider_range, y_slider_range = slider_range.val, slider_range_y.val
    x_slider_range2, y_slider_range2 = slider_range2.val, slider_range_y2.val

    # Combine the data between barriers 1 and 2, and between barriers 3 and 4
    barrier_values = [x_barrier1, x_barrier2, x_barrier3, x_barrier4, x_barrier5, x_barrier6]
    range_values = [ *slider_range.val, *slider_range_y.val,*slider_range2.val, *slider_range_y2.val]
    x_between_barriers = np.concatenate((x[(x >= x_barrier1) & (x <= x_barrier2)], x[(x >= x_barrier3) & (x <= x_barrier4)]))
    
    ax1_1.clear()
    ax1_2.clear()
    for key, y_noisy in y_noisy_list.items():
        y_noisy_between_barriers = np.concatenate((y_noisy[(x >= x_barrier1) & (x <= x_barrier2)], y_noisy[(x >= x_barrier3) & (x <= x_barrier4)]))
        ax1_1.scatter(x_between_barriers, y_noisy_between_barriers, alpha=1, label=f'Data (between limits) imagen {key}', zorder=3) #pensar en colores
        ax1_1.plot(X,Y[key],label=f'spectrum imagen {key}', alpha=0.6, zorder=1)
        # Fit a single linear model to the data between the barriers
        linear_fit_model = Model(linear_fit)
        params = linear_fit_model.make_params(slope=1, intercept=0)
        result = linear_fit_model.fit(y_noisy_between_barriers, x=x_between_barriers, params=params)
        slope_fit, intercept_fit = result.params['slope'].value, result.params['intercept'].value
        ax1_1.plot(x_between_barriers, linear_fit(x_between_barriers, slope_fit, intercept_fit), label=f'Fitted Linear Function for {key}', color='r', zorder=4)
        
        # Calculate the area under the linear function between Barrier 1 and Barrier 4
        area, _ = quad(linear_fit, x_barrier1, x_barrier4, args=(slope_fit, intercept_fit))

        # Calculate the curve after subtracting the area obtained from the fit
        y_curve = y_noisy - linear_fit(x, slope_fit, intercept_fit)
        Y_curve = Y[key] - linear_fit(X, slope_fit, intercept_fit)
        suma = np.sum(y_curve[(x_barrier5 <= x) & (x_barrier6 >= x)])

        # Plot the curve resulting from subtracting the area
        
        ax1_2.plot(X, Y_curve, label=f'Line without continium {key}', color='grey',alpha=0.5)
        ax1_2.hlines(y=[0, 0], xmin=min(x), xmax=max(x), colors='k', linestyles='dashed', zorder=3)
        # Display the area as text inside the plot
    # Plot the fitted linear function   
    ax1_1.axvline(x=line_info["line_center"], color='k', label='Line',linestyle ="--")
    ax1_1.fill_betweenx([-100,100], x_barrier1, x_barrier2,
                        label='left continium', color='purple', alpha=0.2)

    # Create a region between x_barrier3 and x_barrier4
    ax1_1.fill_betweenx([-100,100], x_barrier3, x_barrier4,
                        label='right continium', color='g', alpha=0.2)

# Rest of the code remains the same up to this point

    ax1_1.legend(fontsize='xx-small', frameon=False,ncol=3)
    ax1_1.set_xlim(x_slider_range[0], x_slider_range[1])
    ax1_1.set_ylim(y_slider_range[0], y_slider_range[1])
    ax1_1.set_xlabel('wavelength')
    ax1_1.set_ylabel('Flux')
    
    ax1_2.fill_between(x[(x_barrier5 <= x) & (x_barrier6 >= x)], 
                   -100,100, 
                   label='line core region', color='r', alpha=0.3, zorder=2)
    #ax1_2.axvline(x=x_barrier5, color='m', linestyle='--', label='Barrier 5')
    #ax1_2.axvline(x=x_barrier6, color='c', linestyle='--', label='Barrier 6')
    ax1_2.axvline(x=line_info["line_center"], color='k', label='Line',linestyle ="--")
    ax1_2.set_xlim(x_slider_range2[0], x_slider_range2[1])
    ax1_2.set_ylim(y_slider_range2[0], y_slider_range2[1])
    ax1_2.set_xlabel('wavelenght')
    ax1_2.set_ylabel('Flux')
    ax1_2.legend(fontsize='small', frameon=False,ncol=3)
    
    plt.draw()

def equalility(panda_new,row):
    are_equal,fit_equal,barrier_equal=[],[],"?"
    tolerance = 1e-10
    for i in panda_new.columns:
        is_equal=row[i].values==panda_new[i].values
        if not row[i].values==panda_new[i].values and "y" not in i:
            is_equal = np.abs(row[i].values - panda_new[i].values) < tolerance
        are_equal.append(is_equal)
        if "right" in i or "left" in i:
            fit_equal.append(is_equal)
    return are_equal,fit_equal    
 
def save_data2():
    #global panda_new,panda_read,row,barrier_values,range_values,y_noisy_list,x_between_barriers,y_noisy_between_barriers,line_name,imagen,banda
    for imagen, y_noisy in y_noisy_list.items():
        #falta metodo de error
        panda_new = pd.DataFrame({
            "name":[line_name],
            "imagen":[imagen],
            "band":[band],
            'min_x': [np.min(x)],
            'max_x': [np.max(x)],
            'left_min': [barrier_values[0]],
            'left_max': [barrier_values[1]],
            'right_min': [barrier_values[2]],
            'right_max': [barrier_values[3]],
            'core_min': [barrier_values[4]],
            'core_max': [barrier_values[5]],
            "wavelength":[(barrier_values[4]+barrier_values[5])/2],
            "x_range_min" :[range_values[0]],
            "x_range_max" :[range_values[1]],
            "y_range_min" :[range_values[2]],
            "y_range_max" :[range_values[3]],
            "x_range2_min" :[range_values[4]],
            "x_range2_max" :[range_values[5]],
            "y_range2_min" :[range_values[6]],
            "y_range2_max" :[range_values[7]],
        })

        are_equal,fit_equal=[False],[False]
        if name_file in os.listdir(os.getcwd()):
            panda_read = pd.read_csv(name_file)
            if any((panda_read[["name", "imagen"]].values == [line_name, imagen]).all(axis=1)):
                row=panda_read.loc[((panda_read['name'] == line_name) & (panda_read["imagen"] == imagen))]
                are_equal,fit_equal = equalility(panda_new,row)
        if not all(are_equal):
            panda_new['y_noisy'] = str(y_noisy.tolist())
            if not all(fit_equal):
                print(f"making the bootstraping for {line_name}, {imagen}")
                y_noisy_between_barriers = np.concatenate((y_noisy[(x >= barrier_values[0]) & (x <= barrier_values[1])], y_noisy[(x >= barrier_values[2]) & (x <= barrier_values[3])]))
                mean_slope_fit_estimates,_,mean_intercept_estimates,_,mean_area,std_area, _, _, _ = bootstrap_Area(x_between_barriers,y_noisy_between_barriers, n_bootstrap=5000)
                ycurve=y_noisy-linear_fit(x, mean_slope_fit_estimates,  mean_intercept_estimates)
                panda_new['y_curve'] = str(list(ycurve))
                #print(mean_slope_fit_estimates,mean_intercept_estimates,mean_area,std_area )
                panda_new["slope_fit"]= mean_slope_fit_estimates
                panda_new["intercept_fit"]=mean_intercept_estimates
                panda_new["area_continuo"]=mean_area
                panda_new["area_continuo_error"]=std_area
            else:
                for i in row.columns:
                    if i not in panda_new.columns:
                        panda_new[i]=row[i].values
                ycurve=eval(row['y_curve'].values[0])
            #aqui mal #print(panda_new["core_line"])
            if "cont" in line_name:
                panda_new["core_line"] = 0
            else:   
                panda_new["core_line"]=np.sum(np.array(ycurve)[(barrier_values[4]<=x) & (barrier_values[5]>=x)])
            try:
                if len(panda_read)>0:
                    if any((panda_read[["name", "imagen"]].values == [line_name, imagen]).all(axis=1)):
                        print(f'Data updated in {name_file} for {line_name}, {imagen}')
                        condition = ((panda_read['name'] == line_name) & (panda_read["imagen"] == imagen))
                        row_to_drop = panda_read.loc[condition]
                        panda_read = panda_read.drop(row_to_drop.index)
                        pd.concat([panda_read, panda_new],axis=0).to_csv(name_file, index=False)
                    else:
                        print(f"New row {line_name}, {imagen} in file {name_file}")
                        pd.concat([panda_read, panda_new],axis=0).to_csv(name_file, index=False)
            except:
                    print(f"New row {line_name}, {imagen}")
                    print(f"Creating file {name_file}")
                    panda_new.to_csv(name_file, index=False)
        else:
            print(f"No changes detected in row {line_name}, {imagen}")

def savesubplot(ax,bbox_inches,path):
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    fig_save, ax_save = plt.subplots()
    ax_save.set_xlim(x1, x2)
    ax_save.set_ylim(y1, y2)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=extent.expanded(*bbox_inches))
    plt.close(fig_save)

def on_save_button_clicked(event):
    #global y_curve,y_noisy,S,x_between_barriers,y_noisy_between_barriers,line_name,imagen,banda
    global images,panda_new,panda_read,row,barrier_values,range_values,y_curve,y_noisy,x_between_barriers,y_noisy_between_barriers,line_name,imagen,band,y_noisy_list
    save_data2()
    #save_data()   
    button_save.label.set_visible(True)
    fig.canvas.draw_idle()
    try:
        os.mkdir("images")
    except:
        pass
    savesubplot(ax1_2,(1.15, 1.25),f'images/fit_{line_name}_image_{"".join(list(Y.keys()))}_WC.png')
    savesubplot(ax1_1,(1.15, 1.3),f'images/fit_{line_name}_image_{"".join(list(Y.keys()))}.png')
    print("Image saved")
def on_remove_line_clicked(event):
    panda_previous=pd.read_csv(name_file)
    panda_previous = panda_previous[panda_previous["name"] != line_name]
    panda_previous.to_csv(name_file)
    print(f"deleted {line_name}")
def on_close_all(event):
    global stop_flag
    stop_flag = True  # Set the flag to True to stop the loop
    print("Requested to stop plotting.")
#continium lines should be have a more wide range to select windows 
#maybe add a function to save a restframe like file in cases were i found nice windows to use for other examples
global images
stop_flag=False
def window_analisis(pandas_spectra,name_obj,flux_column="flux_",lens_images=None,bands=None,use_preselected_lines=False,file=None):
    #obj= imagenes 
    #name_obj str
    zs = pandas_spectra.zs.values[0]
    global slider_barrier12,slider_barrier34,slider_barrier56,slider_range,slider_range_y,slider_range2,slider_range_y2,x,ax1_1,ax1_2,X,Y,line_info,y_noisy_list,line_name,band,name_file,button_save,fig
    window=pd.read_csv(os.path.join(module_dir,"rest_frame_windows.csv"))
    w = [[list(np.array(eval(ii.replace(" ",",").replace("nan","np.nan")))*(1+zs)) for ii in window[i].values] for i in window.columns if "line_name" not in i]
    #window[[i for i in window.columns if "line_name" not in i]] = [[np.array(eval(ii.replace(" ",",").replace("nan","np.nan")))*(1+zs) for ii in window[i].values] for i in window.columns if "line_name" not in i]
    window["left_range"] =w[0]
    window['ring_range'] =w[1]
    window['core_range'] =w[2]
    ##,barrier_values,y_curve,y_noisy,x_between_barriers,y_noisy_between_barriers,range_values
    if name_file==None: 
        name_file=f"Flux_cont_core_{name_obj}.csv"
    else:
        name_file = file
    if bands==None:
        bands = pandas_spectra.band.unique()
    for band in bands:
        spectra_band = pandas_spectra[pandas_spectra.band==band]
        spectra_band = spectra_band.dropna(axis=1, how='all')
        if lens_images==None:
            images = [i.replace(flux_column,"") for i in spectra_band.columns if flux_column in i and "G" not in i]
        for r in range(len(window)):
            
            line_info=window.iloc[r].copy()
            line_name=line_info["line_name"]
            if use_preselected_lines==True:
                if name_file in os.listdir(os.getcwd()):
                    panda_read=pd.read_csv(name_file)
                    if not any((panda_read[["name"]].values == [line_name]).all(axis=1)):
                        continue
            #maybe condition
            if "core_range" in line_info.keys() and "line_center" not in line_info.keys():
                mean_rango = np.mean(eval(str(line_info["core_range"])))
                line_info["line_center"] = mean_rango 
                rango=[mean_rango-1000,mean_rango+1000]
            else:
                rango=[line_info["line_center"]-1000,line_info["line_center"]+1000]
            if mean_rango<spectra_band["wavelength"].values[0] or mean_rango>spectra_band["wavelength"].values[-1]:
                continue 
            #######################################
            X,Y = spectra_band["wavelength"].values,{imagen:spectra_band[f"{flux_column}{imagen}"].values for imagen in images}
            x=spectra_band["wavelength"].values[(rango[0]<=spectra_band["wavelength"].values) & (rango[1]>=spectra_band["wavelength"].values)]
            y_noisy_list= {imagen:spectra_band[f"{flux_column}{imagen}"].values[(rango[0]<=spectra_band["wavelength"].values) & (rango[1]>=spectra_band["wavelength"].values)] for imagen in images}
            y_list_D=list(y_noisy_list.values())
            y_up,y_down=np.max(y_list_D)*1.1,np.min(y_list_D)
            q2 = np.percentile(y_list_D, 89)
            q3 = np.percentile(y_list_D, 95) 
            fig = plt.figure(figsize=(15, 15/ aspect_ratio))
            grid = plt.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[3, 1], hspace=0.4)
            ax1_1 = plt.subplot(grid[0, 0])
            ax1_2 = plt.subplot(grid[0, 1])
            
            
            bbox = ax1_1.get_position()
            bbox2 = ax1_2.get_position()
            
            # Add sliders for the barrier positions (as a column inside [0, 1])
            slider_barrier1_ax = plt.axes([bbox.x0, bbox.y0-0.2, bbox.width, 0.03]) #left conti
            slider_barrier4_ax = plt.axes([bbox.x0, bbox.y0-0.25, bbox.width, 0.03]) #right conti
            slider_range_ax = plt.axes([bbox.x0, bbox.y0-0.1, bbox.width, 0.03]) #range wavelenght 
            
            
            slider_range_ax2 = plt.axes([bbox2.x0, bbox2.y0-0.1, bbox2.width, 0.03]) #range wavelenght for core
            slider_barrier6_ax = plt.axes([bbox2.x0, bbox2.y0-0.225, bbox2.width, 0.03]) # line core
            
            y_slider_range_ay2=plt.axes([bbox2.x0+bbox.width+0.05,bbox2.y0, 0.01, bbox.height]) #range flux core
            y_slider_range_ay=plt.axes([bbox.x0-0.07, bbox.y0, 0.01, bbox2.height]) #range flux  
            
            
            
            if band=="NIR":
                y_up = q2*4
            L_valini=[line_info["line_center"]-100,line_info["line_center"]-50,line_info["line_center"]+50,line_info["line_center"]+100,line_info["line_center"]-10,line_info["line_center"]+10]
            range_values_O = [line_info["line_center"]-500, line_info["line_center"]+500,-y_up,y_up,line_info["line_center"]-100,line_info["line_center"]+100,-y_up,y_up]
            if name_file in os.listdir(os.getcwd()):
                panda_read=pd.read_csv(name_file)
                if any((panda_read[["name"]].values == [line_name]).all(axis=1)):
                    row=panda_read.loc[panda_read["name"]==line_name]
                    L_valini =[row[i].values[0] for i in row.columns if "left" in i] + [row[i].values[0] for i in row.columns if "right" in i] +[row[i].values[0] for i in row.columns if "core" in i]
                    try:
                        range_values_O = [row[i].values[0] for i in row.columns if "x_range_" in i]+[row[i].values[0] for i in row.columns if "y_range_" in i]+[row[i].values[0] for i in row.columns if "x_range2_" in i]+[row[i].values[0] for i in row.columns if "y_range2_" in i]
                    except:
                        pass
            ###########################
            slider_barrier12 = RangeSlider(slider_barrier1_ax, "left \ncontinium",np.min(x),line_info["line_center"],valinit=[L_valini[0], L_valini[1]],color="purple",alpha=0.5) 
            slider_barrier34 = RangeSlider(slider_barrier4_ax, "right \ncontinium",line_info["line_center"],np.max(x),valinit=[L_valini[2], L_valini[3]],color="green",alpha=0.2) 
            if band=="NIR":
                slider_barrier56 = RangeSlider(slider_barrier6_ax , "line core",line_info["line_center"]-100,line_info["line_center"]+100,valinit=[L_valini[4], L_valini[5]],color="r",alpha=0.2)
            elif "cont" in line_name:
                slider_barrier56 = RangeSlider(slider_barrier6_ax , "line core",line_info["line_center"]-1,line_info["line_center"]+1,valinit=[L_valini[4], L_valini[5]],color="r",alpha=0.2)
            else:
                slider_barrier56 = RangeSlider(slider_barrier6_ax , "line core",line_info["line_center"]-100,line_info["line_center"]+100,valinit=[L_valini[4], L_valini[5]],color="r",alpha=0.2)
            ###################################
            slider_range = RangeSlider(slider_range_ax, "Range",np.min(X),     np.max(X),valinit=range_values_O[0:2])
            slider_range_y=RangeSlider(y_slider_range_ay, "Range",-y_up,y_up,orientation="vertical",valinit=range_values_O[2:4], valfmt="")
            slider_range2 = RangeSlider(slider_range_ax2, "Range",np.min(x), np.max(x),valinit=range_values_O[4:6])
            slider_range_y2=RangeSlider(y_slider_range_ay2, "Range",-y_up,y_up,orientation="vertical",valinit=range_values_O[6:8], valfmt="")
            #slider_range_y2.set_visible(False) 
            slider_range_y2.valtext.set_visible(False)
            slider_range_y.valtext.set_visible(False)
            slider_range.valtext.set_visible(False)
            slider_range2.valtext.set_visible(False)
            
            ax_save_button = plt.axes([0.4, 0.02, 0.2, 0.04])
            button_save = Button(ax_save_button, 'Save', color='lightgoldenrodyellow', hovercolor='0.975')
            if name_file in os.listdir(os.getcwd()):
                if any((panda_read[["name"]].values == [line_name]).all(axis=1)):
                    ax_save_button = plt.axes([0.6, 0.02, 0.2, 0.04])
                    button_remove = Button(ax_save_button, 'remove line', color='lightgoldenrodyellow', hovercolor='0.975')
                    button_remove.on_clicked(on_remove_line_clicked)
            # Update the plot when slider values change
            ax_save_button = plt.axes([0.01, 0.95, 0.2, 0.04])
            button_close = Button(ax_save_button, 'Close', color='lightgoldenrodyellow', hovercolor='0.975')
            button_close.on_clicked(on_close_all)
            slider_barrier12.on_changed(lambda val: update(val, y_noisy_list))
            slider_barrier34.on_changed(lambda val: update(val, y_noisy_list))
            slider_barrier56.on_changed(lambda val: update(val, y_noisy_list))
            slider_barrier56.on_changed(lambda val: update(val, y_noisy_list))


            slider_range.on_changed(lambda val: update(val, y_noisy_list))
            slider_range_y.on_changed(lambda val: update(val, y_noisy_list))
            slider_range2.on_changed(lambda val: update(val, y_noisy_list))
            slider_range_y2.on_changed(lambda val: update(val, y_noisy_list))
            #slider_barrier8.on_changed(update)
            # Set up the event handler for the button

            button_save.on_clicked(on_save_button_clicked)
            
            ax1_1.set_xlim(np.min(x), np.max(x))
            ax1_1.set_xlabel('wavelength')
            ax1_1.set_ylabel('Flux')
            update(0.01, y_noisy_list)
            titulo = " and ".join(images)
            if len(images)>2:
                titulo = ",".join(images)
            fig.suptitle(f"Fit {line_name} line for image {titulo}", fontsize=16)
            #plt.gca().set_aspect('equal', adjustable='box')
            if stop_flag:
              #  plt.close()
                return
            #else:
            plt.show()
        images=None
            
