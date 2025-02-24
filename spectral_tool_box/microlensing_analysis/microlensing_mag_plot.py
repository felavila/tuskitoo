import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RectangleSelector
import numpy as np
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import os 


def fit_linear_with_errors(x, y, y_uncertainties,Plot=plt,X=None):
    # Define the linear function to fit
    def linear_model(x, a, b):
        return a * x + b

    # Fit the data to the linear model with uncertainties
    params, covariance = curve_fit(linear_model, x, y, sigma=y_uncertainties, absolute_sigma=True)

    # Extract the fit parameters
    a_fit, b_fit = params

    # Calculate the two-sigma (95% confidence) error bars
    sigma_a, sigma_b = np.sqrt(np.diag(covariance))
    a_upper = a_fit +   sigma_a
    a_lower = a_fit -   sigma_a
    b_upper = b_fit +   sigma_b
    b_lower = b_fit -   sigma_b

    # Create the fitted line
    x_fit = np.linspace(min(X)-0.08, max(X)+0.08, 100)
    y_fit = linear_model(x_fit, a_fit, b_fit)

    # Create the upper and lower bounds for the fitted line
    y_upper = linear_model(x_fit, a_upper, b_upper)
    y_lower = linear_model(x_fit, a_lower, b_lower)

   
    Plot.plot(x_fit, y_fit, linestyle='--', label='Fitted Line', color='red')
    Plot.fill_between(x_fit, y_upper, y_lower, color='red', alpha=0.1)

    return #a_fit, b_fit, sigma_a, sigma_b, x_fit, y_fit, y_upper, y_lower
def fit_median(x, y, y_uncertainties,Plot=plt,X=None):
    y_median=np.nanmedian(y)
    absolute_deviations = np.abs(y - y_median)
    mad = np.nanmedian(absolute_deviations)
    Plot.plot([min(X)-0.08, max(X)+0.08], [y_median,y_median], label=None,color="k", linestyle='dashed')
    Plot.fill_between([min(X)-0.08, max(X)+0.08], [y_median-mad,y_median-mad],  [y_median + mad,y_median + mad], color="k", label=None,zorder=0,alpha=0.1)
    return
# Initial data

def microlensing(file,N=10000):
    print("To save press x")
    """"file:f"Ratio_cont_core_{nombre_obj}.csv"""""
    if isinstance(file,str):
        R = pd.read_csv(file).sort_values("wavelength")
    ref,ima = R[["ref","ima"]].drop_duplicates().values[0]
    R_core = R[[bool("cont" not in i) for i in R["line"]]]
    
    x = R["wavelength"].values/N
    x_core =  R_core["wavelength"].values/N
    
    y_cont = R["r_cont"].values
    y_core = R_core["r_core"].values
    y_cont_err = 2*R["e_r_cont"].values
    y_core_err = 2*R_core["e_r_cont"].values
    Y = np.concatenate((y_cont, y_core))
    
    fig, (ax, ax_check) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]},figsize=(14, 10))
    ax_check.set_title("Select Points")
    x_label,y_label= "Wavelength"+ r"$(\mu m)$",f"$m_{ima}-m_{ref}$"
    try:
        states=R["bol_cont"].values.tolist()
        states_core= [bool(x) for x in R["bol_core"].values if x != "0"]
    except:
        states = [True] * len(x)
        states_core = [True] * len(x_core)

    ax.errorbar(x[states], y_cont[states], yerr=y_cont_err[states], fmt="s", label='Continium',c="red")
    fit_linear_with_errors(x[states], y_cont[states], y_cont_err[states],Plot=ax,X=x)
    ax.errorbar(x_core[states_core], y_core[states_core], yerr=y_core_err[states_core], fmt="^", label='Core',c="k")
    fit_median(x_core[states_core], y_core[states_core], y_core_err[states_core],Plot=ax,X=x)
    for i in  list(set(R["line"].values[states].tolist() + R_core["line"].values[states_core].tolist())):
        x_p = R[R["line"]==i]["wavelength"].values/N
        ax.axvline(x_p, linestyle='--', color='grey', alpha=0.5)
        ax.text(x_p, np.max(Y)+0.1, i, rotation=270, ha='left', va='top', fontsize=12, color='k')
    
    selected_line_core,selected_line_cont = R_core["line"].values,R["line"].values
    labels_cont = [i+"_cont" for i in R["line"].values]
    labels_core = [i+"_core" for i in R_core["line"].values]   
    check = CheckButtons( ax=ax_check, labels=labels_cont+labels_core, actives=states+states_core,label_props={'color': ["r"]*(len(labels_cont))+["k"]*len(labels_core)})

    def update(label):
        global selected_line_cont,selected_line_core
        # Clear the previous data points and line fit
        ax.clear()
        selected_x = [x[i] for i in range(len(labels_cont)) if check.get_status()[i] if i<len(labels_cont)]
        selected_y = [y_cont[i] for i in range(len(labels_cont)) if check.get_status()[i] if i<len(labels_cont)]
        selected_y_err = [y_cont_err[i] for i in range(len(labels_cont)) if check.get_status()[i] if i<len(labels_cont)]
        selected_line_cont = [R["line"].values[i] for i in range(len(labels_cont)) if check.get_status()[i] if i<len(labels_cont)]
        ###########################
        selected_x_core = [x_core[i-len(labels_cont)] for i in range(len(labels_core+labels_cont)) if check.get_status()[i] if i>=len(labels_cont)]
        selected_y_core = [y_core[i-len(labels_cont)] for i in range(len(labels_core+labels_cont)) if check.get_status()[i] if i>=len(labels_cont)]
        selected_y_core_err  = [y_core_err[i-len(labels_cont)] for i in range(len(labels_core+labels_cont)) if check.get_status()[i] if i>=len(labels_cont)]
        selected_line_core  = [R_core["line"].values[i-len(labels_cont)] for i in range(len(labels_core+labels_cont)) if check.get_status()[i] if i>=len(labels_cont)]
        lines_plot = list(set(selected_line_core + selected_line_cont))
        SUPER_Y =  list(set(selected_y + selected_y_core))
        if len(selected_x) >= 2:
            ax.errorbar(selected_x, selected_y, yerr=selected_y_err, fmt="s", label='Continium',c="red")
            fit_linear_with_errors(selected_x, selected_y, selected_y_err,Plot=ax,X=x)
        if len(selected_y_core) >= 2:
            ax.errorbar(selected_x_core, selected_y_core, yerr=selected_y_core_err, fmt="^", label='Core',c="k")
            fit_median(x, selected_y_core, selected_y_core_err,Plot=ax,X=x)
        if len(selected_x) == 1:
            ax.errorbar(selected_x_core, selected_y, yerr=selected_y_err, fmt="s", label='Continium',c="red")
        if len(selected_x_core) == 1:
            ax.errorbar(selected_x_core, selected_y_core, yerr=selected_y_core_err, fmt="^", label='Core',c="k")
        for i,nombre in enumerate(lines_plot):
            x_linea = R[R["line"]==nombre]["wavelength"].values[0]/N
            ax.axvline(x_linea, linestyle='--', color='grey', alpha=0.5)
            ax.text(x_linea, np.max(Y)+0.1, nombre, rotation=270, ha='left', va='top', fontsize=12, color='k')
        ax.legend(frameon=False,bbox_to_anchor=(0.1, 1.02, 1, 0.2), loc="lower left",
                   mode="", borderaxespad=0, ncol=3,fontsize=15)
        ax.set_xlim(min(x)-0.08,max(x)+0.08)
        ax.set_ylim(np.min(SUPER_Y)-0.1,np.max(SUPER_Y)+0.1)
        ax.set_xlabel(x_label, fontsize=12)  # Set x-axis label font size
        ax.set_ylabel(y_label, fontsize=12)  # Set y-axis label font size
        plt.draw()

    ax.set_xlabel(x_label, fontsize=12)  # Set x-axis label font size
    ax.set_ylabel(y_label, fontsize=12)  # Set y-axis label font size
    ax.legend(frameon=False,bbox_to_anchor=(0.1, 1.02, 1, 0.2), loc="lower left",
                    mode="", borderaxespad=0, ncol=3,fontsize=15)
    ax.set_xlim(min(x)-0.08,max(x)+0.08)
    ax.set_ylim(np.min(Y)-0.1,np.max(Y)+0.1)
    check.on_clicked(update)
    def save_callback(event):
        global selected_line_cont,selected_line_core
        if event.key == 'x':
            # Calculate the coordinates of the selected region (use zoom coordinates)
            x1, x2 = ax.get_xlim()
            y1, y2 = ax.get_ylim()

            # Create a new figure and axis for the saved plot
            fig_save, ax_save = plt.subplots()
            ax_save.set_xlim(x1, x2)
            ax_save.set_ylim(y1, y2)

            # Copy the relevant content from the original plot (excluding the CheckButtons part)
            #ax_save.imshow(fig.canvas.copy_from_bbox(ax.bbox), interpolation='nearest', origin='upper')
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            try:
                os.mkdir("images")
            except:
                pass
            fig.savefig('images/microlensing.png', bbox_inches=extent.expanded(1.2, 1.3))
            try:
                R["bol_cont"] = [bool(i in selected_line_cont) for i in R["line"].values]
                R["bol_core"] = [bool(i in selected_line_core) if "cont" not in i else 0 for i in R["line"].values]
                R.to_csv(file, index=False)
            except:
                pass
            print("image save")
            plt.close(fig_save)

# Create an event handler for key presses
    fig.canvas.mpl_connect('key_press_event', save_callback)
    plt.show()

    