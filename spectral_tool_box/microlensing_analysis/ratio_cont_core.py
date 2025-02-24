import uncertainties
from uncertainties import unumpy,ufloat,umath
import pandas as pd
import numpy as np 

def _2_5_log10_and_error(args):
    #ima int, error imag int, ref int, error int
    value =  -2.5*umath.log10(ufloat(*args[0:2])/ufloat(*args[2:4]))
    return value.nominal_value,value.s

def ratio_cont_core(file,ref,img=None,save=False,obj_name=None):
    integrated = pd.read_csv(file)
    if isinstance(img,str):
        img = [img]
    elif img==None:
        img = [i for i in integrated.imagen.unique() if i !=ref]
    Ref = integrated.loc[(integrated["imagen"]==ref).values & ([bool("cont" not in i) for i in integrated["name"]])].copy()
    Ima = integrated.loc[np.array([bool(i in img) for i in integrated["imagen"].values]) & ([bool("cont" not in i) for i in integrated["name"]])].copy()
    Ref_c = integrated.loc[(integrated["imagen"]==ref).values & ([bool("cont" in i) for i in integrated["name"]])].copy()
    Ima_c = integrated.loc[np.array([bool(i in img) for i in integrated["imagen"].values]) & ([bool("cont" in i) for i in integrated["name"]])].copy()
    for n,ima in enumerate(img):
        I = Ima[Ima["imagen"]==ima].copy()
        I_c = Ima_c[Ima_c["imagen"]==ima].copy()
        Razon_l = pd.DataFrame([[ref,ima,I["name"].values[i],np.mean(I[["core_max","core_min"]].values[i]),*_2_5_log10_and_error((*I[['area_continuo','area_continuo_error']].values[i], 
        *Ref[['area_continuo','area_continuo_error']].values[i])),*_2_5_log10_and_error((*I[['core_line','area_continuo_error']].values[i], *Ref[['core_line','area_continuo_error']].values[i])),] for i in range(len(I))]
        ,columns=["ref","ima","line","wavelength","r_cont","e_r_cont","r_core","e_r_core"])
        #
        Razon_c = pd.DataFrame([[ref,ima,I_c["name"].values[i],np.mean(I_c[["core_max","core_min"]].values[i]),*_2_5_log10_and_error((*I_c[['area_continuo','area_continuo_error']].values[i], 
        *Ref_c[['area_continuo','area_continuo_error']].values[i])),0,0] for i in range(len(I_c))]
        ,columns=["ref","ima","line","wavelength","r_cont","e_r_cont","r_core","e_r_core"])
        Razon = pd.concat([Razon_c, Razon_l], ignore_index=True)
        Razon["dif"] =  Razon["r_cont"] - Razon["r_core"]
        if save==True:
            Razon.to_csv(f"Ratio_cont_core_{obj_name}_ref_{ref}_ima_{ima}.csv", index=False)
            print(f"Saved ref image {ref} and image {ima}")
