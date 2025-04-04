import numpy as np 


def off(hdr):
    Y_1 = hdr['ESO SEQ CUMOFF Y']
    return  -Y_1

def Ref(header_1,header_2):
    # Extract and convert header values to float
    RA_1 = float(header_1['RA'])
    RA_2 = float(header_2['RA'])
    DEC_1 = float(header_1['DEC'])
    DEC_2 = float(header_2['DEC'])
    dif =  np.sqrt( (RA_1-RA_2)**2 + (DEC_1-DEC_2)**2  )
    return dif * 3600

def ToAng(hdr):
    """
    ???
    """
    # Extract and convert header values to float
    Len = float(hdr['NAXIS2'])
    Pix = float(hdr['CRPIX2'])
    Val = float(hdr['CRVAL2'])
    Del = float(hdr['CDELT2'])
    
    # Generate pixel indices (1-based to match FITS standard)
    pix = np.arange(1, Len + 1)
    
    # Calculate arcsecond values
    arcsec = pix * Del + (Val - Del * Pix)
    
    return arcsec

def correct(hdrs):
    """
    ???
    """
    Ang = [ToAng(hdr) for hdr in hdrs]
    r = [off(hdr) for hdr in hdrs] #off
    return Ang,r


#images
#pca_work:your images  
correct_ = correct(list of headers)
TR = np.zeros((number_images,image_sky_subtracted_flux_corrected.shape[1]*2+1,image_sky_subtracted_flux_corrected.shape[2]))
not_TR = np.zeros((number_images,image_sky_subtracted_flux_corrected.shape[1]*2+1,image_sky_subtracted_flux_corrected.shape[2]))
for i in range(TR.shape[0]):
    an0 = correct_[0][i] - correct_[1][i]
    ind = np.argmin(abs(an0))
    Lx = pca_work[i].shape[0] #number of pixels in x axi# 
    au1 = Lx-ind
    au0 = Lx-au1
    #print(Lx- au0,Lx+au1-1)
    not_TR[i,Lx- au0:Lx+au1,:] = pca_work[i]
    if band != "UVB":
        TR[i,Lx- au0:Lx+au1,:] = pca_work[i]/ telluric[i]
    #au1 = np.arange(ind,Lx)
TR = TR
final_TR = np.median(TR,axis=0)