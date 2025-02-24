from astropy.io import fits
import os
from astropy.wcs import WCS
from astropy.io.fits import Header
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton,minimize
import numpy as np
import pandas as pd
import h5py


def _read_hdf5_recursive(filepath, path='/'):
    """
    Recursively reads an HDF5 file from the specified path and returns its data.

    Parameters:
    - filepath: str, path to the HDF5 file.
    - path: str, starting path within the HDF5 file (default is root '/').
    
    Returns:
    - dict: hierarchical data structure representing the HDF5 file contents.
    """
    def _read_group(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data[key] = _read_group(item)
            elif isinstance(item, h5py.Dataset):
                if isinstance(item[()], bytes):
                    data[key] = item[()].decode('utf-8')
                elif isinstance(item[()], np.ndarray):
                    #if "header" in key:
                     #   data[key] = Header.fromstring(StringIO('\n'.join([card.decode('utf-8') for card in item[()]])).read(), sep='/')
                    data[key] = [x if isinstance(x, np.float64) else x for x in item[()]]
                    #data[key] =  Header.fromstring(StringIO(''.join([card.decode('utf-8') for card in item[()]])).read(), sep='\n')
                else:
                    data[key] = item[()]
            else:
                data[key] = item[()]
        return data

    with h5py.File(filepath, 'r') as f:
        return _read_group(f[path])
def list_to_header(header_list):
    if len(header_list)==3:
        header = fits.Header()
        for a, b, c in np.array(header_list).T:
            try:
                header[a.decode('utf-8')] = (float(b.decode('utf-8')),c.decode('utf-8'))
            except:
                header[a.decode('utf-8')] = (b.decode('utf-8'),c.decode('utf-8'))
        return header
    return Header.fromstring(StringIO('\n'.join([card.decode('utf-8') for card in header_list])).read(), sep='/')  
    
def r_results_lector(file_path):
    data = _read_hdf5_recursive(file_path)
    if "OBJ" in data.keys():
        if "hdr" in data["OBJ"].keys():
            #data["OBJ"].update({"header": list_to_header(data["OBJ"]["hdr"])})
            [data[key].update({"hdr": list_to_header(data[key]["hdr"])}) for key in data.keys() if "OBJ" in key];
        if "header" in data["OBJ"].keys():
            #data["OBJ"].update({"header": list_to_header(data["OBJ"]["header"])})
            #[data[key].update({"header": list_to_header(data[key]["header"])}) for key in data.keys() if "OBJ" in key];
            [data[key].update({"header": list_to_header(data[key]["header"])}) for key in data.keys() if "OBJ" in key];
    return data