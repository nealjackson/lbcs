import astropy; from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

def corr_astro (a1,ra1col,dec1col,a2,ra2col,dec2col,dist):
    a1 = np.array([a1]) if a1.ndim == 1 else a1
    a2 = np.array([a2]) if a2.ndim == 1 else a2
    c1 = SkyCoord(np.asarray(a1[:,ra1col],dtype='f')*u.degree,\
                  np.asarray(a1[:,dec1col],dtype='f')*u.degree)
    c2 = SkyCoord(np.asarray(a2[:,ra2col],dtype='f')*u.degree,\
                  np.asarray(a2[:,dec2col],dtype='f')*u.degree)
    c = c2.search_around_sky (c1,dist*u.deg)
    a = np.asarray(np.column_stack((c[0],c[1],c[2])),dtype='f')
    return a
