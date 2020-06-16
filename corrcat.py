import numpy as np
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u

def correlate (cat1,cra1,cdec1,cat2,cra2,cdec2,dist):
    c=SkyCoord(ra=cat1[:,cra1]*u.degree,dec=cat1[:,cdec1]*u.degree)
    catalog=SkyCoord(ra=cat2[:,cra2]*u.degree,dec=cat2[:,cdec2]*u.degree)
    idxc,idxcatalog,d2d,d3d=catalog.search_around_sky(c,dist*u.deg)
    return np.asarray(np.column_stack((idxc,idxcatalog,d2d)),dtype='float')
