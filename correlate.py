import numpy as np
try:
    import astLib;from astLib import astCoords
    have_astLib = True
except:
    import astropy
    from astropy.coordinates import SkyCoord
    have_astLib = False
import sys;from sys import stdout


def astropy_sep (tra1,tdec1,tra2,tdec2):
    sc1 = SkyCoord(tra1, tdec1, frame = 'fk5', unit='degree')
    sc2 = SkyCoord(tra2, tdec2, frame = 'fk5', unit = 'degree')
    return((sc1.separation(sc2)).degree)

# correlate two arrays sorted in ra. array2 is the bigger one.

def correlate (array1, ra1, dec1, array2, ra2, dec2, dist, \
               mindist=0.0, isabs=False, noisy=True):
    fstart=nfstart=0
    fend=array2.shape[0]
    icou=0
    correl=np.array([])
    decfac=1.0 if isabs else min(1./np.cos(np.deg2rad(array1[:,dec1].max())),\
               1./np.cos(np.deg2rad(array2[:,dec2].max())))
    
    for i in range(array1.shape[0]):
        i10 = np.linspace(0,array1.shape[0],10,dtype='int')
        i100 = np.linspace(0,array1.shape[0],100,dtype='int')
        if i in i10 and noisy:
            sys.stdout.write('*')
            sys.stdout.flush()
        elif i in i100 and noisy:
            sys.stdout.write('.')
            sys.stdout.flush()
        else:
            pass
        fstart=nfstart
        for j in range(fstart,fend):
            r1,d1 = array1[i,ra1],array1[i,dec1]
            r2,d2 = array2[j,ra2],array2[j,dec2]
            radiff = r2-r1
            if radiff<-decfac*dist:
                nfstart=j
            if radiff>decfac*dist:
                break
            if abs(d2-d1)>dist:
                continue
            if isabs:
                adist = np.hypot(r1-r2,d1-d2)
            else:
                adist = astCoords.calcAngSepDeg(r1,d1,r2,d2) if have_astLib \
                        else astropy_sep(r1,d2,r2,d2)

            if adist<dist and abs(radiff)<90.0 and adist>=mindist:
                try:
                    correl=np.vstack((correl,np.array([i,j,adist])))
                except:
                    correl=np.array([[i,j,adist]])

    return correl
            
