import numpy as np
import matplotlib; from matplotlib import pyplot as plt
import astropy; from astropy.coordinates import SkyCoord
import astropy.units as u
LBCS_ROOT='/home/njj/projects/lbcs'

def dist_stations(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2 = np.deg2rad(lat1),np.deg2rad(lon1),\
                          np.deg2rad(lat2),np.deg2rad(lon2)
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = (np.sin(dlat/2.))**2+\
        np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2.))**2
    c = 2.*np.arctan2(np.sqrt(a),np.sqrt(1.-a))
    return 6373.*c

def pos2lengths(infile='IBpos',outfile='IBlengths'):
    pos = np.asarray(np.loadtxt(infile,dtype='str')[:,-2:],dtype='float')
    f=open(outfile,'w')
    blengths = []
    f.write('# EXL   EF   UW   TA   PO   JU   NA   ON   CH   NO   BO   BA   LA   BI\n')
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i!=j:
                lat1,lon1,lat2,lon2 = pos[i,0],pos[i,1],pos[j,0],pos[j,1]
                this_length = dist_stations(lat1,lon1,lat2,lon2)
                f.write('%5d'%int(this_length))
                blengths.append(this_length)
            else:
                f.write('    0')
        f.write('\n')
    f.close()
    return blengths

def blength_hist(outfile=''):
    blengths = pos2lengths()
    matplotlib.rc('font',size=14)
    plt.hist(blengths,range=[0,2000],bins=20)
    plt.xlabel('Baseline length/km')
    plt.ylabel ('Number of baselines')
    if outfile=='':
        plt.show()
    else:
        plt.savefig(outfile,bbox_inches='tight')

def plot_aitoff(a,outfile='',colour='b',marker=','):
    sc = SkyCoord(a,unit=(u.deg,u.deg))
    ra = sc.ra.wrap_at(180*u.deg).radian
    dec = sc.dec.radian
    plt.subplot(111,projection='aitoff')
    plt.grid(True)
    plt.plot(ra,dec,colour+marker,alpha=0.3)
    if outfile=='':
        plt.show()
    else:
        plt.savefig(outfile,bbox_inches='tight')

def plot_lbcs(infile=LBCS_ROOT+'/final_cat/lbcs_stats.sum',outfile=''):
    a = np.asarray((np.loadtxt(infile,dtype='str'))[:,-2:],dtype='float')
