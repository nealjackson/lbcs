import numpy as np,os,sys,astropy
import matplotlib; from matplotlib import pyplot as plt
from lbcs_utils import *
from corrcat import *

#### Experiments with the LBCS catalogue and comparison with First and Wenss
#### Purpose: find out how representative the LBCS sources are
#
# FIRST catalogue: columns are ra dec fpeak fint size pa angle
#
firstcat = '/home/njj/catalogues/first_2014.simple.npy'
if not os.path.isfile('../final_cat/lbcs_stats.cons'):
    lbcs_consolidate('../final_cat/lbcs_stats.sum','../final_cat/lbcs_stats.cons')

#
# LBCS catalogue in usual format but nb we are using the consolidated
# catalogue (where duplicate observations are combined)
#
lbcscat = '../final_cat/lbcs_stats.cons'
first = np.asarray(np.load(firstcat),dtype='float')
#
# select northern FIRST sources
#
#first = first[first[:,4]<2.00]   # point sources
#   problem! - if you just select point sources you don't add
#   fluxes up properly
first = first[first[:,1]>0.00]   # northern sources
lbcs = np.loadtxt(lbcscat,dtype='str')
wenss = np.load('wenss2000.npy')
#
#   WENSS catalogue: columns are: ra dec type fpeak fint
#
c_first = np.asarray(first[:,:2],dtype='float')
c_lbcs = np.asarray(lbcs[:,-2:],dtype='float')
#
#  correlate first and LBCS catalogues to give the array of correlations
#
corr_fl = np.asarray((correlate(c_first,0,1,c_lbcs,0,1,0.01))[:,:2],dtype='int')
fcorr,lcorr=corr_fl[:,0],corr_fl[:,1]
print ('************************************')
print ('Total sources in first,LBCS:',len(c_first),len(c_lbcs))
print ('Total number of First sources in LBCS:',len(np.unique(corr_fl[:,0])))
print ('Total number of LBCS sources in First:',len(np.unique(corr_fl[:,1])))
print ('************************************')
#
# make array lf of sources in LBCS and First. NB if multiple First sources
# are present add their fluxes together.
#   ********* lf array columns: ***************
#   0=LBCS name 1-2=LBCS pos 3-5=LBCS PSX, qual, FT parameters
#   6-10 First params (pflux, iflux, size, axrat, PA)
#
lf = np.array([])
for i in np.unique(lcorr):
    fthis = np.take(first,fcorr[lcorr==i],axis=0)
    imaxthis = np.argwhere(fthis[:,4]==fthis[:,4].max())[0][0]
    this = np.array([lbcs[i,0],lbcs[i,9],lbcs[i,10],lbcs[i,5],lbcs[i,6],\
                     lbcs[i,7],fthis[:,2].sum(),fthis[:,3].sum(),\
                     fthis[imaxthis,4],fthis[imaxthis,5],fthis[imaxthis,6]])
    try:
        lf = np.vstack((lf,this))
    except:
        lf = np.copy(this)

#
#  array lnotf of sources in LBCS but not First
#   ********* lnotf array columns: ***************
#   0=LBCS name 1-2=LBCS pos 3-5=LBCS PSX, qual, FT parameters
#
lnotf = []
for i in np.delete(np.arange(len(lbcs)),lcorr):
    this = np.array([lbcs[i,0],lbcs[i,9],lbcs[i,10],lbcs[i,5],lbcs[i,6],lbcs[i,7]])
    try:
        lnotf = np.vstack((lnotf,this))
    except:
        lnotf = np.copy(this)

plot_aitoff(np.asarray(lnotf[:,1:3],dtype='float'),outfile='temp.png')
plt.clf()
# checked - all these are either non-point sources or outside FIRST region
#   however, if the correlation is 0.01 not 0.05 degrees there are a few
#   which do not have FIRST counterparts.
#  -> all LBCS sources are in FIRST somewhere
#
#  Does not signify very much - 80% of random positions are within 0.05 degrees
#    of a first source
#
#  LBCS: 25000 sources, 3% of random positions within 0.05 degrees of LBCS
# correlated sources (0.05deg = 180 arcsec)
#
#
#  but how big is the ST001 field of view? Effectively 1km->0.1deg?
#     Experiment: find two bright, close LBCS sources and compare ST001 and ST002 s:n
#
#   first thing we need to know: what is the selection effect of the first
#   sources that are in lbcs? I.e. what portion of the flux: spectral index
#   plane.
#
#   Plot flux histograms of FIRST together with LBCS/FIRST sources
#
plt.hist(first[:,3],bins=np.logspace(np.log10(0.1),np.log10(10000.0)),label='First',alpha=0.5)
plt.hist(np.array(list(np.asarray(lf[:,7],dtype='float'))*10),bins=np.logspace(np.log10(0.1),np.log10(10000.0)),label='First/LBCS x10',alpha=0.5)
plt.gca().set_xscale('log')
plt.xlabel ('Flux/mJy'); plt.ylabel ('Number')
plt.legend()
plt.show()
#
#   Now, 2d plot with flux and spectral index. Finding spectral index for First is
#   hard - GB6 at 5GHz, VLSS etc all rather too bright to be useful. So has to be
#   Wenss - but only available at >30 and flux limit 18mJy
#
first_north = first[first[:,1]>30.0]
lf_north = lf[np.asarray(lf[:,2],dtype='float')>30.0]
#
#  correlate FIRST and WENSS, then correlate LBCS-FIRST sources with WENSS
#
corr_fw = correlate(first_north,0,1,wenss,0,1,0.01)   # multiple sources
corr_flw = correlate(np.asarray(lf_north[:,1:3],dtype='float'),0,1,wenss,0,1,0.01)
xs_fw = np.array([]); corr_fw_int = np.asarray(corr_fw[:,:2],dtype='int')
xs_flw = np.array([]); corr_flw_int = np.asarray(corr_flw[:,:2],dtype='int')
#
#  produce xs_fw for all FIRST sources. Columns: WENSS/FIRST spectral index, FIRST
#             parameters (pf,if,width,axrat,angle), WENSS position (6-7) (ra,dec)
#     xs_fw is all sources which are in Wenss and First
#
for i in corr_fw_int:
    this = np.array([0.634*np.log10(first_north[i[0],3]/wenss[i[1],4])])
    this = np.append(this,first_north[i[0],2:7])
    this = np.append(this,wenss[i[1],:2])
    try:
        xs_fw = np.vstack((xs_fw,this))
    except:
        xs_fw = np.copy(this)

#
#  produce xs_flw for all LBCS+FIRST sources. Columns: WENSS/FIRST spectral index,
#       FIRST parameters (1-5) (pf,if,width,axrat,angle)  WENSS pos (6-7) (ra,dec)
#     xs_flw is all sources which are in Wenss, LBCS and First
#     coh and cohstat arrays are measures of which are coherent and which not
#
for i in corr_flw_int:
    this = np.array([0.634*np.log10(float(lf_north[i[0],7])/wenss[i[1],4])])
    this = np.append(this,np.asarray(lf_north[i[0],6:11],dtype='f'))
    this = np.append(this,wenss[i[1],:2])
    coh1,coh2 = lf_north[i[0],3],lf_north[i[0],5]
    cohp,cohs,cohx = coh1.count('P'),coh1.count('S'),coh1.count('X')
    this_coh = np.array([coh1,coh2])
    this_cohstat = cohp/(cohp+cohs+cohx)
    this_cohstat = 0.0 if np.isinf(this_cohstat) else this_cohstat
    try:
        xs_flw = np.vstack((xs_flw,this))
        xs_flw_coh = np.vstack((xs_flw_coh,this_coh))
        xs_flw_cohstat = np.append(xs_flw_cohstat,this_cohstat)
    except:
        xs_flw = np.copy(this)
        xs_flw_coh = np.copy(this_coh)
        xs_flw_cohstat = np.copy(this_cohstat)

xs_fw_point = xs_fw[xs_fw[:,3]<2.0]
cond_point = xs_flw[:,3]<=2.0
cond_ext = xs_flw[:,3]>2.0
cond_coh = xs_flw_cohstat>0.5
cond_notcoh = xs_flw_cohstat<=0.5
plt.subplot(121)
plt.yscale('log')
plt.plot(xs_fw_point[:,0],xs_fw_point[:,2],'y,')
plt.plot(xs_flw[cond_point&cond_coh][:,0],xs_flw[cond_point&cond_coh][:,2],'gx')
plt.plot(xs_flw[cond_point&cond_notcoh][:,0],xs_flw[cond_point&cond_notcoh][:,2],'rx')
plt.xlabel('FIRST/WENSS spectral index');plt.ylabel('FIRST flux')
plt.xlim(-0.8,0.5);plt.ylim(20.0,6000.0)
#
# now do the same but only with the VLBA calibrator list ones
#  i.e. the ones where we KNOW all the flux is compact - so any not
#  coherent in LBCS are not coherent because of a detection limit
#  Hence can compare with the non-VLBA ones
#
os.system('grep + /home/njj/catalogues/vlbaCalib.txt >vlba_north')
vlba = np.loadtxt('vlba_north',dtype='str')

def tfl (array):
    return np.asarray(array,dtype='float')

vlba_ra = 15.*tfl(vlba[:,2])+0.25*tfl(vlba[:,3])+tfl(vlba[:,4])/240.
vlba_dec = tfl(vlba[:,5])+tfl(vlba[:,6])/60.+tfl(vlba[:,7])/3600.
corr_flw_vlba = correlate(xs_flw,6,7,np.column_stack((vlba_ra,vlba_dec)),0,1,0.02)
corr_flw_vlba_vidx = np.asarray(corr_flw_vlba[:,0],dtype='int')
cond_vlba = np.array([],dtype='bool')
for i in range(len(xs_flw)):
    cond_vlba = np.append(cond_vlba, i in corr_flw_vlba_vidx)
plt.subplot(122)
plt.yscale('log')
plt.plot(xs_fw_point[:,0],xs_fw_point[:,2],'y,')
plt.plot(xs_flw[cond_point&cond_vlba&cond_coh][:,0],xs_flw[cond_point&cond_vlba&cond_coh][:,2],'gx')
plt.plot(xs_flw[cond_point&cond_vlba&cond_notcoh][:,0],xs_flw[cond_point&cond_vlba&cond_notcoh][:,2],'rx')
plt.xlabel('FIRST/WENSS spectral index');plt.ylabel('FIRST flux')
plt.xlim(-0.8,0.5);plt.ylim(20.0,6000.0)
plt.show()

#plt.savefig('temp.png')
#plt.clf()

