import numpy as np,os,sys,astropy
import matplotlib; from matplotlib import pyplot as plt
from lbcs_utils import *
from corrcat import *

#### Experiments with the LBCS catalogue and comparison with First and Wenss
#### Purpose: find out how representative the LBCS sources are
#
# FIRST catalogue: columns are ra dec fpeak fint size pa angle
#
def tfl (array):
    return np.asarray(array,dtype='float')

def get_quasars (xs_flw):
    # do things with quasars vs non quasars
    qlink = 'https://www.dropbox.com/s/lf517o6q56t0yum/milliquas.npy?dl=0'
    try:
        quas = np.load('milliquas.npy')
    except:
        os.system('wget '+qlink)
        quas = np.load('milliquas.npy')
    
    qcoord=np.asarray(quas[:,:2],dtype='float')
    flwcoord=np.asarray(xs_flw[:,8:10],dtype='float')
    corrq = correlate(flwcoord,0,1,qcoord,0,1,0.01)
    cond_quas = np.array([],dtype='bool')
    quas_idx = np.asarray(corrq[:,:2],dtype='int')
    quas_type = np.array([],dtype='str')
    for i in range(len(xs_flw)):
        if i in quas_idx[:,0]:
            this_idx = np.argwhere(quas_idx[:,0]==i)[0][0]
            quas_type = np.append(quas_type,quas[quas_idx[this_idx,1],2])
        else:
            quas_type = np.append(quas_type,'G')
        cond_quas = np.append(cond_quas,'q' in quas_type[i] or 'Q' in quas_type[i])
    return quas_type, cond_quas

def makeflw(): # make a data file with all of the LBCS etc information
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
    wenss = wenss[wenss[:,2]!=2.0]   # exclude components of multiple sources
    #
    #   WENSS catalogue: columns are: ra dec type fpeak fint
    #
    c_first, c_lbcs = tfl(first[:,:2]),tfl(lbcs[:,-2:])
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

    plot_aitoff(tfl(lnotf[:,1:3]),outfile='temp.png')
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
    plt.hist(np.array(list(tfl(lf[:,7]))*10),bins=np.logspace(np.log10(0.1),np.log10(10000.0)),\
             label='First/LBCS x10',alpha=0.5)
    plt.gca().set_xscale('log')
    plt.xlabel ('Flux/mJy'); plt.ylabel ('Number')
    plt.legend()
    plt.savefig('lbcs_first_hist.png')
    plt.clf()
    #
    #   Now, 2d plot with flux and spectral index. Finding spectral index for First is
    #   hard - GB6 at 5GHz, VLSS etc all rather too bright to be useful. So has to be
    #   Wenss - but only available at >30 and flux limit 18mJy
    #
    first_north = first[first[:,1]>30.0]
    lf_north = lf[tfl(lf[:,2])>30.0]
    #
    #  correlate FIRST and WENSS, then correlate LBCS-FIRST sources with WENSS
    #
    corr_fw = correlate(first_north,0,1,wenss,0,1,0.01)   # multiple sources
    corr_flw = correlate(tfl(lf_north[:,1:3]),0,1,wenss,0,1,0.01)
    xs_fw = np.array([]); corr_fw_int = np.asarray(corr_fw[:,:2],dtype='int')
    xs_flw = np.array([]); corr_flw_int = np.asarray(corr_flw[:,:2],dtype='int')
    #
    #  produce xs_fw for all FIRST sources. Columns: First row, Wenss row,
    #      WENSS/FIRST spectral index, FIRST
    #      parameters (pf,if,width,axrat,angle), WENSS position (6-7) (ra,dec)
    #     xs_fw is all sources which are in Wenss and First
    #################################################
    #   XS_FW COLS
    #   0    1    2    3    4    5    6    7    8    9   
    #   LIDX WIDX SPIX FPF  FIF  FWID FXR  FPA  WRA  WDEC 
    #################################################
    for i in corr_fw_int:
#        f=open('junk','a')
#        f.write('%d %s %s\n'%(i[0],str(first_north[i[0]]),str(wenss[i[1]])))
#        f.close()
        this = np.array([float(i[0]),float(i[1]),\
                         1.577*np.log10(first_north[i[0],3]/wenss[i[1],4])])
        this = np.append(this,first_north[i[0],2:7])
        this = np.append(this,wenss[i[1],:2])
        try:
            xs_fw = np.vstack((xs_fw,this))
        except:
            xs_fw = np.copy(this)

    f = open('lbcs_fw.dat','w')
    n = f.write('# LIDX   WIDX   SPIX   FPF    FIF    FWID  FXR   FPA     WRA      WDEC\n')
    for i in range(len(xs_fw)):
        n=f.write('%6.0f %6.0f %5.2f %6.1f %6.1f %7.2f %4.1f %5.1f %10.6f %10.6f\n'%\
                (xs_fw[i,0],xs_fw[i,1],xs_fw[i,2],xs_fw[i,3],xs_fw[i,4],\
                 xs_fw[i,5],xs_fw[i,6],xs_fw[i,7],xs_fw[i,8],xs_fw[i,9]))
    f.close()
    #
    #  produce xs_flw for all LBCS+FIRST sources. Columns: lf_north row no,
    #    WENSS row no, WENSS/FIRST spectral index,
    #    FIRST parameters (1-5) (pf,if,width,axrat,angle),
    #    WENSS pos (6-7) (ra,dec)
    #     xs_flw is all sources which are in Wenss, LBCS and First
    #     coh and cohstat arrays are measures of which are coherent and which not
    #
    for i in corr_flw_int:
        this = np.array([float(i[0]),float(i[1]),\
                         1.577*np.log10(float(lf_north[i[0],7])/wenss[i[1],4])])
        this = np.append(this,np.asarray(lf_north[i[0],6:11],dtype='f'))
        this = np.append(this,wenss[i[1],:2])
        this = np.append(this,lf_north[i[0],3])
        this = np.append(this,lf_north[i[0],5])
        try:
            xs_flw = np.vstack((xs_flw,this))
        except:
            xs_flw = np.copy(this)
    # the ones where we KNOW all the flux is compact - so any not
    #  coherent in LBCS are not coherent because of a detection limit
    #  Hence can compare with the non-VLBA ones
    #
    os.system('grep + /home/njj/catalogues/vlbaCalib.txt >vlba_north')
    vlba = np.loadtxt('vlba_north',dtype='str')
    vlba_ra = 15.*tfl(vlba[:,2])+0.25*tfl(vlba[:,3])+tfl(vlba[:,4])/240.
    vlba_dec = tfl(vlba[:,5])+tfl(vlba[:,6])/60.+tfl(vlba[:,7])/3600.
    corr_flw_vlba = correlate(tfl(xs_flw[:,8:10]),0,1,\
                            np.column_stack((vlba_ra,vlba_dec)),0,1,0.02)
    corr_flw_vlba_vidx = np.asarray(corr_flw_vlba[:,0],dtype='int')
    cond_vlba = np.array([],dtype='bool')
    quas_type, cond_quas = get_quasars (xs_flw)
    for i in range(len(xs_flw)):
        cond_vlba = np.append(cond_vlba, i in corr_flw_vlba_vidx)
    f = open('lbcs_flw.dat','w')
    n = f.write('# LIDX   WIDX   SPIX   FPF    FIF    FWID  FXR   FPA     WRA      WDEC  VLBA   QUAS  COH1     COH2\n')
    xz = tfl(xs_flw[:,:10])
    for i in range(len(xs_flw)):
        n=f.write('%6.0f %6.0f %5.2f %6.1f %6.1f %7.2f %4.1f %5.1f %10.6f %10.6f %d %4s %s %s\n'%\
                (xz[i,0],xz[i,1],xz[i,2],xz[i,3],xz[i,4],\
                 xz[i,5],xz[i,6],xz[i,7],xz[i,8],xz[i,9],\
                 cond_vlba[i],quas_type[i],xs_flw[i,10],xs_flw[i,11]))

    f.close()

def tocohstat(a):
    stat = []
    for i in range (len(a)):
        cohp,cohs,cohx = a[i].count('P'),a[i].count('S'),a[i].count('X')
        cohall = cohp+cohs+cohx
        stat.append(float(cohp)/float(cohall) if cohall else 0.0)
    return tfl(stat)        
    
def loadflw():
    a = np.loadtxt('lbcs_flw.dat',dtype='str')
    xs_flw = tfl(a[:,:11])
    xs_flw_coh = np.asarray(a[:,12:14],dtype='str')
    xs_flw_cohstat = tocohstat(xs_flw_coh[:,0])
    xs_flw_quas = a[:,11]
    cond_vlba = np.asarray(a[:,10],dtype='bool')
    cond_point = xs_flw[:,5]<=2.0
    cond_ext = xs_flw[:,5]>2.0
    cond_coh = xs_flw_cohstat>0.5
    cond_notcoh = xs_flw_cohstat<=0.5
    a = np.loadtxt('lbcs_fw.dat',dtype='float')
    xs_fw_point = a[a[:,5]<2.0]
    return xs_flw,xs_flw_coh,xs_flw_cohstat,cond_vlba,cond_point,cond_ext,cond_coh,\
        cond_notcoh,xs_fw_point,xs_flw_quas
    
makeflw()
xs_flw,xs_flw_coh,xs_flw_cohstat,cond_vlba,cond_point,cond_ext,cond_coh,cond_notcoh,\
    xs_fw_point,xs_flw_quas = loadflw()

def plot_spix_flux(panel,arrbkg,arr,xcol,ycol,cond1,cond2,xlab='FIRST/WENSS spectral index',\
                   ylab='FIRST flux/mJy',spmult=0.0):
    plt.subplot(panel)
    plt.yscale('log')
    plt.plot(arrbkg[:,xcol],arrbkg[:,ycol],'y,')
    yarr = arr[cond1][:,ycol]*spmult**arr[cond1][:,xcol] if spmult!=0.0 else arr[cond1][:,ycol]
    plt.plot(arr[cond1][:,xcol],yarr,'gx')
    yarr = arr[cond2][:,ycol]*spmult**arr[cond2][:,xcol] if spmult!=0.0 else arr[cond2][:,ycol]
    plt.plot(arr[cond2][:,xcol],yarr,'rx')
    plt.xlabel(xlab);plt.ylabel(ylab)
    plt.xlim(-1.3,0.7);plt.ylim(20.0,6000.0)
    
# plot everything, non-VLBA and VLBA
plot_spix_flux(121,xs_fw_point,xs_flw,2,4,cond_point&cond_coh,cond_point&cond_notcoh)
plot_spix_flux(122,xs_fw_point,xs_flw,2,4,cond_point&cond_vlba&cond_coh,cond_point&cond_vlba&cond_notcoh)
plt.savefig('lbcs_spix_fluxL.png');plt.clf()
# Now let's try and plot the extrapolated 150MHz flux against spectral
#  index instead: 150MHz flux = FIRST flux * 0.1071**spectral index
plot_spix_flux(121,xs_fw_point,xs_flw,2,4,cond_point&cond_coh,cond_point&cond_notcoh,spmult=0.1071)
plot_spix_flux(122,xs_fw_point,xs_flw,2,4,cond_point&cond_vlba&cond_coh,cond_point&cond_vlba&cond_notcoh,\
               spmult=0.1071,ylab='Implied LBCS flux')
plt.savefig('lbcs_spix_flux150.png');plt.clf()
#
# So it looks like there is a cut about 200mJy for the VLBA sources. So why are there
# some non-VLBA sources detected in LBCS that are apparently very faint?
# let's print them
#
lbcs_point_coh = xs_flw[cond_point&cond_coh]
cond_coh_faint = (lbcs_point_coh[:,4]*0.1071**lbcs_point_coh[:,2]<100.0)
lbcs_point_coh_faint = lbcs_point_coh[cond_coh_faint]
xs_flw_coh_faint = xs_flw_coh[cond_point&cond_coh][cond_coh_faint]
#
#  this faintest one is wenss but with bright point source nearby, so artificially
#colours = ['b','y','g','g','b','y','y','y','b','y','r','r','r']
#
#anames = ['DE601','DE602','DE603','DE604','DE605','FR606','SE607','UK608',\
#          'DE609','PL610','PL611','PL612','IE613']

def getai(xs_flw,xs_flw_coh,cond):   
    alen = [0,2,1,1,0,2,2,2,0,2,3,3,3]
    ai = [np.array([]),np.array([]),np.array([]),np.array([])]
    fai = [np.array([]),np.array([]),np.array([]),np.array([])]
    for i in range(len(xs_flw)):
        for j in range(13):
            thiscoh = xs_flw_coh[i,1][j]
            if thiscoh=='-' or not cond[i]:
                continue
            ai[alen[j]] = np.append(ai[alen[j]],int(thiscoh))
            fai[alen[j]] = np.append(fai[alen[j]],xs_flw[i,4]*0.1071**xs_flw[i,2])
    return ai, fai

def plot_sncoh_f151 (xs_flw,xs_flw_coh,cond,title,outfile):
    aicol = ['b','g','y','r']
    ailab = ['Ef/No/Ju','Ta/Po','Un/Na/On/Ch/Bo','La/Bo/Bi']
    plt.rcParams['font.size'] = 12
    ai,fai = getai(xs_flw,xs_flw_coh,cond)
    for i in range(4):
        yai = np.zeros(10)
        for j in range(10):
            yai[j] = np.median(fai[i][ai[i]==j])
        plt.plot(np.arange(10),yai,color=aicol[i],label=ailab[i])
    plt.ylim(0.0,700.0)
    plt.grid()
    plt.legend()
    plt.xlabel('Coherence number');plt.ylabel('150-MHz extrapolated flux density/mJy')
    plt.title(title)
    plt.savefig(outfile)
    plt.clf()

plot_sncoh_f151 (xs_flw,xs_flw_coh,np.logical_and(cond_point,cond_vlba),'\
      Coherence for LBCS+VLBA sources','lbcs_pointV_coh.png')
plot_sncoh_f151 (xs_flw,xs_flw_coh,cond_point,'Coherence for LBCS sources',\
                 'lbcs_point_coh.png')


cond_quas = []
for i in xs_flw_quas:
    cond_quas.append ('q' in i or 'Q' in i)
plt.hist(xs_flw[:,5],range=[0,30],bins=60,label='all objects',alpha=0.5)
plt.hist(xs_flw[:,5][cond_quas],range=[0,30],bins=60,label='quasars')
plt.legend()
plt.title('LBCS-First-Wenss sources')
plt.xlabel('FIRST major axis size/arcsec')
plt.ylabel('Number')
plt.savefig('lbcs_quas.png')
