#   files you will need:
#   correlate.py; fringemap_v2b.py; wenss2000.npy; lofipi_aips.py
from math import *
from fringemap_v2b import *
from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList, AIPSMessageLog
from AIPSData import AIPSUVData, AIPSImage, AIPSCat
from Wizardry.AIPSData import AIPSUVData as WizAIPSUVData
from scipy import ndimage; from scipy.ndimage import measurements
import matplotlib; from matplotlib import pyplot as plt
#import astLib; from astLib import astCoords
import pyfits; from pyfits import getdata,getheader
import re,sys,pickle,numpy as np,os,glob,time,warnings; from numpy import fft
from lofipi_aips import *
try:
    from correlate import *
except:
    print 'Not importing correlate function, writing fringemaps will fail'

plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.origin']='lower'
warnings.simplefilter('ignore')
INDE,twopi = 3140.892822265625, 2.0*np.pi
# ---------------------------------------------------------------
#   important parameters
#   only fiddle if you know what you are doing
wenss_max = 300              # maximum number of WENSS sources to plot
CLIP = 15.0                  # clipping of bad data for FFTSN statistic
FLOG = 'lofipi_lobos.log'    # logfile name
WEIGHT  = 1                  # weightit parameter in FRING - 1 is best
SOLINT1 = 4                  # FRING SOLINT for first pass (all scan)
SOLINT2 = 0.1                # FRING SOLINT for main pass
SNR1    = 6                  # FRING SOLINT for first pass (all scan)
SNR2    = 2                  # FRING SOLINT for main pass
SUPERTERP = ['ST001','TS001']# possible names for phased superterp
INDISK  = 1                  # aips disk
DELAYWIN = 600               # FRING delay search window
RATEWIN = 5                  # FRING rate search window
# --------------------------------------------------------------------

def dqual(d,clip=False):
    val = ''
    if d.ndim!=3 or d.shape[0]!=2:
        return 'M',0.0 if clip else 'M'
    if d.shape[1]<10 or d.shape[2]<30:
        return 'X',0.0 if clip else 'X'
    a = abs(np.ravel(d))
    aok = a[a!=0.0]
    if len(aok)<0.5*len(a):
        return 'Z',float(len(aok))/float(len(a)) if clip else 'Z'
    if aok.max()>CLIP*np.median(aok):
        return 'A', np.median(aok) if clip else 'A'
    return 'O',0.0 if clip else 'O'

def dget (uvdata,baseline,utstart=-1.0E9,utstop=1.0E9,startvis=0):
    if baseline.ndim==1:
        baseline = np.array([baseline])
    for i in range(len(baseline)):
        baseline[i] = np.sort(baseline[i])
    d = [np.array([])]*len(baseline)
    u = [np.array([])]*len(baseline)
    times = [np.array([])]*len(baseline)
    h = uvdata.header
    npol = min(h['naxis'][h['ctype'].index('STOKES')],2)  # only ll,rr
    nc = h['naxis'][h['ctype'].index('FREQ')]*h['naxis'][h['ctype'].index('IF')]
    for visibility in uvdata:  # shape is (nif,nc,npol,AphiW)
        if visibility.time<utstart:
            continue
        if visibility.time>utstop:   # assumes TB sorted
            break
        idx=-1      # ugly test, change for something more pythonic
        for j in range(len(baseline)):
            if all(baseline[j]==visibility.baseline):
                idx=j
        if idx==-1:
            continue
        if visibility.baseline==[61,70]:
            fs=open('templog','a')
            fs.write('%.2f %.2f %.2f\n'%( visibility.uvw[0],visibility.uvw[1],visibility.uvw[2]))
            fs.close()
        v = visibility.visibility
        dcol = np.zeros((nc,npol),dtype='complex')
        for spw in range(v.shape[0]):
            for i in range(v.shape[1]):
                fq = spw*v.shape[1]+i
                for j in range(npol):
                    dcol[fq,j] = complex(v[spw,i,j,0],v[spw,i,j,1])
        try:
            d[idx] = np.dstack((d[idx],dcol))
            u[idx] = np.column_stack((u[idx],visibility.uvw))
        except:
            d[idx] = np.copy(dcol)
            u[idx] = np.copy(visibility.uvw)
        times[idx] = np.append(times[idx],visibility.time)
    for i in range(len(d)):
        try:
            d[i]=np.rollaxis(d[i],1,0)
        except:
            pass
    uvw = np.zeros((len(baseline),3))
    try: 
        for i in range (len(baseline)):
            uvw[i] = np.array([np.mean(u[i][0]),np.mean(u[i][1]),np.mean(u[i][2])])
    except:
        pass
    return d,uvw,times

def getsn (uvdata, inver, pfring_antennas, nif, refant):
    sn = uvdata.table('SN',inver)     # plot using the user-defined SN table
    times = []
    for i in sn:
        times.append(i['time'])
    times = np.unique(times)
    sd = np.zeros((len(pfring_antennas),4,nif,len(times),2))
    for i in sn:
        indx_t = np.argwhere(times==i['time'])[0][0]
        indx_a = np.argwhere(pfring_antennas==i['antenna_no'])[0][0]
        if nif==1:
            sd[indx_a,0,0,indx_t,0] = i['delay_1']
            sd[indx_a,0,0,indx_t,1] = i['delay_2']
            sd[indx_a,1,0,indx_t,0] = np.arctan2(i['imag1'],i['real1'])
            if i['real1']==INDE:
                sd[indx_a,1,0,indx_t,0] = np.nan
            sd[indx_a,1,0,indx_t,1] = np.arctan2(i['imag2'],i['real2'])
            if i['real2']==INDE:
                sd[indx_a,1,0,indx_t,1] = np.nan
            sd[indx_a,2,0,indx_t,0] = i['rate_1']
            sd[indx_a,2,0,indx_t,1] = i['rate_2']
            sd[indx_a,3,0,indx_t,0] = i['weight_1']
            sd[indx_a,3,0,indx_t,1] = i['weight_2']
        else:
            for j in range(nif):
                sd[indx_a,0,j,indx_t,0] = i['delay_1'][j]
                sd[indx_a,0,j,indx_t,1] = i['delay_2'][j]
                sd[indx_a,1,j,indx_t,0] = np.arctan2(i['imag1'][j],i['real1'][j])
                sd[indx_a,1,j,indx_t,1] = np.arctan2(i['imag2'][j],i['real2'][j])
                sd[indx_a,2,j,indx_t,0] = i['rate_1'][j]
                sd[indx_a,2,j,indx_t,1] = i['rate_2'][j]
                sd[indx_a,3,j,indx_t,0] = i['weight_1'][j]
                sd[indx_a,3,j,indx_t,1] = i['weight_2'][j]
    np.putmask(sd,sd==INDE,np.nan)
    sd[:,0,:,:,:]*=1.E9
    return sd, times

def stats (aipsname,pfring_antennas,pgood,refant,obs,sd,sd1):
    refwhere = np.argwhere(pfring_antennas==refant)[0]
    sd = np.delete (sd,refwhere,0)     # delete refant from arrays
    sd1 = np.delete (sd1,refwhere,0)
    fring_antennas = list(pfring_antennas)
    fring_antennas.remove(refant)
    nant = len(fring_antennas)
    os.system('grep FITLD '+logdir+aipsname+'.log |grep " RA ">tempRA')
    os.system('grep FITLD '+logdir+aipsname+'.log |grep " DEC " >tempDEC')
    f=open('tempRA')
    l=f.readline().split()
    ra=l[3]+':'+l[4]+':'+l[5]
    coord='J'+l[3]+l[4]+l[5].split('.')[0]
    f.close()
    f=open('tempDEC')
    l=f.readline().split()
    dec=l[3]+':'+l[4]+':'+l[5]
    coord+=('+'+l[3]+l[4]+l[5].split('.')[0])
    f.close()
    obs['point_RA'],obs['point_dec']=ra,dec
    flog = open(FLOG,'a')
    flog.write ('%s %s %s: ant '%(coord,ra,dec))
    for i in fring_antennas:
        flog.write('%s '%str(i))
    flog.write('\n')
    flog.close()
    print coord,ra,dec,fring_antennas
    obs['delay_4'], obs['phase_4'], obs['rate_4'], obs['snr_4'] = \
        sd1[:,0,0,0,:], sd1[:,1,0,0,:], sd1[:,2,0,0,:], sd1[:,3,0,0,:]
    obs['delay'], obs['phase'], obs['rate'], obs['snr'] = \
        sd[:,0,0,:,:], sd[:,1,0,:,:], sd[:,2,0,:,:], sd[:,3,0,:,:]
    obs['goodfrac'] = int(pgood)
    os.system('rm tempRA');os.system('rm tempDEC')
    return ra,dec

def get_fftsn (dfft):
    # Sum of brightest 6 pixels in the central 10x10, divided by rms
    dy,dx = dfft.shape[0],dfft.shape[1]
    dfftmax = dfft[max(0,dy/2-10):min(dy-1,dy/2+10),\
                   max(0,dx/2-10):min(dx-1,dx/2+10)]
    dfftmax_ravel = np.sort(np.ravel(dfftmax))
    dfftmax_p = (dfftmax_ravel[-6:]).sum()
    dfft_ravel = np.sort(np.ravel(dfft))
    dfft_ravel = dfft_ravel[:int(0.99*len(dfft_ravel))]
    dsn = dfftmax_p/np.std(dfft_ravel)
    return dsn

def panelplot(sd,aipsname,intl_antennas,intl_name,times,refant,ra,dec):
    ny,nx = len(intl_antennas),6
    fftsn = np.zeros((ny,2))
    pltsquash = 0.5*float(sd.shape[3])/float(sd.shape[2])
    for pol in [0,1]:
        for i in range(ny):
            wdata = WizAIPSUVData (aipsname,'FITS', 1, 1)
            d,uvw,tjunk=dget(wdata,np.array([intl_antennas[i],refant]))
            plt.subplot(nx,ny,i+1,xticks=[],yticks=[])
            if d!=[]:
                dq,val = dqual(d[0],clip=True)
                if dq=='A':
                    np.putmask(d[0][pol],abs(d[0][pol])>CLIP*val,val+0.0j)
                plt.imshow(np.arctan2(d[0][pol].imag,d[0][pol].real))
#                plt.imshow(abs(d[0][pol]))
            if i==2:
                plt.title(aipsname+' '+ra+' '+dec)
            plt.subplot(nx,ny,i+1+ny,xticks=[],yticks=[])
            if d!=[]:
                dfft = abs(fft.fftshift(fft.fft2(d[0][pol])))
                fftsn[i,pol] = get_fftsn(dfft)
                plt.imshow(np.sqrt(dfft),cmap=matplotlib.cm.gray_r)
            plt.subplot(nx,ny,i+1+2*ny,xticks=[],yticks=[])
            plt.plot(times,sd[i,0,0,:,pol],'b+')
            meddelay = np.nanmedian(sd[i,0,0,:,pol])
            meddelay = meddelay if meddelay < 3.e6 else 0
            if meddelay:
                plt.text(times[0],meddelay-50,str(int(meddelay)))
            plt.ylim(meddelay-50.,meddelay+50)
            plt.subplot(nx,ny,i+1+3*ny,xticks=[],yticks=[])
            phx = times[~np.isnan(sd[i,1,0,:,pol])]
            phy = sd[i,1,0,:,pol][~np.isnan(sd[i,1,0,:,pol])]
            medphase=np.median(np.unwrap(phy))
            plt.plot(phx,phy,'b+')
#            plt.ylim(medphase-2.0,medphase+2.0)
            wdata = WizAIPSUVData (aipsname,'SPLIT', 1, 1)
            d,uvw,tjunk=dget(wdata,np.array([intl_antennas[i],refant]))
            if d!=[]:
                dq,val = dqual(d[0],clip=True)
                if dq=='A':
                    np.putmask(d[0][pol],abs(d[0][pol])>CLIP*val,val+0.0j)
            plt.subplot(nx,ny,i+1+4*ny,xticks=[],yticks=[])
            try:
                plt.imshow(np.arctan2(d[0][pol].imag,d[0][pol].real))
            except:
                print 'No calibrated data on ',intl_antennas[i]
                flog=open(FLOG,'a')
                flog.write('No calibrated data on %s\n'%str(intl_antennas[i]))
                flog.close()
            plt.subplot(nx,ny,i+1+5*ny,xticks=[],yticks=[])
            try:
                plt.imshow(np.sqrt(abs(fft.fftshift(fft.fft2(d[0][pol])))),cmap=matplotlib.cm.gray_r)
            except:
                pass
            plt.xlabel(intl_name[i])
        plt.savefig(pngdir+aipsname+('R' if pol else 'L')+'.png',bbox_inches='tight')
        plt.clf()
    return 0.5*(fftsn[:,0]+fftsn[:,1])

def aipszap (aipsname,aipsclass,indisk=1):
    pca = AIPSCat()
    for j in pca[indisk]:
        if j['name']==aipsname and j['klass']==aipsclass:
            if j['type']=='UV':
                data = AIPSUVData(aipsname,aipsclass,indisk,j['seq'])
            else:
                data = AIPSImage(aipsname,aipsclass,indisk,j['seq'])
            data.zap()


##############  BEGINNING OF SCRIPT #####################

#  change the following things
all_antennas = ['DE601','DE602','DE603','DE604','DE605','FR606','SE607','UK608','DE609','PL610','PL611','PL612','TS001','ST001','IE613']
indisk = INDISK
delaywin,ratewin = DELAYWIN,RATEWIN

if len(sys.argv)<3:
    print 'syntax: parseltongue lofipi.py AIPS-no file-template [solint]'
    print '        where files wanted are datadir/file-template*'
    print ' (for standard pipeline output, file-template=\'PP_T\')'
    sys.exit()

AIPS.userno = int(sys.argv[1])
# input files: if contains *, use as template, else try and read file of names
if '*' in sys.argv[2]:
    if not '/' in sys.argv[2]:
        infile = np.sort(glob.glob('./'+sys.argv[2]))
    else:
        infile = np.sort(glob.glob(sys.argv[2]))
else:
    infile = np.loadtxt(sys.argv[2],dtype='S')
print infile
flog=open(FLOG,'a')
flog.write('Processing file %s\n'%infile)
flog.close()
solint = float(sys.argv[3]) if len(sys.argv)>3 else SOLINT2
os.system('mkdir ./logfiles ./pngfiles ./picfiles ./frifiles')
logdir = './logfiles/'
pngdir = './pngfiles/'
picdir = './picfiles/'
fridir = './frifiles/'

###########################################################

for fi in infile:
    aipsname = str(fi.split('/')[-1].split('.')[0])[:12]
#  special provisions:
#  delete FR606 if we are between L264200 and L266700
    fring_antennas = list(all_antennas)    # copy list
    try:
        intaipsname = int(aipsname[1:])
        if 264200<intaipsname<266700:
            fring_antennas.remove('FR606')
    except:
        pass
###########################################################
    obs = dict()
    obs['obsnum'] = aipsname
    if os.path.isfile(logdir+aipsname+'.log'):
        os.system('rm '+logdir+aipsname+'.log')
    print time.ctime(),':Processing ',fi,' with AIPS catalogue name',aipsname
    flog=open(FLOG,'a')
    flog.write('************* %s ************\n:Processing %s with AIPS cat name %s\n'%(time.ctime(),fi,aipsname))
    flog.close()
    aipszap (aipsname,'FITS')
    aipszap (aipsname,'SPLIT')
    pload (fi if './' in fi else './'+fi,aipsname,indisk,'FITS',logdir)
    uvdata = AIPSUVData (aipsname,'FITS', 1, 1)
    cl = uvdata.table('CL',1)
    su = uvdata.table('SU',1) 
    an = uvdata.table('AN',1)
    try:
        nif = len(fq[0]['if_freq'])
    except:
        nif = 1
    source = su[0]['source'].rstrip()
    pfring_antennas,pfring_name,pfring_qual,intl_antennas,intl_name,\
                    refant = [],[],[],[],[],-1
    for i in an:        # Find the reference antenna
        annew = i['anname'].strip('HBA').strip('LBA')[:5]
        if annew in SUPERTERP:
            refant = i['nosta']
            refant_name = annew
    if refant==-1:      # No reference antenna
        print 'Aborting; no reference antenna found'
        sys.exit()
    wdata = WizAIPSUVData (aipsname,'FITS', 1, 1)
    for i in an:
        annew = i['anname'].strip('HBA').strip('LBA')[:5]
        if annew in fring_antennas:
            if i['nosta'] != refant:
                d,uvw,tjunk=dget(wdata,np.array([i['nosta'],refant]))
                if not d[0].sum():     # No data on this antenna
                    print 'No data on antenna',i['nosta']
                    flog=open(FLOG,'a')
                    flog.write('No data on antenna %s\n'%str(i['nosta']))
                    flog.close()
                    continue
                intl_antennas.append(i['nosta'])
                intl_name.append(annew)
            pfring_antennas.append(i['nosta'])
            pfring_name.append(annew)
            dq,val = dqual(d[0])
            pfring_qual.append(dq)
    obs['annames'] = list(pfring_antennas)
    obs['antennas'] = list(pfring_name)
    obs['source'] = source
    obs['anqual'] = list(pfring_qual)
    print 'Found source',obs['source'],'with',obs['antennas'],nif,'IFs'
    flog=open(FLOG,'a')
    flog.write('Found source %s with '%obs['source'])
    for i in obs['antennas']:
        flog.write('%s '%i)
    flog.write ('%d IFs\n'%nif)
    flog.close()
#   SN table 1 has just one solution; SN table 2 has user-defined solint
    pfring (aipsname, refant, pfring_antennas, source, indisk, delaywin, ratewin ,solint=SOLINT1, snr=SNR1, weightit=WEIGHT, logdir=logdir)
    pfring (aipsname, refant, pfring_antennas, source, indisk, delaywin, ratewin ,solint=solint, snr=SNR2, weightit=WEIGHT, logdir=logdir)
    f = open(logdir+aipsname+'.log')
    ngood = nfail = 1
    for i in f:
        if 'good solutions' in i:
            ngood = int((i.split())[2])
        if 'Failed' in i and i[:5]=='FRING':
            nfail = int((i.split())[3])
    f.close()
    pgood = 100.*ngood/(ngood+nfail)
    pclcal (aipsname, indisk, 2, logdir=logdir)
    psplit (aipsname, source, indisk, logdir=logdir)
    pfring_antennas = np.asarray(pfring_antennas,dtype='int')
    sd1, times1 =  getsn (uvdata, 1, pfring_antennas, nif,refant)
    sd, times = getsn (uvdata, 2, pfring_antennas, nif,refant)
    obs['date_obs'] = uvdata.header['date_obs']
    obs['time_start'],obs['time_end'] = times[0],times[-1]
    baselines = np.dstack((intl_antennas,[refant]*len(intl_antennas)))[0]
    wdata = WizAIPSUVData (aipsname,'FITS', 1, 1)
    d,uvw,tjunk=dget(wdata,baselines)
    obs['uvw'] = uvw
    ra,dec = stats (aipsname,pfring_antennas,pgood,refant,obs,sd,sd1)
    obs['fftsn'] = panelplot(sd,aipsname,intl_antennas,intl_name,times,refant,ra,dec)
    flog=open(FLOG,'a')
    sys.stdout.write('FFT s:n ratios ')
    flog.write('FFT s:n ratios:')
    for i in obs['fftsn']:
        flog.write('%.1f '%i)
        sys.stdout.write('%.1f '%i)
    flog.write('\n')
    flog.close(); print '\n'
#    try:
    if True:
        fringemap(AIPS.userno,aipsname,intl_name,refant_name,\
          fsiz=256,imaxoff=2.1,dofilt=True,fridir=fridir,logdir=logdir)
        for i in intl_name:
            aipszap(aipsname+i,'IMAP')
#    except:
#        flog = open(FLOG,'a')
#        flog.write('ERROR: failed to make fringemap for %s\n'%aipsname)
#        print 'ERROR: failed to make fringemap for %s\n'%aipsname
#        flog.close()
    pickle.dump(obs,open(picdir+aipsname+'.pic','wb'))
    aipszap (aipsname,'FITS',indisk)
    aipszap (aipsname,'SPLIT',indisk)
    AIPSMessageLog().zap()
