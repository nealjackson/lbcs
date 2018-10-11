import numpy as np,glob,os,sys
tels = ['DE601','DE602','DE603','DE604','DE605','FR606','SE607','UK608','DE609']
ntels = len(tels)
indir = '../picfiles/'
lbcs = np.loadtxt('lobos_stats.sum',dtype='S')
pics = np.sort(glob.glob(indir+'*.pic'))
d = np.nan*np.ones((len(pics),len(tels),2))
s = np.nan*np.ones((len(pics),len(tels),2))
t = np.array([])
for i in range(len(pics)):
    a = np.load(pics[i])
    lineno = np.argwhere(lbcs[:,0]==a['obsnum'])[0][0]
    quality = lbcs[lineno][5]
    t = np.append(t,lbcs[lineno][3]+':'+lbcs[lineno][4])
    for j in range(len(tels)):
        if quality[j] != 'P':
            continue
        try:
            idx = a['antennas'].index(tels[j])
        except:
            continue
        d[i,j] = a['delay_4'][idx]
        s[i,j] = a['snr_4'][idx]
        

for iline in np.unique(t):
    exd = np.take(d, np.ravel(np.argwhere(t==iline)),axis=0)
    exs = np.take(s, np.ravel(np.argwhere(t==iline)),axis=0)
    np.putmask(exd,exs<400.0,np.nan)
    for j in range(2):
        for i in range(ntels):
            imean = np.nanmean (exd[:,i,j])
            istd = np.nanstd (exd[:,i,j])
            if np.isnan(imean) or np.isnan(istd) or np.isnan(exd[:,i,j]).sum()>20:
                sys.stdout.write ('nan:nan ')
            else:
                swrite = str(int(imean))+':'+str(int(istd))
                sys.stdout.write ('%7s '%swrite)
        sys.stdout.write ('\n')
    
