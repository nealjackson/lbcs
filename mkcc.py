from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSCat
from Wizardry.AIPSData import AIPSUVData as WizAIPSUVData
from Wizardry.AIPSData import AIPSImage as WizAIPSImage
import re, sys, numpy as np, os, pyfits, matplotlib
from matplotlib import pyplot as plt; from pyfits import getdata
import douvcon_casa
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
INDE = 3140.892822265625    # value corresponding to aips INDE
AIPSUSER = 340
AIPS.userno = AIPSUSER
indisk = 1

# new version with CASA (because aips version does not do multiple pol)

def zapexisting(inna,incl,indisk=1):
    for i in AIPSCat()[indisk]:
        if i['name']==inna and i['klass']==incl:
            if i['type']=='MA':
                AIPSImage(inna,incl,indisk,i['seq']).clrstat()
                AIPSImage(inna,incl,indisk,i['seq']).zap()
            else:
                AIPSUVData(inna,incl,indisk,i['seq']).clrstat()
                AIPSUVData(inna,incl,indisk,i['seq']).zap()

def findexisting (inna, incl, indisk=1):
    isseq = 0
    for i in AIPSCat()[indisk]:
        if i['name']==inna and i['klass']==incl:
            isseq = max(isseq,i['seq'])
    return isseq
                
######## mkcc_casa
#   call CASA to make simulation of source:
#   cc is: xoffset/arcs, yoffset/arcs, flux/Jy, bmax/arcs, bmin/arcs, pa

def mkcc_casa (cc,antfile='lofx',freq=140,ra=180.,dec=60.,hastart=0.,\
               haend=0.05,tint=2.,chwid=0.048828125,nchan=64):
    douvcon_casa.douvcon_casa(cc,antfile=antfile,freq=freq,ra=ra,dec=dec,\
               hastart=hastart,haend=haend,tint=tint,chwid=chwid,nchan=nchan)
    fitld = AIPSTask('fitld')
    fitld.datain = './temp.fits'
    fitld.outname = 'UVSIM'
    fitld.outclass = 'FITS'
    fitld.outdisk = indisk
    fitld.go()

# --------------- mkcc_snproc ---------------------------
# mkcc_snproc routine: Takes an SN table from an arbitrary input file
# on an AIPS disk, copies it to the UVSIM.FITS file, and adjusts both
# times and antennas (including both the antenna in each solution and
# the reference antenna), then interpolates to a CL table. This works
# provided that the reference antenna is present throughout the SN.

def mkcc_snproc (inna,incl,outna,outcl,inseq=1,outseq=1,indisk=1):
    indata = WizAIPSUVData(inna,incl,indisk,inseq)
    outdata = WizAIPSUVData(outna,outcl,indisk,outseq)
    tdiff = outdata[0].time - indata[0].time
    # previous line must NOT go after the AN tables are opened
    tacop = AIPSTask('tacop') 
    tacop.indata = indata
    tacop.outdata = outdata
    tacop.inext = 'SN'
    tacop.go()
    outdata = WizAIPSUVData('UVSIM','FITS',1,1)
    outsn = outdata.table('SN',1)
    insn = indata.table('SN',1)
    inan = indata.table('AN',1)
    outan = outdata.table('AN',1)
    inanname, outanname, inanno, outanno = [],[],[],[]
    for i in inan:
        inanname.append(i['anname'])
        inanno.append(i['nosta'])
    for i in outan:
        outanname.append(i['anname'])
        outanno.append(i['nosta'])
    # make the acorr array. This is a list of 2-element arrays. The
    # first element is the station number that should replace the second
    # station number in the solution table
    for i in outanname:
        for j in inanname:
            if i[:5]==j[:5]:
                new = np.array([outanname.index(i),inanname.index(j)])
                try:
                    acorr = np.vstack((acorr,new))
                except:
                    acorr = np.copy(new)
    acorr+=1     # AIPS antenna tables start at 1
    print 'Antenna correspondence table found:'
    print acorr
    # Now fiddle the times in the SN table, and fiddle both the antenna
    # and reference antenna numbers using the acorr array
    icou=0
    for i in outsn:
        i['time'] += tdiff
        i.update()
        try:
            iarr = np.argwhere(acorr[:,1]==i['antenna_no'])[0][0]
            i['antenna_no'] = acorr[iarr,0]
            i.update()
        except:
            pass
        try:
            iarr = np.argwhere(acorr[:,1]==i['refant_1'])[0][0]
            i['refant_1'] = acorr[iarr,0]
            i.update()
            iarr = np.argwhere(acorr[:,1]==i['refant_2'])[0][0]
            i['refant_2'] = acorr[iarr,0]
            i.update()
        except:
            pass
        icou+=1
    # Finally, interpolate to a CL table. This is not present but
    # will be created - ignore error messages about GEODELAY etc
    clcal = AIPSTask('clcal')
    clcal.indata = AIPSUVData(outna,outcl,indisk,outseq)
    clcal.go()

# ------------------- mkcc_apply ----------------
# Gets the simulation file, with a CL table; performs
# a SPLIT of the first source in the file, creates a multi-
# source file and writes it out to disk. This is needed
# because the CL table left by mkcc_snproc will cause crashes
# later (it contains antennas not in the data)
def mkcc_apply (inna,incl,outfile,noise=0.0,indisk=1,inseq=1):
    d = WizAIPSUVData(inna,incl,indisk,inseq)
    s = d.table('SU',1)[0]['source'].strip()
    zapexisting (s,'SPLIT',indisk)
    split = AIPSTask('split')
    split.indata = AIPSUVData(inna,incl,indisk,inseq)
    split.docalib = 1
    split.go()
    zapexisting (s,'UVMOD',indisk)
    uvmod = AIPSTask('uvmod')
    uvmod.indata = AIPSUVData(s,'SPLIT',indisk,inseq)
    uvmod.flux = noise
    uvmod.factor = 1.0
    uvmod.go ()
    zapexisting (s,'MULTI',indisk)
    multi = AIPSTask('multi')
    multi.indata = AIPSUVData(s,'UVMOD',indisk,inseq)
    multi.outdata = AIPSUVData(s,'MULTI',indisk,inseq)
    multi.go()
    os.system ('rm '+outfile)
    fittp = AIPSTask('fittp')
    fittp.indata = AIPSUVData(s,'MULTI',indisk,1)
    fittp.dataout = './'+outfile
    fittp.go()
    
# ---------------  mkcc_do ----------------
# Make a single fits file of a simulation with a given set of CCs
# The antfile must be a file of 4 letters or less, and ST001 should
#       be the LAST telescope in it
# A SN table will be read from a file in the AIPS directory called
#       SNTAB.UVDATA.1 (and will crash if not available)
def mkcc_do(cc,outfile='sim.fits',antfile='lofx'):
    zapexisting ('UVSIM','FITS',indisk)
    mkcc_casa (cc)
    os.system('rm -f temp.fits; rm -fr temp; rm -f casa-*.log; rm -f ipy*.log')
    mkcc_snproc ('SNTAB','UVDATA','UVSIM','FITS')
    mkcc_apply ('UVSIM','FITS',outfile,noise=0.2)

#  here runs a single simulation
cc=np.array([[0.,0.,1.,2.,1.,45.],[0.,1.,0.5,2.,1.,0.]])
mkcc_do(cc,outfile='L617sim.fits')
