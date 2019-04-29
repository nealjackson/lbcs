import numpy as np, astropy,os,sys
from astropy import coordinates
from astropy.coordinates import SkyCoord
casapy = '/pkg/casa-stable-5.0.101/bin/casa'

def douvcon_casa (cc,antfile='lofx',freq=140.,ra=180.,dec=60.,hastart=0.,\
                  haend=0.05,tint=2.,chwid=0.048828125,nchan=64):
    cosd = np.cos(np.deg2rad(dec))
    s = SkyCoord(ra,dec,unit='degree').to_string(style='hmsdms')
    f=open('douvcon_casa','w')
    f.write('direction = "J2000 %s"\n' % s)
    f.write('cl.done()\n')
    f.write('cl.addcomponent(dir="J2000 %s", flux=0.0001, fluxunit="Jy",'% s)
    f.write('  freq="%fMHz", shape="Gaussian", majoraxis="0.1arcmin",'%freq)
    f.write('  minoraxis="0.05arcmin", positionangle="45.0deg")\n')
    f.write('ia.fromshape("Gaussian.im",[256,256,1,%d],overwrite=True)\n'%nchan)
    f.write('cs=ia.coordsys()\n')
    f.write('cs.setunits(["rad","rad","","Hz"])\n')
    f.write('cell_rad=qa.convert(qa.quantity("0.1arcsec"),"rad")["value"]\n')
    f.write('cs.setincrement([-cell_rad,cell_rad],"direction")\n')
    f.write('cs.setreferencevalue([qa.convert("%fdeg","rad")["value"],qa.convert("%fdeg","rad")["value"]],type="direction")\n'%(ra,dec))
    f.write('cs.setreferencevalue("%fMHz","spectral")\n'%freq)
    f.write('cs.setincrement("%fMHz","spectral")\n'%chwid)
    f.write('ia.setcoordsys(cs.torecord())\n')
    f.write('ia.setbrightnessunit("Jy/pixel")\n')
    f.write('ia.modify(cl.torecord(),subtract=False)\n')
    f.write('exportfits(imagename="Gaussian.im",fitsimage="Gaussian.fits",overwrite=True)\n')
    for c in cc:
        cra,cdec = ra-c[0]/(3600.*cosd), dec+c[1]/3600.
        sc = SkyCoord(cra,cdec,unit='degree').to_string(style='hmsdms')
        sstr = 'cl.addcomponent(dir="J2000 %s", ' % sc
        sstr += 'flux=%f, fluxunit="Jy",freq="%fMHz", '%(c[2],freq)
        if len(c)==3:
            sstr += 'shape="point")'
        else:
            sstr += 'shape="Gaussian", majoraxis="%.farcsec", '%c[3]
            sstr += 'minoraxis="%.farcsec",positionangle="%.fdeg")'%(c[4],c[5])
        f.write('%s\n'%sstr)
    f.write('os.system("rm -fr douvcon_casa.cl")\n')
    f.write('cl.rename("douvcon_casa.cl")\n')
    f.write('cl.done()\n')
    f.write('default("simobserve")\n')
    f.write('project = "temp"\n')
    f.write('skymodel = "Gaussian.fits"\n')
    f.write('inwidth = "%fMHz"\n'%chwid)
    f.write('complist = "douvcon_casa.cl"\n')
    f.write('compwidth = "%fMHz"\n'%(chwid*float(nchan)))
    f.write('direction = "J2000 %s"\n' % s)
    f.write('obsmode = "int"\n')
    f.write('integration = "%ds"\n' % int(tint))
    f.write('antennalist = "%s"\n'%antfile)
    f.write('totaltime = "%fs"\n'%((haend-hastart)*3600.))
    f.write('mapsize = "10arcsec"\n')
    f.write('thermalnoise = ""\n')
#    f.write('t_sky = 1000\n')
    f.write('graphics = "none"\n')
    f.write('simobserve()\n')
    f.write('os.system("rm temp.fits")\n')
    f.write('exportuvfits("temp/temp.lofx.ms","temp.fits")\n')
    f.close()
    os.system(casapy + ' --nologger -c douvcon_casa')
    os.system('rm Gaussian.fits;rm -fr Gaussian.im')
    os.system('rm -fr douvcon_casa.cl')
