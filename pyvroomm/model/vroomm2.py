# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 09:16:39 2025

@author: jo
"""

from astropy.io import fits
import numpy as np
from pyechelle.CCD import CCD
from pyechelle.hdfbuilder import HDFBuilder
from pyechelle.spectrograph import InteractiveZEMAX
from pathlib import Path
from pyechelle.telescope import Telescope
from pyechelle.spectrograph import ZEMAX
from pyechelle.simulator import Simulator
from pyechelle.sources import Phoenix,IdealEtalon
from os.path import basename,join,isdir,isfile
from os import mkdir,listdir
from pyvroomm.tools.cfgfile import ConfigParser
from pyechelle.efficiency import BandpassFilter
from natsort import natsorted
import h5py
class vroomm:
    #
    def __init__(self,modelname,wdir="./"):
        self.zemax_file = ""
        self.wdir = Path(wdir)
        self.hdf = join(self.wdir,modelname+".hdf")
        self.simulationdir = join(self.wdir,".simulation")
        self.modelname = modelname
        self.cfgfile = join(self.wdir,"%s.txt"%self.modelname)
        self.simulationname = ""

        
        self.minorder = 65
        self.maxorder = 155
        self.grating_idx = 5 
        self.blaze = 76 
        self.pxsize = 12 
        self.theta = 0 
        self.gamma = 1.5
        #to be configure
      
        #check if we can load the model
        if isfile(self.cfgfile):
            self._load()
        else:#we create a new wdir 
            self._mkdir()
    
    def load_simulation(self,name):
        self.simulationname = name
    def get_all_data(self):
        raise NotImplementedError()
        #fitsfile = join(self.simulationdir,"%s..all.fits"%(self.simulationname))
        
        #if not isfile(fitsfile):
        #    raise Exception("Simulation not runned yet or simulation not loaded yet. Run simulation")
        #data = fits.getdata(fitsfile)
        
        #return data
    def get_order_data(self,order,fiber=1):
        raise NotImplementedError()
        #fitsfile = join(self.simulationdir,"%s..%d.fits"%(self.simulationname,order))
        #if not isfile(fitsfile):
        #    raise Exception("Simulation not runned yet or simulation not loaded yet. Run simulation or load one ")
        #data = fits.getdata(fitsfile)
        
        #return data
    def _load(self):
        if all([isdir(self.wdir),isfile(self.cfgfile)]):
            print("Model Found")
            print("Loading model")
            p = ConfigParser(config_file_path=self.cfgfile)
            self.minorder = p.get_int("minorder")
            self.maxorder = p.get_int("maxorder")
            self.grating_idx = p.get_int("grating_idx") 
            self.blaze = p.get_int("blaze") 
            self.pxsize = p.get_int("pxsize") 
            self.theta = p.get_int("theta") 
            self.gamma = p.get_float("gamma")
            self.modelname = p.get_string("modelname")
        else:
            raise Exception("Failde to load model")
    def get_psf(self,order,fiber=1,ccd=1):
        _tag = "/CCD_%d/fiber_%d/psf_order_%d/"%(ccd,fiber,order)
        psf = [];
        WL = [];
        
        with h5py.File(self.hdf,"r") as f:
            if _tag in f:
                for wl in f[_tag].keys():
                    #print(join(_tag,wl))
                    #print(f[join(_tag,wl)].shape)
                    psf.append( f[join(_tag,wl)][:] )
                    try:
                        WL.append(float(wl.split("_")[-1]))
                    except:
                        WL.append(0.0)
        return np.asarray(psf),np.asarray(WL)
    def get_transormations(self,order,fiber=1,ccd=1):
        _tag = "/CCD_%d/fiber_%d/order%d/"%(ccd,fiber,order)
        with h5py.File(self.hdf,"r") as f:
            if _tag in f:
                return f[_tag][:]
                    
        return []
    def get_spectro(self,order):
        raise NotImplementedError()
        #_orders = range(self.minorder,self.maxorder+1)
        #ls = [ join(self.orderdir,"%s..%d.hdf"%(self.modelname,i)) for i in _orders if isfile( join(self.orderdir,"%s..%d.hdf"%(self.modelname,i))  )]
        #
        #_file = join(self.orderdir,"%s..%d.hdf"%(self.modelname,order))
        #return ZEMAX(_file)
    def _save(self):
        with open(join(self.wdir,self.cfgfile),"w") as f:
            f.write("modelname %s\n"%self.modelname)
            f.write("minorder %s\n"%self.minorder)
            f.write("maxorder %s\n"%self.maxorder)
            f.write("grating_idx %s\n"%self.grating_idx)
            f.write("blaze %s\n"%self.blaze)
            f.write("pxsize %s\n"%self.pxsize)
            f.write("theta %s\n"%self.theta)
            f.write("gamma %s\n"%self.gamma)
    def __str__(self):
        
        txt = "Model: %s\n"%self.modelname
        txt+="HDF: %s\n"%self.hdf
        txt+="Simulation Name: %s\n"%self.simulationname
        txt+="Zemax file: %s\n"%self.zemax_file
        txt+="Working dir: %s\n"%self.wdir
        txt+="Simulation dir: %s\n"%self.simulationdir
        txt+="Config file: %s\n"%self.cfgfile
        txt += "min. order: %s\n"%self.minorder
        txt += "max. order: %s\n"%self.maxorder
        txt += "grating index: %s\n"%self.grating_idx 
        txt += "blaze: %s\n"%self.blaze 
        txt += "pxsize: %s\n"%self.pxsize 
        txt += "theta: %s\n"%self.theta
        txt += "gamma: %s\n"%self.gamma
        return txt
    def _mkdir(self):
        if not isdir(self.wdir):
            mkdir(self.wdir)
        if not isdir(self.simulationdir):
            mkdir(self.simulationdir)
    def _exist(self):
        if isdir(join(self.wdir,self.modelname)):
            raise Exception("File already exist")
    def get_hdf_file(self,order):
        _name = "%s..%d.hdf"%(self.modelname,order)
        return join(self.orderdir,_name)
    
    def get_order_range(self,order,fiber=1):
        if fiber!=1:
            raise NotImplementedError()
        with open( join(self.orderdir,"%s.fiber1.ordes"%self.modelname),"r"   ) as f:
            ll = f.readlines()
        
        for l in ll[1:]:
            o,_min,_max = l.strip().split(",")
            if int(o)==order:
                return (float(_min),float(_max))
        return (0,0)
    def simulate(self,name,mag,science = True, sky = False,exposureTime=10):
        self.simulationname = name
        _orders = range(self.minorder,self.maxorder+1)
        ls = [ join(self.orderdir,"%s..%d.hdf"%(self.modelname,i)) for i in _orders if isfile( join(self.orderdir,"%s..%d.hdf"%(self.modelname,i))  )]

        if len(ls)!=len(_orders):
            
            raise Exception("[Critical] .hdf file length != # of orders")
        #set Cuda if available
        from numba import cuda
        if cuda.is_available():
            _set_cuda = True 
            print("Setting Cuda automatically")
        else:
            _set_cuda = False
        
        for o in _orders:
            print("working on order #%d"%o)
            sim = Simulator(ZEMAX(ls[o-self.minorder]))
            sim.set_ccd(1)
            if sky and science:
                sim.set_fibers([1,2])
            elif sky:
                sim.set_fibers([2])
            elif science:
                sim.set_fibers([1])
            else:
                raise Exception("Need at least one fiber")
                
            #sim.set_fibers(1)
            sim.set_telescope(Telescope(1.6, 2 * 0.286))
            #sim.set_sources([Phoenix(t_eff=3200,log_g=5.0,z=-0.5)])#proxima centauri
            
            sim.set_cuda(_set_cuda)
            sim.set_sources([Phoenix(t_eff=3200,log_g=5.0,z=-0.5,v_mag=mag)])#proxima centauri
            #sim.set_sources([Phoenix(t_eff=3500, log_g=5.0,z=0.5)])#proxima centauri
            #sim.set_atmospheres([True], sky_calc_kwargs={"airmass": 3.0})
            
            
            sim.set_exposure_time(exposureTime)
            _fits = join(self.simulationdir,"%s..%d.fits"%(self.simulationname,o))
            sim.set_output(_fits, overwrite=True)
            
            
            sim.run()
        
        #create a one frame file
        Im = np.zeros((4096,4096))
        for o in _orders:
            Im+=fits.getdata(join(self.simulationdir,"%s..%d.fits"%(self.simulationname,o)))
        fits.PrimaryHDU(data=Im).writeto(join(self.simulationdir,"%s..all.fits"%(self.simulationname)),overwrite=True)
    def create_hdf_orders(self,zmx_model):#"C:/Users/jo/OneDrive/OneDrive - Universite de Montreal/vroomm/pyechelle/VROOMM_F4_optim_biconic_20250513.ZMX"
        """
        Description
        -----------
            This function will open Link to a standalone OpticStudio instance.
            Make sure the license manager and opticstudio is currently running.
        """
        #check if model already has HDF file
        self._exist()
        self.zemax_file = zmx_model
        
        zmx_file = Path(zmx_model)    
        print("Working on: %s"%self.zemax_file)
        zmx = InteractiveZEMAX(name=self.modelname, zemax_filepath=zmx_file)
        # set basic grating specifications
        zmx.set_grating(self.grating_idx, blaze=self.blaze, theta=self.theta, gamma=self.gamma)
        # add CCD information (only one CCD supported so far. So for instruments with multiple CCDs, you have to generate
        # separate models for now.
        zmx.add_ccd(1, CCD(4096, 4096, pixelsize=self.pxsize))
        # Add here as many fiber/fields as you wish. You don't have to fiddle with the fields in OpticStudio. The
        # existing fields will be ignored/deleted.
        zmx.add_field(0., 0., 33, 132, shape='rectangular', name='Science fiber')
        #zmx.add_field(0., 0.150,67,67, shape='circular', name='Sky fiber')
        # Add here a list with the diffraction orders you want to include
        # Adjust settings for the Huygens PSF. Best to check out 'reasonable' parameters manually in ZEMAX first.
        zmx.psf_settings(image_delta=0.0, image_sampling="128x128", pupil_sampling="64x64")
        with open( self.order_list_f1,"w"   ) as f:
            f.write("order,min,max\n")
        for order in range(self.minorder, self.maxorder):
            print("Working on order %d"%order)
            zmx.set_orders(1, 1, list([order]))
            #zmx.set_orders(1, 2, list([order]))
            _name = "%s..%d.hdf"%(self.modelname,order)
            savepath = join(self.orderdir,_name)
            hdf = HDFBuilder(zmx, savepath)
            # this will take a long time...
            hdf.save_to_hdf(n_transformation_per_order=50, n_psfs_per_order=5)
            _min,_max = zmx.get_wavelength_range(order,1,1)
            with open( self.order_list_f1,"a"   ) as f:
                f.write("%d,%f,%f\n"%(order,_min,_max))
        self._save()
if '__main__' in __name__:
    from pyvroomm.model.vroomm2 import vroomm
    
    path = "/home/espressjo/Documents/UdeM/instrument/VROOMM/optical-design/biconic-window-conic"

    V = vroomm("VROOMM_v02a",wdir=path)


