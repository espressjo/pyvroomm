# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 18:04:04 2025

@author: jo
"""

from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
import seaborn as sns
sns.set_theme()
import pandas as pd
from PIL import Image
from pyvroomm.tools.wavelength2colors import stretched_wavelength_to_rgb
from os.path import join
from pathlib import Path
from scipy.integrate import simpson
from os.path import isfile


class colorfullImages:
    def __init__(self,simulation_name,wdir,order_list,normalisation='local'):
        self.wdir = wdir
        self.img =  Image.new('RGB', (4096, 4096))
        self.order = np.zeros((4096,4096))
        self.fname = simulation_name+"..%d.fits"
        if not isfile(order_list):
            raise Exception("Order list file not found. Please provide an order list")
        
        self.df = pd.read_csv(order_list)
        self.wavelength = (0,0)
        self.n = 0
        self.normalisation_flag = normalisation
        self.l1 = 0
        self.l2 = 0
        self._find_orders()
    def _find_orders(self):
        l = self.df["order"].to_numpy()
        self.min_order = int(np.min(l))
        self.max_order = int(np.max(l))
    def get_order(self,n:int):
        self.order = fits.getdata(join(self.wdir,self.fname%n))
        self.order = np.sqrt(self.order)
        if self.normalisation_flag=='local':
            self.normalisation = np.amax(self.order)
        else:
            raise NotImplementedError()
        self.n = n
    def find_xy(self):
        self.x1 = 0
        self.x2 = 4096
        self.y1 = 0
        self.y2 = 4096
        for y in range(4096):
            if np.any( self.order[y,:] !=0):
                self.y1 = y
                break
        for y in range(4096):
            if np.any( self.order[4095-y,:] !=0):
                self.y2 = 4095-y 
                break
        for x in range(0,4096):
            if np.amax(self.order[:,x])>0:
                self.x1 = x 
                break
        for x in range(4096):
            if np.amax(self.order[:,4095-x])>0:
                self.x2 = 4095-x 
                break
        return self.x1,self.x2,self.y1,self.y2
    def f(self,x):
        if x<self.x1:
            return (0,0,0)
        if x>=self.x2:
            return (0,0,0)
        return stretched_wavelength_to_rgb(1e3*self.spline(x))
    def make_spline(self):
        from scipy.interpolate import UnivariateSpline
        X = np.linspace(self.x1,self.x2,self.x2-self.x1)
        Y = np.linspace(self.l1,self.l2,self.x2-self.x1)
        self.spline = UnivariateSpline(X, Y, s=1)
        return
    def order_to_wavelength_range(self):
        result = self.df.loc[self.df['order'] == self.n, ['min', 'max']]
        self.l1 = result['min'].to_numpy()
        self.l2 = result['max'].to_numpy()
        return result['min'].to_numpy(),result['max'].to_numpy()
    def colorfull_picture(self,fname):
        for n in range(self.min_order, self.max_order):
            self.get_order(n)
            self.order_to_wavelength_range()
            self.find_xy()
            self.make_spline()
            print("Working on order=%d"%n)
            #make color
            for y in range(self.y1,self.y2):
                for x in range(4096):
                    if self.order[y,x]==0:
                        continue
                    rgb = self.f(x)#fraction
                    flux = int(round(255*(self.order[y,x]/self.normalisation)))
                    rgb = (int(round(flux*rgb[0])),int(round(flux*rgb[1])),int(round(flux*rgb[2])))
                    self.img.putpixel((x, y), rgb)
        self.img.save(fname)
        print("Image saved successfully.")
            
    
    
if '__main__' in __name__:
    p = colorfullImages("something.ordes",normalisation='local')
    p.colorfull_picture("test.png")
    