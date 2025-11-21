#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 09:33:22 2025

@author: espressjo
"""

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import affine_transform
from pyechelle.optics import convert_matrix
# 1 px ~ 1 km/s

def apply_centered_transform(data, matrix):
    """Apply transformation with content centered in output"""
    h, w = data.shape
    center = np.array([w, h]) / 2
    
    # Find bounding box of transformed content
    y_coords, x_coords = np.where(data > 0)
    if len(x_coords) == 0:
        return np.zeros_like(data)
    
    points = np.column_stack([x_coords, y_coords]) - center
    transformed_points = points @ matrix.T
    
    # Calculate centering offset
    min_coords = transformed_points.min(axis=0)
    max_coords = transformed_points.max(axis=0)
    content_center = (min_coords + max_coords) / 2
    centering_offset = -content_center
    
    # Apply with scipy
    matrix_inv = np.linalg.inv(matrix)
    offset = center - matrix_inv @ (center + centering_offset)
    
    return affine_transform(data, matrix_inv, offset=offset, 
                           output_shape=data.shape, order=0, cval=0)



class rv_line:
    def __init__(self,fwhm=3.0):
        oversampling = 24 
        
        self.fwhm = fwhm 
        self.sigma = fwhm/2.354  *oversampling
        self.pixgrid = np.arange(-int(5*self.sigma),int(5*self.sigma),1)
        self.g = np.exp(-0.5*(self.pixgrid/self.sigma)**2)
        self.g/=np.sum(self.g)
    
    def show(self):
        fig,ax = plt.subplots()
        ax.plot(self.pixgrid,self.g)
        ax.set(xlabel="?",ylabel="Amplitude")
        plt.show()
        #fig, ax = plt.subplots()

class slit:
    def __init__(self):
        self.slit_width_um = 33
        self.slit_height_um = 134
        self.slit_frame_size = (512,512)
        self.pixel_frame_size = (64,64)
        self.middle = self.slit_frame_size[0]//2
        self.pixel_size=12 # micron/pixel
        
        
        
    def load_transformation2(self,data,pyechelle_Atrans):
        
        sx = float(pyechelle_Atrans.sx)*self.pixel_size/self.slit_width_um     #convert the pixel to elements
        sy = float(pyechelle_Atrans.sy)*self.pixel_size/self.slit_height_um#convert the pixel to elements
        shear = float(pyechelle_Atrans.shear) #-0.1924981#radian
        theta = float(pyechelle_Atrans.rot)#radian
        
        stretch = np.array([[sy,0],
                            [0,sx]])

           
        
        
    def load_transformation(self,data,pyechelle_Atrans):
        
        sx = float(pyechelle_Atrans.sx)*self.pixel_size/self.slit_width_um     #convert the pixel to elements
        sy = float(pyechelle_Atrans.sy)*self.pixel_size/self.slit_height_um#convert the pixel to elements

        shear = float(pyechelle_Atrans.shear) #-0.1924981#radian
        theta = float(pyechelle_Atrans.rot)#radian
        
        stretch = np.array([[sy,0],
                            [0,sx]])

        rotation_shear = np.array([[np.cos(theta),-np.sin(theta+shear)],
                             [np.sin(theta),np.cos(theta+shear)]])

        #M = np.array([[sx*np.cos(theta), -sy*np.sin(theta+shear),0  ], [sx*np.sin(theta), sy*np.cos(theta+shear),0   ],[0,0,1]    ])
        
        #M = np.linalg.inv(M)
        
        _stretch = np.linalg.inv(stretch)
        _rotation_shear = np.linalg.inv(rotation_shear)

        M = _stretch @ _rotation_shear
        
        #calculate the offset from the center of 
        #the slit
        mid = self.slit_frame_size[0]//2
        
        _x,_y = self.matrix_offset(mid, mid, M)
        

        trans_data = affine_transform(data, M,offset=(_x,_y),
                                order=0, cval=0)
        
        
        return trans_data
        
        
        
        """
        sx = float(pyechelle_Atrans.sx)*self.pixel_size/self.slit_width_um
        sy = float(pyechelle_Atrans.sy)*self.pixel_size/self.slit_height_um
        shear = float(pyechelle_Atrans.shear)
        rotation = float(pyechelle_Atrans.rot)
        theta = rotation#np.deg2rad(rotation)
        shear_factor = shear#np.tan(shear)
        
        scaling = np.array([[sx,0],
                            [0,sy]])
        shear = np.array([[1,shear_factor],[0,1]])
        rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        matrix = scaling @ rotation @ shear
        
        #test
        matrix = np.array([[sx*np.cos(theta),-sy*np.sin(theta + shear_factor)],
                  [sx*np.sin(theta), sy*np.cos(theta+shear_factor)]])
        
        
        #result = apply_centered_transform(data, matrix)   
        result = affine_transform(data, np.linalg.inv(matrix), #offset=offset, 
                               output_shape=data.shape)

        """
        
        
        
        
    def matrix_offset(self,x,y,M):
        center = np.array((y,x))
        offset = center - M @ center
        #print("[debug] ",float(offset[0]),float(offset[1]))
        return float(offset[0]),float(offset[1])    
    def get_slit_1d(self):
        _tmp =  np.sum(self.slit_pixel_psf_same,axis=0)
        _tmp/=np.sum(_tmp)
        return _tmp
    def initialize(self):
        
        self.X,self.Y = np.meshgrid(np.arange(self.slit_frame_size[0])*self.pixel_scale,np.arange(self.slit_frame_size[1])*self.pixel_scale)
        # Create slit image with constant illumination
        self.slit_img = np.zeros(self.slit_frame_size)
        self.slit_height_elements = int(self.slit_height_um/self.pixel_scale)
        self.slit_width_elements = int(self.slit_width_um/self.pixel_scale)

        self.slit_img[(self.middle-self.slit_height_elements//2) : (self.middle+self.slit_height_elements//2),
                      (self.middle-self.slit_width_elements//2) : (self.middle+self.slit_width_elements//2)]=1
        
        if self.transformation is not None:
            self.slit_img = self.load_transformation(self.slit_img,self.transformation)
        #create the pixel array     
        self.pixel = np.zeros(self.pixel_frame_size)
        self.xpix,self.ypix = np.meshgrid(np.arange(self.pixel.shape[0])*self.pixel_scale,np.arange(self.pixel.shape[1])*self.pixel_scale)
        half_pixel_elements = int((self.pixel_size/self.pixel_scale)//2)
        self.pixel[self.pixel.shape[0]//2-half_pixel_elements:self.pixel.shape[0]//2+half_pixel_elements,
                   self.pixel.shape[0]//2-half_pixel_elements:self.pixel.shape[0]//2+half_pixel_elements]=1    
        
        self.slit_psf = fftconvolve(self.slit_img, self.psf, mode='same')
        
        
        self.slit_pixel_same = fftconvolve(self.slit_img, self.pixel, mode='same')
        self.slit_pixel_full = fftconvolve(self.slit_img, self.pixel, mode='full')
        self.slit_pixel_same/=np.sum(self.slit_pixel_same)
        #convolve both with PSF
        self.slit_pixel_psf_same = fftconvolve(self.slit_pixel_same, self.psf, mode='same')
        self.slit_pixel_psf_full = fftconvolve(self.slit_pixel_full, self.psf, mode='full')
        self.slit_pixel_psf_same/=np.sum(self.slit_pixel_psf_same)
    def get_slit_px(self):
        return self.slit_pixel_psf_same
    def __str__(self):
        t = ""
        t+="Order: %d\n"%self.order
        return t
    def load(self,psf,order,image_delta=0.5,transformation=None):
        self.pixel_scale = image_delta
        self.psf = psf
        self.order = order
        _y,_x = psf.shape
        self.transformation = transformation
        self.psfpix,self.psfpixy = np.meshgrid(np.arange(_y)*self.pixel_scale,np.arange(_x)*self.pixel_scale)
        self.initialize()
    
    def show_all(self):

        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        #slit
        axes[0][0].pcolor(self.X, self.Y, self.slit_img, cmap='inferno')
        axes[0][0].set_title("Slit")
        axes[0][0].set(xlabel="$\mu$m",ylabel="$\mu$m")
        #pixel
        axes[0][1].pcolor(self.xpix, self.ypix, self.pixel, cmap='inferno')
        axes[0][1].set_title("Pixel")
        axes[0][1].set(xlabel="$\mu$m",ylabel="$\mu$m")
            
        axes[0][2].pcolor(self.psfpix, self.psfpixy, self.psf, cmap='inferno')
        axes[0][2].set_title("PSF")
        axes[0][2].set(xlabel="$\mu$m",ylabel="$\mu$m")
        #convolve pixel+slit
        
        
        
        axes[1][0].pcolor(self.X,self.Y,self.slit_pixel_same, cmap='inferno')
        axes[1][0].set_title("Slit + Pixel")
        axes[1][0].set(xlabel="$\mu$m",ylabel="$\mu$m")
        
        axes[1][1].pcolor(self.X,self.Y,self.slit_pixel_psf_same, cmap='inferno')
        axes[1][1].set_title("Slit + Pixel + PSF")
        axes[1][1].set(xlabel="$\mu$m",ylabel="$\mu$m")
        
        axes[1][2].pcolor(self.X/self.pixel_size,self.Y/self.pixel_size,self.slit_pixel_psf_same, cmap='inferno')
        axes[1][2].set_title("Slit + Pixel + PSF")
        axes[1][2].set(xlabel="pixel (12$\mu$m)",ylabel="pixel (12$\mu$m)")
        plt.tight_layout()
        plt.show()
        return fig
    def get_convoluted_slit(self,fwhm=3.0):
        feature = rv_line(fwhm=fwhm)
        rvline = fftconvolve(feature.g,self.get_slit_1d(),'same')
        return rvline
    
    
    
    




if '__main__' in __name__:
    
 
    cylindric_psf = fits.getdata("/home/espressjo/cylindric-0deg.fits")
    biconic_psf = fits.getdata("/home/espressjo/biconic-0deg.fits")
    image = np.zeros((4096,4096))
    p = "/home/espressjo/tmp2"
    from os.path import join
    from skimage.transform import resize
    from pyvroomm.model.vroomm import vroomm
    wdir = "/home/espressjo/Documents/UdeM/instrument/VROOMM/optical-design/biconic-cylindric/vroomm-model"
    cylindric = vroomm("vroomm-cylindric-0deg-sampling",wdir=wdir)
    
    order=67
    wl  = cylindric.get_spectro(order).get_wavelength_range(order,1,1)
    _min,_max = wl
    wl = np.mean(wl)
    X = np.linspace(_min,_max,5)
    
    trans = cylindric.get_spectro(order).get_transformation(wl, order)
    _check = convert_matrix(trans)
    cyl_slit = slit()
    cyl_slit.load(cylindric_psf[order-65], order,transformation=trans)
    cyl_slit.show_all()
    
    # for o in range(65,155):
    #     wl  = cylindric.get_spectro(order).get_wavelength_range(order,1,1)
    #     _min,_max = wl
    #     wl = np.mean(wl)
    #     M = cylindric.get_spectro(order).get_transformation(wl, order)
    #     print("order: ","%.3f rad, "%M.rot,"%.1f"%(M.rot*180/np.pi))
    # from sys import exit
    # exit(0)
    """
    X = np.linspace(_min,_max,10)
    
    trans = cylindric.get_spectro(order).get_transformation(wl, order)
    
        
    cyl_slit = slit()
    cyl_slit.load(cylindric_psf[order-65], order,transformation=trans)
    _tx = [];
    _ty = [];    
    cyl_slit.show_all()
    
    for x in X:
        trans = cylindric.get_spectro(order).get_transformation(x, order)
        print("[debug] rotation: ",trans.rot," Shear: %.4f"%trans.shear)
        _tx.append(float(trans.tx))
        _ty.append(float(trans.ty))
    cyl_slit.show_all()
    
    """
    
    """
    for order in range(70,71):
        wl  = cylindric.get_spectro(order).get_wavelength_range(order,1,1)
        _min,_max = wl
        wl = np.mean(wl)
        X = np.linspace(_min,_max,100)
        for wl in X:
            trans = cylindric.get_spectro(order).get_transformation(wl, order)
            x = int(int(trans.tx))
            y = int( int(trans.ty))
            if x<0 or x>=4096:
                continue
            if y<0 or y>=4096:
                continue
            
            print(x,y)
            print(order)
            cyl_slit = slit()
            cyl_slit.load(cylindric_psf[order-65], order)
            #cyl_slit.show_all()
            _slit = cyl_slit.get_slit_px()
            
            
            small = resize(_slit, (512//24, 512//24), anti_aliasing=True)
            image[y:y+int(512//24),x:x+int(512//24)] = small*1e6
            image[y,x] = 1e6
    fits.PrimaryHDU(data=image).writeto("/home/espressjo/test65.fits",overwrite=True)
    plt.imshow(image)
    plt.show()
            
    """
