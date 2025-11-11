#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:30:16 2025

@author: espressjo
"""
import numpy as np

def psfCrossSection(im,imageDelta=0.5):
    """
     Will return the RMS and geometric radius of a PSF array
    
     Parameters
     ----------
     im : TYPE
         DESCRIPTION.
    
     Returns
     -------
     rms : TYPE
         DESCRIPTION.
     geo : TYPE
         DESCRIPTION.
    
     """
    
    im = np.asarray(im)
    y_max,x_max = np.unravel_index(np.argmax(im), im.shape)
    
    r = [];
    flux = [];
    for x in range(im.shape[0]):
        for y in range(im.shape[0]):
            r.append(np.sqrt(  (x-x_max)**2 + (y-y_max)**2  ))
            flux.append(im[y,x])
    _rms = np.sqrt(np.sum(np.asarray(flux)**2)/im.size)
    _geo = np.percentile(np.asarray(flux),95)
    data = zip(r,flux)
    data = sorted(data) #.sorted()
    r,flux = zip(*data)
    return np.asarray(r),np.asarray(flux)


def calculate_rms_radius(image,image_delta, threshold=None, background_percentile=10):
    """
    Calculate the RMS (Root Mean Square) radius of a PSF spot in a 2D image.
    
    This function handles irregular spot shapes by:
    - Computing the centroid weighted by intensity
    - Optionally removing background noise
    - Calculating RMS radius from the centroid
    
    Parameters:
    -----------
    image : np.ndarray
        2D array representing the PSF intensity distribution
    threshold : float, optional
        Minimum intensity threshold. Pixels below this are ignored.
        If None, uses background_percentile to estimate threshold.
    background_percentile : float, optional
        Percentile used to estimate background level (default: 10)
        Only used if threshold is None.
    
    Returns:
    --------
    dict containing:
        'rms_radius': float - RMS radius in pixels
        'centroid': tuple - (x, y) centroid position
        'total_intensity': float - Sum of intensities above threshold
    
    Example:
    --------
    >>> # Create a Gaussian-like PSF
    >>> x = np.linspace(-10, 10, 100)
    >>> y = np.linspace(-10, 10, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> psf = np.exp(-(X**2 + Y**2) / (2 * 3**2))
    >>> result = calculate_rms_radius(psf)
    >>> print(f"RMS radius: {result['rms_radius']:.2f} pixels")
    """
    
    # Ensure image is a numpy array and float type
    image = np.asarray(image, dtype=float)
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    if image.size == 0:
        raise ValueError("Image is empty")
    
    # Handle threshold
    if threshold is None:
        # Estimate background level and set threshold slightly above it
        background = np.percentile(image, background_percentile)
        # Use mean of background region as threshold
        threshold = background + 0.1 * (np.max(image) - background)
    
    # Create mask of pixels above threshold
    mask = image > threshold
    
    if not np.any(mask):
        raise ValueError(f"No pixels above threshold {threshold:.2e}. "
                        f"Max intensity: {np.max(image):.2e}")
    
    # Get coordinates
    height, width = image.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Apply mask to get only relevant pixels
    intensities = image[mask]
    x_vals = x_coords[mask]
    y_vals = y_coords[mask]
    
    # Calculate weighted centroid
    total_intensity = np.sum(intensities)
    x_centroid = np.sum(x_vals * intensities) / total_intensity
    y_centroid = np.sum(y_vals * intensities) / total_intensity
    
    # Calculate squared distances from centroid
    dx = x_vals - x_centroid
    dy = y_vals - y_centroid
    r_squared = dx**2 + dy**2
    
    # Calculate RMS radius (weighted by intensity)
    rms_radius_squared = np.sum(r_squared * intensities) / total_intensity
    rms_radius = np.sqrt(rms_radius_squared)
    
    return {
        'rms_radius': rms_radius*image_delta,
        'centroid': (x_centroid, y_centroid),
        'total_intensity': total_intensity,
        'n_pixels': np.sum(mask)
    }


def calculate_rms_radius_simple(image,image_delta):
    """
    Simplified version that uses all pixels (no thresholding).
    Good for clean images without background noise.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array representing the PSF intensity distribution
    
    Returns:
    --------
    float : RMS radius in pixels
    """
    image = np.asarray(image, dtype=float)
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    height, width = image.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate centroid
    total = np.sum(image)
    if total == 0:
        raise ValueError("Image has zero total intensity")
    
    x_centroid = np.sum(x_coords * image) / total
    y_centroid = np.sum(y_coords * image) / total
    
    # Calculate RMS radius
    dx = x_coords - x_centroid
    dy = y_coords - y_centroid
    r_squared = dx**2 + dy**2
    
    rms_radius = np.sqrt(np.sum(r_squared * image) / total)
    
    return rms_radius*image_delta



def calculate_geometric_radius(image, image_delta,percentile=95, threshold=None, background_percentile=10):
    """
    Calculate the geometric radius of a PSF spot containing a specified percentile
    of the total energy (integrated intensity).
    
    This function handles irregular spot shapes by:
    - Computing the centroid weighted by intensity
    - Calculating radius that contains the specified percentile of energy
    - Optionally removing background noise
    
    Parameters:
    -----------
    image : np.ndarray
        2D array representing the PSF intensity distribution
    percentile : float, optional
        Percentile of energy to enclose (default: 95)
        E.g., 95 means radius contains 95% of total energy
    threshold : float, optional
        Minimum intensity threshold. Pixels below this are ignored.
        If None, uses background_percentile to estimate threshold.
    background_percentile : float, optional
        Percentile used to estimate background level (default: 10)
        Only used if threshold is None.
    
    Returns:
    --------
    dict containing:
        'geometric_radius': float - Radius containing specified energy percentile
        'centroid': tuple - (x, y) centroid position
        'total_intensity': float - Sum of intensities above threshold
        'enclosed_intensity': float - Intensity within geometric radius
    
    Example:
    --------
    >>> # Create a Gaussian-like PSF
    >>> x = np.linspace(-10, 10, 100)
    >>> y = np.linspace(-10, 10, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> psf = np.exp(-(X**2 + Y**2) / (2 * 3**2))
    >>> result = calculate_geometric_radius(psf, percentile=95)
    >>> print(f"95% energy radius: {result['geometric_radius']:.2f} pixels")
    """
    
    # Ensure image is a numpy array and float type
    image = np.asarray(image, dtype=float)
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    if image.size == 0:
        raise ValueError("Image is empty")
    
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    # Handle threshold
    if threshold is None:
        # Estimate background level and set threshold slightly above it
        background = np.percentile(image, background_percentile)
        # Use mean of background region as threshold
        threshold = background + 0.1 * (np.max(image) - background)
    
    # Create mask of pixels above threshold
    mask = image > threshold
    
    if not np.any(mask):
        raise ValueError(f"No pixels above threshold {threshold:.2e}. "
                        f"Max intensity: {np.max(image):.2e}")
    
    # Get coordinates
    height, width = image.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Apply mask to get only relevant pixels
    intensities = image[mask]
    x_vals = x_coords[mask]
    y_vals = y_coords[mask]
    
    # Calculate weighted centroid
    total_intensity = np.sum(intensities)
    x_centroid = np.sum(x_vals * intensities) / total_intensity
    y_centroid = np.sum(y_vals * intensities) / total_intensity
    
    # Calculate distances from centroid
    dx = x_vals - x_centroid
    dy = y_vals - y_centroid
    distances = np.sqrt(dx**2 + dy**2)
    
    # Sort pixels by distance from centroid
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_intensities = intensities[sorted_indices]
    
    # Calculate cumulative energy
    cumulative_intensity = np.cumsum(sorted_intensities)
    
    # Find radius that contains the desired percentile of energy
    target_intensity = (percentile / 100.0) * total_intensity
    
    # Find the index where cumulative intensity exceeds target
    idx = np.searchsorted(cumulative_intensity, target_intensity)
    
    # Handle edge cases
    if idx >= len(sorted_distances):
        idx = len(sorted_distances) - 1
    
    geometric_radius = sorted_distances[idx]
    enclosed_intensity = cumulative_intensity[idx]
    
    return {
        'geometric_radius': geometric_radius*image_delta,
        'centroid': (x_centroid, y_centroid),
        'total_intensity': total_intensity,
        'enclosed_intensity': enclosed_intensity,
        'energy_fraction': enclosed_intensity / total_intensity,
        'percentile': percentile,
        'n_pixels': np.sum(mask)
    }


def calculate_geometric_radius_simple(image,image_delta, percentile=95):
    """
    Simplified version that uses all pixels (no thresholding).
    Good for clean images without background noise.
    
    Parameters:
    -----------
    image : np.ndarray
        2D array representing the PSF intensity distribution
    percentile : float, optional
        Percentile of energy to enclose (default: 95)
    
    Returns:
    --------
    float : Geometric radius containing specified energy percentile
    """
    image = np.asarray(image, dtype=float)
    
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    height, width = image.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate centroid
    total = np.sum(image)
    if total == 0:
        raise ValueError("Image has zero total intensity")
    
    x_centroid = np.sum(x_coords * image) / total
    y_centroid = np.sum(y_coords * image) / total
    
    # Calculate distances
    dx = x_coords - x_centroid
    dy = y_coords - y_centroid
    distances = np.sqrt(dx**2 + dy**2).flatten()
    intensities = image.flatten()
    
    # Sort by distance
    sorted_indices = np.argsort(distances)
    sorted_intensities = intensities[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    # Find radius for target percentile
    cumulative = np.cumsum(sorted_intensities)
    target = (percentile / 100.0) * total
    idx = np.searchsorted(cumulative, target)
    
    if idx >= len(sorted_distances):
        idx = len(sorted_distances) - 1
    
    return sorted_distances[idx]*image_delta


