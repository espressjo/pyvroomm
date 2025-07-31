import numpy as np
import matplotlib.pyplot as plt
from pyechelle.simulator import Simulator  # Assuming you have pyechelle installed
from pyechelle.spectrograph import ZEMAX
def calculate_spectrograph_efficiency(model_file, wavelengths, input_flux):
    """
    Calculate the efficiency of a spectrograph using PyEchelle simulations.
    
    Parameters:
    -----------
    model_file : str
        Path to the spectrograph model file (ZEMAX-derived)
    wavelengths : array
        Wavelength array in nm
    input_flux : array  
        Input flux at each wavelength (photons/s/nm or similar units)
    
    Returns:
    --------
    efficiency : array
        Wavelength-dependent efficiency (0-1)
    """
    
    # Initialize the simulator
    sim = Simulator(ZEMAX(model_file))
    sim.set_ccd(1)
    sim.set_fibers([1])
    
    # Arrays to store results
    output_flux = np.zeros_like(input_flux)
    efficiency = np.zeros_like(input_flux)
    
    print(f"Calculating efficiency for {len(wavelengths)} wavelengths...")
    
    for i, (wl, flux_in) in enumerate(zip(wavelengths, input_flux)):
        if i % 10 == 0:  # Progress indicator
            print(f"Processing wavelength {i+1}/{len(wavelengths)}: {wl:.1f} nm")
        
        # Create a monochromatic input spectrum
        # You might need to adjust this based on your PyEchelle version
        spectrum = create_monochromatic_spectrum(wl, flux_in)
        
        
        # Simulate the spectrum through the spectrograph
        simulated_2d = sim.simulate_spectrum(spectrum, wavelength=wl)
        try:
            simulated_2d = sim.simulate_spectrum(spectrum, wavelength=wl)
            from matplotlib import pyplot as plt
            x,y = simulated_2d
            plt.plot(x,y)
            plt.show()
            
            # Extract the total flux from the 2D simulation
            # This depends on how your extraction works
            flux_out = extract_total_flux(simulated_2d)
            
            # Calculate efficiency
            if flux_in > 0:
                efficiency[i] = flux_out / flux_in
            else:
                efficiency[i] = 0
                
            output_flux[i] = flux_out
            
        except Exception as e:
            print(f"Error at wavelength {wl:.1f} nm: {e}")
            efficiency[i] = 0
            output_flux[i] = 0
    
    return efficiency, output_flux

def create_monochromatic_spectrum(wavelength, flux, width=0.1):
    """
    Create a monochromatic input spectrum for simulation.
    
    Parameters:
    -----------
    wavelength : float
        Central wavelength in nm
    flux : float
        Flux value
    width : float
        Spectral width in nm (narrow line)
    """
    # Create a narrow Gaussian line
    wl_array = np.linspace(wavelength - 5*width, wavelength + 5*width, 100)
    spectrum = flux * np.exp(-0.5 * ((wl_array - wavelength) / width)**2)
    
    return wl_array, spectrum

def extract_total_flux(simulated_2d):
    """
    Extract total flux from 2D simulated spectrum.
    
    This is a simplified version - you'll need to adapt based on:
    - Your extraction method (optimal vs simple aperture)
    - Order separation and identification
    - Background subtraction
    """
    # Simple approach: sum all pixels above background
    background = np.median(simulated_2d)  # Rough background estimate
    signal_pixels = simulated_2d > background + 3*np.std(simulated_2d)
    
    total_flux = np.sum(simulated_2d[signal_pixels] - background)
    
    return max(0, total_flux)  # Ensure non-negative

def plot_efficiency_curve(wavelengths, efficiency, title="Spectrograph Efficiency"):
    """
    Plot the efficiency curve with nice styling.
    """
    plt.figure(figsize=(12, 7))
    
    # Main efficiency curve
    plt.plot(wavelengths, efficiency * 100, 'b-', linewidth=2.5, 
             label='Total Efficiency', alpha=0.8)
    
    # Fill under curve
    plt.fill_between(wavelengths, efficiency * 100, alpha=0.3, color='blue')
    
    # Styling
    plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency (%)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add some statistics
    max_eff = np.max(efficiency) * 100
    mean_eff = np.mean(efficiency) * 100
    plt.text(0.02, 0.98, f'Peak Efficiency: {max_eff:.1f}%\nMean Efficiency: {mean_eff:.1f}%', 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example usage
def main():
    # Define your wavelength range (adjust for your spectrograph)
    wavelengths = np.linspace(0.510422,0.520468, 50)  # 400-700 nm, 50 points
    
    # Define input flux (flat spectrum for efficiency measurement)
    input_flux = np.ones_like(wavelengths) * 1000  # 1000 photons/s/nm
    
    # Path to your spectrograph model
    model_file = "/home/espressjo/Documents/UdeM/instrument/VROOMM/optical-design/biconic-cylindric/vroomm-model/.hdf/vroomm-biconic..119.hdf"
    
    try:
        # Calculate efficiency
        efficiency, output_flux = calculate_spectrograph_efficiency(
            model_file, wavelengths, input_flux)
        
        # Plot results
        plot_efficiency_curve(wavelengths, efficiency, 
                            "My Spectrograph Efficiency Curve")
        
        # Save results
        np.savetxt('efficiency_results.txt', 
                  np.column_stack([wavelengths, efficiency, input_flux, output_flux]),
                  header='Wavelength(nm) Efficiency Input_Flux Output_Flux',
                  fmt='%.3f')
        
        print("Efficiency calculation complete!")
        print(f"Results saved to efficiency_results.txt")
        print(f"Peak efficiency: {np.max(efficiency)*100:.1f}%")
        print(f"Mean efficiency: {np.mean(efficiency)*100:.1f}%")
        
    except FileNotFoundError:
        print("Model file not found. Please check the path.")
    except ImportError:
        print("PyEchelle not installed. Install with: pip install pyechelle")

if __name__ == "__main__":
    model_file = "/home/espressjo/Documents/UdeM/instrument/VROOMM/optical-design/biconic-cylindric/vroomm-model/.hdf/vroomm-biconic..119.hdf"

    spectro = ZEMAX(model_file)  
    print(spectro.get_wavelength_range(119,1,1))
    eff = spectro.get_efficiency(1, 1)
    #sim = Simulator(ZEMAX(model_file))
    #sim.set_ccd(1)
    #sim.set_fibers([1])
    #sim.get_efficiency()

# Alternative: Calculate efficiency from existing observations
def efficiency_from_observations(observed_spectrum, input_spectrum, wavelengths):
    """
    Calculate efficiency from real observations if you have both
    input and output spectra.
    """
    # Interpolate to common wavelength grid if needed
    from scipy.interpolate import interp1d
    
    # Simple efficiency calculation
    efficiency = observed_spectrum / input_spectrum
    
    # Handle division by zero and outliers
    efficiency = np.where(input_spectrum > 0, efficiency, 0)
    efficiency = np.clip(efficiency, 0, 1)  # Keep between 0 and 1
    
    return efficiency