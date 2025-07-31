import h5py
import numpy as np
import os
import glob
from pathlib import Path

def combine_pyechelle_orders(order_files, output_file, verbose=True):
    """
    Combine multiple PyEchelle order HDF5 files into a single file.
    
    Parameters:
    -----------
    order_files : list
        List of paths to individual order HDF5 files
    output_file : str
        Path for the combined output file
    verbose : bool
        Print progress information
    """
    
    if verbose:
        print(f"Combining {len(order_files)} order files into {output_file}")
    
    with h5py.File(output_file, 'w') as output_h5:
        
        # Create main groups for organization
        orders_group = output_h5.create_group('orders')
        metadata_group = output_h5.create_group('metadata')
        
        # Store global metadata
        metadata_group.attrs['n_orders'] = len(order_files)
        metadata_group.attrs['pyechelle_version'] = 'combined'
        
        order_numbers = []
        
        for i, order_file in enumerate(order_files):
            if verbose and i % 10 == 0:
                print(f"Processing order {i+1}/{len(order_files)}: {order_file}")
            
            try:
                with h5py.File(order_file, 'r') as order_h5:
                    
                    # Extract order number from filename or HDF5 metadata
                    order_num = extract_order_number(order_file, order_h5)
                    order_numbers.append(order_num)
                    
                    # Create group for this order
                    order_group = orders_group.create_group(f'order_{order_num:03d}')
                    
                    # Copy all datasets and groups recursively
                    copy_hdf5_tree(order_h5, order_group)
                    
                    # Add order-specific metadata
                    order_group.attrs['order_number'] = order_num
                    order_group.attrs['source_file'] = os.path.basename(order_file)
                    
            except Exception as e:
                print(f"Error processing {order_file}: {e}")
                continue
        
        # Store order number array for easy access
        metadata_group.create_dataset('order_numbers', data=np.array(sorted(order_numbers)))
        
        if verbose:
            print(f"Successfully combined {len(order_numbers)} orders")
            print(f"Order numbers: {min(order_numbers)} to {max(order_numbers)}")

def extract_order_number(filename, h5_file):
    """
    Extract order number from filename or HDF5 file metadata.
    Adapt this function based on your naming convention.
    """
    
    # Method 1: From filename (e.g., "order_045.h5" or "pyechelle_order45.h5")
    import re
    basename = os.path.basename(filename)
    
    # Try different patterns
    patterns = [
        r'order_?(\d+)',  # order_045, order45
        r'(\d+)\.h5',     # 045.h5
        r'ord(\d+)',      # ord45
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Method 2: From HDF5 metadata (if stored)
    try:
        if 'order_number' in h5_file.attrs:
            return int(h5_file.attrs['order_number'])
        if 'metadata' in h5_file and 'order' in h5_file['metadata'].attrs:
            return int(h5_file['metadata'].attrs['order'])
    except:
        pass
    
    # Method 3: Default to filename without extension
    print(f"Warning: Could not extract order number from {filename}, using 0")
    return 0

def copy_hdf5_tree(source, destination):
    """
    Recursively copy all datasets and groups from source to destination.
    """
    for key, item in source.items():
        if isinstance(item, h5py.Dataset):
            # Copy dataset
            destination.create_dataset(key, data=item[:], 
                                     compression=item.compression,
                                     compression_opts=item.compression_opts)
            # Copy attributes
            for attr_key, attr_val in item.attrs.items():
                destination[key].attrs[attr_key] = attr_val
                
        elif isinstance(item, h5py.Group):
            # Create group and recurse
            new_group = destination.create_group(key)
            copy_hdf5_tree(item, new_group)
            # Copy group attributes
            for attr_key, attr_val in item.attrs.items():
                new_group.attrs[attr_key] = attr_val

def inspect_combined_file(filename):
    """
    Inspect the structure of the combined HDF5 file.
    """
    print(f"\n=== Inspecting {filename} ===")
    
    with h5py.File(filename, 'r') as f:
        print(f"File structure:")
        print_hdf5_structure(f, indent=0)
        
        if 'metadata' in f:
            print(f"\nMetadata:")
            for key, val in f['metadata'].attrs.items():
                print(f"  {key}: {val}")
            
            if 'order_numbers' in f['metadata']:
                orders = f['metadata']['order_numbers'][:]
                print(f"  Available orders: {len(orders)} orders from {min(orders)} to {max(orders)}")

def print_hdf5_structure(obj, indent=0):
    """
    Recursively print HDF5 file structure.
    """
    prefix = "  " * indent
    for key in obj.keys():
        item = obj[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: Dataset {item.shape} {item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}{key}: Group")
            if indent < 3:  # Limit depth to avoid clutter
                print_hdf5_structure(item, indent + 1)

def load_order_data(combined_file, order_number):
    """
    Load data for a specific order from the combined file.
    """
    with h5py.File(combined_file, 'r') as f:
        order_key = f'order_{order_number:03d}'
        if order_key not in f['orders']:
            available = list(f['orders'].keys())
            raise ValueError(f"Order {order_number} not found. Available: {available}")
        
        order_group = f['orders'][order_key]
        
        # Return a dictionary with all datasets for this order
        order_data = {}
        for key, item in order_group.items():
            if isinstance(item, h5py.Dataset):
                order_data[key] = item[:]
            elif isinstance(item, h5py.Group):
                # Handle nested groups if needed
                order_data[key] = {}
                for subkey, subitem in item.items():
                    if isinstance(subitem, h5py.Dataset):
                        order_data[key][subkey] = subitem[:]
        
        return order_data

# Example usage functions
def main():
    """
    Example of how to use the combining functions.
    """
    
    # Method 1: Combine files matching a pattern
    from natsort import natsorted
    order_files = glob.glob("/home/espressjo/Documents/UdeM/instrument/VROOMM/optical-design/biconic-cylindric/vroomm-model/.hdf/vroomm-biconic..*.hdf")  # Adjust pattern as needed
    order_files = natsorted(order_files)  # Ensure consistent ordering

    if not order_files:
        print("No order files found! Check your file pattern.")
        # Create some dummy files for demonstration
        create_dummy_order_files()
        order_files = glob.glob("order_*.hdf")
    
    print(f"Found {len(order_files)} order files")
    
    # Combine them
    output_file = "combined_echelle_orders.hdf"
    combine_pyechelle_orders(order_files[:3], output_file)
    
    # Inspect the result
    inspect_combined_file(output_file)
    
    # Test loading specific order
    try:
        order_data = load_order_data(output_file, 45)  # Load order 45
        print(f"\nLoaded order 45 data keys: {list(order_data.keys())}")
    except ValueError as e:
        print(f"Error loading order: {e}")

def create_dummy_order_files():
    """
    Create some dummy order files for testing.
    """
    print("Creating dummy order files for demonstration...")
    
    for order_num in [42, 43, 44, 45, 46]:
        filename = f"order_{order_num:03d}.h5"
        
        with h5py.File(filename, 'w') as f:
            # Simulate typical PyEchelle structure
            f.attrs['order_number'] = order_num
            
            # Some dummy datasets
            f.create_dataset('wavelength', data=np.linspace(500 + order_num, 600 + order_num, 1000))
            f.create_dataset('flux', data=np.random.normal(1000, 100, 1000))
            f.create_dataset('noise', data=np.random.normal(10, 2, 1000))
            
            # Metadata group
            meta = f.create_group('metadata')
            meta.attrs['central_wavelength'] = 550 + order_num
            meta.attrs['blaze_angle'] = 45.0 + order_num * 0.1
            
    print("Dummy files created: order_042.h5 through order_046.h5")

if __name__ == "__main__":
    main()