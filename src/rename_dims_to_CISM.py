import os
import xarray as xr
import argparse

def rename_and_fillna_in_directory(directory):
    # Loop through all .nc files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.nc'):
            filepath = os.path.join(directory, filename)
            
            # Open the dataset
            ds = xr.open_dataset(filepath)
            
            # Rename dimensions if they exist
            dims_to_rename = {}
            if 'x' in ds.dims:
                dims_to_rename['x'] = 'x1'
            if 'y' in ds.dims:
                dims_to_rename['y'] = 'y1'
            
            if dims_to_rename:
                ds = ds.rename(dims_to_rename)
            
            # Fill NaN values with zero in all data variables
            ds = ds.fillna(0)
            
            # Save the modified dataset, overwriting the original file
            ds.to_netcdf(filepath)
            print(f"Updated {filename}: renamed dimensions and filled NaNs with zero.")
        else:
            print(f"Skipped {filename}: not a NetCDF file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename dimensions x,y to x1,y1 in .nc files in a directory.')
    parser.add_argument("-d", "--dir", dest="directory", type=str, help='Path to the directory containing .nc files')

    args = parser.parse_args()

    rename_and_fillna_in_directory(args.directory)