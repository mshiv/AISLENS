import os
import xarray as xr
import argparse

def rename_time_dimension(directory):
    # Loop through all .nc files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.nc'):
            filepath = os.path.join(directory, filename)
            
            # Open the dataset
            ds = xr.open_dataset(filepath)
            
            # Rename dimension 'Time' to 'time' if it exists
            if 'Time' in ds.dims:
                ds = ds.rename({'Time': 'time'})
                print(f"Renamed 'Time' to 'time' in {filename}")
                
                # Create a new filename with 'MOD' appended before the extension
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_MOD{ext}"
                new_filepath = os.path.join(directory, new_filename)

                # Save the modified dataset, overwriting the original file
                ds.to_netcdf(new_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename dimension 'Time' to 'time' in .nc files in a directory.")
    parser.add_argument("-d", "--dir", dest="directory", type=str, help='Path to the directory containing .nc files')

    args = parser.parse_args()

    rename_time_dimension(args.directory)