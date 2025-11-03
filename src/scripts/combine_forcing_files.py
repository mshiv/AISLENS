import subprocess
import xarray as xr
from pathlib import Path
from aislens.config import config

def combine_forcing_files_nco(file1_path, file2_path, output_path, 
                             time1_start, time1_end, time2_start, time2_end):
    """
    Combine two forcing files using NCO, adding specific timesteps.
    
    Parameters:
    file1_path: Path to first NetCDF file
    file2_path: Path to second NetCDF file  
    output_path: Path for output file
    time1_start, time1_end: Time indices for first file
    time2_start, time2_end: Time indices for second file
    """
    
    # Create temporary files for extracted time slices
    temp_file1 = Path(output_path).parent / "temp_file1.nc"
    temp_file2 = Path(output_path).parent / "temp_file2.nc"
    
    try:
        # Extract time slices
        subprocess.run([
            "ncks", "-O", 
            f"-d", f"Time,{time1_start},{time1_end}",
            str(file1_path), str(temp_file1)
        ], check=True)
        
        subprocess.run([
            "ncks", "-O",
            f"-d", f"Time,{time2_start},{time2_end}", 
            str(file2_path), str(temp_file2)
        ], check=True)
        
        # Add the files
        subprocess.run([
            "ncbo", "-O", "-o", str(output_path),
            "--op_typ=add", str(temp_file1), str(temp_file2)
        ], check=True)
        
        print(f"Successfully combined files and saved to {output_path}")
        
    finally:
        # Clean up temporary files
        if temp_file1.exists():
            temp_file1.unlink()
        if temp_file2.exists():
            temp_file2.unlink()

if __name__ == "__main__":
    # Example usage
    file1 = config.DIR_MALI_ISMIP6_FORCINGS / "ISMIP6_SSP585_UKESM_FLOATINGBMB_TREND.nc" 
    file2 = config.DIR_FORCINGS / "forcing_realization_0.nc"
    output = config.DIR_MALI_ISMIP6_FORCINGS / "combined_forcing.nc"
    
    # Add timesteps 0-50 from file1 with timesteps 10-60 from file2
    combine_forcing_files_nco(file1, file2, output, 0, 50, 10, 60)
