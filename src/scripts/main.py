# This script executes the entire workflow of the project and is the main entry point.
# It imports all the necessary modules and functions from the src package
# and executes them in the correct order.
# STEPS:
#   1. Prepare data using prepare_data.py
#   2. Calculate draft dependence using calculate_draft_dependence.py
#   3. Create forcing files using generate_forcings.py
#   4. Regrid forcings and draft dependence to the MALI grid using regrid_to_mali.py

import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_script(script_name):
    subprocess.run(["python", script_name])

def main():
    # Step 1: Run prepare_data.py
    run_script("prepare_data.py")
    
    # Step 2 & 3: Run calculate_draft_dependence.py and generate_forcings.py in parallel
    with ThreadPoolExecutor() as executor:
        executor.submit(run_script, "calculate_draft_dependence.py")
        executor.submit(run_script, "generate_forcings.py")
    
    # Step 4: Run regrid_to_mali.py
    run_script("regrid_to_mali.py")

if __name__ == "__main__":
    main()