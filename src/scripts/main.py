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
from aislens.config import config
from aislens.utils import collect_directories, initialize_directories

def run_script(script_name):
    subprocess.run(["python", script_name])

def main():
    # Step 0: Initialize all directories
    print("Initializing data directories...")
    dirs_to_create = collect_directories(config)
    initialize_directories(dirs_to_create)
    print("DATA DIRECTORIES INITIALIZED SUCCESSFULLY.")

    # Step 1: Run prepare_data.py
    print("Running prepare_data.py...")
    run_script("prepare_data.py")
    print("prepare_data.py COMPLETED SUCCESSFULLY.")
    
    # Step 2 & 3: Run calculate_draft_dependence.py and generate_forcings.py in parallel
    with ThreadPoolExecutor() as executor:
        print("Running calculate_draft_dependence.py and generate_forcings.py in parallel...")
        executor.submit(run_script, "calculate_draft_dependence.py")
        executor.submit(run_script, "generate_forcings.py")
        print("calculate_draft_dependence.py and generate_forcings.py COMPLETED SUCCESSFULLY.")

    # Step 4: Run regrid_to_mali.py
    print("Running regrid_to_mali.py...")
    run_script("regrid_to_mali.py")
    print("regrid_to_mali.py COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()