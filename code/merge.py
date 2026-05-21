import os
import h5py
import glob
from tqdm import tqdm

def merge_hdf5_files(target_dir, output_filename="merge.hdf5"):
    output_path = os.path.join(target_dir, output_filename)
    
    # 1. Find all HDF5 files in the directory
    search_pattern = os.path.join(target_dir, "*.h5*")
    all_files = glob.glob(search_pattern)
    
    # Remove the target merge file from the list if it already exists
    if output_path in all_files:
        all_files.remove(output_path)
        
    if not all_files:
        print(f"No .h5 or .hdf5 files found in {target_dir}")
        return

    print(f"Found {len(all_files)} HDF5 files to merge.")
    
    # 2. Open the new master file in 'append' mode ('a') or 'write' mode ('w')
    with h5py.File(output_path, 'w') as h5_out:
        
        # 3. Iterate through each file
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            
            with h5py.File(file_path, 'r') as h5_in:
                events = list(h5_in.keys())
                
                # Copy every event (earthquake) into the master file
                for event_id in tqdm(events, desc=f"Merging {file_name}"):
                    
                    # Safety check: Prevent crashing if two years share an event ID
                    if event_id not in h5_out:
                        h5_in.copy(event_id, h5_out)
                    else:
                        # If the event exists, we merge the stations inside it
                        for station_id in h5_in[event_id].keys():
                            if station_id not in h5_out[event_id]:
                                h5_in[event_id].copy(station_id, h5_out[event_id])

    print(f"\nAll files successfully merged into: {output_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Point this to the folder containing your 2019.h5, 2020.h5, etc.
    DATA_DIRECTORY = "../datasets/waveform_h5/"
    
    merge_hdf5_files(DATA_DIRECTORY, output_filename="merged_bigger.hdf5")