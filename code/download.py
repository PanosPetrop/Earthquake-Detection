import os
from huggingface_hub import hf_hub_download

# Use "." for the current directory, or a specific folder name like "data"
save_dir = "../datasets" 
os.makedirs(save_dir, exist_ok=True)

print("Downloading NCEDC from Hugging Face Repository...")

filenames = ["2018", "2017", "2016", "2015"]

for filename in filenames:
    # Constructing the path within the repo
    repo_filename = f"waveform_h5/{filename}.h5"
    
    h5_path = hf_hub_download(
        repo_id="AI4EPS/quakeflow_nc", 
        repo_type="dataset", 
        filename=repo_filename, 
        local_dir=save_dir
    )
    print(f"Download complete! File saved at: {h5_path}")