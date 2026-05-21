import h5py

def print_structure(name, obj):
    print(name)

file_path = 'datasets/waveform_h5/2015.h5'

with h5py.File(file_path, 'r') as f:
    # This will recursively visit every group and dataset
    print("Keys στο αρχείο:", list(f.keys()))

    dataset = f['nc72572906']['NN.POC..EH']

    print(f"Name of dataset : {dataset.name} ")
    print(f"Shape of dataset : {dataset.shape} ")
    print(f"Attr items : {dataset.attrs['p_phase_score']} ")
    print(f"Attr keys : {dataset.attrs.keys()} ")

    print(f"Attr keys : {dataset.attrs['phase_type']} ")

    arr = dataset.attrs['phase_type']

    if arr[0] == 'P' and arr[1] == 'S': 
        print('love u') 