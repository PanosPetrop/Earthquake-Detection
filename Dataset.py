

import torch
import h5py
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
class NCEDCDataset(Dataset):
    def __init__(self, h5_file, mode="train", window_size=3000, trace_list=None):
        self.h5_file = h5_file
        self.mode = mode
        self.window_size = window_size
        self.h5 = None 
        
        
        if trace_list is not None:
            self.valid_traces = trace_list
        else:
            self.valid_traces = self._scan_file() 

    # Μαζεύει όλα τα P και S waves με τα αντίστοιχα indexes (και απο διαφορετικές οπτικές - αισθητήρες) σε μια μεγάλη λίστα 
    def _scan_file(self):
        traces = []
        print(f"🔍 Scanning {self.h5_file} for valid P and S picks...")
        with h5py.File(self.h5_file, 'r') as h5_in:
            # Λουπαρε για κάθε event
            for event_id in tqdm(h5_in.keys(), desc="Indexing HDF5"): 
                # Πάρε το group του event με βάση το event id
                event_group = h5_in[event_id] 
                # Λουπαρε για κάθε μέτρησης σταθμού για το συγκεκριμένο event
                for station_id in event_group.keys(): 
                    # παρε τα χαρακτηριστικά του events (atrributes)
                    attrs = event_group[station_id].attrs 
                     # αν υπάρχει το phase_type και το phase_index ως attributes
                    if 'phase_type' in attrs and 'phase_index' in attrs:
                        # ελέγχουμε αν τα δεδομένα είναι σε μορφή byte και τότε τα μετατρέπουμε σε utf-8
                        p_types = [p.decode('utf-8') if isinstance(p, bytes) else p for p in attrs['phase_type']] 
                        p_indices = attrs['phase_index'] 
                        # αν στο event υπάρχει και P και S wave τότε 
                        if 'P' in p_types and 'S' in p_types: 
                            # Βρές τα αντίστοιχα indexes για το P και το S wave και κανε τα python int
                            p_idx = int(p_indices[p_types.index('P')]) 
                            s_idx = int(p_indices[p_types.index('S')]) 
                            # βαλε το event στην λίστα
                            traces.append((event_id, station_id, p_idx, s_idx)) 

                           
        return traces

    # πλήθος των έγκυρων events
    def __len__(self):
        return len(self.valid_traces)

    # συνάρτηση μετασχηματισμού των events σε παράθυρο 3000 χρονικών events 
    def __getitem__(self, idx):
        # Lazy loading H5 για πολυεπεξεργαστική συμβατότητα
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_file, 'r', swmr=True)

        # εξαγωγή των χαρακτηριστικών του event με ID -> idx
        event_id, station_id, p_pick, s_pick = self.valid_traces[idx]
        # εξαγωγή όλόκληρου του waveform για x,y,z
        raw_waveform = self.h5[event_id][station_id][:]
        
        # Σιγουρεύουμε οτι ειναι σε γραμμές -> κανάλι, στήλες -> χρόνος
        if raw_waveform.shape[0] != 3:
            raw_waveform = raw_waveform.T
        # πλήθος χρονικών στιγμών    
        raw_length = raw_waveform.shape[1]

        
        if self.mode == "train":
            # Σε training mode το P-wave θα βρίσκεται σε κάποια απο τις χρονικές στιγμές 500 έως self.window_size - 500
            offset = random.randint(500, self.window_size - 500)
        else:
            # Σε testing mode το P-wave θα  είναι πάντα στην χρονική στιγμή 1000
            offset = 1000

        # Υπολιγμσός θέσης παραθύρου σε σχέση με τα αρχικά δεδομένα
        start_idx = p_pick - offset
        end_idx = start_idx + self.window_size

        # Δημιουργία παραθύρου με μηδενικές τιμές
        cropped_waveform = np.zeros((3, self.window_size), dtype=np.float32)

        # Οριοθέτηση παραθύρου για να βρίσκονται οι τιμές μέσα στα όρια
        data_start = max(0, start_idx)
        data_end = min(raw_length, end_idx)
        
        crop_start = max(0, -start_idx)
        crop_end = crop_start + (data_end - data_start)

        if data_end > data_start:
            cropped_waveform[:, crop_start:crop_end] = raw_waveform[:, data_start:data_end]

        # Υπολογιμσός για το πού βρίσκονται τα p και s waves μέσα στο νεο παράθυρο
        new_p_pick = p_pick - start_idx
        new_s_pick = s_pick - start_idx

        # Κανονικοποίηση τιμών με z-score
        for c in range(3):
            valid_slice = cropped_waveform[c, crop_start:crop_end]
            if len(valid_slice) > 0:
                mean_val = np.mean(valid_slice)
                std_val = np.std(valid_slice)
                if std_val > 1e-6:
                    cropped_waveform[c] = (cropped_waveform[c] - mean_val) / std_val
                else:
                    cropped_waveform[c] = cropped_waveform[c] - mean_val
        #print(cropped_waveform)
        # Αν το s-wave βγει εκτός ορίων τότε το κανουμε mask
        if new_s_pick >= self.window_size or new_s_pick < 0:
            new_s_pick = -1.0

        return torch.tensor(cropped_waveform, dtype=torch.float32), \
               torch.tensor([new_p_pick, new_s_pick], dtype=torch.float32)


from torch.utils.data import DataLoader
import random

# Bασική συνάρτηση που καλείται απο την main, διαμορφώνει το datset σε train, validation και test
def create_dataloaders(h5_path, batch_size=16, splits=(0.8, 0.1, 0.1), random_seed=42):
    # 1. Αρχική ανάγνωση του dataset
    temp_ds = NCEDCDataset(h5_path)
    # μάζεψε όλα τα αποδεκτά events
    all_traces = temp_ds.valid_traces 
    
    # 2. Επειδή στο all_traces υπάρχουν ίδια events απο διαφορετικούς αισθητήρες φτιάχνουμε ενα dictionary με βάση το event id
    events_dict = {}
    print('------------------------------------------------------------------------------')
    for trace in tqdm (all_traces, desc= 'Creating a dictionary based on unique events'):
        event_id = trace[0]
        if event_id not in events_dict:
            events_dict[event_id] = []
        events_dict[event_id].append(trace)
    
    # λίστα με διακριτά event ids
    unique_events = list(events_dict.keys())
    #print(unique_events) 
    
    
    # 3. Ταξινόμηση των events χρονολογικά βασιζόμενοι στo event id
    unique_events.sort() 
    
    # 4. Διαχωρισμός του dataset
    total_events = len(unique_events)
    # 0...split[0]%  - 1 training
    train_end = int(total_events * splits[0])
    # split[0]% + split[1] - 1 validation 
    val_end = train_end + int(total_events * splits[1])
    
    train_events = unique_events[:train_end]
    val_events = unique_events[train_end:val_end]
    test_events = unique_events[val_end:]
   
    # 5. Φτιάχνουμε λιστες που έχουν κάθε event ξεχωριστά χωρις leak
    train_traces = [trace for ev in train_events for trace in events_dict[ev]]
    val_traces = [trace for ev in val_events for trace in events_dict[ev]]
    test_traces = [trace for ev in test_events for trace in events_dict[ev]]

    # εκ νέου τυχαιοποίηση των events χωρις leak
    random.seed(random_seed)
    random.shuffle(train_traces)
    random.shuffle(val_traces)
    random.shuffle(test_traces)
    
    
    print("-" * 60)
    print("DATASET SPLIT BY EVENT (STRICT TEMPORAL ISOLATION)")
    print("-" * 60)
    print(f"Training Events    : {len(train_events):,} (Total Traces: {len(train_traces):,})")
    print(f"  Event ID Range   : {train_events[0]}  --->  {train_events[-1]}")
    print("-" * 60)
    print(f"Validation Events  : {len(val_events):,} (Total Traces: {len(val_traces):,})")
    print(f"  Event ID Range   : {val_events[0]}  --->  {val_events[-1]}")
    print("-" * 60)
    print(f"Test Events        : {len(test_events):,} (Total Traces: {len(test_traces):,})")
    print(f"  Event ID Range   : {test_events[0]}  --->  {test_events[-1]}")
    print("-" * 60)
    
    # 6. Δημιουργία αντικειμένων NCEDCDataset τα οποία θα έχουν ως valid_traces τα ήδη διαμορωμένα από την create_dataloaders
    train_ds = NCEDCDataset(h5_path, mode="train", trace_list=train_traces)
    val_ds = NCEDCDataset(h5_path, mode="val", trace_list=val_traces)
    test_ds = NCEDCDataset(h5_path, mode="test", trace_list=test_traces)
    
    # 7. Δημιουργία dataloaders απο την torch για αποδοτικότερη διαχείριση των δεδομένων κατα την εκμάθηση
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# if __name__ == "__main__":

    #debugging

    # batch_size = 32
    # random_seed = 10
    # num_epochs = 10


    # H5_PATH = "datasets/waveform_h5/merged_bigger.hdf5" #DATASET PATH
    # train_loader, val_loader, test_loader = create_dataloaders(H5_PATH, batch_size=batch_size, random_seed=random_seed)

    # print(f' train loader: {train_loader}')
    # print(f' val loader: {val_loader}')
    # print(f' test loader: {test_loader}')