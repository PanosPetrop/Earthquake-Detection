from Dataset import create_dataloaders
from Model import *
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import torch

def create_boxcar_targets(picks, window_size=3000, coda_multiplier=1.5):
    """
    Δημιουργούμε boxcar targets για την ανίχνευση ανωμαλιών βασισμένοι στα διαμορφωμένα P and S waves.
    
        picks: Είναι της μορφής (batch_size, 2) όπου 
               picks[:, 0] = P-wave index, picks[:, 1] = S-wave index.
        window_size: Όσα και το παράθυρο που δημιουρήθηκε στο Dataset.py
        coda_multiplier: Ελέγχει το θα διαρκέσει το event μετά το S wave σε σχέση με την διάρκεια του (S - P) 
                         
    Επιστρέφει
        targets: Επιστρέφει έναν tensor (batch_size, window_size, 1) οπου θα περιέχει 1 κατα την διάρκεια του σεισμούς και
        0 σε περιόδους ηρεμίας
    """
    B = picks.shape[0] 
    #print(B)
    
    # 1. Προετοιμασία των waves για broadcast
    p_picks = picks[:, 0].unsqueeze(1) 
    s_picks = picks[:, 1].unsqueeze(1) 
    #print(f" p picks {p_picks}")
    
    # 2. Υπολογισμός διάρκειας σεισμούς παρατήνοντας τον event και μετά το s_wave με την βοήθεια του coda_multiplier
    p_to_s_duration = s_picks - p_picks
    event_end = s_picks + ( coda_multiplier * p_to_s_duration)
    
    # 3. Δημιουργία window_size χρονικά βήματα
    time_axis = torch.arange(window_size, device=picks.device).unsqueeze(0)
    
    # 4. Δημιουργία boolean mask: 
    # Γυρίζει true (1) μόνο όταν το time_axis βρίσκεται ανάμεσα στο p_picks και το evend_end
    mask = (time_axis >= p_picks) & (time_axis < event_end)
    
    # 5. boolean σε float
    targets = mask.float()
    
    # 6. Προετοιμασία για LSTM και  Loss Function
    targets = targets.unsqueeze(-1) # Final shape: (B, window_size, 1)
    # print(f"traget:{targets}")
    # print(f'1 {targets.shape[0]}')
    # print(f'2 {targets[0].shape[0]}')
    # print(f'3 {targets[0][0].shape[0]}')
    return targets

import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_detection_results(waveform, target, prediction, sample_rate=100, sample_idx=0):
    """
    Plots the 3-component seismic waveform alongside the target boxcar and model predictions.
    Saves the output to a PNG file.
    """
    
    # 1. Εναλλαγή tensor σε numpy 
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(prediction):
        # Για το prediction, εφαρμόζουμε sigmoid για να πάρουμε πιθανότητες 0 εως 1
        prediction = torch.sigmoid(prediction).detach().cpu().numpy() 

    # 2. Εξάγετε το συγκεκριμένο δείγμα από την παρτίδα χρησιμοποιώντας το sample_idx
    if waveform.ndim == 3: 
        waveform = waveform[sample_idx]
    if target.ndim == 3: 
        target = target[sample_idx]
    if prediction.ndim == 3: 
        prediction = prediction[sample_idx]
        
    # 3. κάνε τα target και prediction 1D arrays για το plotting
    target = target.squeeze()
    prediction = prediction.squeeze()

    # Φτιάξε τον άξονα του χρόνου με βάση το sample_rate και το μήκος του waveform
    time_steps = waveform.shape[0]
    time_axis = np.arange(time_steps) / sample_rate

    # 4. Δημιουργία του plot με δύο υπο-διαγράμματα (subplots)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)

    
    ax1.plot(time_axis, waveform[:, 0], color='gray', alpha=0.6, label='Channel E')
    ax1.plot(time_axis, waveform[:, 1], color='blue', alpha=0.4, label='Channel N')
    ax1.plot(time_axis, waveform[:, 2], color='black', alpha=0.8, linewidth=1.2, label='Channel Z')
    
    ax1.set_title("Seismic Waveform", fontsize=14, pad=10)
    ax1.set_ylabel("Amplitude", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

   
    ax2.fill_between(time_axis, 0, target, color='lightgray', alpha=0.5, label='Ground Truth Boxcar')
    ax2.plot(time_axis, target, color='black', linestyle='--', linewidth=1.5)
    
    ax2.plot(time_axis, prediction, color='red', linewidth=2, label='Model Prediction (Probability)')
    ax2.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, label='0.5 Threshold')

    ax2.set_title("Event Detection", fontsize=14, pad=10)
    ax2.set_xlabel("Time (seconds)", fontsize=12)
    ax2.set_ylabel("Probability", fontsize=12)
    ax2.set_ylim(-0.05, 1.05) 
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)


    plt.tight_layout(pad=2.0, h_pad=2.0)
    plt.savefig(f"../test_results/test_{sample_idx}.png")
    plt.close(fig) 

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    batch_size = 32
    random_seed = 10
    num_epochs = 10
    
    model = EventDetectionLSTM().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    H5_PATH = "../datasets/waveform_h5/merged_bigger.hdf5" #DATASET PATH
    train_loader, val_loader, test_loader = create_dataloaders(H5_PATH, batch_size=batch_size, random_seed=random_seed)
    #print(train_loader)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            #print(batch)
            waveforms, picks = batch
            # debugging
            # print(f"batch are: ", batch)
            # print(f"waveforms are: ", waveforms)
            # print(f"waveform leagth: {waveforms[0][0].shape[0]}")
            # print(f"picks are: ", picks)
            # print(f"picks leagth: {picks.shape[0]}")
            waveforms, picks = waveforms.to(device), picks.to(device)
            waveforms = waveforms.permute(0, 2, 1)
            optim.zero_grad()
            logits = model(waveforms)
            
            # Φτιάξε τα targets με την συνάρτηση create_boxcar_targets και υπολόγισε το loss
            targets = create_boxcar_targets(picks)
            targets = targets.to(device)
            
            loss = criterion(logits, targets)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
           
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        total_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                waveforms, picks = batch
                waveforms, picks = waveforms.to(device), picks.to(device)
                waveforms = waveforms.permute(0, 2, 1)
                logits = model(waveforms)
                
                targets = create_boxcar_targets(picks)
                targets = targets.to(device)
                
                val_loss = criterion(logits, targets)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Average Validation Loss: {avg_val_loss:.4f}")
    
    
    # Τώρα που το μοντέλο έχει εκπαιδευτεί, ας το αξιολογήσουμε στο test set και να υπολογίσουμε τις μετρικές 
    model.eval()
    total_test_loss = 0.0

    all_targets = []
    all_preds = []
    
    with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
                waveforms, picks = batch
                waveforms, picks = waveforms.to(device), picks.to(device)
                waveforms = waveforms.permute(0, 2, 1)
                
                logits = model(waveforms)
                targets = create_boxcar_targets(picks).to(device)
                
                # Υπολογισμός Loss
                test_loss = criterion(logits, targets)
                total_test_loss += test_loss.item()

                # Μετατροπή logits σε binary predictions (0 ή 1)
                # Χρησιμοποιούμε sigmoid και κατώφλι 0.5
                probs = torch.sigmoid(logits)
                #print(probs)
                preds = (probs > 0.5).float()

                # Μετατροπή σε numpy και flatten για τις μετρικές
                all_targets.append(targets.cpu().numpy().flatten())
                all_preds.append(preds.cpu().numpy().flatten())

                # Plotting για τα πρώτα δείγματα
                if i < 5: 
                    plot_detection_results(waveforms, targets, logits, sample_idx=0) # sample_idx=0 για το πρώτο του batch
                    
        # Συγκέντρωση όλων των αποτελεσμάτων σε δύο μεγάλα arrays
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Υπολογισμός Μετρικών
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    avg_test_loss = total_test_loss / len(test_loader)

    print("\n" + "="*30)
    print("TEST SET RESULTS")
    print(f"Average Loss: {avg_test_loss:.4f}")
    print(f"Precision:    {precision:.4f}  (Πόσο σίγουρο είναι το μοντέλο όταν βρίσκει σεισμό)")
    print(f"Recall:       {recall:.4f}  (Πόσο ποσοστό των σεισμών όντως εντόπισε)")
    print(f"F1-Score:     {f1:.4f}  (Ισορροπία μεταξύ των δύο)")
    print("="*30)
                
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    

    