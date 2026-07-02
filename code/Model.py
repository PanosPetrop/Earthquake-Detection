import torch
import torch.nn as nn

class EventDetectionLSTM(nn.Module):
    def __init__(self, input_channels=3, cnn_features=16, hidden_size=64, lstm_layers=2):
        super(EventDetectionLSTM, self).__init__()
        
        # --- 1. The CNN Block ---
        # Εξάγει τοπικά features από την συχνότητα και την φάση αντίστοιχα από τα waveform
        self.cnn = nn.Sequential(
            # Layer 1: Wider kernel (11) to catch broader frequency 
            # Πιο πλατής πυρήνας για να εντοπίζει τις μεγάλες διακυμάνσης της συνχότητας
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_features, kernel_size=15, padding='same'),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            
            # Layer 2: Tighter kernel (7) to refine the features
            # Λεπτότερος πυρήνας για αναδιατυπώσει τα χαρακτηριστικά
            nn.Conv1d(in_channels=cnn_features, out_channels=cnn_features * 2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(cnn_features * 2),
            nn.ReLU()
        )
        
        # --- 2. The LSTM Block ---
        # 
        # Διαβάζει τις ακολουθίες που παρέχει το CNN layer για να αναγωρίσει αν υπάρχει γεγονός
        self.lstm = nn.LSTM(
             input_size=cnn_features * 2, 
            #input_size=input_channels,
            hidden_size=hidden_size, 
            num_layers=2, # 2 layers
            batch_first=True,
            bidirectional=True 
        )
        
        # --- 3.  Output Classifier ---
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Η μορφή του X: (Batch, TimeSteps, Channels)  [32, 3000, 3]
        
        # Conv1d περιμένει (Batch, Channels, TimeSteps)
        # Εναλλαγή των διαστάσεων για το CNNs
        x = x.permute(0, 2, 1) # Shape : [32, 3, 3000]
        
        # Πέρασε μέσα από το CNN
        x = self.cnn(x) # Shape : [32, 32_features, 3000]
        
        # CRITICAL:  LSTM (with batch_first=True) περιμένει (Batch, TimeSteps, Features)
        # Εναλλαγή των διαστάσεων για το LSTM
        x = x.permute(0, 2, 1) # Shape becomes: [32, 3000, 32_features]
        
        # Πέρασε μέσα από το LSTM
        lstm_out, _ = self.lstm(x) # Shape: [32, 3000, hidden_size*2]
        
        # Πέρασε μέσα από το γραμμικό layer για να πάρουμε τα logits
        logits = self.fc(lstm_out) # Shape: [32, 3000, 1]
        
        return logits