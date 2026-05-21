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
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_features, kernel_size=21, padding='same'),
            nn.BatchNorm1d(cnn_features),
            nn.ReLU(),
            
            # Layer 2: Tighter kernel (7) to refine the features
            # Λεπτότερος πυρήνας για αναδιατυπώσει τα χαρακτηριστικά
            nn.Conv1d(in_channels=cnn_features, out_channels=cnn_features * 2, kernel_size=7, padding='same'),
            nn.BatchNorm1d(cnn_features * 2),
            nn.ReLU()
        )
        
        # --- 2. The LSTM Block ---
        # 
        # Διαβάζει τις ακολουθίες που παρέχει το CNN layer για να αναγωρίσει αν υπάρχει γεγονός
        self.lstm = nn.LSTM(
            input_size=cnn_features * 2, 
            hidden_size=hidden_size, 
            num_layers=lstm_layers, # Upgraded to 2 layers
            batch_first=True,
            bidirectional=True 
        )
        
        # --- 3. The Output Classifier ---
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Incoming 'x' shape: (Batch, TimeSteps, Channels) e.g., [32, 3000, 3]
        
        # CRITICAL: PyTorch Conv1d expects (Batch, Channels, TimeSteps)
        # We must swap the dimensions before the CNN
        x = x.permute(0, 2, 1) # Shape becomes: [32, 3, 3000]
        
        # Pass through CNN
        x = self.cnn(x)        # Shape becomes: [32, 32_features, 3000]
        
        # CRITICAL: PyTorch LSTM (with batch_first=True) expects (Batch, TimeSteps, Features)
        # We must swap them back
        x = x.permute(0, 2, 1) # Shape becomes: [32, 3000, 32_features]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x) # Shape: [32, 3000, hidden_size*2]
        
        # Pass through Linear layer to get final boxcar envelope
        logits = self.fc(lstm_out) # Shape: [32, 3000, 1]
        
        return logits