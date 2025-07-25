import torch
import torch.nn as nn
import torch.nn.functional as F

class OneClassLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=1):
        """
        LSTM model for anomaly detection in pose sequences.
        
        Args:
            input_size (int): Size of input features (keypoints)
            hidden_size (int): Size of LSTM hidden layer
            num_layers (int): Number of LSTM layers
        """
        super(OneClassLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Forward pass with fixed window size.
        
        Args:
            x: Either a tensor of shape [batch_size, window_size, input_size] for sequences only
               or a tuple of (sequences, labels)
            
        Returns:
            tuple: (predictions, labels) if labels are provided, otherwise predictions only
        """
        # Check if input is a tuple (sequences, labels)
        if isinstance(x, tuple):
            sequences, labels = x
        else:
            sequences = x
            labels = None
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(sequences)
        
        # Extract the output from the last time step
        final_out = lstm_out[:, -1, :]
        
        # Project to single output value
        predictions = self.linear(final_out).squeeze(-1)
        
        # Return predictions and labels if available
        if labels is not None:
            return predictions, labels
        return predictions