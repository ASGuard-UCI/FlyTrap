import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLoss(nn.Module):
    """
    Binary classification loss for LSTM sequence classification.
    """
    def __init__(self):
        super(LSTMLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, model):
        """
        Compute binary classification loss.
        
        Args:
            outputs (torch.Tensor): Model outputs, shape [batch_size]
            model (OneClassLSTM): The LSTM model for weight regularization
            
        Returns:
            tuple: (classification_loss, total_loss)
        """
        # Extract predictions and labels (assuming outputs is a tuple of (predictions, labels))
        predictions, labels = outputs
        
        # Convert labels to float for BCE loss
        labels = labels.float()
        
        # Binary classification loss
        classification_loss = self.bce_loss(predictions, labels)
        
        return classification_loss