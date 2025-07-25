import os
import argparse
import torch.cuda as cuda

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from vogues.loss import LSTMLoss
from vogues.model import OneClassLSTM
from vogues.data import create_data_loader


def train_model(model, data_loader, num_epochs, learning_rate, weight_decay, device, lstm_loss):
    """
    Train the binary classification LSTM model.
    
    Args:
        model (OneClassLSTM): The LSTM model
        data_loader (DataLoader): DataLoader for training data with labels
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for regularization
        device (str): Device to train on ('cuda' or 'cpu')
        lstm_loss (LSTMLoss): Loss function instance
    
    Returns:
        model: Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Add cosine learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Maximum number of iterations
        eta_min=learning_rate * 0.01  # Minimum learning rate (1% of initial)
    )
    
    # Get the dataset from the data loader
    # dataset = data_loader.dataset
    
    # Initialize variables for training statistics
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        correct_predictions = 0
        total_samples = 0
        
        # Regenerate negative samples at the start of each epoch
        # dataset.regenerate_negative_samples()
        
        for batch_sequences, batch_labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to device
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass through the model
            predictions = model(batch_sequences)
            
            # Compute loss
            loss = lstm_loss((predictions, batch_labels), model)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Calculate accuracy
            predicted_labels = (predictions > 0).float()
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_samples += batch_labels.size(0)
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch statistics
        avg_loss = np.mean(epoch_losses)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - loss: {avg_loss:.4e} accuracy: {accuracy:.4f} lr: {current_lr:.2e}")
    
    return model


# ---------------------------
# Example usage with actual data
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train one-class LSTM on keypoint data')
    parser.add_argument('--data_dir', type=str, default='output/ucf101_results', help='Directory containing JSON files with keypoint data')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of LSTM hidden units')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--window_size', type=int, default=20, help='Fixed window size for sequences')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save_path', type=str, default='model_noise_0.01.pt', help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loader
    print(f"Loading data from {args.data_dir}")
    data_loader = create_data_loader(
        args.data_dir, 
        batch_size=args.batch_size,
        window_size=args.window_size,
        shuffle=True
    )
    
    # Instantiate the LSTM model
    model = OneClassLSTM(408, args.hidden_size, args.num_layers).to(device)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model
    print("Starting training...")
    trained_model = train_model(
        model, 
        data_loader, 
        num_epochs=args.num_epochs, 
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        lstm_loss=LSTMLoss()
    )
    
    # Save the trained model
    save_dict = {
        'model_state_dict': trained_model.state_dict(),
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'window_size': args.window_size,
        'input_size': 408,
    }
    torch.save(save_dict, args.save_path)
    print(f"Model saved to {args.save_path}")
    
    print("Training complete!")
    print("To classify sequences, use model(sequence) > 0.5")
    print("If result is True, the sequence is normal; otherwise, it is anomalous.")