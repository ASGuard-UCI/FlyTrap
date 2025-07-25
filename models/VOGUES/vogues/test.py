import torch
import argparse
import json
import os
from vogues.model import OneClassLSTM
from vogues.data import KeypointDataset


def print_model_parameters(model):
    """
    Print detailed information about the LSTM model parameters.
    
    Args:
        model (OneClassLSTM): The LSTM model
    """
    print("\nModel Parameter Information:")
    print("-" * 50)
    
    # Print LSTM parameters
    for name, param in model.lstm.named_parameters():
        if param.requires_grad:
            # Calculate matrix rank if it's a 2D tensor
            if len(param.shape) == 2:
                rank = torch.matrix_rank(param).item()
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Rank: {rank}")
                print(f"  Full rank: {rank == min(param.shape)}")
                print(f"  Mean: {param.mean().item():.4f}")
                print(f"  Std: {param.std().item():.4f}")
            else:
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {param.mean().item():.4f}")
                print(f"  Std: {param.std().item():.4f}")
    
    # Print linear layer parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 2:
                rank = torch.matrix_rank(param).item()
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Rank: {rank}")
                print(f"  Full rank: {rank == min(param.shape)}")
                print(f"  Mean: {param.mean().item():.4f}")
                print(f"  Std: {param.std().item():.4f}")
            else:
                print(f"\n{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {param.mean().item():.4f}")
                print(f"  Std: {param.std().item():.4f}")
    
    print("-" * 50)


def create_corrupted_sequences(json_path, window_size=10, num_corruptions=10):
    """
    Create corrupted sequences from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        window_size (int): Size of each sequence window
        num_corruptions (int): Number of corrupted sequences to create
    
    Returns:
        torch.Tensor: Tensor of corrupted sequences
    """
    # Load and process the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group data by track ID (idx)
    tracks = {}
    for entry in data:
        idx = entry.get('idx')
        if idx is not None:
            if idx not in tracks:
                tracks[idx] = []
            tracks[idx].append(entry)
    
    # Process the first track
    if not tracks:
        raise ValueError("No tracks found in the JSON file")
    
    track_data = list(tracks.values())[0]  # Get the first track
    if all('image_id' in entry for entry in track_data):
        track_data.sort(key=lambda x: int(x['image_id'].split('.')[0]))
    
    # Extract and normalize keypoints
    full_sequence = []
    for entry in track_data:
        # Normalize keypoints using image shape
        height, width, _ = entry['img_shape']
        normalized = []
        for i in range(0, len(entry['keypoints']), 3):
            if i + 2 < len(entry['keypoints']):
                x = entry['keypoints'][i] / width
                y = entry['keypoints'][i+1] / height
                score = entry['keypoints'][i+2]
                normalized.extend([x, y, score])
        full_sequence.append(normalized)
    
    # Convert to tensor
    full_sequence = torch.tensor(full_sequence, dtype=torch.float32)
    
    # Create corrupted sequences
    corrupted_sequences = []
    for i in range(num_corruptions):
        # Get a window of the original sequence
        start_idx = i % (len(full_sequence) - window_size + 1)
        sequence = full_sequence[start_idx:start_idx + window_size].clone()
        
        # Corrupt the sequence with random noise
        noise = torch.rand_like(sequence)  # Random noise in [0, 10]
        sequence = noise  # Replace the entire sequence with noise
        
        corrupted_sequences.append(sequence)
    
    return torch.stack(corrupted_sequences)


def create_benign_sequences(json_path, window_size=10, num_sequences=10):
    """
    Create benign sequences from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        window_size (int): Size of each sequence window
        num_sequences (int): Number of benign sequences to create
    
    Returns:
        torch.Tensor: Tensor of benign sequences
    """
    # Load and process the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group data by track ID (idx)
    tracks = {}
    
    for entry in data:
        idx = entry.get('idx')
        if idx is not None:
            if idx not in tracks:
                tracks[idx] = []
            tracks[idx].append(entry)

    # Process each track
    benign_sequences = []
    for track_data in tracks.values():
        # Sort by image_id if available
        if all('image_id' in entry for entry in track_data):
            track_data.sort(key=lambda x: int(x['image_id'].split('.')[0]))
        
        # Extract and normalize keypoints
        full_sequence = []
        for entry in track_data:
            # Normalize keypoints using image shape
            height, width, _ = entry['img_shape']
            normalized = []
            for i in range(0, len(entry['keypoints']), 3):
                if i + 2 < len(entry['keypoints']):
                    x = entry['keypoints'][i] / width
                    y = entry['keypoints'][i+1] / height
                    score = entry['keypoints'][i+2]
                    normalized.extend([x, y, score])
            full_sequence.append(normalized)
        
        # Skip if sequence is too short
        if len(full_sequence) < window_size:
            continue
            
        # Convert to tensor
        full_sequence = torch.tensor(full_sequence, dtype=torch.float32)
        
        # Create sequences using sliding windows
        for i in range(min(num_sequences, len(full_sequence) - window_size + 1)):
            sequence = full_sequence[i:i + window_size].clone()
            benign_sequences.append(sequence)
            
            if len(benign_sequences) >= num_sequences:
                break
                
        if len(benign_sequences) >= num_sequences:
            break
    
    if not benign_sequences:
        raise ValueError("No valid sequences could be created from the data")
        
    
    return torch.stack(benign_sequences[:num_sequences])


def create_random_sequences(input_size, window_size=10, num_sequences=10):
    """
    Create fully random sequences with values between [0, 1].
    
    Args:
        input_size (int): Size of each input vector
        window_size (int): Size of each sequence window
        num_sequences (int): Number of random sequences to create
    
    Returns:
        torch.Tensor: Tensor of random sequences
    """
    return torch.rand(num_sequences, window_size, input_size)


def test_model(model_path, json_path):
    """
    Test the trained model with corrupted sequences from a JSON file and random sequences.
    
    Args:
        model_path (str): Path to the saved model
        json_path (str): Path to the JSON file to create corrupted sequences from
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same parameters
    model = OneClassLSTM(
        input_size=408,
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Get the rho value and move to device
    rho = 0.5
    
    # Print model parameter information
    print_model_parameters(model)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create corrupted sequences
    print(f"\nCreating corrupted sequences from: {json_path}")
    corrupted_sequences = create_corrupted_sequences(json_path, window_size=20, num_corruptions=10)
    corrupted_sequences = corrupted_sequences.to(device)
    
    # Create random sequences
    print("\nCreating random sequences")
    random_sequences = create_random_sequences(
        input_size=408,
        window_size=20,
        num_sequences=10
    ).to(device)
    
    # Create benign sequences
    print("\nCreating benign sequences from: {json_path}")
    benign_sequences = create_benign_sequences(json_path, window_size=20, num_sequences=10)
    benign_sequences = benign_sequences.to(device)
    
    # Get predictions for corrupted sequences
    with torch.no_grad():
        corrupted_outputs = torch.sigmoid(model(corrupted_sequences))
        random_outputs = torch.sigmoid(model(random_sequences))
        benign_outputs = torch.sigmoid(model(benign_sequences))
    
    # Apply decision function
    corrupted_decisions = torch.sign(corrupted_outputs - rho)
    random_decisions = torch.sign(random_outputs - rho)
    benign_decisions = torch.sign(benign_outputs - rho)
    # Print results
    print("\nTest Results:")
    print("-" * 50)
    print(f"Model loaded from: {model_path}")
    print(f"Window size: 10")
    print(f"Input size: 408")
    print("-" * 50)
    
    print("\nCorrupted Sequences Results:")
    print("-" * 50)
    for i in range(10):
        status = "Normal" if corrupted_decisions[i] >= 0 else "Anomalous"
        print(f"Corrupted Sequence {i+1}:")
        print(f"  Output: {corrupted_outputs[i].item():.4f}")
        print(f"  Decision: {status}")
        print("-" * 50)
    
    print("\nRandom Sequences Results:")
    print("-" * 50)
    for i in range(10):
        status = "Normal" if random_decisions[i] >= 0 else "Anomalous"
        print(f"Random Sequence {i+1}:")
        print(f"  Output: {random_outputs[i].item():.4f}")
        print(f"  Decision: {status}")
        print("-" * 50)
        
    print("\nBenign Sequences Results:")
    print("-" * 50)
    for i in range(10):
        status = "Normal" if benign_decisions[i] >= 0 else "Anomalous"
        print(f"Benign Sequence {i+1}:")
        print(f"  Output: {benign_outputs[i].item():.4f}")
        print(f"  Decision: {status}")
        print("-" * 50)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test one-class LSTM model with corrupted and random sequences')
    parser.add_argument('--model_path', type=str, default='model.pt', help='Path to the saved model')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the JSON file to create corrupted sequences from')
    
    args = parser.parse_args()
    
    test_model(args.model_path, args.json_path)
