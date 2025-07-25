import os
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader


class KeypointDataset(Dataset):
    def __init__(self, data_dir, window_size=10):
        """
        Dataset for loading keypoint sequences from JSON files with fixed window size.
        Also generates negative samples by randomly concatenating frames from different sequences.
        
        Args:
            data_dir (str): Directory containing JSON files with keypoint data
            window_size (int): Fixed window size for sequences
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.sequences = []
        self.labels = []
        self.all_frames = []  # Store all frames for negative sample generation
        self.positive_sequences = []  # Store original positive sequences
        
        # Process all JSON files in the directory
        self._load_data()
        
    def regenerate_negative_samples(self):
        """Regenerate all negative samples while keeping positive samples unchanged."""
        # Clear current sequences and labels
        self.sequences = self.positive_sequences.copy()
        self.labels = [1] * len(self.positive_sequences)
        
        # Generate new negative samples
        self._generate_negative_samples(self.all_frames)
        
        # Print statistics
        num_positive_samples = sum(self.labels)
        num_negative_samples = len(self.labels) - num_positive_samples
        print(f"Regenerated negative samples. Total samples: {len(self.sequences)} (Positive: {num_positive_samples}, Negative: {num_negative_samples})")
    
    def _load_data(self):
        """Load and preprocess all JSON files in the data directory using sliding windows."""
        
        subfolders = os.listdir(self.data_dir)
        json_files = []
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.data_dir, subfolder)
            if os.path.isdir(subfolder_path):
                json_files.extend([os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith('.json')])
            
        # Collect all normalized keypoint frames
        self.all_frames = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Group data by track ID (idx)
                tracks = {}
                for entry in data:
                    idx = entry.get('idx')
                    if idx is not None:
                        if idx not in tracks:
                            tracks[idx] = []
                        tracks[idx].append(entry)
                
                # Process each track as a separate sequence
                for idx, track_data in tracks.items():
                    # Sort by image_id if available to ensure temporal order
                    if all('image_id' in entry for entry in track_data):
                        track_data.sort(key=lambda x: int(x['image_id'].split('.')[0]))
                    
                    # Extract and normalize keypoints
                    full_sequence = []
                    for entry in track_data:
                        # Normalize keypoints using image shape
                        norm_keypoints = self._normalize_keypoints(
                            entry['keypoints'], 
                            entry['img_shape']
                        )
                        full_sequence.append(norm_keypoints)
                        self.all_frames.append(norm_keypoints)  # Add to all frames for negative sample generation
                    
                    # Skip if sequence is shorter than window_size
                    if len(full_sequence) < self.window_size:
                        continue
                    
                    # Create sliding windows for positive samples
                    for i in range(len(full_sequence) - self.window_size + 1):
                        window = full_sequence[i:i+self.window_size]
                        self.sequences.append(torch.tensor(window, dtype=torch.float32))
                        self.labels.append(1)  # 1 for positive samples
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Store original positive sequences
        self.positive_sequences = self.sequences.copy()
        
        # Generate negative samples if all_frames is not empty
        if self.all_frames:
            self._generate_negative_samples(self.all_frames)
            
            num_positive_samples = sum(self.labels)
            num_negative_samples = len(self.labels) - num_positive_samples
            print(f"Total samples: {len(self.sequences)} (Positive: {num_positive_samples}, Negative: {num_negative_samples})")
    
    def _generate_negative_samples(self, all_frames):
        """Generate negative samples using different methods."""
        num_positive_samples = len(self.sequences)
        num_negative_samples = num_positive_samples
        
        # Define negative sample generation methods
        negative_generators = [
            self._generate_random_frame_sequence,
            self._generate_reduced_confidence_sequence,
            self._generate_shuffled_sequence,
            self._generate_pure_random_sequence,
            self._generate_shuffled_coordinates_sequence,
            self._generate_random_concat_sequence
        ]
        
        num_methods = len(negative_generators)
        samples_per_method = num_negative_samples // num_methods
        
        print(f"Generating {num_negative_samples} negative samples using {num_methods} methods...")
        
        # Generate samples for each method
        for generator in negative_generators:
            for _ in range(samples_per_method):
                sequence, label = generator(all_frames, num_positive_samples)
                self.sequences.append(sequence)
                self.labels.append(label)
        
        # Handle any remaining samples due to division
        remaining = num_negative_samples - (samples_per_method * num_methods)
        for i in range(remaining):
            generator = negative_generators[i % num_methods]
            sequence, label = generator(all_frames, num_positive_samples)
            self.sequences.append(sequence)
            self.labels.append(label)
    
    def _generate_random_frame_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample by randomly sampling frames."""
        random_frames = []
        for _ in range(self.window_size):
            random_frame = random.choice(all_frames)
            random_frames.append(random_frame)
        
        return torch.tensor(random_frames, dtype=torch.float32), 0
    
    def _generate_reduced_confidence_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample by reducing confidence scores in a benign sequence."""
        # Sample a random sequence from the positive samples
        start_idx = random.randint(0, num_positive_samples - 1)
        benign_sequence = self.sequences[start_idx].clone()
        
        # Randomly choose a point to start the anomaly
        anomaly_start = random.randint(1, self.window_size - 1)
        
        # Create anomalous sequence by reducing confidence scores
        anomalous_sequence = benign_sequence.clone()
        for i in range(anomaly_start, self.window_size):
            # Reduce confidence scores to a small fraction of original
            # anomalous_sequence[i, 0::3] += random.uniform(0.01, 0.05)
            # anomalous_sequence[i, 1::3] += random.uniform(0.01, 0.05)
            anomalous_sequence[i, 2::3] *= random.uniform(0.01, 0.3)
        
        return anomalous_sequence, 0
    
    def _generate_shuffled_coordinates_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample by shuffling x,y coordinates while keeping scores in order."""
        # Sample a random sequence from the positive samples
        start_idx = random.randint(0, num_positive_samples - 1)
        benign_sequence = self.sequences[start_idx].clone()
        anomalous_sequence = benign_sequence.clone()
        
        # Get the first frame to determine the number of keypoints
        first_frame = benign_sequence[0]
        num_keypoints = len(first_frame) // 3
        
        if num_keypoints > 0:
            # Create a single permutation for all frames
            xy_flat = torch.zeros(num_keypoints * 2)  # Just to get the length
            shuffle_idx = torch.randperm(len(xy_flat))
            
            # Process each frame with the same shuffle order
            for i in range(self.window_size):
                # Get the current frame
                frame = benign_sequence[i]
                
                # Reshape to separate x, y, and score
                keypoints = frame[:num_keypoints*3].view(num_keypoints, 3)
                
                # Extract x,y coordinates and scores
                xy_coords = keypoints[:, :2].clone()  # Get all x,y pairs
                scores = keypoints[:, 2].clone()      # Get all scores
                
                # Flatten and shuffle x,y coordinates using the same permutation
                xy_flat = xy_coords.reshape(-1)
                xy_shuffled = xy_flat[shuffle_idx].reshape(num_keypoints, 2)
                
                # Create new keypoints with shuffled coordinates and original scores
                new_keypoints = torch.cat([xy_shuffled, scores.unsqueeze(1)], dim=1)
                
                # Update the frame in the sequence
                anomalous_sequence[i, :num_keypoints*3] = new_keypoints.reshape(-1)
        
        return anomalous_sequence, 0

    
    def _generate_shuffled_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample by replacing one frame with random noise."""
        # Sample a random sequence from the positive samples
        start_idx = random.randint(0, num_positive_samples - 1)
        benign_sequence = self.sequences[start_idx].clone()
        
        # Randomly choose a frame to replace with noise
        noise_frame_idx = random.randint(0, self.window_size - 1)
        
        # Generate random noise with same shape as a frame
        noise = torch.rand_like(benign_sequence[0])
        
        # Replace the chosen frame with noise
        anomalous_sequence = benign_sequence.clone()
        anomalous_sequence[noise_frame_idx] = noise
        
        return anomalous_sequence, 0

    def _generate_pure_random_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample using pure random noise between [0, 1]."""
        # Generate random noise with same shape as a frame
        noise_sequence = torch.rand(self.window_size, len(all_frames[0]))
        
        return noise_sequence, 0
    
    def _generate_random_concat_sequence(self, all_frames, num_positive_samples):
        """Generate a negative sample by concatenating two random sequences at a random point."""
        # Sample two random sequences from the positive samples
        seq1_idx = random.randint(0, num_positive_samples - 1)
        seq2_idx = random.randint(0, num_positive_samples - 1)
        
        # Get the sequences
        seq1 = self.sequences[seq1_idx].clone()
        seq2 = self.sequences[seq2_idx].clone()
        
        # Randomly choose a concatenation point (between 1 and window_size-1)
        concat_point = random.randint(1, self.window_size - 1)
        
        # Create new sequence by concatenating first part of seq1 and second part of seq2
        anomalous_sequence = torch.cat([
            seq1[:concat_point],
            seq2[concat_point:]
        ], dim=0)
        
        return anomalous_sequence, 0


    
    def _normalize_keypoints(self, keypoints, img_shape):
        """
        Normalize keypoint coordinates using image dimensions.
        
        Args:
            keypoints (list): List of [x1, y1, score1, x2, y2, score2, ...]
            img_shape (list): Image dimensions [height, width, channels]
        
        Returns:
            list: Normalized keypoints
        """
        height, width, _ = img_shape
        normalized = []
        
        # Process each triplet [x, y, score]
        for i in range(0, len(keypoints), 3):
            if i + 2 < len(keypoints):
                # Normalize x by width, y by height, keep score as is
                x = keypoints[i] / width
                y = keypoints[i+1] / height
                score = keypoints[i+2]
                
                normalized.extend([x, y, score])
        
        return normalized
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # random add small noise to the sequence
        sequence = self.sequences[idx]
        sequence[0::3] += (torch.rand_like(sequence[0::3]) * 0.02 - 0.01)  # x coordinates: ±0.01
        sequence[1::3] += (torch.rand_like(sequence[1::3]) * 0.02 - 0.01)  # y coordinates: ±0.01
        sequence[2::3] += (torch.rand_like(sequence[2::3]) * 0.2 - 0.1)    # scores: ±0.1
        sequence = sequence.clamp(0, 1)
        return sequence, self.labels[idx]


def create_data_loader(data_dir, batch_size=32, window_size=10, shuffle=True):
    """
    Create a DataLoader for keypoint sequences with fixed window size.
    
    Args:
        data_dir (str): Directory containing JSON files with keypoint data
        batch_size (int): Batch size for training
        window_size (int): Fixed window size for sequences
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        DataLoader: PyTorch DataLoader for training
    """
    dataset = KeypointDataset(data_dir, window_size)
    print(f"Dataset length: {len(dataset)}")
    print(f"Positive samples: {sum(dataset.labels)}")
    print(f"Negative samples: {len(dataset.labels) - sum(dataset.labels)}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )