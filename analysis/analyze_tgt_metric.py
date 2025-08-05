import os
import numpy as np
import json
import seaborn as sns
import re
import matplotlib.pyplot as plt
import glob
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze target metric results')
parser.add_argument('--input_dir', type=str, required=True, 
                    help='Input directory containing result JSON files')
args = parser.parse_args()

# Define the input directory from command line argument
input_dir = args.input_dir

# Find all files with 'results' in their names
result_files = glob.glob(os.path.join(input_dir, '*15.0_results*.json'))
print(f"Found {len(result_files)} result files")

def extract_person_location(filename):
    """Extract person and location from filename"""
    # Expected format: person1_grass1_instance1_scale15.0_benign_results_epoch-1.json
    # or: person1_grass1_instance1_scale15.0_results_epoch-1.json
    basename = os.path.basename(filename)
    
    # Extract person and location
    match = re.search(r'person(\d+)_([a-zA-Z]+)\d*_', basename)
    if match:
        person = f'person{match.group(1)}'
        location = match.group(2)  # Remove numbers from location (e.g., grass1 -> grass)
        return person, location
    else:
        print(f"Warning: Could not parse filename {basename}")
        return None, None

def extract_person_location_from_video_name(video_name):
    """Extract person and location from video name within the data"""
    # Video names might be like: person1_bareground1_instance2, person2_grass1_instance1, etc.
    match = re.search(r'person(\d+)_([a-zA-Z]+)\d*_', video_name)
    if match:
        person = f'person{match.group(1)}'
        location = match.group(2)  # Remove numbers from location
        return person, location
    else:
        return None, None

def _compute_single(data, score_threshold, area_threshold):
    successful_attacks = 0

    for frame in data:
        # Extract target score and bounding boxes
        target_score = frame['out']['target_score']
        out_bbox = frame['out']['target_bbox']
        gt_bbox = frame['gt']['target_bbox']

        # Compute areas of bounding boxes
        out_area = out_bbox[2] * out_bbox[3]
        gt_area = gt_bbox[2] * gt_bbox[3]

        # check center of the bounding box
        x1, y1, w, h = out_bbox
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        x1, y1, w, h = gt_bbox
        x2 = x1 + w
        y2 = y1 + h
        # TODO: [ ] modify the metric: the condition should be the bbox is fully inside the gt_bbox
        if center_x >= x1 and center_x <= x2 and center_y >= y1 and center_y <= y2:
            # Check success conditions
            if target_score > score_threshold and out_area < gt_area * area_threshold:
                successful_attacks += 1

    total_frames = len(data)
    success_rate = successful_attacks / total_frames if total_frames > 0 else 0

    return success_rate

def compute_metrics_for_file(filepath):
    """Compute metrics for a single file using the file's person and location as reference"""
    print(f'Analyzing {filepath}...')
    
    # Extract person and location from filename (this becomes the reference)
    reference_person, reference_location = extract_person_location(filepath)
    if reference_person is None or reference_location is None:
        return None
    
    print(f'  Reference Person: {reference_person}, Reference Location: {reference_location}')
    
    # Define thresholds
    area_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Initialize counters for different metrics
    effectiveness = 0
    effectiveness_count = 0
    uni_person = 0
    uni_person_count = 0
    uni_location = 0
    uni_location_count = 0
    uni_person_location = 0
    uni_person_location_count = 0
    
    # Load data
    data = json.load(open(filepath))
    
    total_frame_count = 0
    
    for video_name, video_data in data.items():
        frame_count = len(video_data)
        total_frame_count += frame_count
        
        # Extract person and location from video name
        video_person, video_location = extract_person_location_from_video_name(video_name)
        if video_person is None or video_location is None:
            print(f"    Warning: Could not parse video name {video_name}")
            continue
        
        # Compute success rates for all threshold combinations
        success_rates = []
        for area_threshold in area_thresholds:
            for score_threshold in score_thresholds:
                success_rate = _compute_single(video_data, score_threshold, area_threshold)
                success_rates.append(success_rate)
        mASR = np.mean(success_rates)
        
        print(f"    Video: {video_name} -> Person: {video_person}, Location: {video_location}, mASR: {mASR:.3f}, Frames: {frame_count}")
        
        # Categorize based on video's person and location vs reference
        if video_person == reference_person and video_location == reference_location:
            # Effectiveness: same person and same location as reference
            effectiveness += mASR * frame_count
            effectiveness_count += frame_count
        elif video_person != reference_person and video_location == reference_location:
            # Universality to person: different person, same location as reference
            uni_person += mASR * frame_count
            uni_person_count += frame_count
        elif video_person == reference_person and video_location != reference_location:
            # Universality to location: same person as reference, different location
            uni_location += mASR * frame_count
            uni_location_count += frame_count
        else:
            # Universality to both: different person and different location from reference
            uni_person_location += mASR * frame_count
            uni_person_location_count += frame_count
    
    return {
        'filepath': filepath,
        'reference_person': reference_person,
        'reference_location': reference_location,
        'total_frames': total_frame_count,
        'effectiveness': effectiveness,
        'effectiveness_count': effectiveness_count,
        'uni_person': uni_person,
        'uni_person_count': uni_person_count,
        'uni_location': uni_location,
        'uni_location_count': uni_location_count,
        'uni_person_location': uni_person_location,
        'uni_person_location_count': uni_person_location_count
    }

# Process all files
all_results = []
for filepath in result_files:
    result = compute_metrics_for_file(filepath)
    if result is not None:
        all_results.append(result)

# Aggregate results across all files (weighted by frame count)
total_effectiveness = 0
total_effectiveness_count = 0
total_uni_person = 0
total_uni_person_count = 0
total_uni_location = 0
total_uni_location_count = 0
total_uni_person_location = 0
total_uni_person_location_count = 0

print("\n" + "="*80)
print("INDIVIDUAL FILE RESULTS:")
print("="*80)

for result in all_results:
    print(f"\nFile: {os.path.basename(result['filepath'])}")
    print(f"Reference: {result['reference_person']}, {result['reference_location']}")
    print(f"Total frames: {result['total_frames']}")
    
    if result['effectiveness_count'] > 0:
        effectiveness_rate = (result['effectiveness'] / result['effectiveness_count']) * 100
        print(f"Effectiveness: {effectiveness_rate:.1f}% ({result['effectiveness_count']} frames)")
    else:
        print(f"Effectiveness: No data")
    
    if result['uni_person_count'] > 0:
        uni_person_rate = (result['uni_person'] / result['uni_person_count']) * 100
        print(f"Uni Person: {uni_person_rate:.1f}% ({result['uni_person_count']} frames)")
    else:
        print(f"Uni Person: No data")
    
    if result['uni_location_count'] > 0:
        uni_location_rate = (result['uni_location'] / result['uni_location_count']) * 100
        print(f"Uni Location: {uni_location_rate:.1f}% ({result['uni_location_count']} frames)")
    else:
        print(f"Uni Location: No data")
    
    if result['uni_person_location_count'] > 0:
        uni_person_location_rate = (result['uni_person_location'] / result['uni_person_location_count']) * 100
        print(f"Uni Person Location: {uni_person_location_rate:.1f}% ({result['uni_person_location_count']} frames)")
    else:
        print(f"Uni Person Location: No data")
    
    # Aggregate for overall calculation
    total_effectiveness += result['effectiveness']
    total_effectiveness_count += result['effectiveness_count']
    total_uni_person += result['uni_person']
    total_uni_person_count += result['uni_person_count']
    total_uni_location += result['uni_location']
    total_uni_location_count += result['uni_location_count']
    total_uni_person_location += result['uni_person_location']
    total_uni_person_location_count += result['uni_person_location_count']

# Compute overall metrics
print("\n" + "="*80)
print("OVERALL RESULTS (weighted by frame count across all files):")
print("="*80)

if total_effectiveness_count > 0:
    overall_effectiveness = (total_effectiveness / total_effectiveness_count) * 100
    print(f"Effectiveness: {overall_effectiveness:.2f}% ({total_effectiveness_count} total frames)")
else:
    print("Effectiveness: No data")

if total_uni_person_count > 0:
    overall_uni_person = (total_uni_person / total_uni_person_count) * 100
    print(f"Universality to Person: {overall_uni_person:.2f}% ({total_uni_person_count} total frames)")
else:
    print("Universality to Person: No data")

if total_uni_location_count > 0:
    overall_uni_location = (total_uni_location / total_uni_location_count) * 100
    print(f"Universality to Location: {overall_uni_location:.2f}% ({total_uni_location_count} total frames)")
else:
    print("Universality to Location: No data")

if total_uni_person_location_count > 0:
    overall_uni_person_location = (total_uni_person_location / total_uni_person_location_count) * 100
    print(f"Universality to Person & Location: {overall_uni_person_location:.2f}% ({total_uni_person_location_count} total frames)")
else:
    print("Universality to Person & Location: No data")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Total files processed: {len(all_results)}")
print(f"Total frame counts: E={total_effectiveness_count}, UP={total_uni_person_count}, UL={total_uni_location_count}, UPL={total_uni_person_location_count}")