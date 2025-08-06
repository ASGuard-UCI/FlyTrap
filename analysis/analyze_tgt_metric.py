import os
import numpy as np
import json
import re
import glob
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze target metric results')
parser.add_argument('--input_dir', type=str, required=True, 
                    help='Input directory containing result JSON files')
args = parser.parse_args()

# Define the input directory from command line argument
input_dir = args.input_dir

# Find all files with 'results' in their names but exclude 'benign_results'
result_files = glob.glob(os.path.join(input_dir, '*_results*.json'))
# Filter out files with 'benign_results' in the name
result_files = [f for f in result_files if '_benign_results' not in f]
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
    
    # Initialize lists to store mASR values for each video
    effectiveness_masrs = []
    uni_person_masrs = []
    uni_location_masrs = []
    uni_person_location_masrs = []
    
    # Load data
    data = json.load(open(filepath))
    
    total_frame_count = 0
    total_video_count = 0
    
    for video_name, video_data in data.items():
        frame_count = len(video_data)
        total_frame_count += frame_count
        total_video_count += 1
        
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
            effectiveness_masrs.append(mASR)
        elif video_person != reference_person and video_location == reference_location:
            # Universality to person: different person, same location as reference
            uni_person_masrs.append(mASR)
        elif video_person == reference_person and video_location != reference_location:
            # Universality to location: same person as reference, different location
            uni_location_masrs.append(mASR)
        else:
            # Universality to both: different person and different location from reference
            uni_person_location_masrs.append(mASR)
    
    # Compute file-level averages
    effectiveness_avg = np.mean(effectiveness_masrs) if effectiveness_masrs else 0
    uni_person_avg = np.mean(uni_person_masrs) if uni_person_masrs else 0
    uni_location_avg = np.mean(uni_location_masrs) if uni_location_masrs else 0
    uni_person_location_avg = np.mean(uni_person_location_masrs) if uni_person_location_masrs else 0
    
    return {
        'filepath': filepath,
        'reference_person': reference_person,
        'reference_location': reference_location,
        'total_frames': total_frame_count,
        'total_videos': total_video_count,
        'effectiveness_avg': effectiveness_avg,
        'uni_person_avg': uni_person_avg,
        'uni_location_avg': uni_location_avg,
        'uni_person_location_avg': uni_person_location_avg,
        'effectiveness_count': len(effectiveness_masrs),
        'uni_person_count': len(uni_person_masrs),
        'uni_location_count': len(uni_location_masrs),
        'uni_person_location_count': len(uni_person_location_masrs)
    }

# Process all files
all_results = []
for filepath in result_files:
    result = compute_metrics_for_file(filepath)
    if result is not None:
        all_results.append(result)

# Aggregate results across all files (simple average across files)
file_effectiveness_avgs = []
file_uni_person_avgs = []
file_uni_location_avgs = []
file_uni_person_location_avgs = []

print("\n" + "="*80)
print("INDIVIDUAL FILE RESULTS:")
print("="*80)

for result in all_results:
    print(f"\nFile: {os.path.basename(result['filepath'])}")
    print(f"Reference: {result['reference_person']}, {result['reference_location']}")
    print(f"Total videos: {result['total_videos']}, Total frames: {result['total_frames']}")
    
    if result['effectiveness_count'] > 0:
        effectiveness_rate = result['effectiveness_avg'] * 100
        print(f"Effectiveness: {effectiveness_rate:.1f}% ({result['effectiveness_count']} videos)")
        file_effectiveness_avgs.append(result['effectiveness_avg'])
    else:
        print(f"Effectiveness: No data")
    
    if result['uni_person_count'] > 0:
        uni_person_rate = result['uni_person_avg'] * 100
        print(f"Uni Person: {uni_person_rate:.1f}% ({result['uni_person_count']} videos)")
        file_uni_person_avgs.append(result['uni_person_avg'])
    else:
        print(f"Uni Person: No data")
    
    if result['uni_location_count'] > 0:
        uni_location_rate = result['uni_location_avg'] * 100
        print(f"Uni Location: {uni_location_rate:.1f}% ({result['uni_location_count']} videos)")
        file_uni_location_avgs.append(result['uni_location_avg'])
    else:
        print(f"Uni Location: No data")
    
    if result['uni_person_location_count'] > 0:
        uni_person_location_rate = result['uni_person_location_avg'] * 100
        print(f"Uni Person Location: {uni_person_location_rate:.1f}% ({result['uni_person_location_count']} videos)")
        file_uni_person_location_avgs.append(result['uni_person_location_avg'])
    else:
        print(f"Uni Person Location: No data")

# Compute overall metrics
print("\n" + "="*80)
print("OVERALL RESULTS (simple average across all files):")
print("="*80)

if len(file_effectiveness_avgs) > 0:
    overall_effectiveness = np.mean(file_effectiveness_avgs) * 100
    print(f"Effectiveness: {overall_effectiveness:.2f}% ({len(file_effectiveness_avgs)} total files)")
else:
    print("Effectiveness: No data")

if len(file_uni_person_avgs) > 0:
    overall_uni_person = np.mean(file_uni_person_avgs) * 100
    print(f"Universality to Person: {overall_uni_person:.2f}% ({len(file_uni_person_avgs)} total files)")
else:
    print("Universality to Person: No data")

if len(file_uni_location_avgs) > 0:
    overall_uni_location = np.mean(file_uni_location_avgs) * 100
    print(f"Universality to Location: {overall_uni_location:.2f}% ({len(file_uni_location_avgs)} total files)")
else:
    print("Universality to Location: No data")

if len(file_uni_person_location_avgs) > 0:
    overall_uni_person_location = np.mean(file_uni_person_location_avgs) * 100
    print(f"Universality to Person & Location: {overall_uni_person_location:.2f}% ({len(file_uni_person_location_avgs)} total files)")
else:
    print("Universality to Person & Location: No data")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Total files processed: {len(all_results)}")
print(f"Total file counts: E={len(file_effectiveness_avgs)}, UP={len(file_uni_person_avgs)}, UL={len(file_uni_location_avgs)}, UPL={len(file_uni_person_location_avgs)}")