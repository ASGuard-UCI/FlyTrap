import os
import numpy as np
import json
import seaborn as sns
import re
import matplotlib.pyplot as plt
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze result metrics from JSON files')
    parser.add_argument('--file', type=str,
                       help='Path to the JSON result file to analyze')
    parser.add_argument('--files', type=str, nargs='+',
                       help='Multiple JSON result files to analyze (alternative to --file)')
    return parser.parse_args()





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


def analyze_files(files):
    mASR_all = dict()
    for file in files:
        print(f'Analyzing {file}...')
        area_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        effectiveness = 0
        effectiveness_count = 0
        uni_person = 0
        uni_person_count = 0
        uni_location = 0
        uni_location_count = 0
        uni_person_location = 0
        uni_person_count_location = 0

        # location = []
        # person = []
        
        location = ['bareground', 'grass']
        person = ['person3', 'person4']

        # Extract person and location from the file name
        match = re.search(r'person(\d+)_([^/]+)', file)
        if match:
            person.append(f'person{match.group(1)}')
            location.append(match.group(2))

        # print('parsed person:', person)
        # print('parsed location:', location)

        data = json.load(open(file))
        for video_name, video_data in data.items():
            success_rates = []
            for area_threshold in area_thresholds:
                for score_threshold in score_thresholds:
                    success_rate = _compute_single(video_data, score_threshold, area_threshold)
                    success_rates.append(success_rate)
            mASR = np.mean(success_rates)
            
            mASR_all[file] = success_rates
            
            if (any(person_ in video_name for person_ in person)) and (any(location_ in video_name for location_ in location)):
                effectiveness += mASR
                effectiveness_count += 1
            elif (any(location_ in video_name for location_ in location)):
                uni_person += mASR
                uni_person_count += 1
            elif (any(person_ in video_name for person_ in person)):
                uni_location += mASR
                uni_location_count += 1
            else:
                uni_person_location += mASR
                uni_person_count_location += 1

        # print('Area Threshold:' , area_thresholds)
        # print('Score Threshold:', score_thresholds)
        print('Effectiveness:', round((effectiveness / effectiveness_count) * 100, 2))
        print('Universaity to Location:', round((uni_location / uni_location_count) * 100, 2))
        print('Universaity to Person:', round((uni_person / uni_person_count) * 100, 2))
        print('Universaity to Both:', round((uni_person_location / uni_person_count_location) * 100, 2))


if __name__ == "__main__":
    args = parse_args()
    
    # Determine which files to analyze
    if args.files:
        files = args.files
    elif args.file:
        files = [args.file]
    else:
        print("Error: Please provide either --file or --files argument")
        sys.exit(1)
    
    analyze_files(files)