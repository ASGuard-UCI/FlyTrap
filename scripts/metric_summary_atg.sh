#!/bin/bash

echo -e "=== Computing the metric for MixFormer under PercepGuard ==="
python analysis/analyze_result_metric.py --file work_dirs/mixformer_percepguard/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for MixFormer under VOGUES ==="
python analysis/analyze_result_metric.py --file work_dirs/mixformer_vogues/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-AlexNet under PercepGuard ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_alex_percepguard/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-AlexNet under VOGUES ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_alex_vogues/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-ResNet under PercepGuard ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_resnet_percepguard/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-ResNet under VOGUES ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_resnet_vogues/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-Mobile under PercepGuard ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_mob_percepguard/json_files/results_epoch-1.json
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-Mobile under VOGUES ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_mob_vogues/json_files/results_epoch-1.json
echo -e "==========================================\n\n"