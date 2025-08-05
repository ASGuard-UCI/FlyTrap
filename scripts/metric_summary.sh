#!/bin/bash

echo -e "=== Computing the metric for MixFormer w/o PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/mixformer/json_files/results_epoch-1.json
echo -e "\n## Please refer effectiveness to Table I in the paper (FlyTrap w/o PDP - MixFormer)"
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for MixFormer w/ PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/mixformer_pdp/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/ PDP - MixFormer)\n## Please refer to universality to Table II (MixFormer) in the paper"
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-AlexNet w/o PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_alex/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/o PDP - SiamRPN-AlexNet) in the paper."
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-AlexNet w/ PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_alex_pdp/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/ PDP - SiamRPN-AlexNet)\n## Please refer to universality to Table II (SiamRPN-AlexNet) in the paper."
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-Mobile w/o PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_mob/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/o PDP - SiamRPN-Mobile) in the paper."
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-Mobile w/ PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_mob_pdp/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/ PDP - SiamRPN-Mobile)\n## Please refer to universality to Table II (SiamRPN-Mobile) in the paper."
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-ResNet w/o PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_resnet/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/o PDP - SiamRPN-ResNet) in the paper."
echo -e "==========================================\n\n"

echo -e "=== Computing the metric for SiamRPN-ResNet w/ PDP ==="
python analysis/analyze_result_metric.py --file work_dirs/siamrpn_resnet_pdp/json_files/results_epoch-1.json
echo -e "\n## Please refer to effectiveness to Table I (FlyTrap w/ PDP - SiamRPN-ResNet)\n## Please refer to universality to Table II (SiamRPN-ResNet) in the paper."
echo -e "==========================================\n\n"