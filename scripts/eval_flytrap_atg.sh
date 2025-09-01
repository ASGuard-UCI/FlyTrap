#!/bin/bash

# Load environment variables
. env.sh

python tools/main.py config/final/mixformer_percepguard.py
python tools/main.py config/final/mixformer_vogues.py

python tools/main.py config/final/siamrpn_alex_percepguard.py
python tools/main.py config/final/siamrpn_alex_vogues.py

python tools/main.py config/final/siamrpn_resnet_percepguard.py
python tools/main.py config/final/siamrpn_resnet_vogues.py

python tools/main.py config/final/siamrpn_mob_percepguard.py
python tools/main.py config/final/siamrpn_mob_vogues.py