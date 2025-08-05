#!/bin/bash

# Load environment variables
. env.sh

# Evaluate FlyTrap w/o PDP against MixFormer
python tools/main.py config/final/mixformer.py

# Evaluate FlyTrap w/ PDP against MixFormer
python tools/main.py config/final/mixformer_pdp.py

# Evaluate FlyTrap w/o PDP against SiamRPN-AlexNet
python tools/main.py config/final/siamrpn_alex.py

# Evaluate FlyTrap w/ PDP against SiamRPN-AlexNet
python tools/main.py config/final/siamrpn_alex_pdp.py

# Evaluate FlyTrap w/o PDP against SiamRPN-Mobile
python tools/main.py config/final/siamrpn_mob.py

# Evaluate FlyTrap w/ PDP against SiamRPN-Mobile
python tools/main.py config/final/siamrpn_mob_pdp.py

# Evaluate FlyTrap w/o PDP against SiamRPN-ResNet
python tools/main.py config/final/siamrpn_resnet.py

# Evaluate FlyTrap w/ PDP against SiamRPN-ResNet
python tools/main.py config/final/siamrpn_resnet_pdp.py