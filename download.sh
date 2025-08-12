#!/bin/bash

mkdir -p ckpts && cd ckpts
gdown 1j3SzI6Dno9GQVEUUoMmD3KEeyLoNdWom
gdown 1g8RN18tdgXhb5G-zCzgTi7M0KMG_3Al4
gdown 19J3juew8Hco_1dPrKLvk2dhpYv3edhoT
gdown 1vQMj6g8E2suQrs-5OTFEgUWJfFe3BRPT

cd ..
mkdir -p models/MixFormer/models && cd models/MixFormer/models
gdown 1kOOEDi_wA7u0kWKJUbVxEqHv6INIPjgn
cd ..

cd models/VOGUES/
gdown 1exyOvOkueX2c-6e-oFr2YY38r3ImFCn8
mkdir -p pretrained_models && cd pretrained_models 
gdown 1fCNYPSlM06XQnDFCyo_qmuU72OzfUL3l
cd ../../../