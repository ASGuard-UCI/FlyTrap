cd models/pysot && python setup.py build_ext --inplace
cd ../..
cd models/MixFormer && python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
cd ../..
cd models/AlphaPose && python setup.py build develop
cd ../..
cd models/VOGUES && python setup.py build develop
cd ../..